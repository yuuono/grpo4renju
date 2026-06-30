"""GRPO training for Renju policy improvement."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RenjuDataset
from .model import RenjuTransformerModel
from .reward import (
    GrpoRewardEvaluator,
    board_is_full,
    current_winner,
)
from .rules import (
    BLACK,
    BOARD_CELLS,
    WHITE,
    board_with_move,
    infer_player,
    move_number,
    winner_after_move,
)
from .rules_tss import legal_move_masks
from .tokenizer import RenjuTokenizer
from .train import build_optimizer
from .utils import (
    JsonlCsvLogger,
    ensure_mlflow_experiment,
    flatten_config,
    get_run_output_dir,
    select_device,
    set_seed,
)


@dataclass(slots=True)
class TrajectoryStep:
    # 軌跡内の1つのステップを表すデータクラス
    board: list[int]  # ボード状態
    action: int  # 実行されたアクション
    actor: int  # アクションを実行したプレイヤー
    old_log_prob: float  # アクションのログ確率
    local_reward: float  # ローカルリワード
    group_index: int  # グループ内のインデックス
    chosen: bool  # このステップで選択されたかどうか
    learn: bool  # GRPO更新に使うステップかどうか


@dataclass(slots=True)
class Trajectory:
    # ポリシーロールアウトで生成された完全な軌跡
    steps: list[TrajectoryStep]  # 軌跡内のステップリスト
    winner: int | None  # ゲームの勝者（Noneはドロー）
    final_board: list[int]  # 最終的なボード状態
    total_reward: float  # 軌跡全体のリワード
    actual_plies: int  # 実際のプレイ数
    policy_player: int | None = None  # この軌跡でpolicyが担当した色


def masked_log_probs(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    """非合法手をマスクしたログ確率を計算する"""
    # 非合法手のロジットを-infで埋め、ログソフトマックスを計算
    # これにより非合法手に0の確率を割り当てる
    return torch.log_softmax(logits.masked_fill(~legal_masks, float("-inf")), dim=-1)


@torch.no_grad()
def sample_actions_from_logits(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    sample_count: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """masked logitsから合法手だけを温度付きでサンプリングする共通処理"""
    # 温度チェック
    if temperature <= 0:
        raise ValueError("grpo.temperature must be positive.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    # 非合法手をマスクして温度でスケーリング
    masked_logits = logits.masked_fill(~legal_masks, float("-inf")) / temperature
    # ソフトマックスで確率に変換
    probabilities = torch.softmax(masked_logits, dim=-1)
    sampled_actions: list[torch.Tensor] = []
    # 局面ごとに合法手数が違うため、置換あり/なしを行単位で決める
    for row_index in range(probabilities.size(0)):
        legal_count = int(legal_masks[row_index].sum().item())
        if legal_count <= 0:
            raise ValueError("No legal moves available.")
        replacement = legal_count < sample_count
        sampled_actions.append(
            torch.multinomial(
                probabilities[row_index],
                num_samples=sample_count,
                replacement=replacement,
            )
        )
    actions = torch.stack(sampled_actions, dim=0)
    # サンプリングしたアクションのログ確率を計算
    log_probs = torch.log(probabilities.gather(dim=-1, index=actions).clamp_min(1e-12))
    # アクションとログ確率を返す
    return actions, log_probs


def build_legal_masks(boards: list[list[int]], device: torch.device) -> torch.Tensor:
    """複数のボード状態について合法手マスクをバッチで生成する"""
    # tss.so が利用可能なら複数局面をまとめてC++に渡す
    masks = legal_move_masks(boards)
    # テンソルに変換して返す
    return torch.tensor(masks, dtype=torch.bool, device=device)


def normalize_group_advantages(rewards: torch.Tensor, epsilon: float) -> torch.Tensor:
    """グループ内でリワードを正規化してアドバンテージを計算する"""
    # グループ内での平均をカウント
    mean = rewards.mean(dim=-1, keepdim=True)
    # グループ内での標準偏差を計算
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    # (報酬 - 平均) / (標準偏差 + epsilon) で正規化（advantage に変換）
    return (rewards - mean) / (std + epsilon)


def compute_flat_grpo_loss(
    policy_model: RenjuTransformerModel,
    reference_model: RenjuTransformerModel,
    input_ids: torch.Tensor,
    legal_masks: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """flatなサンプル列に対してGRPO/PPO風lossを計算する共通実装"""
    # gatherで使える形にするため、アクションを列ベクトルへ整形する
    flat_actions = actions.reshape(-1, 1)
    # 旧方策のlog-probを1次元に平坦化する
    flat_old_log_probs = old_log_probs.reshape(-1)
    # アドバンテージを1次元に平坦化する
    flat_advantages = advantages.reshape(-1)

    # 現方策モデルで各入力のロジットを計算する
    logits = policy_model(input_ids)
    # 合法手マスクを適用した行動log-prob分布を作る
    log_probs = masked_log_probs(logits, legal_masks)
    # 実際に選択された行動のlog-probを取り出す
    selected_log_probs = log_probs.gather(dim=-1, index=flat_actions).squeeze(-1)

    # 参照方策側の値は勾配不要なのでno_gradで計算する
    with torch.no_grad():
        # 参照モデルで各入力のロジットを計算する
        reference_logits = reference_model(input_ids)
        # 参照モデルの合法手付きlog-prob分布を作る
        reference_log_probs = masked_log_probs(reference_logits, legal_masks)
        # 参照モデルで選択行動のlog-probを取り出す
        selected_reference_log_probs = reference_log_probs.gather(
            dim=-1,
            index=flat_actions,
        ).squeeze(-1)

    # PPO比率 r = exp(new_log_prob - old_log_prob) を計算する
    ratio = torch.exp(selected_log_probs - flat_old_log_probs)
    # PPOクリップ範囲で比率を制限する
    clipped_ratio = ratio.clamp(
        1.0 - float(cfg.grpo.clip_epsilon),
        1.0 + float(cfg.grpo.clip_epsilon),
    )
    # クリップ版surrogateと通常surrogateの小さい方を使ってpolicy lossを計算する
    policy_loss = -torch.minimum(ratio * flat_advantages, clipped_ratio * flat_advantages).mean()
    # 参照方策とのlog比を計算する
    log_ratio_to_ref = selected_reference_log_probs - selected_log_probs
    # 参照方策に対する近似KL項を計算する
    kl = (torch.exp(log_ratio_to_ref) - log_ratio_to_ref - 1.0).mean()
    # 合法手上の確率分布を作る（非合法手は0にする）
    probabilities = torch.exp(log_probs).masked_fill(~legal_masks, 0.0)
    # エントロピー計算用に非合法手のlog-probを0へ置き換える
    safe_log_probs = log_probs.masked_fill(~legal_masks, 0.0)
    # 方策エントロピー（探索促進項）を計算する
    entropy = -(probabilities * safe_log_probs).sum(dim=-1).mean()
    # 最終loss = policy_loss + KL重み - entropy重み で合成する
    loss = policy_loss + float(cfg.grpo.kl_beta) * kl - float(cfg.grpo.entropy_coef) * entropy
    # 損失テンソルとログ用メトリクス辞書を返す
    return loss, {
        # 合成後の総損失
        "loss": float(loss.detach().cpu()),
        # PPO主体のpolicy損失
        "policy_loss": float(policy_loss.detach().cpu()),
        # 参照方策とのKL
        "kl": float(kl.detach().cpu()),
        # 方策エントロピー
        "entropy": float(entropy.detach().cpu()),
    }


def final_reward_for_actor(winner: int | None, actor: int, cfg: DictConfig) -> float:
    """ゲーム終了時のプレイヤーのターミナルリワードを計算する"""
    # 勝者がいない場合（ドロー）はドロー報酬を返す
    if winner is None:
        return float(cfg.grpo.step_group.draw_reward)
    # アクターが勝った場合は最終結果重みを返す
    if winner == actor:
        return float(cfg.grpo.step_group.final_result_weight)
    # アクターが負けた場合は負の最終結果重みを返す
    return -float(cfg.grpo.step_group.final_result_weight)


def discounted_final_bonus(
    winner: int | None,
    actor: int,
    group_index: int,
    actual_plies: int,
    cfg: DictConfig,
) -> float:
    """ゲーム終了時のリワードを時間割引係数で減衰させたボーナスを計算する
    終局勝敗ボーナスを時間割引して返す。
    group_index の手が終局から何手前かを distance として計算し、
    gamma^distance を掛けることで credit assignment （各行動の貢献度をどう割り当てるか）を調整する。
        - gamma = 1.0: 割引なし（全手に同じ強さで伝播）
        - gamma < 1.0: 終局に近い手ほど強く、序盤の手ほど弱く伝播
    最終的な符号（勝ち正・負け負・引き分け）は final_reward_for_actor 側で決定する。
    """
    # 割引係数を計算（時間経過に応じた減衰）
    # distance = アクションが選択されてからの経過ステップ数
    distance = max(actual_plies - group_index - 1, 0)
    # ガンマ係数のdistance乗で割引を適用
    return (float(cfg.grpo.step_group.gamma) ** distance) * final_reward_for_actor(winner, actor, cfg)


@torch.no_grad()
def sample_policy_actions(
    model: RenjuTransformerModel,
    tokenizer: RenjuTokenizer,
    board: list[int] | list[list[int]] | None,
    sample_count: int,
    temperature: float,
    device: torch.device,
    *,
    input_ids: torch.Tensor | None = None,
    legal_masks: torch.Tensor | None = None,
    return_tensors: bool = False,
) -> tuple[list[int], list[float]] | tuple[list[list[int]], list[list[float]]] | tuple[torch.Tensor, torch.Tensor]:
    """現在局面からon-policyで指定数のアクションをサンプリングする"""
    # 既存呼び出しのため、単局面/複数局面のboard指定を受け付ける。
    is_batch = bool(board) and isinstance(board[0], list) if board is not None else True
    boards = board if is_batch else [board] if board is not None else None
    # ボード状態をモデルの入力形式にエンコード。state objectiveでは既存input_idsを使う。
    if input_ids is None:
        if boards is None:
            raise ValueError("board or input_ids must be provided.")
        input_ids = torch.stack([tokenizer.encode_input(single_board) for single_board in boards]).to(device)
    else:
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    # 合法手のマスクを作成。state objectiveでは既存legal_masksを使う。
    if legal_masks is None:
        if boards is None:
            raise ValueError("board or legal_masks must be provided.")
        legal_masks = build_legal_masks(boards, device)
    else:
        legal_masks = legal_masks.to(device)
    # モデルでロジットを生成
    logits = model(input_ids)
    actions, log_probs = sample_actions_from_logits(
        logits,
        legal_masks,
        sample_count=sample_count,
        temperature=temperature,
    )
    if return_tensors:
        return actions, log_probs
    # アクションをPythonリストに変換
    action_rows = [[int(action) for action in row] for row in actions.detach().cpu().tolist()]
    # 各アクションの対数確率を取得
    old_log_prob_rows = [[float(value) for value in row] for row in log_probs.detach().cpu().tolist()]
    if is_batch:
        return action_rows, old_log_prob_rows
    return action_rows[0], old_log_prob_rows[0]


def choose_candidate_index(rewards: list[float], cfg: DictConfig) -> int:
    """候補アクションの中からリワードに基づいて1つを選択する"""
    # リワードリストが空でないかチェック
    if not rewards:
        raise ValueError("Cannot choose from an empty candidate list.")
    # 選択方式を取得
    selection = str(cfg.grpo.step_group.action_selection)
    # ベスト方式：最高リワードの手を選択
    if selection == "best":
        return max(range(len(rewards)), key=lambda index: rewards[index])
    # ソフトマックス方式：確率に基づき選択
    if selection == "softmax":
        # 温度パラメータを取得
        temperature = float(cfg.grpo.step_group.selection_temperature)
        # 温度がポジティブでないかチェック
        if temperature <= 0:
            raise ValueError("grpo.step_group.selection_temperature must be positive.")
        # リワードをテンソルに変換
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        # ソフトマックスで確率を計算
        probabilities = torch.softmax(reward_tensor / temperature, dim=0)
        # 確率分布からインデックスをサンプリング
        return int(torch.multinomial(probabilities, num_samples=1).item())
    # サポートされていない方式の場合はエラー
    raise ValueError(f"Unsupported grpo.step_group.action_selection: {selection}")


def log_grpo_step_metrics(
    step_metrics_logger: JsonlCsvLogger,
    cfg: DictConfig,
    global_step: int,
    local_metrics: dict[str, object],
    mlflow_metrics: dict[str, float],
) -> None:
    """ステップ単位のJSONL/CSVログと間隔付きMLflowログをまとめて記録する"""
    # ステップ単位メトリクスをJSONL/CSVへ記録する
    step_metrics_logger.log(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **local_metrics,
        }
    )

    # 所定間隔ごとにMLflowへ主要メトリクスを送信する
    if global_step % int(cfg.grpo.log_every_steps) == 0:
        for metric_name, metric_value in mlflow_metrics.items():
            mlflow.log_metric(metric_name, metric_value, step=global_step)


def resolve_policy_players(cfg: DictConfig, rollout_index: int | None = None) -> list[int | None]:
    """trajectory設定からpolicy担当色を決める。

    rollout_index=Noneの場合はtrajectory_group用に生成すべきpolicy担当色リストを返す。
    rollout_index指定時はstep_group用に、そのrolloutで使うpolicy担当色を1要素リストで返す。
    Noneは黒白どちらの手番もpolicyの学習対象にすることを表す。
    """
    learning_player = str(cfg.grpo.step_group.get("learning_player", "both"))
    opponent = str(cfg.grpo.step_group.get("opponent", "self"))
    if opponent not in {"self", "reference"}:
        raise ValueError(f"Unsupported grpo.step_group.opponent: {opponent}")

    if learning_player == "black":
        return [BLACK]
    if learning_player == "white":
        return [WHITE]
    if learning_player != "both":
        raise ValueError(f"Unsupported grpo.step_group.learning_player: {learning_player}")

    if rollout_index is None:
        return [BLACK, WHITE]
    if opponent == "self":
        return [None]
    return [BLACK if rollout_index % 2 == 0 else WHITE]


@torch.no_grad()
def rollout_policy_episode(
    start_board: list[int],
    policy_player: int | None,
    policy_model: RenjuTransformerModel,
    reference_model: RenjuTransformerModel,
    tokenizer: RenjuTokenizer,
    reward_evaluator: GrpoRewardEvaluator,
    cfg: DictConfig,
    device: torch.device,
    *,
    sample_count: int,
    default_opponent: str,
    reward_mode: str,
) -> Trajectory:
    """policy/referenceモデルでepisodeを進め、学習用Trajectoryを作る共通処理"""
    # 開始局面を破壊しないようにコピーして作業用盤面を作成する
    board = start_board.copy()
    # 軌跡中の各手を格納する配列を初期化する
    steps: list[TrajectoryStep] = []
    # 開始局面時点で既に勝敗が付いているかを確認する
    winner = current_winner(board)
    # このロールアウトで許可する最大手数を設定から取得する
    max_plies = int(cfg.grpo.step_group.max_plies)
    # 実際に進めた手数カウンタを0で初期化する
    actual_plies = 0
    # 相手方の挙動モード（self/reference）を設定から取得する
    opponent = str(cfg.grpo.step_group.get("opponent", default_opponent))
    # 未対応のopponent設定値は早期にエラーにする
    if opponent not in {"self", "reference"}:
        raise ValueError(f"Unsupported grpo.step_group.opponent: {opponent}")

    # 未終局かつ盤面が埋まっておらず、最大手数未満の間だけ1手ずつ進める
    while winner is None and not board_is_full(board) and actual_plies < max_plies:
        try:
            # 現在盤面から手番プレイヤー（黒/白）を推定する
            actor = infer_player(board)
            # この手がpolicy担当色の学習対象手かどうかを判定する
            # """この手番をpolicyの学習対象として扱うかを返す。"""
            learn = policy_player is None or actor == policy_player
            # 学習対象手またはself対戦時はpolicy、それ以外はreferenceで着手を生成する
            acting_model = policy_model if learn or opponent == "self" else reference_model
            # 現在局面から指定数の候補手をサンプリングし、その旧log-probも取得する
            candidate_actions, old_log_probs = sample_policy_actions(
                acting_model,
                tokenizer,
                board,
                sample_count=sample_count,
                temperature=float(cfg.grpo.temperature),
                device=device,
            )
        except ValueError:
            # 合法手が無いなどで着手不能ならロールアウトを終了する
            break

        # 各候補アクションのローカルリワードを計算する
        local_rewards = reward_evaluator.evaluate_batch([board], [candidate_actions])[0]
        # 候補が1手ならそのまま、複数候補なら報酬に基づいて実際に進める手を選ぶ
        chosen_index = 0 if len(candidate_actions) == 1 else choose_candidate_index(local_rewards, cfg)
        # 選択されたアクションを取得する
        chosen_action = candidate_actions[chosen_index]
        # 各候補アクションに対してTrajectoryStepを作成する
        for action_index, (action, old_log_prob, local_reward) in enumerate(
            zip(candidate_actions, old_log_probs, local_rewards, strict=True)
        ):
            steps.append(
                TrajectoryStep(
                    board=board.copy(),
                    action=action,
                    actor=actor,
                    old_log_prob=old_log_prob,
                    local_reward=float(local_reward) * float(cfg.grpo.step_group.tss_weight),
                    group_index=actual_plies,
                    chosen=action_index == chosen_index,
                    learn=learn,
                )
            )

        # 選択した手を盤面へ反映して次局面へ遷移する
        board = board_with_move(board, chosen_action, actor)
        # 直前の着手で勝敗が決まったかを更新する
        winner = winner_after_move(board, chosen_action, actor)
        # 実際に進めた手数を1増やす
        actual_plies += 1

    # ループ終了後も未終局なら現在盤面から最終勝者を再判定する
    if winner is None:
        winner = current_winner(board)

    if reward_mode == "trajectory_group":
        if policy_player is None:
            raise ValueError("trajectory_group reward requires a concrete policy_player.")
        # 学習対象手のみのローカル報酬を合計し、終局結果に基づく最終報酬を加算する
        total_reward = sum(step.local_reward for step in steps if step.learn)
        total_reward += final_reward_for_actor(winner, policy_player, cfg)
    elif reward_mode == "step_group":
        # 実際に選ばれた学習対象手のローカル報酬と、割引済み終局ボーナスを合計する
        total_reward = sum(step.local_reward for step in steps if step.chosen and step.learn)
        total_reward += sum(
            discounted_final_bonus(winner, step.actor, step.group_index, actual_plies, cfg)
            for step in steps
            if step.chosen and step.learn
        )
    else:
        raise ValueError(f"Unsupported rollout reward_mode: {reward_mode}")

    # 収集した軌跡情報をTrajectoryとして返す
    return Trajectory(
        # ステップ列
        steps=steps,
        # 最終勝者
        winner=winner,
        # 最終盤面
        final_board=board,
        # 軌跡総報酬
        total_reward=total_reward,
        # 実際に進めた手数
        actual_plies=actual_plies,
        # policy担当色
        policy_player=policy_player,
    )

def select_episode_starts(
    episode: list[list[int]],
    start_positions: str,
    min_start_ply: int,
    max_start_ply: int,
) -> list[list[int]]:
    """エピソード内からプレイ番号の条件に合う開始位置を選択する"""
    # エピソードが空の場合は空リストを返す
    if not episode:
        return []
    # 指定範囲内のプレイ番号を持つボード状態をフィルタリング
    eligible = [
        board
        for board in episode
        # プレイ番号が最小と最大の範囲内かをチェック
        if min_start_ply <= move_number(board) <= max_start_ply
    ]
    # "first"の場合は最初の1つだけ返す
    if start_positions == "first":
        return eligible[:1] if eligible else []
    # "all"の場合は全て返す
    if start_positions == "all":
        return eligible
    # "early"の場合は最初の1つだけ返す
    if start_positions == "early":
        return eligible[:1] if eligible else []
    # 指定されていない方式の場合はエラー
    raise ValueError(f"Unsupported grpo.step_group.start_positions: {start_positions}")


def reconstruct_episode_start_boards(dataset: RenjuDataset, cfg: DictConfig) -> list[list[int]]:
    """データセットからゲームエピソードを復元してロールアウト開始位置を抽出する"""
    # エピソード開始位置のボード状態を格納
    start_boards: list[list[int]] = []
    # 現在処理中のエピソード内のボード状態を格納
    current_episode: list[list[int]] = []
    # 前回のプレイ番号を記録
    previous_ply: int | None = None
    # エピソード開始位置の最大プレイ番号を取得
    max_start_ply = int(cfg.grpo.step_group.max_start_ply)
    # エピソード開始位置の最小プレイ番号を取得
    min_start_ply = int(cfg.grpo.step_group.min_start_ply)
    # エピソード開始位置の選択方法を取得
    start_positions = str(cfg.grpo.step_group.start_positions)

    # データセットの全サンプルを処理
    for input_ids, _label in dataset.samples:
        # 最後の要素を除外してボード状態を取得
        board = input_ids[:-1].tolist()
        # ボード状態から現在のプレイ番号を計算
        ply = move_number(board)
        # プレイ番号が前回より小さい場合は新しいエピソード開始
        if previous_ply is not None and ply <= previous_ply:
            # 現在のエピソードから開始位置を選択してリストに追加
            start_boards.extend(
                select_episode_starts(current_episode, start_positions, min_start_ply, max_start_ply)
            )
            # 新しいエピソード用にリセット
            current_episode = []
        # ボード状態を現在のエピソードに追加
        current_episode.append(board)
        # 前回のプレイ番号を更新
        previous_ply = ply

    # 最後のエピソードを処理
    if current_episode:
        # 最後のエピソードから開始位置を選択してリストに追加
        start_boards.extend(
            select_episode_starts(current_episode, start_positions, min_start_ply, max_start_ply)
        )

    # 重複を削除するかチェック
    if bool(cfg.grpo.step_group.deduplicate):
        # 既に見たボード状態をセットで管理
        seen: set[tuple[int, ...]] = set()
        # 重複削除後のボード状態リスト
        deduplicated: list[list[int]] = []
        # 各ボード状態を処理
        for board in start_boards:
            # ボード状態をタプルに変換（ハッシュ化用）
            key = tuple(board)
            # 既に見たボード状態の場合はスキップ
            if key in seen:
                continue
            # 新しいボード状態をセットに追加
            seen.add(key)
            # 重複削除リストに追加
            deduplicated.append(board)
        # 重複削除版に置き換え
        start_boards = deduplicated

    # 最大開始位置数の制限をチェック
    max_positions = int(cfg.grpo.step_group.max_start_positions)
    # 制限がある場合は超過分をカット
    if max_positions > 0 and len(start_boards) > max_positions:
        start_boards = start_boards[:max_positions]

    # 開始位置がない場合は空のボード状態を追加（開始局面）
    if not start_boards:
        start_boards.append([0] * BOARD_CELLS)
    # 開始位置リストを返す
    return start_boards

def build_start_prompts(dataset: RenjuDataset, cfg: DictConfig) -> list[list[int]]:
    """ロールアウト開始位置（プロンプト）をモードに応じて構築する"""
    # 開始局面のソースを取得
    source = str(cfg.grpo.step_group.source)
    # "self_play"の場合は空のボード状態だけを返す
    if source == "self_play":
        return [[0] * BOARD_CELLS]
    # "dataset"の場合はデータセットから開始位置を抽出
    if source == "dataset":
        return reconstruct_episode_start_boards(dataset, cfg)
    # "mixed"の場合は空のボード状態とデータセット開始位置を組み合わせ
    if source == "mixed":
        return [[0] * BOARD_CELLS] + reconstruct_episode_start_boards(dataset, cfg)
    # 指定されていないソースの場合はエラー
    raise ValueError(f"Unsupported grpo.step_group.source: {source}")


def save_grpo_checkpoint(
    model: RenjuTransformerModel,
    cfg: DictConfig,
    output_path: Path,
    epoch: int,
    global_step: int,
    mean_reward: float,
) -> None:
    """訓練済みモデルと設定をチェックポイントファイルに保存する"""
    # チェックポイント辞書を構築
    checkpoint = {
        # モデルの重みをCPU上に移動してコピー
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        # 設定を辞書形式で保存
        "config": OmegaConf.to_container(cfg, resolve=True),
        # エポック番号
        "epoch": epoch,
        # グローバルステップ数
        "global_step": global_step,
        # 平均リワード
        "mean_reward": mean_reward,
    }
    # チェックポイントをファイルに保存
    torch.save(checkpoint, output_path)


def maybe_save_step_checkpoint(
    model: RenjuTransformerModel,
    cfg: DictConfig,
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
    mean_reward: float,
) -> Path | None:
    """指定ステップごとにモデルチェックポイントを条件付きで保存する"""
    # 何ステップごとにチェックポイントを保存するかを取得
    save_every_steps = cfg.grpo.get("save_every_steps")
    # 設定されていない、または0以下の場合は保存しない
    if save_every_steps is None or int(save_every_steps) <= 0:
        return None
    # グローバルステップ数が0以下、または保存間隔に達していない場合は保存しない
    if global_step <= 0 or global_step % int(save_every_steps) != 0:
        return None

    # チェックポイントファイルのパスを構築
    output_path = checkpoint_dir / f"step_{global_step:06d}_grpo_model.pt"
    # チェックポイントを保存
    save_grpo_checkpoint(
        model,
        cfg,
        output_path,
        epoch,
        global_step,
        mean_reward,
    )
    # チェックポイントパスを出力
    print(f"step_grpo_checkpoint={output_path.resolve()}", flush=True)
    # 保存されたパスを返す
    return output_path

# -------------------------------------------------------------------------------------------# 
# ------------- objective trajectory-group GRPO training loop -------------# 
# -------------------------------------------------------------------------------------------# 

def flatten_trajectories(
    trajectories: list[Trajectory],
    cfg: DictConfig,
    *,
    mode: str,
) -> tuple[list[list[int]], list[int], list[float], list[float]]:
    """trajectory群を指定モードのadvantage計算で学習用サンプルへ平坦化する"""
    boards: list[list[int]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    all_advantages: list[float] = []

    if mode == "trajectory_group":
        # 同一起点・同一policy色のtrajectory群を、trajectory総報酬で正規化する。
        learnable_trajectories = [
            trajectory
            for trajectory in trajectories
            if any(step.learn for step in trajectory.steps)
        ]
        if not learnable_trajectories:
            return [], [], [], []
        rewards = torch.tensor(
            [[trajectory.total_reward for trajectory in learnable_trajectories]],
            dtype=torch.float32,
        )
        advantages = normalize_group_advantages(
            rewards,
            epsilon=float(cfg.grpo.advantage_epsilon),
        ).squeeze(0)
        for trajectory, advantage in zip(learnable_trajectories, advantages.tolist(), strict=True):
            for step in trajectory.steps:
                if not step.learn:
                    continue
                boards.append(step.board)
                actions.append(step.action)
                old_log_probs.append(step.old_log_prob)
                all_advantages.append(float(advantage))
        return boards, actions, old_log_probs, all_advantages

    if mode == "step_group":
        # 各手番group内で、候補手の局所報酬+選択手の割引終局ボーナスを正規化する。
        for trajectory in trajectories:
            group_indexes = sorted({step.group_index for step in trajectory.steps})
            for group_index in group_indexes:
                group_steps = [
                    step
                    for step in trajectory.steps
                    if step.group_index == group_index and step.learn
                ]
                if not group_steps:
                    continue
                values = [
                    step.local_reward
                    + (
                        discounted_final_bonus(
                            trajectory.winner,
                            step.actor,
                            step.group_index,
                            trajectory.actual_plies,
                            cfg,
                        )
                        if step.chosen
                        else 0.0
                    )
                    for step in group_steps
                ]
                advantages = normalize_group_advantages(
                    torch.tensor(values, dtype=torch.float32).unsqueeze(0),
                    epsilon=float(cfg.grpo.advantage_epsilon),
                ).squeeze(0)
                for step, advantage in zip(group_steps, advantages.tolist(), strict=True):
                    boards.append(step.board)
                    actions.append(step.action)
                    old_log_probs.append(step.old_log_prob)
                    all_advantages.append(float(advantage))
        return boards, actions, old_log_probs, all_advantages

    raise ValueError(f"Unsupported flatten_trajectories mode: {mode}")


def train_trajectory_group_grpo_loop(
    cfg: DictConfig,
    dataset: RenjuDataset,
    tokenizer: RenjuTokenizer,
    policy_model: RenjuTransformerModel,
    reference_model: RenjuTransformerModel,
    optimizer: torch.optim.Optimizer,
    reward_evaluator: GrpoRewardEvaluator,
    device: torch.device,
    step_metrics_logger: JsonlCsvLogger,
    epoch_metrics_logger: JsonlCsvLogger,
    latest_checkpoint_path: Path,
    best_checkpoint_path: Path,
) -> None:
    """同一開始局面から複数trajectoryを生成し、trajectory単位の相対報酬でGRPO更新する"""
    # データセット/設定からロールアウト開始局面候補（プロンプト）を作成する
    prompts = build_start_prompts(dataset, cfg)
    # 開始局面が1つも作れなかった場合は学習を継続できないため例外を送出する
    if not prompts:
        raise ValueError("No trajectory_group prompts were built.")

    # 1開始局面あたりに生成するtrajectory本数（比較グループサイズ）を取得する
    trajectory_group_size = int(cfg.grpo.trajectory_group.group_size)
    # 相対比較を成立させるため、グループサイズが2未満なら例外を送出する
    if trajectory_group_size < 2:
        raise ValueError("grpo.trajectory_group.group_size must be at least 2 for group-relative advantages.")

    # 学習対象とするpolicy担当色（黒/白/両方）の一覧を設定から決定する
    policy_players = resolve_policy_players(cfg)
    # 全体更新ステップ数カウンタを初期化する
    global_step = 0
    # ベスト報酬を最小値で初期化する
    best_reward = float("-inf")
    # プロンプト巡回用カーソルを初期化する
    prompt_cursor = 0
    # 最大更新ステップ数設定を取得する（Noneなら無制限）
    max_steps = cfg.grpo.max_steps
    # 1エポックあたりの内部ステップ数を決定する
    steps_per_epoch = int(max_steps) if max_steps is not None else max(1, len(prompts))
    # 1内部ステップで処理する開始局面数を取得する
    prompts_per_step = int(cfg.grpo.step_group.prompts_per_step)

    # 指定エポック数だけ学習ループを回す
    for epoch in range(1, int(cfg.grpo.max_epochs) + 1):
        # エポック進捗を表示するtqdmプログレスバーを作成する
        progress = tqdm(
            range(steps_per_epoch),
            desc=f"Trajectory-group GRPO epoch {epoch}/{cfg.grpo.max_epochs}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        # このエポックで観測した平均報酬を蓄積するリスト
        epoch_rewards: list[float] = []
        # このエポックで観測した損失を蓄積するリスト
        epoch_losses: list[float] = []

        # エポック内の各更新ステップを処理する
        for _ in progress:
            # 最大ステップ上限に達していればエポック内ループを打ち切る
            if max_steps is not None and global_step >= int(max_steps):
                break

            # 学習用に平坦化した盤面配列を初期化する
            flat_boards: list[list[int]] = []
            # 学習用に平坦化した行動配列を初期化する
            flat_actions: list[int] = []
            # 学習用に平坦化した旧log-prob配列を初期化する
            flat_old_log_probs: list[float] = []
            # 学習用に平坦化したadvantage配列を初期化する
            flat_advantages: list[float] = []
            # グループ単位の総報酬を記録する配列を初期化する
            group_scores: list[float] = []
            # 各trajectoryの実手数を記録する配列を初期化する
            rollout_lengths: list[int] = []

            # この更新ステップのTSS統計をリセットする
            reward_evaluator.reset_stats()
            # 生成フェーズなのでpolicyを評価モードに切り替える
            policy_model.eval()
            # 参照モデルは常に評価モードで使用する
            reference_model.eval()

            # 設定された個数の開始局面をこの更新ステップで処理する
            for _prompt_index in range(prompts_per_step):
                # カーソルを循環利用して開始局面を1つ選ぶ
                prompt = prompts[prompt_cursor % len(prompts)]
                # 次回に向けてカーソルを進める
                prompt_cursor += 1
                # 各policy担当色ごとにtrajectoryグループを作る
                for policy_player in policy_players:
                    # 同じ開始局面・同じ担当色で複数trajectoryを生成する
                    trajectories = [
                        rollout_policy_episode(
                            prompt,
                            policy_player,
                            policy_model,
                            reference_model,
                            tokenizer,
                            reward_evaluator,
                            cfg,
                            device,
                            sample_count=1,
                            default_opponent="reference",
                            reward_mode="trajectory_group",
                        )
                        for _group_index in range(trajectory_group_size)
                    ]
                    # trajectory報酬で正規化した学習サンプルに平坦化する
                    group_boards, group_actions, group_old_log_probs, group_advantages = flatten_trajectories(
                        trajectories,
                        cfg,
                        mode="trajectory_group",
                    )
                    # 学習対象手が無い場合はこのグループをスキップする
                    if not group_advantages:
                        continue

                    # 盤面サンプルを全体バッファへ追加する
                    flat_boards.extend(group_boards)
                    # 行動サンプルを全体バッファへ追加する
                    flat_actions.extend(group_actions)
                    # 旧log-probを全体バッファへ追加する
                    flat_old_log_probs.extend(group_old_log_probs)
                    # advantageを全体バッファへ追加する
                    flat_advantages.extend(group_advantages)
                    # 生成した各trajectoryの総報酬を記録する
                    group_scores.extend([trajectory.total_reward for trajectory in trajectories])
                    # 生成した各trajectoryの長さを記録する
                    rollout_lengths.extend([trajectory.actual_plies for trajectory in trajectories])

            # 学習サンプルが1件も無ければこの更新ステップを飛ばす
            if not flat_boards:
                continue
            # ここまでのTSS統計を取得する
            tss_stats = reward_evaluator.consume_stats()

            # 学習フェーズに入るためpolicyを訓練モードに戻す
            policy_model.train()
            # 盤面をモデル入力テンソルへ変換する
            input_ids = torch.stack([tokenizer.encode_input(board) for board in flat_boards]).to(device)
            # 各盤面の合法手マスクを作成する
            legal_masks = build_legal_masks(flat_boards, device)
            # 行動をLongTensorへ変換する
            actions = torch.tensor(flat_actions, dtype=torch.long, device=device)
            # 旧log-probをFloatTensorへ変換する
            old_log_probs = torch.tensor(flat_old_log_probs, dtype=torch.float32, device=device)
            # advantageをFloatTensorへ変換する
            advantages = torch.tensor(flat_advantages, dtype=torch.float32, device=device)

            # 更新メトリクス格納先を初期化する
            update_metrics: dict[str, float] = {}
            # PPO風の複数更新エポックを回す
            for _ in range(int(cfg.grpo.update_epochs)):
                # 既存勾配をクリアする
                optimizer.zero_grad(set_to_none=True)
                # 現在ミニバッチのGRPO損失とメトリクスを計算する
                loss, update_metrics = compute_flat_grpo_loss(
                    policy_model,
                    reference_model,
                    input_ids,
                    legal_masks,
                    actions,
                    old_log_probs,
                    advantages,
                    cfg,
                )
                # 逆伝播で勾配を計算する
                loss.backward()
                # 設定が有効なら勾配クリッピングを適用する
                if cfg.train.gradient_clip_norm is not None and cfg.train.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        policy_model.parameters(),
                        cfg.train.gradient_clip_norm,
                    )
                # オプティマイザでパラメータを更新する
                optimizer.step()

            # 更新ステップ数を1進める
            global_step += 1
            # 今回グループの平均報酬を計算する
            reward_mean = sum(group_scores) / len(group_scores)
            # 今回グループの最小報酬を計算する
            reward_min = min(group_scores)
            # 今回グループの最大報酬を計算する
            reward_max = max(group_scores)
            # 今回グループの報酬標準偏差を計算する
            reward_std = float(torch.tensor(group_scores, dtype=torch.float32).std(unbiased=False))
            # 今回trajectory長の平均を計算する
            length_mean = sum(rollout_lengths) / len(rollout_lengths)
            # エポック報酬ログへ追加する
            epoch_rewards.append(reward_mean)
            # エポック損失ログへ追加する
            epoch_losses.append(update_metrics["loss"])

            # ステップ保存条件を満たせばチェックポイントを保存する
            step_checkpoint_path = maybe_save_step_checkpoint(
                policy_model,
                cfg,
                latest_checkpoint_path.parent,
                epoch,
                global_step,
                reward_mean,
            )

            # ステップ単位の詳細ログとMLflow主要ログをまとめて記録する
            log_grpo_step_metrics(
                step_metrics_logger,
                cfg,
                global_step,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_size": prompts_per_step,
                    "group_size": trajectory_group_size,
                    "trajectory_group_length_mean": length_mean,
                    "sample_count": len(flat_boards),
                    "tss_deep_count": tss_stats["tss_deep_count"],
                    "tss_batch_ms": tss_stats["tss_batch_ms"],
                    "tss_batch_calls": tss_stats["tss_batch_calls"],
                    "tss_thread_count": tss_stats["tss_thread_count"],
                    "step_checkpoint": str(step_checkpoint_path.resolve()) if step_checkpoint_path else "",
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "reward_min": reward_min,
                    "reward_max": reward_max,
                    "advantage_mean": float(advantages.mean().detach().cpu()),
                    "advantage_std": float(advantages.std(unbiased=False).detach().cpu()),
                    "loss": update_metrics["loss"],
                    "policy_loss": update_metrics["policy_loss"],
                    "kl": update_metrics["kl"],
                    "entropy": update_metrics["entropy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                {
                    "trajectory_group_reward": reward_mean,
                    "trajectory_group_length": length_mean,
                    "grpo_loss": update_metrics["loss"],
                    "grpo_kl": update_metrics["kl"],
                },
            )

            # プログレスバー表示用の要約値を更新する
            progress.set_postfix(
                reward=f"{reward_mean:.4f}",
                len=f"{length_mean:.1f}",
                loss=f"{update_metrics['loss']:.4f}",
                kl=f"{update_metrics['kl']:.4f}",
            )

        # エポックのプログレスバーを閉じる
        progress.close()
        # 1件も更新できなかった場合は学習ループを終了する
        if not epoch_rewards:
            break

        # エポック平均報酬を計算する
        epoch_mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        # エポック平均損失を計算する
        epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
        # エポック平均報酬をMLflowへ記録する
        mlflow.log_metric("trajectory_group_epoch_reward", epoch_mean_reward, step=epoch)
        # エポック平均損失をMLflowへ記録する
        mlflow.log_metric("trajectory_group_epoch_loss", epoch_mean_loss, step=epoch)

        # 最新チェックポイントを毎エポック保存する
        save_grpo_checkpoint(
            policy_model,
            cfg,
            latest_checkpoint_path,
            epoch,
            global_step,
            epoch_mean_reward,
        )
        # ベスト報酬更新時はベストチェックポイントを保存する
        if epoch_mean_reward > best_reward:
            # ベスト報酬値を更新する
            best_reward = epoch_mean_reward
            # ベストモデルを保存する
            save_grpo_checkpoint(
                policy_model,
                cfg,
                best_checkpoint_path,
                epoch,
                global_step,
                epoch_mean_reward,
            )
            # ベスト報酬をMLflowへ記録する
            mlflow.log_metric("trajectory_group_best_reward", best_reward, step=epoch)

        # エポック集計メトリクスをJSONL/CSVへ記録する
        epoch_metrics_logger.log(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "global_step": global_step,
                "reward_mean": epoch_mean_reward,
                "loss_mean": epoch_mean_loss,
                "best_reward": best_reward,
                "latest_checkpoint": str(latest_checkpoint_path.resolve()),
                "best_checkpoint": str(best_checkpoint_path.resolve()) if best_checkpoint_path.exists() else "",
            }
        )

        # 端末へエポック要約を出力する
        print(
            f"epoch={epoch} "
            f"trajectory_group_reward={epoch_mean_reward:.4f} "
            f"trajectory_group_loss={epoch_mean_loss:.4f} "
            f"best_reward={best_reward:.4f}",
            flush=True,
        )

        # グローバル最大ステップに到達したら学習を終了する
        if cfg.grpo.max_steps is not None and global_step >= int(cfg.grpo.max_steps):
            break

# ------------------------------------------------------------------------------# 
# ------------- objective step-group GRPO training loop -------------# 
# ------------------------------------------------------------------------------# 

def train_step_group_grpo_loop(
    cfg: DictConfig,
    dataset: RenjuDataset,
    tokenizer: RenjuTokenizer,
    policy_model: RenjuTransformerModel,
    reference_model: RenjuTransformerModel,
    optimizer: torch.optim.Optimizer,
    reward_evaluator: GrpoRewardEvaluator,
    device: torch.device,
    step_metrics_logger: JsonlCsvLogger,
    epoch_metrics_logger: JsonlCsvLogger,
    latest_checkpoint_path: Path,
    best_checkpoint_path: Path,
) -> None:
    """step_group objectiveのGRPO訓練ループを実行する"""
    # ロールアウト開始位置を構築
    prompts = build_start_prompts(dataset, cfg)
    # プロンプトが存在するか確認
    if not prompts:
        raise ValueError("No step_group prompts were built.")

    # グローバルステップカウンタを初期化
    global_step = 0
    # 最高リワードを初期化
    best_reward = float("-inf")
    # プロンプト選択用のカーソルを初期化
    prompt_cursor = 0
    # 最大ステップ数を取得
    max_steps = cfg.grpo.max_steps
    # エポック内のステップ数を計算
    steps_per_epoch = int(max_steps) if max_steps is not None else max(1, len(prompts))
    # 1ステップで処理するプロンプト数を取得
    prompts_per_step = int(cfg.grpo.step_group.prompts_per_step)
    # グループサイズを取得
    group_size = int(cfg.grpo.group_size)

    # 各エポック処理
    for epoch in range(1, int(cfg.grpo.max_epochs) + 1):
        # プログレスバーを作成
        progress = tqdm(
            range(steps_per_epoch),
            desc=f"Step-group GRPO epoch {epoch}/{cfg.grpo.max_epochs}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        # エポック内のリワードを記録
        epoch_rewards: list[float] = []
        # エポック内のロスを記録
        epoch_losses: list[float] = []

        # エポック内の各ステップを処理
        for _ in progress:
            # 最大ステップ数に達した場合はブレーク
            if max_steps is not None and global_step >= int(max_steps):
                break

            # フラット化されたボード、アクション、対数確率、アドバンテージ
            flat_boards: list[list[int]] = []
            flat_actions: list[int] = []
            flat_old_log_probs: list[float] = []
            flat_advantages: list[float] = []
            # グループのリワード（軌跡全体のリワード）
            group_scores: list[float] = []
            # 軌跡のプレイ数
            rollout_lengths: list[int] = []

            # リワード評価器の統計情報をリセット
            reward_evaluator.reset_stats()
            # ポリシーモデルを評価モードに設定
            policy_model.eval()
            reference_model.eval()
            # 指定数のプロンプトを処理
            for _prompt_index in range(prompts_per_step):
                # プロンプトリストからプロンプトを選択
                prompt = prompts[prompt_cursor % len(prompts)]
                rollout_index = prompt_cursor
                policy_player = resolve_policy_players(cfg, rollout_index=rollout_index)[0]
                # カーソルをインクリメント
                prompt_cursor += 1
                # ポリシーロールアウトで軌跡を生成
                trajectories = [
                    rollout_policy_episode(
                        prompt,
                        policy_player,
                        policy_model,
                        reference_model,
                        tokenizer,
                        reward_evaluator,
                        cfg,
                        device,
                        sample_count=int(cfg.grpo.group_size),
                        default_opponent="self",
                        reward_mode="step_group",
                    )
                ]
                # 軌跡グループをフラット化
                group_boards, group_actions, group_old_log_probs, group_advantages = flatten_trajectories(
                    trajectories,
                    cfg,
                    mode="step_group",
                )
                # アドバンテージがない場合はスキップ
                if not group_advantages:
                    continue

                # フラット化データをバッチに追加
                flat_boards.extend(group_boards)
                flat_actions.extend(group_actions)
                flat_old_log_probs.extend(group_old_log_probs)
                flat_advantages.extend(group_advantages)
                # 軌跡のリワードと長さを記録
                group_scores.extend([trajectory.total_reward for trajectory in trajectories])
                rollout_lengths.extend([trajectory.actual_plies for trajectory in trajectories])

            # ボードが無い場合はスキップ
            if not flat_boards:
                continue
            # リワード評価器の統計情報を取得
            tss_stats = reward_evaluator.consume_stats()

            # ポリシーモデルを訓練モードに設定
            policy_model.train()
            # ボード状態をモデル入力形式にエンコード
            input_ids = torch.stack([tokenizer.encode_input(board) for board in flat_boards]).to(device)
            # 合法手マスクを構築
            legal_masks = build_legal_masks(flat_boards, device)
            # アクションをテンソル化
            actions = torch.tensor(flat_actions, dtype=torch.long, device=device)
            # 対数確率をテンソル化
            old_log_probs = torch.tensor(flat_old_log_probs, dtype=torch.float32, device=device)
            # アドバンテージをテンソル化
            advantages = torch.tensor(flat_advantages, dtype=torch.float32, device=device)

            # ロス計算用の辞書を初期化
            update_metrics: dict[str, float] = {}
            # 複数エポック更新を実行
            for _ in range(int(cfg.grpo.update_epochs)):
                # 勾配をリセット
                optimizer.zero_grad(set_to_none=True)
                # flatなtrajectoryサンプルに対してGRPOロスを計算
                loss, update_metrics = compute_flat_grpo_loss(
                    policy_model,
                    reference_model,
                    input_ids,
                    legal_masks,
                    actions,
                    old_log_probs,
                    advantages,
                    cfg,
                )
                # 逆伝播でロスを計算
                loss.backward()
                # 勾配クリップを適用
                if cfg.train.gradient_clip_norm is not None and cfg.train.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        policy_model.parameters(),
                        cfg.train.gradient_clip_norm,
                    )
                # オプティマイザーでステップを実行
                optimizer.step()

            # グローバルステップをインクリメント
            global_step += 1
            # リワード平均を計算
            reward_mean = sum(group_scores) / len(group_scores)
            # リワード最小値を取得
            reward_min = min(group_scores)
            # リワード最大値を取得
            reward_max = max(group_scores)
            # リワード標準偏差を計算
            reward_std = float(torch.tensor(group_scores, dtype=torch.float32).std(unbiased=False))
            # 軌跡長平均を計算
            length_mean = sum(rollout_lengths) / len(rollout_lengths)
            # エポックリワード記録に追加
            epoch_rewards.append(reward_mean)
            # エポックロス記録に追加
            epoch_losses.append(update_metrics["loss"])
            # ステップチェックポイント保存を試行
            step_checkpoint_path = maybe_save_step_checkpoint(
                policy_model,
                cfg,
                latest_checkpoint_path.parent,
                epoch,
                global_step,
                reward_mean,
            )

            # ステップ単位の詳細ログとMLflow主要ログをまとめて記録する
            log_grpo_step_metrics(
                step_metrics_logger,
                cfg,
                global_step,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_size": prompts_per_step,
                    "group_size": group_size,
                    "step_group_length_mean": length_mean,
                    "sample_count": len(flat_boards),
                    "tss_deep_count": tss_stats["tss_deep_count"],
                    "tss_batch_ms": tss_stats["tss_batch_ms"],
                    "tss_batch_calls": tss_stats["tss_batch_calls"],
                    "tss_thread_count": tss_stats["tss_thread_count"],
                    "step_checkpoint": str(step_checkpoint_path.resolve()) if step_checkpoint_path else "",
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "reward_min": reward_min,
                    "reward_max": reward_max,
                    "advantage_mean": float(advantages.mean().detach().cpu()),
                    "advantage_std": float(advantages.std(unbiased=False).detach().cpu()),
                    "loss": update_metrics["loss"],
                    "policy_loss": update_metrics["policy_loss"],
                    "kl": update_metrics["kl"],
                    "entropy": update_metrics["entropy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                {
                    "step_group_reward": reward_mean,
                    "step_group_length": length_mean,
                    "grpo_loss": update_metrics["loss"],
                    "grpo_kl": update_metrics["kl"],
                },
            )

            progress.set_postfix(
                reward=f"{reward_mean:.4f}",
                len=f"{length_mean:.1f}",
                loss=f"{update_metrics['loss']:.4f}",
                kl=f"{update_metrics['kl']:.4f}",
            )

        # プログレスバーを閉じる
        progress.close()
        # エポックリワード記録が空の場合はブレーク
        if not epoch_rewards:
            break

        # エポック平均リワードを計算
        epoch_mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        # エポック平均ロスを計算
        epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
        mlflow.log_metric("step_group_epoch_reward", epoch_mean_reward, step=epoch)
        mlflow.log_metric("step_group_epoch_loss", epoch_mean_loss, step=epoch)

        # 最新チェックポイントを保存
        save_grpo_checkpoint(
            policy_model,
            cfg,
            latest_checkpoint_path,
            epoch,
            global_step,
            epoch_mean_reward,
        )
        # 新しい最高リワードの場合
        if epoch_mean_reward > best_reward:
            # 最高リワードを更新
            best_reward = epoch_mean_reward
            # 最高チェックポイントを保存
            save_grpo_checkpoint(
                policy_model,
                cfg,
                best_checkpoint_path,
                epoch,
                global_step,
                epoch_mean_reward,
            )
            mlflow.log_metric("step_group_best_reward", best_reward, step=epoch)

        # エポックメトリクスをログに記録
        epoch_metrics_logger.log(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "global_step": global_step,
                "reward_mean": epoch_mean_reward,
                "loss_mean": epoch_mean_loss,
                "best_reward": best_reward,
                "latest_checkpoint": str(latest_checkpoint_path.resolve()),
                "best_checkpoint": str(best_checkpoint_path.resolve())
                if best_checkpoint_path.exists()
                else "",
            }
        )
        print(
            f"epoch={epoch} "
            f"step_group_reward={epoch_mean_reward:.4f} "
            f"step_group_loss={epoch_mean_loss:.4f} "
            f"best_reward={best_reward:.4f}",
            flush=True,
        )

        if max_steps is not None and global_step >= int(max_steps):
            break



# ----------------------------------------------------------------------------------------------------------------------------------
# ------------- objective state GRPO training loop ------------------------------------------------------------------------------# 
# ----------------------------------------------------------------------------------------------------------------------------------

def boards_from_input_ids(input_ids: torch.Tensor) -> list[list[int]]:
    """入力IDからボード状態を抽出する"""
    # 入力IDから最後の行を除いてボード状態を抽出
    # 最後の1要素はセパレータトークンなので除外
    return [row[:-1].detach().cpu().tolist() for row in input_ids]

def compute_rewards(
    boards: list[list[int]],
    actions: torch.Tensor,
    reward_evaluator: GrpoRewardEvaluator,
) -> torch.Tensor:
    """複数のアクションについてTSS/ルール由来の局面リワードだけを計算する"""
    # GPUテンソルをCPU上のPythonリストに変換
    sampled_actions = actions.detach().cpu().tolist()
    # リワード評価器でローカルリワードを計算
    local_rewards = reward_evaluator.evaluate_batch(boards, sampled_actions)
    # リワードをテンソルに変換して返す
    return torch.tensor(local_rewards, dtype=torch.float32, device=actions.device)

def train_state_grpo_loop(
    cfg: DictConfig,
    dataloader: DataLoader,
    tokenizer: RenjuTokenizer,
    policy_model: RenjuTransformerModel,
    reference_model: RenjuTransformerModel,
    optimizer: torch.optim.Optimizer,
    reward_evaluator: GrpoRewardEvaluator,
    device: torch.device,
    step_metrics_logger: JsonlCsvLogger,
    epoch_metrics_logger: JsonlCsvLogger,
    latest_checkpoint_path: Path,
    best_checkpoint_path: Path,
) -> None:
    """state objectiveのGRPO訓練ループを実行する"""
    # 全体更新ステップ数カウンタを初期化する
    global_step = 0
    # ベスト報酬を最小値で初期化する
    best_reward = float("-inf")

    # 指定エポック数だけ学習ループを回す
    for epoch in range(1, int(cfg.grpo.max_epochs) + 1):
        # データローダーを反復するプログレスバーを作成する
        progress = tqdm(
            dataloader,
            desc=f"GRPO epoch {epoch}/{cfg.grpo.max_epochs}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        # このエポックで観測した平均報酬を蓄積するリスト
        epoch_rewards: list[float] = []
        # このエポックで観測した損失を蓄積するリスト
        epoch_losses: list[float] = []

        # データローダーからバッチを順に取り出して処理する
        for input_ids, _labels in progress:
            # 最大ステップ上限に達していればエポック内ループを打ち切る
            if cfg.grpo.max_steps is not None and global_step >= int(cfg.grpo.max_steps):
                break

            # 入力IDを学習デバイスへ転送する
            input_ids = input_ids.to(device)
            # 入力IDから各局面の盤面配列を復元する
            boards = boards_from_input_ids(input_ids)
            # 各局面に対する合法手マスクを生成する
            legal_masks = build_legal_masks(boards, device)

            # サンプリングと報酬計算は勾配不要のためno_gradで実行する
            with torch.no_grad():
                # グループ行動とその旧log-probをサンプリングする
                actions, old_log_probs = sample_policy_actions(
                    policy_model,
                    tokenizer,
                    boards,
                    sample_count=int(cfg.grpo.group_size),
                    temperature=float(cfg.grpo.temperature),
                    device=device,
                    input_ids=input_ids,
                    legal_masks=legal_masks,
                    return_tensors=True,
                )
                # 報酬評価の統計をリセットする
                reward_evaluator.reset_stats()
                # TSS/ルール由来の局面リワードだけを計算する
                rewards = compute_rewards(
                    boards,
                    actions,
                    reward_evaluator,
                )
                # 報酬評価統計を取得する
                tss_stats = reward_evaluator.consume_stats()
                # グループ内報酬を正規化してadvantageを作る
                advantages = normalize_group_advantages(
                    rewards,
                    epsilon=float(cfg.grpo.advantage_epsilon),
                )

            # 更新メトリクス格納先を初期化する
            update_metrics: dict[str, float] = {}
            # state objectiveは1局面に複数候補を持つので、loss入力をflatサンプル列へ変換する
            group_size = actions.shape[1]
            flat_input_ids = input_ids.repeat_interleave(group_size, dim=0)
            flat_legal_masks = legal_masks.repeat_interleave(group_size, dim=0)
            flat_actions = actions.reshape(-1)
            flat_old_log_probs = old_log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            # PPO風の複数更新エポックを回す
            for _ in range(int(cfg.grpo.update_epochs)):
                # 既存勾配をクリアする
                optimizer.zero_grad(set_to_none=True)
                # 現在バッチのGRPO損失とメトリクスを計算する
                loss, update_metrics = compute_flat_grpo_loss(
                    policy_model,
                    reference_model,
                    flat_input_ids,
                    flat_legal_masks,
                    flat_actions,
                    flat_old_log_probs,
                    flat_advantages,
                    cfg,
                )
                # 逆伝播で勾配を計算する
                loss.backward()
                # 設定が有効なら勾配クリッピングを適用する
                if cfg.train.gradient_clip_norm is not None and cfg.train.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        policy_model.parameters(),
                        cfg.train.gradient_clip_norm,
                    )
                # オプティマイザでパラメータを更新する
                optimizer.step()

            # 更新ステップ数を1進める
            global_step += 1
            # バッチ報酬平均を計算する
            mean_reward = float(rewards.mean().detach().cpu())
            # バッチ報酬標準偏差を計算する
            reward_std = float(rewards.std(unbiased=False).detach().cpu())
            # バッチ報酬最小値を計算する
            reward_min = float(rewards.min().detach().cpu())
            # バッチ報酬最大値を計算する
            reward_max = float(rewards.max().detach().cpu())
            # バッチadvantage平均を計算する
            advantage_mean = float(advantages.mean().detach().cpu())
            # バッチadvantage標準偏差を計算する
            advantage_std = float(advantages.std(unbiased=False).detach().cpu())
            # エポック報酬ログへ追加する
            epoch_rewards.append(mean_reward)
            # エポック損失ログへ追加する
            epoch_losses.append(update_metrics["loss"])
            # ステップ保存条件を満たせばチェックポイントを保存する
            step_checkpoint_path = maybe_save_step_checkpoint(
                policy_model,
                cfg,
                latest_checkpoint_path.parent,
                epoch,
                global_step,
                mean_reward,
            )
            # ステップ単位の詳細ログとMLflow主要ログをまとめて記録する
            log_grpo_step_metrics(
                step_metrics_logger,
                cfg,
                global_step,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_size": int(input_ids.size(0)),
                    "group_size": int(cfg.grpo.group_size),
                    "rollout_length_mean": None,
                    "sample_count": int(actions.numel()),
                    "tss_deep_count": tss_stats["tss_deep_count"],
                    "tss_batch_ms": tss_stats["tss_batch_ms"],
                    "tss_batch_calls": tss_stats["tss_batch_calls"],
                    "tss_thread_count": tss_stats["tss_thread_count"],
                    "step_checkpoint": str(step_checkpoint_path.resolve()) if step_checkpoint_path else "",
                    "reward_mean": mean_reward,
                    "reward_std": reward_std,
                    "reward_min": reward_min,
                    "reward_max": reward_max,
                    "advantage_mean": advantage_mean,
                    "advantage_std": advantage_std,
                    "loss": update_metrics["loss"],
                    "policy_loss": update_metrics["policy_loss"],
                    "kl": update_metrics["kl"],
                    "entropy": update_metrics["entropy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                {
                    "grpo_reward": mean_reward,
                    "grpo_loss": update_metrics["loss"],
                    "grpo_policy_loss": update_metrics["policy_loss"],
                    "grpo_kl": update_metrics["kl"],
                    "grpo_entropy": update_metrics["entropy"],
                },
            )
            # プログレスバー表示用の要約値を更新する
            progress.set_postfix(
                reward=f"{mean_reward:.4f}",
                loss=f"{update_metrics['loss']:.4f}",
                kl=f"{update_metrics['kl']:.4f}",
            )

        # エポックのプログレスバーを閉じる
        progress.close()
        # 1件も更新できなかった場合は学習ループを終了する
        if not epoch_rewards:
            break

        # エポック平均報酬を計算する
        epoch_mean_reward = sum(epoch_rewards) / len(epoch_rewards)
        # エポック平均損失を計算する
        epoch_mean_loss = sum(epoch_losses) / len(epoch_losses)
        # エポック平均報酬をMLflowへ記録する
        mlflow.log_metric("grpo_epoch_reward", epoch_mean_reward, step=epoch)
        # エポック平均損失をMLflowへ記録する
        mlflow.log_metric("grpo_epoch_loss", epoch_mean_loss, step=epoch)

        # 最新チェックポイントを毎エポック保存する
        save_grpo_checkpoint(
            policy_model,
            cfg,
            latest_checkpoint_path,
            epoch,
            global_step,
            epoch_mean_reward,
        )
        # ベスト報酬更新時はベストチェックポイントを保存する
        if epoch_mean_reward > best_reward:
            # ベスト報酬値を更新する
            best_reward = epoch_mean_reward
            # ベストモデルを保存する
            save_grpo_checkpoint(
                policy_model,
                cfg,
                best_checkpoint_path,
                epoch,
                global_step,
                epoch_mean_reward,
            )
            # ベスト報酬をMLflowへ記録する
            mlflow.log_metric("grpo_best_reward", best_reward, step=epoch)

        # エポック集計メトリクスをJSONL/CSVへ記録する
        epoch_metrics_logger.log(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "global_step": global_step,
                "reward_mean": epoch_mean_reward,
                "loss_mean": epoch_mean_loss,
                "best_reward": best_reward,
                "latest_checkpoint": str(latest_checkpoint_path.resolve()),
                "best_checkpoint": str(best_checkpoint_path.resolve()) if best_checkpoint_path.exists() else "",
            }
        )

        # 端末へエポック要約を出力する
        print(
            f"epoch={epoch} "
            f"grpo_reward={epoch_mean_reward:.4f} "
            f"grpo_loss={epoch_mean_loss:.4f} "
            f"best_reward={best_reward:.4f}",
            flush=True,
        )

        # グローバル最大ステップに到達したら学習を終了する
        if cfg.grpo.max_steps is not None and global_step >= int(cfg.grpo.max_steps):
            break

# ----------------------------------------------------------------------------------------------------------------------------------
# ------------- GRPO training objective分岐と共通初期化 -------------# 
# ----------------------------------------------------------------------------------------------------------------------------------

def build_grpo_step_logger(run_output_dir: Path) -> JsonlCsvLogger:
    """ステップごとの詳細メトリクスをログする出力ロガーを作成する関数"""
    # ステップレベルのメトリクスログを作成
    return JsonlCsvLogger(
        # JSONLファイルのパス
        jsonl_path=run_output_dir / "metrics" / "grpo_steps.jsonl",
        # CSVファイルのパス
        csv_path=run_output_dir / "metrics" / "grpo_steps.csv",
        # ログするフィールド名
        fieldnames=[
            "timestamp",
            "epoch",
            "global_step",
            "batch_size",
            "group_size",
            "trajectory_group_length_mean",
            "step_group_length_mean",
            "rollout_length_mean",
            "sample_count",
            "tss_deep_count",
            "tss_batch_ms",
            "tss_batch_calls",
            "tss_thread_count",
            "step_checkpoint",
            "reward_mean",
            "reward_std",
            "reward_min",
            "reward_max",
            "advantage_mean",
            "advantage_std",
            "loss",
            "policy_loss",
            "kl",
            "entropy",
            "learning_rate",
        ],
    )

def build_grpo_epoch_logger(run_output_dir: Path) -> JsonlCsvLogger:
    """エポック単位の集計メトリクスをログする出力ロガーを作成する関数"""
    # エポックレベルのメトリクスログを作成
    return JsonlCsvLogger(
        # JSONLファイルのパス
        jsonl_path=run_output_dir / "metrics" / "grpo_epochs.jsonl",
        # CSVファイルのパス
        csv_path=run_output_dir / "metrics" / "grpo_epochs.csv",
        # ログするフィールド名
        fieldnames=[
            "timestamp",
            "epoch",
            "global_step",
            "reward_mean",
            "loss_mean",
            "best_reward",
            "latest_checkpoint",
            "best_checkpoint",
        ],
    )

def build_model_from_checkpoint(checkpoint: dict, cfg: DictConfig) -> RenjuTransformerModel:
    """チェックポイントから設定を復元してモデルを構築する"""
    # チェックポイントから設定を取得
    checkpoint_config = checkpoint.get("config")
    # チェックポイントのモデル設定、またはデフォルト設定を使用
    model_cfg = checkpoint_config["model"] if checkpoint_config is not None else cfg.model
    # モデル設定からRenjuTransformerModelを構築
    return RenjuTransformerModel(
        vocab_size=model_cfg["token_vocab_size"],  # トークン語彙サイズ
        max_seq_len=model_cfg["max_seq_len"],  # 最大シーケンス長
        d_model=model_cfg["d_model"],  # モデル次元
        nhead=model_cfg["nhead"],  # アテンション頭数
        num_layers=model_cfg["num_layers"],  # トランスフォーマー層数
        dim_feedforward=model_cfg["dim_feedforward"],  # フィードフォワード層の次元
        dropout=model_cfg["dropout"],  # ドロップアウト率
        activation=model_cfg["activation"],  # 活性化関数
        norm_first=model_cfg["norm_first"],  # 正規化を先行させるか
        num_move_labels=model_cfg["num_move_labels"],  # 手のラベル数
    )

def load_policy_and_reference(
    cfg: DictConfig,
    device: torch.device,
) -> tuple[RenjuTransformerModel, RenjuTransformerModel, dict]:
    """チェックポイントからポリシーモデルと参照モデルをロードする"""
    # チェックポイントパスを取得
    checkpoint_path = cfg.grpo.checkpoint_path
    # チェックポイントが指定されているか確認
    if not checkpoint_path:
        raise ValueError("Set grpo.checkpoint_path to a supervised checkpoint before GRPO training.")

    # チェックポイントをロード
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # ポリシーモデルを構築
    policy_model = build_model_from_checkpoint(checkpoint, cfg).to(device)
    # 参照モデルを構築
    reference_model = build_model_from_checkpoint(checkpoint, cfg).to(device)
    # ポリシーモデルの重みをロード
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    # 参照モデルの重みをロード
    reference_model.load_state_dict(checkpoint["model_state_dict"])
    # 参照モデルを評価モードに設定
    reference_model.eval()
    # 参照モデルの勾配計算を無効化
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    # ポリシーモデル、参照モデル、チェックポイントを返す
    return policy_model, reference_model, checkpoint


def train_grpo(cfg: DictConfig) -> None:
    """GRPO訓練を実行する（トラジェクトリベースまたは状態ベースの選択肢がある）"""
    # ランダムシード を設定
    set_seed(cfg.seed)
    # 実行出力ディレクトリを取得
    run_output_dir = get_run_output_dir()
    # ステップメトリクスロガーを構築
    step_metrics_logger = build_grpo_step_logger(run_output_dir)
    # エポックメトリクスロガーを構築
    epoch_metrics_logger = build_grpo_epoch_logger(run_output_dir)
    # トークナイザーを初期化
    tokenizer = RenjuTokenizer(
        sep_token_id=cfg.data.sep_token_id,
        move_id_offset=cfg.data.move_id_offset,
    )
    # データセットを読み込み
    dataset = RenjuDataset(cfg.data.path, tokenizer=tokenizer, max_rows=cfg.data.max_rows)
    # データローダーを構築
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.grpo.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )

    # デバイスを選択
    device = select_device(cfg.train.device)
    # ポリシーモデルと参照モデルをロード
    policy_model, reference_model, _ = load_policy_and_reference(cfg, device)
    # ポリシーモデルを訓練モードに設定
    policy_model.train()
    # オプティマイザーを構築
    optimizer = build_optimizer(policy_model, cfg)
    # リワード評価器を初期化
    reward_evaluator = GrpoRewardEvaluator(cfg)

    # 出力ルートディレクトリを作成
    output_root = Path(cfg.train.output_root)
    # チェックポイントディレクトリを作成
    checkpoint_dir = output_root / cfg.grpo.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # 設定ディレクトリを作成
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    # 解決済み設定ファイルパスを指定
    resolved_config_path = config_dir / "resolved_grpo_config.yaml"
    # 設定をYAMLファイルに保存
    OmegaConf.save(cfg, resolved_config_path, resolve=True)

    # MLflow実験をセットアップ（省略）
    ensure_mlflow_experiment(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        artifact_root=cfg.mlflow.artifact_root,
    )
    # MLflowの実験を設定（省略）
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # MLflow実行名を生成（省略）
    run_name = f"{cfg.mlflow.run_name_prefix}-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # 最新チェックポイントパスを指定
    latest_checkpoint_path = checkpoint_dir / cfg.grpo.checkpoint_name
    # 最高チェックポイントパスを指定
    best_checkpoint_path = checkpoint_dir / cfg.grpo.best_checkpoint_name

    # MLflowの実行コンテキストで処理（省略）
    with mlflow.start_run(run_name=run_name):
        # MLflowにパラメータをログ（省略）
        mlflow.log_params(flatten_config(cfg))
        # MLflowに設定ファイルをアップロード（省略）
        mlflow.log_artifact(str(resolved_config_path), artifact_path="configs")

        objective = str(cfg.grpo.objective)

        # 目的関数に応じた訓練を実行
        if objective == "trajectory_group":
            train_trajectory_group_grpo_loop(
                cfg,
                dataset,
                tokenizer,
                policy_model,
                reference_model,
                optimizer,
                reward_evaluator,
                device,
                step_metrics_logger,
                epoch_metrics_logger,
                latest_checkpoint_path,
                best_checkpoint_path,
            )
            mlflow.log_artifact(str(latest_checkpoint_path), artifact_path="checkpoints")
            if best_checkpoint_path.exists():
                mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
            if cfg.mlflow.log_model:
                policy_model.eval()
                mlflow.pytorch.log_model(policy_model, name="trajectory_group_grpo_model")
            print(f"grpo_checkpoint={latest_checkpoint_path.resolve()}", flush=True)
            if best_checkpoint_path.exists():
                print(f"best_grpo_checkpoint={best_checkpoint_path.resolve()}", flush=True)
            return

        if objective == "step_group":
            # ステップ候補比較型GRPO訓練を実行
            train_step_group_grpo_loop(
                cfg,
                dataset,
                tokenizer,
                policy_model,
                reference_model,
                optimizer,
                reward_evaluator,
                device,
                step_metrics_logger,
                epoch_metrics_logger,
                latest_checkpoint_path,
                best_checkpoint_path,
            )
            # MLflowに最新チェックポイントをアップロード（省略）
            mlflow.log_artifact(str(latest_checkpoint_path), artifact_path="checkpoints")
            # 最高チェックポイントが存在する場合はアップロード（省略）
            if best_checkpoint_path.exists():
                mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
            # MLflowにモデルをログ（省略）
            if cfg.mlflow.log_model:
                policy_model.eval()
                mlflow.pytorch.log_model(policy_model, name="step_group_grpo_model")
            # チェックポイントパスを出力（省略）
            print(f"grpo_checkpoint={latest_checkpoint_path.resolve()}", flush=True)
            # 最高チェックポイントが存在する場合は出力（省略）
            if best_checkpoint_path.exists():
                print(f"best_grpo_checkpoint={best_checkpoint_path.resolve()}", flush=True)
            # 処理を終了
            return

        # 目的関数が"state"でない場合はエラー
        if objective != "state":
            raise ValueError(f"Unsupported grpo.objective: {cfg.grpo.objective}")

        train_state_grpo_loop(
            cfg,
            dataloader,
            tokenizer,
            policy_model,
            reference_model,
            optimizer,
            reward_evaluator,
            device,
            step_metrics_logger,
            epoch_metrics_logger,
            latest_checkpoint_path,
            best_checkpoint_path,
        )

        # MLflowに最新チェックポイントをアップロード（省略）
        mlflow.log_artifact(str(latest_checkpoint_path), artifact_path="checkpoints")
        # 最高チェックポイントが存在する場合はアップロード（省略）
        if best_checkpoint_path.exists():
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
        # MLflowにモデルをログ（省略）
        if cfg.mlflow.log_model:
            policy_model.eval()
            mlflow.pytorch.log_model(policy_model, name="grpo_model")

    # チェックポイントパスを出力（省略）
    print(f"grpo_checkpoint={latest_checkpoint_path.resolve()}", flush=True)
    # 最高チェックポイントが存在する場合は出力（省略）
    if best_checkpoint_path.exists():
        print(f"best_grpo_checkpoint={best_checkpoint_path.resolve()}", flush=True)
