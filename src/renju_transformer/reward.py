"""Reward helpers for GRPO policy improvement."""

from __future__ import annotations

import ctypes
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .rules import (
    BLACK,
    BOARD_CELLS,
    EMPTY,
    WHITE,
    board_winner,
    board_with_move,
    count_four_directions,
    count_open_three_directions,
    infer_player,
    is_forbidden_for_black,
    legal_move_mask,
    winner_after_move,
)


def other_player(player: int) -> int:
    # 相手プレイヤーを返す（黒なら白、白なら黒）
    return WHITE if player == BLACK else BLACK


def legal_moves_for_player(board: list[int], player: int) -> list[int]:
    # 指定プレイヤーの合法手を全て取得
    if player == infer_player(board):
        # 現在のプレイヤーの場合、legal_move_maskを使用
        return [index for index, is_legal in enumerate(legal_move_mask(board)) if is_legal]

    # 別のプレイヤーの場合の合法手を検索
    legal_moves: list[int] = []
    for index, cell in enumerate(board):
        # 空いているマスを確認
        if cell != EMPTY:
            continue
        # 黒の場合、禁止手を除外
        if player == BLACK and is_forbidden_for_black(board, index):
            continue
        # 合法手をリストに追加
        legal_moves.append(index)
    return legal_moves


def immediate_winning_moves(board: list[int], player: int) -> list[int]:
    # 指定プレイヤーが即座に勝つことができる手をすべて取得
    winning_moves: list[int] = []
    # 合法手を列挙
    for move in legal_moves_for_player(board, player):
        # この手を打った後のボード状態を計算
        next_board = board_with_move(board, move, player)
        # この手で勝つかどうかを確認
        if winner_after_move(next_board, move, player) == player:
            # 勝つ手ならリストに追加
            winning_moves.append(move)
    return winning_moves



@dataclass(slots=True)
class TssEvaluation:
    # TSS評価の結果を格納するデータクラス
    score: float = 0.0  # TSSスコア
    forced_win: bool = False  # 強制勝利の可能性
    forced_loss: bool = False  # 強制敗北の可能性
    win_depth: int | None = None  # 勝利までの手数（深さ）
    loss_depth: int | None = None  # 敗北までの手数（深さ）


@dataclass(slots=True)
class TssRequest:
    # TSS評価リクエストのデータクラス
    board: list[int]  # ボード状態
    player: int  # 評価対象のプレイヤー
    move: int  # 評価対象の手


@dataclass(slots=True)
class TssBatchStats:
    # TSS処理の統計情報を格納するデータクラス
    deep_count: int = 0  # 深い探索の実行回数
    thread_count: int = 0  # 使用スレッド数
    elapsed_ms: float = 0.0  # 経過時間（ミリ秒）


class _CtypesTssResult(ctypes.Structure):
    # C言語の共有ライブラリから返される構造体
    _fields_ = [
        ("score", ctypes.c_double),  # スコア
        ("forced_win", ctypes.c_int),  # 強制勝利フラグ
        ("forced_loss", ctypes.c_int),  # 強制敗北フラグ
        ("win_depth", ctypes.c_int),  # 勝利までの深さ
        ("loss_depth", ctypes.c_int),  # 敗北までの深さ
    ]


class _CtypesTssRewardConfig(ctypes.Structure):
    # リワード計算の設定を格納するC言語構造体
    _fields_ = [
        ("illegal", ctypes.c_double),  # 非合法手のリワード
        ("immediate_win", ctypes.c_double),  # 即勝利のリワード
        ("immediate_loss", ctypes.c_double),  # 即敗北のリワード
        ("allow_immediate_loss", ctypes.c_double),  # 相手の即勝利手を作るリワード
        ("block_immediate_win", ctypes.c_double),  # 相手の即勝利手をブロックするリワード
        ("tss_weight", ctypes.c_double),  # TSSスコアの重み
        ("tss_forced_win", ctypes.c_double),  # TSS強制勝利のリワード
        ("tss_forced_loss", ctypes.c_double),  # TSS強制敗北のリワード
        ("create_four", ctypes.c_double),  # 四目を作るリワード
        ("create_open_three", ctypes.c_double),  # オープン三目を作るリワード
        ("staged", ctypes.c_int),  # ステージング有効フラグ
        ("shallow_depth", ctypes.c_int),  # 浅い探索の深さ
        ("deep_top_k", ctypes.c_int),  # 深い探索のトップK候補数
        ("deep_score_threshold", ctypes.c_double),  # 深い探索のスコア閾値
    ]


class _CtypesTssRewardResult(ctypes.Structure):
    # リワード計算の結果を格納するC言語構造体
    _fields_ = [
        ("reward", ctypes.c_double),  # 計算されたリワード
        ("tss_score", ctypes.c_double),  # TSSスコア
        ("forced_win", ctypes.c_int),  # 強制勝利フラグ
        ("forced_loss", ctypes.c_int),  # 強制敗北フラグ
        ("win_depth", ctypes.c_int),  # 勝利までの深さ
        ("loss_depth", ctypes.c_int),  # 敗北までの深さ
        ("illegal", ctypes.c_int),  # 非合法フラグ
        ("terminal", ctypes.c_int),  # 終了状態フラグ
    ]


class _CtypesTssBatchStats(ctypes.Structure):
    # バッチ処理の統計情報を格納するC言語構造体
    _fields_ = [
        ("deep_count", ctypes.c_int),  # 深い探索の実行回数
        ("thread_count", ctypes.c_int),  # 使用スレッド数
        ("elapsed_ms", ctypes.c_double),  # 経過時間（ミリ秒）
    ]


class SharedLibraryTssClient:
    """Calls a tss shared library through its batch C API."""

    def __init__(
        self,
        library_path: str | None,
        max_depth: int,
        candidate_limit: int,
        required: bool,
        parallel_threads: int,
    ) -> None:
        # パラメータを保存
        self.library_path = library_path
        self.max_depth = max_depth
        self.candidate_limit = candidate_limit
        self.required = required
        self.parallel_threads = parallel_threads
        # 最後のバッチ統計を初期化
        self.last_batch_stats = TssBatchStats()
        # 共有ライブラリのハンドルを初期化
        self._library: ctypes.CDLL | None = None
        self._function = None
        self._reward_function = None
        self._reward_function_with_stats = None

        # ライブラリパスが指定されていない場合は早期終了
        if not library_path:
            return

        try:
            # ライブラリパスを展開・解決
            resolved_path = str(Path(library_path).expanduser().resolve())
            # 共有ライブラリをロード
            library = ctypes.CDLL(resolved_path)
            # tss_evaluate_batch関数を取得
            function = library.tss_evaluate_batch
            # 引数の型を指定
            function.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(_CtypesTssResult),
            ]
            # 戻り値の型を指定
            function.restype = ctypes.c_int

            # tss_evaluate_reward_batch関数を取得（オプション）
            reward_function = getattr(library, "tss_evaluate_reward_batch", None)
            if reward_function is not None:
                # 引数の型を指定
                reward_function.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(_CtypesTssRewardConfig),
                    ctypes.POINTER(_CtypesTssRewardResult),
                ]
                # 戻り値の型を指定
                reward_function.restype = ctypes.c_int

            # tss_evaluate_reward_batch_with_stats関数を取得（オプション）
            reward_function_with_stats = getattr(library, "tss_evaluate_reward_batch_with_stats", None)
            if reward_function_with_stats is not None:
                # 引数の型を指定
                reward_function_with_stats.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(_CtypesTssRewardConfig),
                    ctypes.POINTER(_CtypesTssRewardResult),
                    ctypes.c_int,
                    ctypes.POINTER(_CtypesTssBatchStats),
                ]
                # 戻り値の型を指定
                reward_function_with_stats.restype = ctypes.c_int
        except (AttributeError, OSError) as exc:
            # ライブラリロード失敗時
            if required:
                # 必須の場合は例外を発生
                raise RuntimeError(f"Failed to load TSS shared library: {exc}") from exc
            # 必須でない場合は処理を続行
            return

        # ロード成功時、関数を保存
        self._library = library
        self._function = function
        self._reward_function = reward_function
        self._reward_function_with_stats = reward_function_with_stats

    @property
    def available(self) -> bool:
        # ライブラリが正常にロードされたかどうかを返す
        return self._function is not None

    def evaluate_many(self, requests: list[TssRequest]) -> list[TssEvaluation] | None:
        # リクエストが空の場合は空リストを返す
        if not requests:
            return []
        # ライブラリが利用不可の場合はNoneを返す
        if self._function is None:
            return None

        # リクエスト数を取得
        request_count = len(requests)
        # フラットなボード配列、プレイヤー、手を格納するリストを準備
        flat_boards: list[int] = []
        players: list[int] = []
        moves: list[int] = []
        # 各リクエストから情報を抽出
        for request in requests:
            # ボード配列を展開して追加
            flat_boards.extend(request.board)
            # プレイヤーを追加
            players.append(request.player)
            # 手を追加
            moves.append(request.move)

        # PythonリストをCタイプの配列に変換
        board_array = (ctypes.c_int * len(flat_boards))(*flat_boards)
        player_array = (ctypes.c_int * request_count)(*players)
        move_array = (ctypes.c_int * request_count)(*moves)
        # 結果配列を準備
        result_array = (_CtypesTssResult * request_count)()

        # C言語関数を呼び出し
        status = self._function(
            board_array,
            player_array,
            move_array,
            request_count,
            int(self.max_depth),
            int(self.candidate_limit),
            result_array,
        )
        # エラーをチェック
        if status != 0:
            if self.required:
                # 必須の場合は例外を発生
                raise RuntimeError(f"TSS shared library returned error code {status}.")
            # 必須でない場合はNoneを返す
            return None

        # 結果をPythonオブジェクトに変換して返す
        return [
            TssEvaluation(
                score=float(result.score),
                forced_win=bool(result.forced_win),
                forced_loss=bool(result.forced_loss),
                win_depth=result.win_depth if result.win_depth >= 0 else None,
                loss_depth=result.loss_depth if result.loss_depth >= 0 else None,
            )
            for result in result_array
        ]

    def evaluate_rewards(
        self,
        boards: list[list[int]],
        actions: list[list[int]],
        reward_cfg: Any,
        staged_cfg: Any,
    ) -> list[list[float]] | None:
        # リワード計算関数が利用不可の場合はNoneを返す
        if self._reward_function is None and self._reward_function_with_stats is None:
            return None

        # リクエスト数（全ボード×各ボード毎のアクション数）をカウント
        request_count = sum(len(board_actions) for board_actions in actions)
        # リクエストが空の場合は空リストを返す
        if request_count == 0:
            self.last_batch_stats = TssBatchStats()
            return [[] for _ in actions]

        # フラットなボード配列と手を準備
        flat_boards: list[int] = []
        moves: list[int] = []
        # 各ボード毎のアクション数を記録
        widths: list[int] = []
        for board, board_actions in zip(boards, actions, strict=True):
            # このボード毎のアクション数を記録
            widths.append(len(board_actions))
            # 各アクションに対して
            for action in board_actions:
                # ボード状態を追加
                flat_boards.extend(board)
                # 手を追加
                moves.append(action)

        # PythonリストをCタイプの配列に変換
        board_array = (ctypes.c_int * len(flat_boards))(*flat_boards)
        move_array = (ctypes.c_int * request_count)(*moves)
        # 結果配列を準備
        result_array = (_CtypesTssRewardResult * request_count)()
        # リワード設定をC言語構造体に変換
        config = _CtypesTssRewardConfig(
            float(reward_cfg.illegal),
            float(reward_cfg.immediate_win),
            float(reward_cfg.immediate_loss),
            float(reward_cfg.allow_immediate_loss),
            float(reward_cfg.block_immediate_win),
            float(reward_cfg.tss_weight),
            float(reward_cfg.tss_forced_win),
            float(reward_cfg.tss_forced_loss),
            float(reward_cfg.create_four),
            float(reward_cfg.create_open_three),
            int(staged_cfg.enabled),
            int(staged_cfg.shallow_depth),
            int(staged_cfg.deep_top_k),
            float(staged_cfg.deep_score_threshold),
        )

        # 統計情報を準備
        stats = _CtypesTssBatchStats()
        # 統計付きの関数がある場合はそちらを使用
        if self._reward_function_with_stats is not None:
            # 統計付きでC言語関数を呼び出し
            status = self._reward_function_with_stats(
                board_array,
                None,
                move_array,
                request_count,
                int(self.max_depth),
                int(self.candidate_limit),
                ctypes.byref(config),
                result_array,
                int(self.parallel_threads),
                ctypes.byref(stats),
            )
            # 統計情報を保存
            self.last_batch_stats = TssBatchStats(
                deep_count=int(stats.deep_count),
                thread_count=int(stats.thread_count),
                elapsed_ms=float(stats.elapsed_ms),
            )
        else:
            # 統計なしでC言語関数を呼び出し
            status = self._reward_function(
                board_array,
                None,
                move_array,
                request_count,
                int(self.max_depth),
                int(self.candidate_limit),
                ctypes.byref(config),
                result_array,
            )
            # 統計情報をクリア
            self.last_batch_stats = TssBatchStats()
        # エラーをチェック
        if status != 0:
            if self.required:
                # 必須の場合は例外を発生
                raise RuntimeError(f"TSS reward shared library returned error code {status}.")
            # 必須でない場合はNoneを返す
            return None

        # 結果をボード単位のリストに変換して返す
        rewards: list[list[float]] = []
        cursor = 0
        for width in widths:
            # このボード毎のリワード値を抽出
            rewards.append([float(result_array[cursor + offset].reward) for offset in range(width)])
            # カーソルを次のボードに進める
            cursor += width
        return rewards


class ExternalTssClient:
    """Calls an optional C++ TSS executable using a JSON stdin/stdout protocol."""

    def __init__(
        self,
        command: str | None,
        timeout_seconds: float,
        max_depth: int,
        candidate_limit: int,
        required: bool,
    ) -> None:
        # パラメータを保存
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.max_depth = max_depth
        self.candidate_limit = candidate_limit
        self.required = required

    def evaluate(self, board: list[int], player: int, move: int) -> TssEvaluation | None:
        # コマンドが指定されていない場合はNoneを返す
        if not self.command:
            return None

        # JSONリクエストを準備
        request = {
            "board": board,
            "player": player,
            "move": move,
            "max_depth": self.max_depth,
            "candidate_limit": self.candidate_limit,
        }
        try:
            # 外部コマンドを実行（JSON入出力）
            completed = subprocess.run(
                shlex.split(self.command),
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            # コマンド実行失敗時
            if self.required:
                # 必須の場合は例外を発生
                raise RuntimeError(f"TSS command failed: {exc}") from exc
            # 必須でない場合はNoneを返す
            return None

        try:
            # 出力をJSONでパース
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            # JSON解析失敗時
            if self.required:
                # 必須の場合は例外を発生
                raise RuntimeError(f"TSS command returned invalid JSON: {completed.stdout!r}") from exc
            # 必須でない場合はNoneを返す
            return None

        # JSONペイロードをTssEvaluationに変換
        return _parse_tss_payload(payload)

    def evaluate_many(self, requests: list[TssRequest]) -> list[TssEvaluation] | None:
        # リクエストが空の場合は空リストを返す
        if not requests:
            return []
        # 各リクエストを順に評価
        results: list[TssEvaluation] = []
        for request in requests:
            # 単一の評価を実行
            result = self.evaluate(request.board, request.player, request.move)
            # 失敗時はNoneを返す
            if result is None:
                return None
            # 結果を追加
            results.append(result)
        return results


def _parse_tss_payload(payload: dict[str, Any]) -> TssEvaluation:
    # JSONペイロードをTssEvaluationに変換
    return TssEvaluation(
        score=float(payload.get("score", 0.0)),  # スコアを取得（デフォルト0.0）
        forced_win=bool(payload.get("forced_win", False)),  # 強制勝利フラグを取得
        forced_loss=bool(payload.get("forced_loss", False)),  # 強制敗北フラグを取得
        win_depth=_optional_int(payload.get("win_depth")),  # 勝利までの深さを取得
        loss_depth=_optional_int(payload.get("loss_depth")),  # 敗北までの深さを取得
    )


def _optional_int(value: Any) -> int | None:
    # 値をオプショナル整数に変換
    if value is None:
        # Noneはそのまま返す
        return None
    # その他の値は整数に変換
    return int(value)


class GrpoRewardEvaluator:
    def __init__(self, cfg: DictConfig) -> None:
        # 設定を保存
        self.cfg = cfg
        # 共有ライブラリのTSSクライアントを初期化
        self.shared_tss = SharedLibraryTssClient(
            library_path=cfg.grpo.tss.library_path,
            max_depth=cfg.grpo.tss.max_depth,
            candidate_limit=cfg.grpo.tss.candidate_limit,
            required=cfg.grpo.tss.required,
            parallel_threads=int(cfg.grpo.tss.get("parallel_threads", 1)),
        )
        # 外部実行ファイルのTSSクライアントを初期化
        self.external_tss = ExternalTssClient(
            command=cfg.grpo.tss.command,
            timeout_seconds=cfg.grpo.tss.timeout_seconds,
            max_depth=cfg.grpo.tss.max_depth,
            candidate_limit=cfg.grpo.tss.candidate_limit,
            required=cfg.grpo.tss.required,
        )
        # 統計情報をリセット
        self.reset_stats()

    def reset_stats(self) -> None:
        # 統計情報を初期化
        self._stats = {
            "tss_deep_count": 0,  # 深い探索の実行回数
            "tss_batch_ms": 0.0,  # バッチ処理の経過時間
            "tss_batch_calls": 0,  # バッチ処理の呼び出し回数
            "tss_thread_count": 0,  # 使用スレッド数
        }

    def consume_stats(self) -> dict[str, float | int]:
        # 累積された統計情報をコピー
        stats = dict(self._stats)
        # 統計情報をリセット
        self.reset_stats()
        # コピーした統計情報を返す
        return stats

    def _record_shared_tss_stats(self) -> None:
        # 共有TSSから最後のバッチ統計を取得
        stats = self.shared_tss.last_batch_stats
        # 深い探索の実行回数を累積
        self._stats["tss_deep_count"] += stats.deep_count
        # バッチ処理の経過時間を累積
        self._stats["tss_batch_ms"] += stats.elapsed_ms
        # バッチ処理の呼び出し回数を増加
        self._stats["tss_batch_calls"] += 1
        # 使用スレッド数を最大値で更新
        self._stats["tss_thread_count"] = max(
            int(self._stats["tss_thread_count"]),
            stats.thread_count,
        )

    def evaluate_local(self, board: list[int], action: int) -> float:
        # ボードサイズを確認
        if len(board) != BOARD_CELLS:
            raise ValueError(f"Expected {BOARD_CELLS} board cells, got {len(board)}.")

        # 現在のプレイヤーを推定
        player = infer_player(board)
        # 手の有効性を確認
        if action < 0 or action >= BOARD_CELLS:
            # 無効な手はリワード（非合法手）を返す
            return float(self.cfg.grpo.reward.illegal)

        # 合法手のマスクを取得
        legal_mask = legal_move_mask(board)
        # 手が合法かどうかを確認
        if not legal_mask[action]:
            # 非合法手はリワード（非合法手）を返す
            return float(self.cfg.grpo.reward.illegal)

        # TSSを使わない基本的なリワードを計算
        base = self._evaluate_without_tss(board, action, player)
        # ゲーム終了の場合はリワードをそのまま返す
        if base["done"]:
            return float(base["reward"])

        # 次のボード状態を取得
        next_board = base["next_board"]
        if not isinstance(next_board, list):
            raise TypeError("Internal error: expected next_board list.")
        # TSSで評価
        tss_result = self._evaluate_tss_many([TssRequest(next_board, player, action)])[0]
        # リワード = 基本リワード + TSSスコア + 形状スコア
        return (
            float(base["reward"])
            + self._tss_score_from_result(tss_result, next_board, player, action)
            + self._shape_score(next_board, action, player)
        )

    def evaluate_batch(self, boards: list[list[int]], actions: list[list[int]]) -> list[list[float]]:
        # 共有ライブラリのリワード評価を試す
        shared_rewards = self.shared_tss.evaluate_rewards(
            boards,
            actions,
            self.cfg.grpo.reward,
            self.cfg.grpo.tss.staged,
        )
        # 共有ライブラリが利用可能な場合は結果を返す
        if shared_rewards is not None:
            # 統計情報を記録
            self._record_shared_tss_stats()
            return shared_rewards

        # バッチリワードを初期化
        batch_rewards = [[0.0 for _ in board_actions] for board_actions in actions]
        # ペンディング中のTSSリクエストと位置情報を格納
        pending_requests: list[TssRequest] = []
        pending_locations: list[tuple[int, int, list[int], int, int]] = []

        # 各ボードと手について処理
        for board_index, (board, board_actions) in enumerate(zip(boards, actions, strict=True)):
            # 現在のプレイヤーを推定
            player = infer_player(board)
            # 各手について処理
            for action_index, action in enumerate(board_actions):
                # TSSを使わない基本的なリワードを計算
                base = self._evaluate_without_tss(board, action, player)
                # リワードを記録
                batch_rewards[board_index][action_index] = float(base["reward"])
                # ゲーム終了している場合はスキップ
                if base["done"]:
                    continue

                # 次のボード状態を取得
                next_board = base["next_board"]
                if not isinstance(next_board, list):
                    raise TypeError("Internal error: expected next_board list.")
                # ペンディング情報を記録（後でTSSで一括評価するため）
                pending_locations.append((board_index, action_index, next_board, player, action))
                # TSSリクエストを追加
                pending_requests.append(TssRequest(next_board, player, action))

        # 全ペンディングリクエストをTSSで評価
        tss_results = self._evaluate_tss_many(pending_requests)
        # 各結果に対してリワードを更新
        for result, (board_index, action_index, next_board, player, action) in zip(
            tss_results,
            pending_locations,
            strict=True,
        ):
            # TSSスコアを加算
            batch_rewards[board_index][action_index] += self._tss_score_from_result(
                result,
                next_board,
                player,
                action,
            )
            # 形状スコア（四目、オープン三目）を加算
            batch_rewards[board_index][action_index] += self._shape_score(next_board, action, player)

        return batch_rewards

    def _evaluate_without_tss(self, board: list[int], action: int, player: int) -> dict[str, object]:
        # リワードを初期化
        reward = 0.0
        # 相手プレイヤーを取得
        opponent = other_player(player)
        # 相手が既に持っている即勝利手を取得
        opponent_wins_before = set(immediate_winning_moves(board, opponent))

        # この手を打った後のボード状態を計算
        next_board = board_with_move(board, action, player)
        # ゲームの勝者を確認
        winner = winner_after_move(next_board, action, player)
        # 自分が勝つ場合
        if winner == player:
            # 即勝利のリワードを返す
            return {"reward": float(self.cfg.grpo.reward.immediate_win), "done": True}
        # 相手が勝つ場合
        if winner == opponent:
            # 即敗北のリワードを返す
            return {"reward": float(self.cfg.grpo.reward.immediate_loss), "done": True}

        # この手が相手の即勝利手をブロックしている場合
        if action in opponent_wins_before:
            # ブロック手のリワードを加算
            reward += float(self.cfg.grpo.reward.block_immediate_win)

        # 手を打った後の相手の即勝利手を取得
        opponent_wins_after = immediate_winning_moves(next_board, opponent)
        # 相手が即勝利手を持つようになった場合
        if opponent_wins_after:
            # 許容できる敗北のリワードを加算
            reward += float(self.cfg.grpo.reward.allow_immediate_loss)

        # 非終了状態で、次のボード状態を返す
        return {"reward": reward, "done": False, "next_board": next_board}

    def _evaluate_tss_many(self, requests: list[TssRequest]) -> list[TssEvaluation]:
        # リクエストが空の場合は空リストを返す
        if not requests:
            return []

        # 共有ライブラリで評価を試す
        shared_results = self.shared_tss.evaluate_many(requests)
        # 共有ライブラリが利用可能な場合は結果を返す
        if shared_results is not None:
            return shared_results

        # 外部実行ファイルで評価を試す
        external_results = self.external_tss.evaluate_many(requests)
        # 外部実行ファイルが利用可能な場合は結果を返す
        if external_results is not None:
            return external_results

        # フォールバック機能が無効な場合は空のスコアを返す
        if not bool(self.cfg.grpo.tss.use_fallback):
            return [TssEvaluation() for _ in requests]

        # フォールバック評価を実行（形状スコアを使用）
        return [
            self._fallback_tss_like_score(request.board, request.player, request.move)
            for request in requests
        ]

    def _tss_score(self, board_after_action: list[int], player: int, action: int) -> float:
        # 単一のリクエストでTSSを評価
        tss_result = self._evaluate_tss_many([TssRequest(board_after_action, player, action)])[0]
        # 結果からスコアを計算
        return self._tss_score_from_result(tss_result, board_after_action, player, action)

    def _tss_score_from_result(
        self,
        tss_result: TssEvaluation,
        _board_after_action: list[int],
        _player: int,
        _action: int,
    ) -> float:
        # TSSスコアをコピー
        score = tss_result.score
        # 強制勝利の場合
        if tss_result.forced_win:
            # 深さを考慮した強制勝利ボーナスを加算
            score += self._depth_scaled(
                float(self.cfg.grpo.reward.tss_forced_win),
                tss_result.win_depth,
            )
        # 強制敗北の場合
        if tss_result.forced_loss:
            # 深さを考慮した強制敗北ペナルティを加算
            score += self._depth_scaled(
                float(self.cfg.grpo.reward.tss_forced_loss),
                tss_result.loss_depth,
            )
        # TSSスコアに重みを付けて返す
        return float(self.cfg.grpo.reward.tss_weight) * score

    def _fallback_tss_like_score(
        self,
        board_after_action: list[int],
        player: int,
        action: int,
    ) -> TssEvaluation:
        # 相手プレイヤーを取得
        opponent = other_player(player)
        # この手で作った四目の数を数える
        own_fours = count_four_directions(board_after_action, action, player)
        # この手で作ったオープン三目の数を数える
        own_open_threes = count_open_three_directions(board_after_action, action, player)
        # 相手の即勝利手を取得
        opponent_forcing_moves = immediate_winning_moves(board_after_action, opponent)

        # 四目が2個以上で強制勝利と判定
        forced_win = own_fours >= 2
        # 相手の即勝利手がある場合は強制敗北と判定
        forced_loss = bool(opponent_forcing_moves)
        # スコアを初期化
        score = 0.0
        # 四目が1個の場合
        if own_fours == 1:
            # スコアを加算
            score += 0.18
        # オープン三目がある場合
        if own_open_threes >= 1:
            # オープン三目の数に応じてスコアを加算（最大0.24）
            score += min(0.12 * own_open_threes, 0.24)
        # 強制敗北の場合
        if forced_loss:
            # ペナルティを減算
            score -= 0.35

        # 評価結果を返す
        return TssEvaluation(
            score=score,
            forced_win=forced_win,
            forced_loss=forced_loss,
            win_depth=3 if forced_win else None,
            loss_depth=1 if forced_loss else None,
        )

    def _shape_score(self, board_after_action: list[int], action: int, player: int) -> float:
        # この手で作った四目の数を数える
        fours = count_four_directions(board_after_action, action, player)
        # この手で作ったオープン三目の数を数える
        open_threes = count_open_three_directions(board_after_action, action, player)
        # スコア = min(四目数, 1) * 四目のリワード + min(オープン三目数, 2) * オープン三目のリワード
        return (
            min(fours, 1) * float(self.cfg.grpo.reward.create_four)
            + min(open_threes, 2) * float(self.cfg.grpo.reward.create_open_three)
        )

    @staticmethod
    def _depth_scaled(base_score: float, depth: int | None) -> float:
        # 深さが指定されていない、または深さが1以下の場合は基本スコアをそのまま返す
        if depth is None or depth <= 1:
            return base_score
        # 深さを考慮した減衰を適用（深いほど減衰）
        return base_score / (depth**0.5)


def board_is_full(board: list[int]) -> bool:
    # 全マスが埋まっているかどうかを確認
    return all(cell != EMPTY for cell in board)


def current_winner(board: list[int]) -> int | None:
    # 現在のボード状態での勝者を確認
    return board_winner(board)
