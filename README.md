# RenjuTransformer

PyTorch + Hydra + MLflow + uv で構成した、五目並べの「次の1手」予測用 TransformerEncoder プロジェクトです。

## セットアップ

```powershell
uv sync
```

## 学習

`data.csv` は `mcts.cpp` が出力した 1 行 227 列の CSV を想定します。

```powershell
uv run python .\renju-transformer.py
```

設定は Hydra で上書きできます。

```powershell
uv run python .\renju-transformer.py data.path=sample-log.csv train.max_epochs=5 train.batch_size=32 model.d_model=256
```

## 合成データ生成

`mcts.cpp` は、Renju のルールと禁じ手を考慮した自己対戦データ生成器です。各手について、`board(225) + SEP(228) + move_id` の 1 行 CSV を標準出力に書き出します。試合進捗と勝敗結果は標準エラー出力に書き出します。

### ビルド

```powershell
g++ -std=c++17 -O2 -pthread .\mcts.cpp -o .\mcts.exe
```

### Usage

```powershell
.\mcts.exe 100000 --simulations 1000 --parallel 28 > data.csv 2> error.log
```

この例では次を行います。

- `100000` 試合の自己対戦を実行
- 1 手あたり `1000` 回の MCTS シミュレーションを実行
- `28` スレッドで試合単位に並列化
- 学習用 CSV を `output.csv` に保存
- 進捗と勝敗ログを `error.log` に保存

主な引数は次です。

- `<games>`: 総試合数
- `--simulations <N>`: 1 手あたりの MCTS シミュレーション回数
- `--parallel <N>`: 並列スレッド数
- `--seed <N>`: 乱数 seed
- `--candidate-limit <N>`: 探索対象に残す候補手の上限
- `--rollout-limit <N>`: rollout の最大手数
- `--exploration <C>`: UCT の探索定数
- `--trace-plies`: 標準エラー出力に各手の進捗も出す

ヘルプは次で表示できます。

```powershell
.\mcts.exe --help
```

## 推論

学習済み checkpoint と盤面を与えると、次の一手 ID を出力します。

```powershell
uv run python .\renju-transformer.py mode=predict predict.checkpoint_path=artifacts/checkpoints/best_model.pt predict.board_path=board.txt
```

`board.txt` は 225 個の `0,1,2` をカンマ区切りで並べた 1 行ファイルです。

## GRPO 強化学習

教師あり学習済み checkpoint を policy model と reference model の初期値として読み込み、合法手マスクつきの GRPO で更新できます。

```powershell
uv run python .\renju-transformer.py mode=grpo grpo.checkpoint_path=artifacts/checkpoints/best_model.pt
```

報酬は次を合成します。

- 即勝ち、即負け、相手の即勝ちブロック
- TSS による強制勝ち、強制負けの評価
- 四、開三などの小さな形評価
- `step_group` / `trajectory_group` では終局勝敗 bonus

`grpo.objective` は3種類あります。

- `state`: CSV の各行を独立した局面として扱う局面単位 GRPO。
- `step_group`: 開始局面から試合を進めながら、各局面で複数候補手を比較する方式。
- `trajectory_group`: 同じ開始局面から複数の試合軌跡を生成し、trajectory単位の合計報酬で比較する方式。

`step_group` は次の流れで動きます。

1. 開始局面を1つ選びます。
2. policy担当の手番では policy から、reference担当の手番では固定 reference から `grpo.group_size` 個の候補手をサンプルします。
3. 各候補手に C++ TSS 報酬、即勝ち/即負け、四、開三などの局面報酬を付けます。
4. 同じ局面内の候補を1つの GRPO group として保存します。
5. 次局面へ進む手は `grpo.step_group.action_selection` で選びます。`softmax` なら `softmax(reward / grpo.step_group.selection_temperature)`、`best` なら報酬最大の手を選びます。
6. 終局または `grpo.step_group.max_plies` まで進めます。
7. 実際に採用された手系列だけに、終局勝敗 bonus を割引して足します。
8. policy担当手番の候補 group だけで advantage を正規化し、GRPO 更新します。reference担当手番は盤面を進めるために使いますが、更新対象には入れません。

この実装では、`grpo.group_size` は「1 step で作る試合数」ではなく、「各局面で比較する候補手数」です。例えば `grpo.group_size=4` なら、各局面で4手を候補として出し、その4手の報酬差から policy を更新します。

`grpo.step_group.opponent=reference` では、片方を学習中の policy、もう片方を固定 reference model として対戦させます。`grpo.step_group.learning_player=both` は、policy が黒を持つ試合と白を持つ試合を交互に作り、黒番/白番どちらの policy 手も学習する設定です。`learning_player=black` または `white` にすると、policy がその色を持つ試合だけを生成します。`opponent=self` に戻すと従来どおり policy が両側を打ちます。

終局勝敗 bonus の割引は次です。

```text
final_bonus_t = gamma ** distance_to_terminal * final_result_reward(actor_t)
```

`gamma` は `grpo.step_group.gamma` です。`distance_to_terminal` は、その手から終局までに実際に進んだ手数です。最後の採用手は `gamma ** 0`、1手前は `gamma ** 1` になります。`final_result_reward(actor_t)` は、その手を打ったプレイヤーが最終的に勝てば `+grpo.step_group.final_result_weight`、負ければ `-grpo.step_group.final_result_weight`、引き分けなら `grpo.step_group.draw_reward` です。非採用候補には終局勝敗 bonus は足さず、その局面の TSS/ルール報酬だけを使います。

`grpo.step_group.gamma=1.0` は終局勝敗を全採用手へ同じ強さで伝えます。`0.95` や `0.99` にすると、終局に近い手ほど強く評価され、序盤の手への勝敗 credit は弱くなります。

`trajectory_group` は次の流れで動きます。

1. 開始局面を1つ選びます。
2. 同じ開始局面から、黒をpolicyが担当するtrajectoryを `grpo.trajectory_group.group_size` 本生成します。
3. 同じ開始局面から、白をpolicyが担当するtrajectoryも `grpo.trajectory_group.group_size` 本生成します。
4. 各手の報酬は C++ TSS/ルール評価から返された局面報酬を使います。
5. 各trajectoryの報酬は、policyが実際に打った手のTSS報酬を単純合計し、終局勝敗bonusを1回足したものです。
6. 同じ開始局面かつ同じpolicy担当色のgroup内で、trajectory報酬を正規化してadvantageを作ります。
7. そのtrajectory内でpolicyが実際に打った手すべてに同じadvantageを配ります。
8. reference/opponentが打った手は盤面を進めるためだけに使い、更新対象には入れません。

`trajectory_group` では、同じ開始局面から複数本を比較するため、`grpo.temperature=1.0` から `1.2` 程度にしてtrajectoryのばらつきを作るのがおすすめです。勝敗bonusを使わない場合は `grpo.step_group.final_result_weight=0` にしてください。

trajectory group GRPO の自己対戦で `tss.so` を使う実行例:

```bash
g++ -std=c++17 -O3 -fPIC -shared -pthread tss.cpp -o tss.so

nohup uv run python renju-transformer.py mode=grpo \
  grpo.objective=trajectory_group \
  grpo.checkpoint_path=models/pretrained.pt \
  grpo.tss.library_path=./tss.so \
  grpo.tss.required=true \
  grpo.tss.use_fallback=false \
  grpo.tss.max_depth=3 \
  grpo.tss.candidate_limit=8 \
  grpo.tss.staged.deep_top_k=2 \
  grpo.tss.staged.deep_score_threshold=0.3 \
  grpo.tss.parallel_threads=0 \
  grpo.step_group.source=mixed \
  grpo.step_group.opponent=reference \
  grpo.step_group.learning_player=both \
  grpo.step_group.prompts_per_step=4 \
  grpo.trajectory_group.group_size=4 \
  grpo.temperature=1.1 \
  grpo.step_group.max_plies=120 \
  grpo.step_group.final_result_weight=1.0 \
  grpo.reward.allow_immediate_loss=-1.2 \
  grpo.reward.tss_forced_loss=-1.0 \
  grpo.reward.block_immediate_win=0.5 \
  grpo.save_every_steps=100 \
  grpo.max_steps=1000 \
  > nohup-grpo.out 2>&1 &
```

この例は、各開始局面から黒policy groupを4本、白policy groupを4本生成します。各trajectoryではpolicy/referenceが1手ずつサンプルして試合を進め、policy担当手のTSS報酬合計と終局勝敗bonusでtrajectoryを比較します。即死防止を強めるため、相手の即勝ちを許す手と TSS 強制負けのペナルティを少し強くし、ブロック報酬も上げています。

step_group GRPO の試合復元開始例:

```bash
uv run python renju-transformer.py mode=grpo \
  grpo.objective=step_group \
  grpo.checkpoint_path=models/pretrained.pt \
  grpo.tss.library_path=./tss.so \
  grpo.tss.required=true \
  grpo.tss.use_fallback=false \
  grpo.step_group.source=mixed \
  grpo.step_group.opponent=reference \
  grpo.step_group.learning_player=both \
  grpo.step_group.start_positions=all \
  grpo.step_group.max_start_ply=20 \
  grpo.step_group.prompts_per_step=2 \
  grpo.group_size=4 \
  grpo.step_group.max_plies=120 \
  grpo.step_group.gamma=0.99 \
  grpo.save_every_steps=100 \
  grpo.max_steps=1000
```

`grpo.step_group.source=self_play` は空盤面を開始局面にします。`grpo.step_group.source=dataset` では、`data.csv.gz` の行順から「石数が減る、または空盤面に戻る」位置を試合境界として推定し、復元した試合内の局面を開始局面に使います。`grpo.step_group.source=mixed` は空盤面と dataset 開始局面を混ぜます。

dataset 開始局面は `grpo.step_group.min_start_ply` から `grpo.step_group.max_start_ply` の石数でフィルタします。`grpo.step_group.start_positions=first` は各試合から最初に条件を満たす局面だけ、`all` は条件を満たす全局面を使います。`grpo.step_group.deduplicate=true` なら同一盤面を重複除去し、`grpo.step_group.max_start_positions` で上限を切ります。CSV には `game_id` や `winner` が保存されていないため、元試合の勝敗を読むのではなく、開始局面から policy で続きを生成して終局勝敗を評価します。

### TSS 評価器

`tss.cpp` は GRPO 報酬に差し込むための Threat Space Search 評価器です。推奨は共有ライブラリ `tss.so` を Python から `ctypes` で一度ロードし、GRPO の step 内で複数リクエストを batch C API にまとめて渡す方式です。

Linux or macOS:

```powershell
g++ -std=c++17 -O3 -fPIC -shared -pthread tss.cpp -o tss.so
```

Linux/macOS で実行する場合:

```bash
g++ -std=c++17 -O3 -fPIC -shared -pthread tss.cpp -o tss.so

uv run python renju-transformer.py mode=grpo \
  grpo.checkpoint_path=models/pretrained.pt \
  grpo.tss.library_path=./tss.so \
  grpo.tss.required=true \
  grpo.tss.use_fallback=false \
  grpo.batch_size=8 \
  grpo.group_size=4 \
  grpo.save_every_steps=100 \
  grpo.tss.max_depth=3 \
  grpo.tss.candidate_limit=8 \
  grpo.tss.parallel_threads=0 \
  grpo.tss.staged.enabled=true \
  grpo.tss.staged.shallow_depth=1 \
  grpo.tss.staged.deep_top_k=2 \
  grpo.tss.staged.deep_score_threshold=0.3
```

長時間実行する場合:

```bash
nohup uv run python renju-transformer.py mode=grpo \
  grpo.checkpoint_path=artifacts/checkpoints/best_model.pt \
  grpo.tss.library_path=./tss.so \
  grpo.tss.required=true \
  grpo.tss.use_fallback=false \
  grpo.batch_size=16 \
  grpo.group_size=8 \
  grpo.save_every_steps=100 \
  grpo.tss.max_depth=3 \
  grpo.tss.candidate_limit=8 \
  grpo.tss.parallel_threads=0 \
  grpo.tss.staged.enabled=true \
  grpo.tss.staged.shallow_depth=1 \
  grpo.tss.staged.deep_top_k=2 \
  grpo.tss.staged.deep_score_threshold=0.3 \
  > nohup-grpo.out 2>&1 &
```

実行開始時に `run_output_dir=outputs/YYYY-MM-DD/HH-MM-SS` が表示されます。`nohup-grpo.out` を見なくても、このディレクトリ配下の `logs/stdout.log`、`logs/stderr.log`、`metrics/grpo_steps.csv`、`metrics/grpo_steps.jsonl` で実行状況を確認できます。

GRPO から共有ライブラリを使う場合は `grpo.tss.library_path` に指定します。

```powershell
uv run python .\renju-transformer.py mode=grpo ^
  grpo.checkpoint_path=artifacts/checkpoints/best_model.pt ^
  grpo.tss.library_path=.\tss.so ^
  grpo.tss.required=true ^
  grpo.tss.use_fallback=false
```

共有ライブラリが指定されている場合は、`grpo.tss.command` より優先されます。GRPO はまず報酬まで C++ 側でまとめて計算する `tss_evaluate_reward_batch_with_stats` を使います。古い `tss.so` などでこの関数が見つからない場合は、`tss_evaluate_reward_batch`、さらに古い場合は TSS のみを返す `tss_evaluate_batch` にフォールバックします。

デフォルトでは段階評価を使います。全リクエストに軽い static/shallow 評価を行い、強制負け候補や報酬絶対値が `grpo.tss.staged.deep_score_threshold` 以上のものから上位 `grpo.tss.staged.deep_top_k` 個だけを `grpo.tss.max_depth` の深い TSS に進めます。それ以外は浅い評価の報酬を使います。深い探索を全手にかけたい場合は `grpo.tss.staged.enabled=false` にします。

`tss_evaluate_reward_batch_with_stats` では、深い TSS に進んだ `deep_indexes` を C++ 内で request 単位に並列評価します。`grpo.tss.parallel_threads=0` は自動で CPU スレッド数を使い、`1` は直列、`4` なら最大4スレッドです。並列化されるのは深い TSS 評価だけで、static/shallow 評価と trajectory の手順進行は直列です。

GRPO が使う報酬 batch C API は次です。`boards` は着手前の盤面を `num_requests * 225` 個並べた配列、`moves` は各盤面で評価する着手 index です。`players` は `null` にできます。その場合は石数から手番を推定します。

```c
typedef struct {
    double illegal;
    double immediate_win;
    double immediate_loss;
    double allow_immediate_loss;
    double block_immediate_win;
    double tss_weight;
    double tss_forced_win;
    double tss_forced_loss;
    double create_four;
    double create_open_three;
    int staged;
    int shallow_depth;
    int deep_top_k;
    double deep_score_threshold;
} TssRewardConfig;

typedef struct {
    double reward;
    double tss_score;
    int forced_win;
    int forced_loss;
    int win_depth;
    int loss_depth;
    int illegal;
    int terminal;
} TssRewardResult;

typedef struct {
    int deep_count;
    int thread_count;
    double elapsed_ms;
} TssBatchStats;

int tss_evaluate_reward_batch_with_stats(
    const int* boards,
    const int* players,
    const int* moves,
    int num_requests,
    int max_depth,
    int candidate_limit,
    const TssRewardConfig* reward_config,
    TssRewardResult* results,
    int parallel_threads,
    TssBatchStats* stats
);

int tss_evaluate_reward_batch(
    const int* boards,
    const int* players,
    const int* moves,
    int num_requests,
    int max_depth,
    int candidate_limit,
    const TssRewardConfig* reward_config,
    TssRewardResult* results
);
```

合法手判定も `tss.so` から呼べます。`src/renju_transformer/rules_tss.py` は `ctypes` wrapper で、`tss.so` が見つからない場合や古い `tss.so` の場合は Python 実装にフォールバックします。`RENJU_TSS_LIBRARY_PATH` で読み込む共有ライブラリを明示できます。

合法手マスクの batch 化は「複数局面をまとめて渡す」方式です。入力は `num_boards * 225` 個の盤面配列、出力は `num_boards * 225` 個の 0/1 マスクです。1つの局面内の225手を別々の request にするのではなく、複数の局面それぞれについて225手ぶんの合法手マスクを返します。GRPO の `build_legal_masks` ではこの batch API を使います。

```c
int tss_is_forbidden_for_black(
    const int* board,
    int move
);

int tss_legal_move_mask(
    const int* board,
    int player,
    int* mask_out
);

int tss_legal_move_mask_batch(
    const int* boards,
    const int* players,
    int num_boards,
    int* masks_out
);
```

互換用の TSS batch C API は次です。

```c
typedef struct {
    double score;
    int forced_win;
    int forced_loss;
    int win_depth;
    int loss_depth;
} TssResult;

int tss_evaluate_batch(
    const int* boards,
    const int* players,
    const int* moves,
    int num_requests,
    int max_depth,
    int candidate_limit,
    TssResult* results
);
```

従来どおり、標準入力で JSON を受け取り、標準出力に JSON を返す実行ファイルとしても使えます。

```powershell
g++ -std=c++17 -O2 -pthread .\tss.cpp -o .\tss.exe
```

実行ファイルとして GRPO から使う場合は `grpo.tss.command` に指定します。この方式は互換用で、`tss.so` の batch C API より遅くなります。

```powershell
uv run python .\renju-transformer.py mode=grpo ^
  grpo.checkpoint_path=artifacts/checkpoints/best_model.pt ^
  grpo.tss.command=.\tss.exe ^
  grpo.tss.required=true ^
  grpo.tss.use_fallback=false
```

入力 JSON:

```json
{"board":[0,0, ... 225 cells ...],"player":1,"move":112,"max_depth":7,"candidate_limit":24}
```

出力 JSON:

```json
{"score":0.0,"forced_win":false,"forced_loss":false,"win_depth":null,"loss_depth":null}
```

### 現在の TSS 実装の性質と弱点

この `tss.cpp` は、GRPO 報酬に使いやすい軽量な強制手順探索です。攻撃側は即勝ち、四、開三などの forcing move を優先して伸ばし、防御側は発生した即勝ち点を受ける、という形で探索します。返す値は勝率ではなく、局面内に見つかった戦術的な強制勝ち、強制負け、脅威の強さを報酬化しやすくしたものです。

そのため、次の弱点があります。

- 古典的な Threat Space Search の cost square / rest square / gain square を完全には管理していません。複数脅威が干渉する局面では、強制勝ちを過大評価または過小評価する可能性があります。
- 防御側の応手は主に「攻撃側の即勝ち点を受ける」形に絞っています。実戦では、相手が反撃の強制勝ちを作る、防御しながら別の脅威を作る、という手があり、その評価は限定的です。
- 候補手を `candidate_limit` で切るため、探索が速い一方で、遠い位置や一見静かな好手を見落とすことがあります。
- 開三は候補優先度と static score に強く使っていますが、すべての開三連鎖を厳密に証明するわけではありません。
- `forced_win=true` は「この実装の探索範囲内で強制勝ちを見つけた」という意味です。完全解析の証明ではありません。
- `forced_win=false` は「勝ち筋がない」ではなく、「この深さ・候補幅・簡略化した TSS では見つからなかった」という意味です。
- `score` は確率ではありません。GRPO のグループ内比較に使うためのヒューリスティック報酬です。

GRPO では、この TSS 評価を単独の真理として扱うより、即勝ち/即負け、合法手、短い rollout 勝敗 bonus、reference model との KL と組み合わせて使うことを想定しています。より厳密にしたい場合は、cost/rest square を持つ古典 TSS、より広い防御応手生成、MCTS または value model との混合評価を追加します。

### 実行ログ

Hydra の実行ディレクトリ `outputs/YYYY-MM-DD/HH-MM-SS/` に、実行時のログと設定を保存します。`nohup` で実行した場合も、このディレクトリを見れば標準出力、標準エラー、Python logger の内容を確認できます。

主な出力は次です。

- `resolved_config.yaml`: 実行時に解決された設定
- `logs/run.log`: Python logger のログ
- `logs/stdout.log`: 標準出力
- `logs/stderr.log`: 標準エラー
- `metrics/grpo_steps.jsonl`: GRPO step ごとの JSONL metrics
- `metrics/grpo_steps.csv`: GRPO step ごとの CSV metrics
- `metrics/grpo_epochs.jsonl`: GRPO epoch ごとの JSONL metrics
- `metrics/grpo_epochs.csv`: GRPO epoch ごとの CSV metrics

`grpo_steps.csv/jsonl` には TSS の効果測定用に `tss_deep_count`、`tss_batch_ms`、`tss_batch_calls`、`tss_thread_count` も出力します。`tss_deep_count` が多く、`tss_batch_ms` が大きい場合は TSS が主なボトルネックです。`tss_thread_count` が `1` のままなら、`grpo.tss.parallel_threads` または `tss.so` の再ビルドを確認してください。

GRPO checkpoint は `train.output_root/grpo_checkpoints/` に保存します。`latest_grpo_model.pt` と `best_grpo_model.pt` は epoch 終了時に更新します。`grpo.save_every_steps=100` なら、100 step ごとに `step_000100_grpo_model.pt`、`step_000200_grpo_model.pt` のような途中 checkpoint も保存します。保存された step checkpoint のパスは `grpo_steps.csv/jsonl` の `step_checkpoint` にも出ます。

## MLflow

追跡 DB は SQLite、artifact はローカルディレクトリです。

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

SSH 先で MLflow UI を見る場合は、SSH 先のリポジトリ直下で次を起動します。

```bash
uv run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

長時間起動する場合:

```bash
nohup uv run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000 \
  > mlflow-ui.log 2>&1 &
```

手元 PC からは別ターミナルで port forwarding します。

```bash
ssh -N -L 5000:127.0.0.1:5000 user@ssh-host
```

その後、手元のブラウザで `http://127.0.0.1:5000` を開きます。手元の 5000 番が使用中なら、左側だけ変えます。

```bash
ssh -N -L 5001:127.0.0.1:5000 user@ssh-host
```

この場合は `http://127.0.0.1:5001` を開きます。

## 設定

設定はすべて `config/` 配下の Hydra 管理です。

- `config/data/`: データセット
- `config/model/`: TransformerEncoder
- `config/train/`: 学習条件
- `config/optimizer/`: 最適化
- `config/scheduler/`: スケジューラ
- `config/mlflow/`: 実験管理
- `config/predict/`: 推論
- `config/grpo/`: GRPO と TSS 報酬
