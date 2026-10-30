# leisaac-docker

[LeIsaac](https://github.com/LightwheelAI/leisaac)（Isaac Sim / IsaacLab 上で SO101 と LeKiwi を動かすシミュレーション環境）を Docker で動かすための環境です。対象はテレオペ、記録、リプレイ、GR00T N1.5 による閉ループ推論です。

- **sim**: Isaac Sim 5.1 + IsaacLab 2.3 + LeIsaac v0.4.0。テレオペ、リプレイ、推論クライアントを担当する
- **gr00t**: GR00T N1.5 の推論サーバ。sim とは ZMQ の `:5555` で通信する
- **train**: LeRobot 0.4.2。sim の記録を LeRobot 形式に変換し、ポリシー（ACT、SmolVLA、GR00T）を学習する。学習したポリシーの推論サーバ（`:5556`）もこのイメージで動かす（[docs/sweep.md](docs/sweep.md)）

3 つは `docker/Dockerfile` の multi-stage target として作ります。分けている理由は [docs/decisions.md](docs/decisions.md) を参照してください。

## 必要環境

| 項目 | 確認済みの構成 |
|---|---|
| GPU | NVIDIA RTX 5070 12GB（Blackwell / sm_120）。sim と gr00t を同時に動かすには 12GB 以上が必要 |
| ドライバ | 580.178.04 |
| OS | Ubuntu 22.04、X11 |
| Docker | Docker Engine + Compose v2 + nvidia-container-toolkit |
| ディスク | イメージ約 60GB（sim 30GB、gr00t 16GB、train 15GB）とビルドキャッシュ |
| RAM | 32GB。推論中は Kit が約 11GB 使う |

## クイックスタート（PickOrange）

```bash
make build                 # sim / gr00t / train のイメージをビルドする（初回は約 20 分）
make assets                # アセットと学習済みポリシーを取得して検証する（約 8GB。手動で入手するものは案内が出る）
make list-envs             # 登録されているタスクを表示する（15 件）

# テレオペと記録（GUI）。B で開始、N で成功として保存、R でやり直し
make teleop DATASET=demo   # 保存先は data/datasets/demo.hdf5。同じ名前が既にあれば止まる（追記は RESUME=1）
make replay DATASET=demo

# GR00T N1.5 による閉ループ推論
# 学習済みポリシーは make assets が data/policies/leisaac-pick-orange-v0 に取得する
make eval EVAL_ROUNDS=10              # ヘッドレスで実行（成功率は約 50〜70%。docs/evaluation.md）
make eval HEADLESS=0 EVAL_ROUNDS=1    # GUI で表示して実行

make check                 # 動作確認一式。結果は RESULT: PASS / FAIL で表示される
make down                  # 推論サーバなど、常駐しているものを止める
```

設定は `.env` とコマンドラインで指定します（優先順位: コマンドライン > `.env` > 既定値。シェルの環境変数は使いません）。`cp .env.example .env` で作り、`make config` で実際に使われる値を確認できます。

- sim の設定はすべてのタスクで共通です。既定は VRAM 12GB 向けで、物理演算を CPU で行い、テクスチャの予算を絞っています（`SIM_DEVICE=cpu TEX_BUDGET=0.2`）。VRAM に余裕がある環境では `.env` で変えられます。
- 記録時の設定は `data/datasets/<name>.sim.env` に残ります。記録したときと `SIM_DEVICE` が違うと、再生と追記は止まります（`FORCE=1` で続行）。

sim は常駐させず、`make` のコマンドごとに使い捨てのコンテナで実行します。実行中のものを強制終了するときは `make stop` です。

- 初回の起動では、シェーダのキャッシュを作るのに数分かかります。
- 推論中の VRAM のピークは約 8.6GB / 12.2GB です（既定の CPU 物理の場合。GPU 物理では 11.4GB になるので、ブラウザなど GPU を使うアプリを閉じてください）。

## Makefile

| ターゲット | 内容 |
|---|---|
| `build` | 両方のイメージをビルドする |
| `assets` | `assets.json` のアセットと学習済みポリシーを取得してハッシュで検証する。`GROUP=pick-orange` や `GROUP=lekiwi` で絞れる |
| `down` / `stop` | 常駐しているサービスを止める / 実行中の sim のコマンドを強制終了する |
| `list-envs` | タスクの一覧を表示する |
| `smoke` | `TASK` をヘッドレスで作り、ゼロ action で数ステップ動かす（起動確認。`DEVICE` で action の構成を選ぶ） |
| `sweep-check` | Sweep タスク（`LeIsaac-LeKiwi-Sweep-Floor-v0`）の確認。データの約束事と上流の変換の一致、ゴミの配置、成功の判定を確かめ、front と wrist の画像を `data/sweep/check/` に保存する（[docs/sweep.md](docs/sweep.md)） |
| `sweep-eval` | Sweep タスクを `EVAL_POLICY` で `EVAL_ROUNDS` 回解き、成功率を測る。`planner` は計画器（特権情報を使う）、`remote` は学習したポリシー（先に `policy-up` を実行する）。結果は `data/sweep/eval/<日時>/` に残す |
| `datagen` | Sweep タスクの自動生成。計画器が成功したエピソードだけを `NUM` 個、`data/datasets/$(DATASET).hdf5` に記録する（例: `make datagen DATASET=sweep_floor NUM=50`） |
| `sweep-replay` | `datagen` の記録を再生し、記録どおりに成功するかを確かめる。`EPISODES="0 3"` で選ぶ。`HEADLESS=0` で GUI に表示する |
| `convert` | `datagen` の記録を LeRobot 形式（lerobot 0.4.2、v3.0、30fps、動画は AV1）に変換し、`data/lerobot/$(DATASET)/` に出力する。読み直して、すべてのフレームを HDF5 と照合する（train イメージ） |
| `train` | `data/lerobot/$(DATASET)/` で `TRAIN_POLICY`（`act` / `smolvla` / `groot`）を学習し、`data/train/$(TRAIN_NAME)/` に出力する。GR00T はローカルでは学習できない（Colab で行う） |
| `colab-data` / `import-policy` | Colab に渡すデータセットを `data/colab/$(DATASET).tar` にする / Colab の学習結果 `FILE=<tar>` を `data/train/$(TRAIN_NAME)/` に展開する（Drive との受け渡しはブラウザで手動。Colab 側は `notebooks/train_colab.ipynb`） |
| `sweep-inspect` | `datagen` の記録を Kit なしで確かめる。一覧表、上から見た図（`overview.png`）、動画（`episode_*.mp4`）を `data/sweep/inspect/$(DATASET)/` に作る |
| `teleop` / `replay` | テレオペと記録 / 記録したデータの再生。Leader は `DEVICE=so101leader LEADER_PORT=/dev/ttyACMx`。再生は記録時と `SIM_DEVICE` が違えば止まる（`FORCE=1` で続行） |
| `teleop-lekiwi` / `replay-lekiwi` | LeKiwi 用の別名。`TASK` を LeKiwi にし、`DEVICE=keyboard` / `gamepad` を LeKiwi 用に読み替えて teleop / replay を実行する。事前に `make assets GROUP=lekiwi` で Loft シーン（約 4.2GB）を取得する |
| `policy-up` / `policy-down` / `policy-logs` | 学習したポリシー（`data/train/$(TRAIN_NAME)/checkpoints/$(TRAIN_CHECKPOINT)/`）の推論サーバを起動して healthy まで待つ / 停止する / ログを表示する |
| `gr00t-up` / `gr00t-down` / `gr00t-logs` | 推論サーバを起動して healthy まで待つ / 停止する / ログを表示する |
| `eval` | 閉ループ推論を実行する |
| `config` | `.env` とコマンドラインを反映した、実際に使われる設定を表示する |
| `check` | タスク一覧、サーバとの疎通と action の形状、ヘッドレス推論 1 ラウンド、LeKiwi の起動（Loft がなければ SKIP）、Sweep タスクの確認（`sweep-check`）、変換、学習、推論サーバの確認（train。合成データ）、VRAM ピークを確認する |
| `lock` / `lock-check` | `docker/envs/*/uv.lock` を更新する / 最新であることを確認する |
| `shell-sim` / `shell-gr00t` | コンテナのシェルに入る |

| 変数 | 既定値 |
|---|---|
| `TASK` | `LeIsaac-SO101-PickOrange-v0` |
| `DEVICE` | `keyboard`（ほかに `gamepad`、`so101leader`、`lekiwi-leader` など） |
| `DATASET` | `demo` |
| `HEADLESS` | `1`（`eval` のみ） |
| `EVAL_ROUNDS` | `10` |
| `POLICY_DIR` | `/workspace/data/policies/leisaac-pick-orange-v0` |
| `TRAIN_POLICY` / `TRAIN_NAME` | `act` / `$(DATASET)-$(TRAIN_POLICY)` |
| `TRAIN_STEPS` / `TRAIN_BATCH` | 空（ポリシーごとの既定。`tools/sweep_train.py`） |
| `EVAL_POLICY` / `TRAIN_CHECKPOINT` | `planner` / `last`（`sweep-eval` のポリシーと、推論サーバが読むチェックポイント） |
| `EVAL_SEED` | 空（`sweep-eval` のシード。空なら現在時刻） |

## ディレクトリ構成

```
assets.json           # 外部リソース（アセット、学習済みポリシー）の取得元とハッシュ
compose.yaml          # サービス定義と git の ref（x-versions）
compose.leader.yaml   # Leader アーム接続時に重ねる設定
Makefile
docker/
  Dockerfile          # base → sim / gr00t / train
  envs/{sim,gr00t,train}/  # pyproject.toml（依存の宣言）と uv.lock（ハッシュ付きの固定）
tools/
  gr00t_server.py     # VRAM 節約のラッパーを付けて GR00T の推論サーバを起動する
  probe_gr00t.py      # サーバとの疎通と action の形状を確認する
  policy_server.py    # make policy-up の本体（lerobot で学習したポリシーの推論サーバ。train イメージ）
  sim_smoke.py        # make smoke の本体（タスクを作り、ゼロ action で数ステップ動かす）
  leisaac_run.py      # sim のスクリプトを、自前のタスクを登録してから実行するラッパー（sim のコマンドはすべてこれを通る）
  leisaac_ext/        # 自前のタスク（Sweep。docs/sweep.md）
                      #   contract.py: データの約束事、remote.py: 推論サーバとの通信、broom.py: 箒付きの LeKiwi、
                      #   sweep_floor.py: 床のタスクの定義、sweep_mdp.py: 配置、ランダム化、成功の判定、
                      #   sweep_planner.py: 計画器
  sweep_check.py      # make sweep-check の本体
  sweep_eval.py       # make sweep-eval / datagen の本体（ポリシーに N ラウンド解かせて成功率を測る。--record で記録）
  sweep_replay.py     # make sweep-replay の本体（記録の再生と再現の確認）
  sweep_inspect.py    # make sweep-inspect の本体（記録の一覧、図、動画。Kit なし）
  sweep_convert.py    # make convert の本体（HDF5 → LeRobot 形式と照合。train イメージ）
  sweep_train.py      # make train の本体（lerobot-train を呼ぶ。Colab でも同じものを使う）
  train_check.py      # make check の 6 番目（contract と lerobot の LeKiwi の一致、合成データの変換、学習、推論サーバ）
  fetch_assets.py     # make assets の本体（標準ライブラリのみ。ホストの python3 で動く）
  check.sh            # make check の本体
notebooks/
  train_colab.ipynb   # Colab で学習する（GitHub から clone し、同じ lock から環境を作って sweep_train.py を実行する）
data/                 # データセット、ポリシー、キャリブレーション、check のログ（git 管理外）
assets/               # USD アセット（git 管理外）
docs/
```

## ドキュメント

- [docs/versions.md](docs/versions.md): バージョンの組と更新手順
- [docs/decisions.md](docs/decisions.md): 設計判断と、再設計前後の比較
- [docs/troubleshooting.md](docs/troubleshooting.md): 実際に起きた問題と対処
- [docs/evaluation.md](docs/evaluation.md): 推論評価の内容と、結果の判定方法
- [docs/manual-tasks.md](docs/manual-tasks.md): 手作業のチェックリスト
- [docs/lekiwi-learning.md](docs/lekiwi-learning.md): LeKiwi の学習と推論の調査結果と選択肢（Phase 4 の判断材料）
- [docs/sweep.md](docs/sweep.md): LeKiwi の掃き掃除タスク（前提、データの約束事、進み具合）
- [AGENTS.md](AGENTS.md): コーディングエージェント向けの要点
