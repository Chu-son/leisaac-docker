# leisaac-docker

[LeIsaac](https://github.com/LightwheelAI/leisaac)（Isaac Sim / IsaacLab 上で SO101 と LeKiwi を動かすシミュレーション環境）を Docker で動かすための環境です。対象はテレオペ、記録、リプレイ、GR00T N1.5 による閉ループ推論です。

- **sim**: Isaac Sim 5.1 + IsaacLab 2.3 + LeIsaac v0.4.0。テレオペ、リプレイ、推論クライアントを担当する
- **gr00t**: GR00T N1.5 の推論サーバ。sim とは ZMQ の `:5555` で通信する

2 つは `docker/Dockerfile` の multi-stage target として作ります。分けている理由は [docs/decisions.md](docs/decisions.md) を参照してください。

## 必要環境

| 項目 | 確認済みの構成 |
|---|---|
| GPU | NVIDIA RTX 5070 12GB（Blackwell / sm_120）。sim と gr00t を同時に動かすには 12GB 以上が必要 |
| ドライバ | 580.178.04 |
| OS | Ubuntu 22.04、X11 |
| Docker | Docker Engine + Compose v2 + nvidia-container-toolkit |
| ディスク | イメージ約 46GB（sim 30GB、gr00t 16GB）とビルドキャッシュ |
| RAM | 32GB。推論中は Kit が約 11GB 使う |

## クイックスタート（PickOrange）

```bash
make build                 # sim / gr00t のイメージをビルドする（初回は約 20 分）
make list-envs             # 登録されているタスクを表示する（15 件）

# テレオペと記録（GUI）。B で開始、N で成功として保存、R でやり直し
make teleop DATASET=demo   # 保存先は data/datasets/demo.hdf5。同じ名前が既にあれば止まる（追記は RESUME=1）
make replay DATASET=demo

# GR00T N1.5 による閉ループ推論
# 事前に学習済みポリシーを data/policies/leisaac-pick-orange-v0 に置いておく
make eval EVAL_ROUNDS=10              # ヘッドレスで実行（成功率は約 50%。docs/evaluation.md）
make eval HEADLESS=0 EVAL_ROUNDS=1    # GUI で表示して実行

make check                 # 動作確認一式。結果は RESULT: PASS / FAIL で表示される
make down                  # 推論サーバなど、常駐しているものを止める
```

sim は常駐させず、`make` のコマンドごとに使い捨てのコンテナで実行します。実行中のものを強制終了するときは `make stop` です。

- 初回の起動では、シェーダのキャッシュを作るのに数分かかります。
- 推論の前に、ブラウザなど GPU を使うアプリを閉じてください。VRAM のピークは 11.4GB / 12.2GB です。

## Makefile

| ターゲット | 内容 |
|---|---|
| `build` | 両方のイメージをビルドする |
| `down` / `stop` | 常駐しているサービスを止める / 実行中の sim のコマンドを強制終了する |
| `list-envs` | タスクの一覧を表示する |
| `teleop` / `replay` | テレオペと記録 / 記録したデータの再生。Leader は `DEVICE=so101leader LEADER_PORT=/dev/ttyACMx` |
| `teleop-lekiwi` | LeKiwi をテレオペする。Loft シーンが必要（[docs/manual-tasks.md](docs/manual-tasks.md)） |
| `download-assets` | `lekiwi.usd` を取得する |
| `gr00t-up` / `gr00t-down` / `gr00t-logs` | 推論サーバを起動して healthy まで待つ / 停止する / ログを表示する |
| `eval` | 閉ループ推論を実行する |
| `check` | タスク一覧、サーバとの疎通と action の形状、ヘッドレス推論 1 ラウンド、VRAM ピークを確認する |
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

## ディレクトリ構成

```
compose.yaml          # サービス定義と git の ref（x-versions）
compose.leader.yaml   # Leader アーム接続時に重ねる設定
Makefile
docker/
  Dockerfile          # base → sim / gr00t
  envs/{sim,gr00t}/   # pyproject.toml（依存の宣言）と uv.lock（ハッシュ付きの固定）
tools/
  gr00t_server.py     # VRAM 節約のラッパーを付けて GR00T の推論サーバを起動する
  probe_gr00t.py      # サーバとの疎通と action の形状を確認する
  check.sh            # make check の本体
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
- [AGENTS.md](AGENTS.md): コーディングエージェント向けの要点
