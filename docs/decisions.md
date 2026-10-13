# 設計判断の記録

## 1. イメージを sim と gr00t の 2 つに分ける

LeIsaac の推論は、sim がクライアントとなり、ZMQ でポリシーサーバ（GR00T）に問い合わせる構成です。依存関係も実際に確認したところ、1 つの Python 環境にはまとめられませんでした。

| パッケージ | sim（IsaacLab / leisaac）の要求 | GR00T `4af2b62` の要求 |
|---|---|---|
| gymnasium | `==1.2.0` | `==1.0.0` |
| protobuf | `>=4.25.8`（実際に入るのは 7.x） | `==4.25.1` |
| transformers | 制約なし（5.x） | `==4.51.3` |

検討した案と結論:

- **1 環境にまとめる**: 上の表の要求が両立しないため不可能。
- **1 イメージの中に venv を 2 つ持つ**: 実現はできるが採用しなかった。
  - イメージが大きくなる。
  - GR00T だけ更新したい場合でも、sim 部分まで作り直すことになる。
  - 推論サーバを別のマシンに移しにくくなる。
- **採用した案**: multi-stage の 1 ファイル（`docker/Dockerfile`）で、共通の `base` から target `sim` と `gr00t` を作る。
  - 名前は役割を表すものにした。以前の「brain」は何をするのか分からなかったため。
  - 別のポリシー（openpi など）を追加するときは、target を 1 つ足す。

## 2. Isaac Sim は pip 版を使う

- leisaac と IsaacLab の手順が pip 版を前提に書かれているため。
- NGC の Isaac Sim コンテナを土台にする案は採用しなかった。そのコンテナに入っている Python 環境と leisaac を組み合わせる必要があり、手間が増えるため。

## 3. GR00T は NVIDIA の本家リポジトリを commit で固定する

- leisaac v0.4.0 のクライアントが対象としている `4af2b62` に固定した。
- leisaac のドキュメントで案内されている fork（EverNorif `bd21973`）は古く、gripper の形状が合わない。詳細は [versions.md](versions.md) を参照。

## 4. 推論も同じ PC（VRAM 12GB）で行う

- RTX 5070 の 12GB に sim（約 5.5GB）と GR00T を同時に載せている。
- そのための工夫:
  - `tools/gr00t_server.py` で、GR00T の `Gr00tPolicy._load_model` を差し替える。読み込んだ後に `lm_head` を `Identity` に置き換え、行動の推論では使わない語彙の logits を確保しないようにしている。上流のコードは変更していない。
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定する。
- 実測のピークは 11.4GB / 12.2GB で、余裕はほとんどない。VRAM が足りなくなった場合は、gr00t サービスを別のマシンに移す。その場合は sim 側の `--policy_host` を変えるだけで済む。

## 5. Dockerfile の再設計（D1〜D11）

| # | 変更 | 理由 |
|---|---|---|
| D1 | 土台を `cuda:12.8.0-devel` から `cuda:12.8.1-base` に変更 | CUDA ランタイムは wheel に含まれており、nvcc は使っていない |
| D2 | Miniforge（`latest`）と conda をやめ、uv と venv にする | 再現性とインストール速度のため。conda は Python を入れるためだけに使っていた |
| D3 | `UV_CONSTRAINT` で torch / torchvision / numpy を固定する | 以前の「後から入れ直して上書きする」方式は、インストールの順序に依存して壊れやすかった |
| D4 | `docker/lock/*.txt` で間接依存も固定する | ビルドした日によって中身が変わらないようにするため |
| D4' | sim の依存（isaacsim、IsaacLab、leisaac）を 1 回の `uv pip install` でまとめて解決する | 段階的に入れると、後の段階が isaacsim-kernel の完全一致ピン（typing-extensions、click、idna）を暗黙に上書きする。その最終状態を lock にすると、前の段階に適用したときに解決できなくなる（実際にビルドが失敗した）。isaacsim だけを先に入れる層は、キャッシュのために残している |
| D5 | バージョンを `compose.yaml` の `x-versions` に集める | 一緒に更新すべき値を 1 か所にまとめるため。build args に直接渡せるので、`versions.env` は作らなかった |
| D6 | ホストと同じ UID/GID の `dev` ユーザーで動かす | `data/` に root 所有のファイルができないようにするため |
| D7 | GR00T サーバを compose のサービスにし、healthcheck を付ける | `make eval` が、サーバの準備が終わるまで待てるようにするため |
| D8 | GPU の指定を `gpus: all` だけにする | `runtime: nvidia` と `deploy` で二重に書いていたため |
| D9 | multi-stage にし、イメージ名を `leisaac-sim` / `leisaac-gr00t` に変える | 1 章の理由による |
| D10 | Dockerfile を `docker/` に移し、`.dockerignore` を追加する | `data/`（数 GB）がビルドコンテキストに含まれないようにするため |
| D11 | `make check` で自動検証する。推論は 1 ラウンドが最後まで完走するかだけを見て、成功率は記録のみ | 変更のたびに、同じ手順で動作を確かめられるようにするため。成功率は試行ごとにばらつくので、性能は `make eval EVAL_ROUNDS=10` で別に測る |

### 再設計前と後の比較（どちらも同じ RTX 5070 マシンで計測）

| 項目 | 旧（Dockerfile.sim / .brain） | 新（docker/Dockerfile） |
|---|---|---|
| sim イメージ | 46.4GB | 30.3GB |
| GR00T イメージ | 32.6GB | 17.1GB |
| ビルド時間（キャッシュなし） | 未計測 | sim 約 15 分、gr00t 約 3 分（base は共有） |
| ヘッドレス推論 | 成功 1/1 | 累計で約 50%（docs/evaluation.md） |
| VRAM ピーク（sim + gr00t） | 11.4GB | 11.2〜11.4GB |
| 実行ユーザー | root | dev（ホストの UID） |
| GUI 起動時の VRAM | 7.0GB | 7.0GB |

### isaacsim のピンと推論の成功率（2026-10-13 の検証）

- ピンどおりに 1 回で解決すると、transformers 5.17、huggingface-hub 1.16、tyro 0.9 など一部が古い版になる。
- 5 ラウンドの A/B 比較では、成功率が下がったように見えた（ピンどおり 1/5、以前の版 4/5）。
- そこで 10 ラウンドずつ測り直した:
  - 以前の版を override で強制した環境: 3/10
  - ピンどおりの環境: 7/10
- 結論: 有意な差はない。上流のピンどおりの構成を採用し、override は使わない。10 ラウンドの結果でも 3/10 と 7/10 の両方が出ており、これは評価そのもののばらつきの範囲に収まる。
- 教訓: PickOrange の学習済みポリシーは、成功率が試行ごとに大きくばらつく。5 ラウンド程度では判断しない。

## 6. 不採用にした案

- **ビルドキャッシュを削除する**: 再ビルドの時間を優先した。
- **`xhost +` で X の接続を許可する**: 誰でも接続できるようになってしまう。`~/.Xauthority` をマウントすれば足りる。
