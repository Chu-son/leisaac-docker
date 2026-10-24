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

### sim は物理演算を CPU にし、テクスチャの予算を絞る（全タスク共通。2026-10-24 の計測）

- LeKiwi（Loft シーン）は、既定の設定だと sim だけで VRAM のピークが 11.4GB / 12.2GB（GUI のテレオペ時）になり、ブラウザを開いていると OOM になった。GR00T は起動していない。
- 計測方法: ゼロ action またはテレオペ（記録なし）で起動し、`nvidia-smi` を 1 秒ごとに記録した。値はデスクトップ分（約 1.0GB）を含む GPU 全体の使用量。

| 設定 | ヘッドレス | GUI テレオペ |
|---|---|---|
| 既定（GPU 物理、テクスチャ予算 0.6） | 11.2GB | 11.4GB |
| テクスチャ予算 0.2 | 10.7GB | 10.6GB |
| `--rendering_mode performance` | 10.9GB | — |
| CPU 物理 | 7.7GB | 8.1GB |
| CPU 物理 + テクスチャ予算 0.2 | 5.4GB | 6.1GB |
| CPU 物理 + テクスチャ予算 0.2 + performance | — | 5.8GB |
| 参考: PickOrange（GPU 物理） | 6.2GB | — |
| 参考: PickOrange + GR00T（`make check`、GPU 物理 → CPU 物理） | 11.0GB → 8.6GB | — |

- 決定: すべてのタスクで `SIM_DEVICE=cpu`、`TEX_BUDGET=0.2` を既定にする。タスクごとの例外は作らない（設定がタスクによって変わると、記録・再生・評価の条件を追いにくくなるため）。VRAM に余裕がある環境では、`.env` に `SIM_DEVICE=cuda` と `TEX_BUDGET=`（空）を書けば、GPU 物理と Kit の既定に戻る。performance は効果が小さく、画質が変わるので使わない。
- 設定の入り口は `.env` とコマンドラインだけにする（優先順位: コマンドライン > `.env` > 既定値）。make は `-include` した `.env` の値を環境変数より優先し、compose は逆に環境変数を優先するため、シェルの環境変数での指定はサポートしない。compose が参照する変数（`POLICY_DIR` など）は make が export し、make と compose で同じ値になるようにしている。
- 記録時の設定は `data/datasets/<name>.sim.env` に残し、追記（`RESUME=1`）と再生のときに `SIM_DEVICE` を照合する。違えば止める（`FORCE=1` で続行）。`TEX_BUDGET` は描画だけに効き、再生する action には影響しないので、記録するが照合しない。
- 効果が大きいのは、Loft の大量のコリジョンメッシュを GPU の PhysX に載せなくなること。テクスチャ予算は、CPU 物理と組み合わせたときに効く（GPU 物理のままではほとんど減らない）。カメラ画像（front / wrist）を比べた限りでは、画質の差は見られなかった。
- 副作用:
  - Loft の飾りの紙箱（`PaperEN/Box001`）は変形体（deformable）で、CPU 物理では扱えないため動かない（起動時にエラーが出るが、続行する）。タスクで使うティッシュ（`Tissue005`）は剛体なので、影響はない。
  - 物理演算の結果は GPU と CPU で一致しない。記録したときと同じ `SIM_DEVICE` で再生する（上の照合で守る）。10/24 より前に GPU で記録したデータ（`demo.hdf5`、`lekiwi_test.hdf5`）には `.sim.env` がなく、再生時に警告が出る。`SIM_DEVICE=cuda` で再生しないと結果がずれるが、記録が大きいと下記の理由で 12GB に収まらない。その場合は CPU で記録し直す。
  - replay は、エピソード全体（カメラ画像を含む）を `--device` に読み込む。GPU 物理だと、1.4GB の記録の再生で CUDA OOM になった。CPU 物理なら読み込み先はホストの RAM になる（再生時のピークは VRAM 5.9GB、RAM 19.2GB）。
  - CPU 物理のほうが、ステップは速い（ヘッドレスで約 2.7 倍）。
  - PickOrange の学習済みポリシーは GPU 物理のデータで学習されているが、CPU 物理での成功率は 20/30 で、劣化は見られなかった（docs/evaluation.md）。

## 5. Dockerfile・実行環境の再設計（D1〜D13）

| # | 変更 | 理由 |
|---|---|---|
| D1 | 土台を `cuda:12.8.0-devel` から `cuda:12.8.1-base` に変更 | CUDA ランタイムは wheel に含まれており、nvcc は使っていない |
| D2 | Miniforge（`latest`）と conda をやめ、uv と venv にする | 再現性とインストール速度のため。conda は Python を入れるためだけに使っていた |
| D3 | torch / torchvision / numpy を pyproject.toml に宣言し、torch の取得元の index を `[tool.uv.sources]` で明示する | 以前の「後から入れ直して上書きする」方式は、インストールの順序に依存して壊れやすかった |
| D4 | target ごとに uv プロジェクト（pyproject.toml と uv.lock）にし、`uv sync --locked` でインストールする | ビルドした日によって中身が変わらないようにするため。以前は、インストール結果を `uv pip freeze` で書き出して制約にしていたが、孤立したパッケージが混ざり、lock の生成を 2 回やり直す必要があった。uv.lock は取得元の URL とハッシュを記録し、ref や pyproject との食い違いはビルドの失敗として検知できる |
| D4' | sim の依存（isaacsim、IsaacLab、leisaac）を 1 回の解決で決める | 段階的に入れると、後の段階が isaacsim-kernel の完全一致ピンを暗黙に上書きする。uv プロジェクトでは、自然に 1 回の解決になる |
| D5 | git の ref を `compose.yaml` の `x-versions` に集め、leisaac は tag ではなく commit で固定する | 一緒に更新すべき値を 1 か所にまとめるため。tag は付け替えられる。Python パッケージのバージョンは pyproject.toml だけに書く |
| D6 | ホストと同じ UID/GID の `dev` ユーザーで動かす | `data/` に root 所有のファイルができないようにするため |
| D7 | GR00T サーバを compose のサービスにし、healthcheck を付ける | `make eval` が、サーバの準備が終わるまで待てるようにするため |
| D8 | GPU の指定を `gpus: all` だけにする | `runtime: nvidia` と `deploy` で二重に書いていたため |
| D9 | multi-stage にし、イメージ名を `leisaac-sim` / `leisaac-gr00t` に変える | 1 章の理由による |
| D10 | Dockerfile を `docker/` に移し、`.dockerignore` を追加する | `data/`（数 GB）がビルドコンテキストに含まれないようにするため |
| D11 | `make check` で自動検証する。推論は 1 ラウンドが最後まで完走するかだけを見て、成功率は記録のみ | 変更のたびに、同じ手順で動作を確かめられるようにするため。成功率は試行ごとにばらつくので、性能は `make eval EVAL_ROUNDS=10` で別に測る |
| D12 | sim を常駐させず、コマンドごとに `docker compose run --rm` で実行する | 常駐コンテナにはパッケージの追加などの状態が残り、実際に A/B 試験の変更が `down` まで残った。使い捨てなら、毎回イメージのとおりの状態で動く。止めるときもコンテナごと消せばよい（`make stop`）。推論サーバ（gr00t）は役割に合うので常駐のまま |
| D13 | 外部リソースを `assets.json` に、取得元と sha256 つきで記録し、`tools/fetch_assets.py` で取得・検証する | 以前は手で置いたもので、入手元の記録がなく、新しい環境では再現できなかった。ポリシーは revision を固定した。Loft は公式の配布物にないが、HF の `leisaac_env_backup` にあったため、revision 固定の `hf-dir` として取得する（ファイルごとの LFS sha256 とツリーハッシュで検証。初回のハッシュは取得物から記録）。上流では Loft の床・壁のテクスチャが `lightwheel_bedroom/LW_Loft/textures` に誤って置かれているため、`extra` で `lightwheel_loft/textures` に取り込む（取り込まないと床が黒くなる）。手動入手が必要なもの用に `manual` も残す |

### 再設計前と後の比較（どちらも同じ RTX 5070 マシンで計測）

| 項目 | 旧（Dockerfile.sim / .brain） | 新（docker/Dockerfile） |
|---|---|---|
| sim イメージ | 46.4GB | 30.2GB |
| GR00T イメージ | 32.6GB | 16.1GB |
| ビルド時間（キャッシュなし） | 未計測 | sim 約 15 分、gr00t 約 3 分（base は共有） |
| ヘッドレス推論 | 成功 1/1 | 累計で約 53%（docs/evaluation.md） |
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

## 6. 既知の制約（意図して受け入れているもの）

- **UID/GID を build arg で焼き込む**: 単一マシンでの運用なので受け入れる。ホストの UID が違う環境では、再ビルドが必要になる。
- **`index-strategy = "unsafe-best-match"`（sim の pyproject.toml）**: isaacsim が pypi.nvidia.com にあるため必要。index をまたいで版を選ぶ設定なので、lock を作るときは、依存の混同（同名の別パッケージが紛れ込むこと）のリスクがある。ビルド時は uv.lock の取得元とハッシュで入れるため、lock を更新したときの差分を確認する。
- **Python のバージョンが compose の mount 先のパスにある**: Kit のキャッシュは `.../python3.11/site-packages/isaacsim/kit/cache` にある。Python を上げるときは、compose.yaml と Dockerfile のこのパスも直す。
- **`/dev/input` 全体をコンテナから読める**: ゲームパッドを使うための仕様。キーボードを含むすべての入力デバイスが読める状態になる。
- **`decord` のインストール時に `built for a different platform` と表示される**: 推論では動画のデコードを使わないため、影響はない。学習でデータを読む段階になったら、動作を確認する。

## 7. 不採用にした案

- **ビルドキャッシュを削除する**: 再ビルドの時間を優先した。
- **`xhost +` で X の接続を許可する**: 誰でも接続できるようになってしまう。`~/.Xauthority` をマウントすれば足りる。
