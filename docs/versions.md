# バージョン固定

バージョンは 2 か所で管理します。

| 種類 | 置き場所 | 固定のしかた |
|---|---|---|
| git の ref（leisaac、Isaac-GR00T） | `compose.yaml` の `x-versions` | commit の SHA。build args として `docker/Dockerfile` に渡す |
| Python パッケージ | `docker/envs/{sim,gr00t}/pyproject.toml` | 宣言はここ。`uv.lock` で、推移的な依存も含めて URL とハッシュまで固定する |

## 現在の組み合わせ

| 項目 | 値 | 置き場所 |
|---|---|---|
| leisaac | v0.4.0（`1651c321e9b0c1bb54233211fc7b3cd70d8373d5`）。IsaacLab v2.3.0 をサブモジュールとして含む | x-versions の `LEISAAC_REF` |
| Isaac-GR00T | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` | x-versions の `GR00T_REF` |
| Isaac Sim | 5.1.0 | sim の pyproject.toml |
| torch / torchvision | 2.7.0 / 0.22.0（cu128） | 各 pyproject.toml（取得元の index も `[tool.uv.sources]` で明示） |
| numpy | 1.26.0 | 各 pyproject.toml |
| flash-attn | 2.8.3（`cu12torch2.7cxx11abiTRUE-cp311` の wheel の URL） | gr00t の pyproject.toml |
| ベースイメージ | `nvidia/cuda:12.8.1-base-ubuntu22.04` | Dockerfile |
| uv | 0.12.15 | Dockerfile |
| Python | 3.11 | Dockerfile と各 pyproject.toml（`requires-python`） |

- ベースイメージが devel 版ではないのは、CUDA ランタイムを torch / isaacsim の wheel が同梱しているため。

## 組で更新しなければならない値

- **`LEISAAC_REF` と `GR00T_REF`**
  - leisaac の `Gr00tServicePolicyClient` は、特定の Isaac-GR00T commit の応答形式を前提にしている。
  - 例: v0.4.0 のクライアントは `action.gripper` を `(16, 1)` として受け取る。
  - leisaac のドキュメントにある EverNorif の fork（`bd21973`）は古く、`(16,)` を返すため `ValueError` で失敗する。
  - leisaac を上げるときは、その版のクライアントがどの GR00T commit を対象にしているかを確認し、`GR00T_REF` も合わせて更新する。
- **torch、CUDA、flash-attn の wheel**
  - RTX 50 系（Blackwell、sm_120）には cu128 以降の torch が必要。
  - flash-attn の wheel は、torch のマイナーバージョン、CUDA の系列、Python のバージョンに合わせたものを選ぶ必要がある。
  - torch を変えるときは、sim と gr00t の両方の pyproject.toml と、flash-attn の URL を一緒に変える。

## lock の仕組み

- `uv.lock` は、`pyproject.toml` に加えて、leisaac / IsaacLab / GR00T のソースの依存宣言にも対応している。
- ref か pyproject.toml を変えたのに `make lock` を忘れると、`uv sync --locked` が失敗してビルドが止まる。古い lock のまま気付かずに進むことはない。
- 3 つのパッケージ（h5py、pandas、tyro）は、どの setup.py にも宣言されていないが、IsaacLab と leisaac のスクリプトが使うため、sim の pyproject.toml に書いている。
- `unsafe-best-match`（index をまたいで最良の版を選ぶ設定）は、sim の pyproject.toml にだけ書いている。isaacsim が pypi.nvidia.com にあるため。ビルド時は uv.lock に記録した取得元とハッシュで入れるので、この設定は lock を作るときだけ効く。

## 更新手順

1. `compose.yaml` の ref、または `docker/envs/*/pyproject.toml` を変更する。
2. `make lock` を実行する。既存の固定は維持され、必要な部分だけが更新される。全パッケージを最新にしたいときは `make lock UPGRADE=1`。
3. `make build`
4. `make check` が PASS し、`make eval EVAL_ROUNDS=10` の成功率が大きく下がっていないことを確認する（判定方法は [evaluation.md](evaluation.md)）。
5. `docker/envs/*/uv.lock` を含めてコミットする。
6. 確認だけしたいときは `make lock-check`。lock が最新でなければ失敗する。
