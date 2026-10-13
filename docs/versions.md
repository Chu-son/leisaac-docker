# バージョン固定

バージョンの組み合わせは `compose.yaml` の `x-versions` にまとめてあり、build args として `docker/Dockerfile` に渡されます。

| 変数 | 値 | 使う target |
|---|---|---|
| `ISAACSIM_VERSION` | `5.1.0` | sim |
| `LEISAAC_REF` | `v0.4.0`（IsaacLab v2.3.0 をサブモジュールとして含む） | sim |
| `GR00T_REF` | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`（NVIDIA/Isaac-GR00T） | gr00t |
| `TORCH_VERSION` / `TORCHVISION_VERSION` | `2.7.0` / `0.22.0`（cu128） | base（両方） |

Dockerfile 側だけで固定している値:

| 項目 | 値 | 理由 |
|---|---|---|
| ベースイメージ | `nvidia/cuda:12.8.1-base-ubuntu22.04` | CUDA ランタイムは torch / isaacsim の wheel が同梱しているため、devel 版は使わない |
| uv | `ghcr.io/astral-sh/uv:0.12.15` | |
| Python | 3.11 | Isaac Sim 5.x の要件 |
| flash-attn | `2.8.3+cu12torch2.7cxx11abiTRUE-cp311` の wheel | GR00T の Qwen 系バックボーンが FA2 を必須とする。RTX 5070（sm_120）で動作を確認済み |

## 組で更新しなければならない値

- **`LEISAAC_REF` と `GR00T_REF`**
  - leisaac の `Gr00tServicePolicyClient` は、特定の Isaac-GR00T commit の応答形式を前提にしている。
  - 例: v0.4.0 のクライアントは `action.gripper` を `(16, 1)` として受け取る。
  - leisaac のドキュメントにある EverNorif の fork（`bd21973`）は古く、`(16,)` を返すため `ValueError` で失敗する。
  - leisaac を上げるときは、その版のクライアントがどの GR00T commit を対象にしているかを確認し、`GR00T_REF` も合わせて更新する。
- **`TORCH_VERSION`、CUDA、flash-attn wheel**
  - RTX 50 系（Blackwell、sm_120）には cu128 以降の torch が必要。
  - flash-attn の wheel は torch のマイナーバージョン、CUDA の系列、Python のバージョンに合わせたものを選ぶ必要がある。
- **`docker/constraints.txt`**
  - torch / torchvision / numpy を全 target で固定している。
  - 後から入れるパッケージがこれらを入れ替えるのを防ぐためのもの。numpy は Isaac Sim 5.1 に合わせて 1.26.0 にしている。

## 依存のロック

- `docker/lock/{sim,gr00t}.txt` は、ビルドしたイメージの `uv pip freeze` の結果。ビルド時に constraint として効く。
- 一覧から除いているもの:
  - editable install（leisaac、IsaacLab、GR00T）
  - torch / torchvision（`constraints.txt` で管理）
  - flash-attn（wheel の URL で管理）

## 更新手順

1. `compose.yaml` の `x-versions` を変更する。torch を変える場合は `docker/constraints.txt` と flash-attn wheel も変更する。
2. `docker/lock/*.txt` を枠だけ（コメント行のみ）に戻す。古いロックが残っていると、新しい依存の解決を妨げる。
3. `make build`
4. `make check` が PASS し、`make eval EVAL_ROUNDS=10` の成功率が下がっていないことを確認する。
5. `make lock` でロックを再生成してコミットする。
