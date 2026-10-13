# AGENTS.md

LeIsaac（Isaac Sim + IsaacLab）と GR00T N1.5 の推論サーバを Docker で動かすリポジトリです。全体像は [README.md](README.md) を参照してください。

## 構成
- `docker/Dockerfile`: multi-stage 構成。`base` から `sim`（Isaac Sim + leisaac）と `gr00t`（推論サーバ）を作る。
- `compose.yaml`: サービスの定義と、バージョンの組（`x-versions`）。`gr00t` は profile `inference` に属する。
- `tools/`: gr00t イメージに含まれ、`/opt/tools` に置かれる。check.sh だけはホストで実行する。
- 操作はすべて `make` で行う。ターゲットの一覧は `make help` で表示できる。

## 不変条件（壊さないこと）
- **sim と gr00t は 1 つの環境にまとめない。** gymnasium と protobuf の要求が衝突する（docs/decisions.md）。
- **`LEISAAC_REF` と `GR00T_REF` は必ず組で更新する。** 片方だけ変えると action の形状が合わなくなる（docs/versions.md）。
- torch / torchvision / numpy は `docker/constraints.txt` で固定している。後から入れ直したり、上書きでバージョンを戻したりする書き方はしない。
- sim の依存は 1 回の `uv pip install` でまとめて解決する。段階的に入れると isaacsim-kernel の完全一致ピンが暗黙に崩れる。`uv pip check` で出る不整合は、`wheel` と `packaging` の 1 件だけになる（venv の seed によるもので、実害はない）。
- コンテナは非 root ユーザー `dev`（ホストの UID/GID）で動かす。root を前提にしたパスや所有者にしない。
- VRAM は 12GB でほぼ上限まで使っている。`tools/gr00t_server.py` の lm_head の差し替えと、`expandable_segments` を外さない。

## 変更したら
1. `make build`
2. `make check`。`RESULT: PASS` になること。ログは `data/check/<日時>/` に残る。check の推論は「完走したか」だけを見る。ポリシーの性能に影響しうる変更をしたときは、`make eval EVAL_ROUNDS=10` も実行する。判定方法は docs/evaluation.md に従う。1 回の結果で差があると結論しない。
3. 依存を変えた場合は、`docker/lock/*.txt` を枠だけに戻してからビルドし、check が通った後に `make lock` を実行して結果をコミットする。

## 運用上の注意
- 推論や check は、ホストで重い処理（並列ビルドなど）と同時に実行しない。RAM が足りなくなり、OOM killer が Kit を終了させる。
- プロセスを止めるときは `pkill -9 -f "[p]olicy_inference.py"` のように書く。`[]` を付けないと、実行したシェル自身にも一致してしまう。Kit は SIGTERM を無視することがある。
- `data/datasets/` と `data/policies/` はユーザーのデータなので、削除したり所有者を変えたりしない。
- ホストに変更を加える操作（xhost、sudo、イメージやボリュームの削除）は、事前にユーザーの確認を取る。

## コミット
- Conventional Commits の形式で書く（例: `fix(docker): ...`、`docs: ...`）。件名は英語の命令形にする。
- AI による署名やトレーラー（Co-Authored-By など）は付けない。
