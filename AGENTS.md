# AGENTS.md

LeIsaac（Isaac Sim + IsaacLab）と GR00T N1.5 の推論サーバを Docker で動かすリポジトリです。全体像は [README.md](README.md) を参照してください。

## 構成
- `docker/Dockerfile`: multi-stage 構成。`base` から `sim`（Isaac Sim + leisaac）と `gr00t`（推論サーバ）を作る。
- `docker/envs/{sim,gr00t}/`: Python 依存の宣言（pyproject.toml）と固定（uv.lock）。
- `assets.json`: 外部リソース（USD アセット、学習済みポリシー）の取得元とハッシュ。`make assets` で取得・検証する。新しいアセットを使うときはここに追記する。
- `.env.example`: 設定の雛形。設定の入り口は `.env` とコマンドラインだけ（優先順位: コマンドライン > `.env` > 既定値）。`.env` は Makefile と compose の両方が読み、git には含めない。Makefile に変数を足したら、ここと `make config` にも書く。compose が参照する変数は Makefile で export する。
- sim の設定（`SIM_DEVICE` など）は全タスクで共通にする。タスクごとの例外を作らない。
- `compose.yaml`: サービスの定義と、git の ref（`x-versions`）。`gr00t` は profile `inference` に属する。
- `tools/`: gr00t イメージに含まれ、`/opt/tools` に置かれる。check.sh だけはホストで実行する。
- 操作はすべて `make` で行う。ターゲットの一覧は `make help` で表示できる。

## 不変条件（壊さないこと）
- **sim と gr00t は 1 つの環境にまとめない。** gymnasium と protobuf の要求が衝突する（docs/decisions.md）。
- **`LEISAAC_REF` と `GR00T_REF` は必ず組で更新する。** 片方だけ変えると action の形状が合わなくなる（docs/versions.md）。
- Python の依存は `docker/envs/<target>/pyproject.toml` に宣言し、`uv.lock` で固定する。`pip install` や `uv pip install` でイメージを組み立てない。ref や pyproject を変えたら `make lock` を実行する（忘れるとビルドが失敗する）。
- torch の取得元は `[tool.uv.sources]` で cu128 の index に固定している。isaacsim のバージョン指定を上書きする override は、必要な根拠が出るまで入れない（docs/decisions.md）。
- コンテナは非 root ユーザー `dev`（ホストの UID/GID）で動かす。root を前提にしたパスや所有者にしない。
- VRAM は 12GB でほぼ上限まで使っている。`tools/gr00t_server.py` の lm_head の差し替えと、`expandable_segments` を外さない。

## 変更したら
1. `make build`
2. `make check`。`RESULT: PASS` になること。ログは `data/check/<日時>/` に残る。check の推論は「完走したか」だけを見る。ポリシーの性能に影響しうる変更をしたときは、`make eval EVAL_ROUNDS=10` も実行する。判定方法は docs/evaluation.md に従う。1 回の結果で差があると結論しない。
3. 依存や ref を変えた場合は、`make lock` で `uv.lock` を更新し、`make build` と `make check` を通してから、uv.lock を含めてコミットする。

## 運用上の注意
- 推論や check は、ホストで重い処理（並列ビルドなど）と同時に実行しない。RAM が足りなくなり、OOM killer が Kit を終了させる。
- sim のコマンドは `docker compose run --rm` で都度実行する（常駐させない）。止めるときは `make stop`（コンテナごと強制終了する）。Kit は SIGTERM を無視することがあるため、`pkill` ではなくコンテナを消す。
- `data/datasets/` と `data/policies/` はユーザーのデータなので、削除したり所有者を変えたりしない。
- ホストに変更を加える操作（xhost、sudo、イメージやボリュームの削除）は、事前にユーザーの確認を取る。

## コミット
- Conventional Commits の形式で書く（例: `fix(docker): ...`、`docs: ...`）。件名は英語の命令形にする。
- AI による署名やトレーラー（Co-Authored-By など）は付けない。
