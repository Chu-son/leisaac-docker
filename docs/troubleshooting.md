# トラブルシューティング

この環境の構築中に実際に起きた問題と、その対処をまとめています。

## 起動・表示

### Isaac Sim が起動途中で止まり、何も表示されない
- 原因: pip 版の Isaac Sim は、初回起動時に EULA への同意を対話的に求める。この入力待ちで止まっている。pip 版では `ACCEPT_EULA` は効かない。
- 対処: sim イメージには `OMNI_KIT_ACCEPT_EULA=YES` を設定済み。自分でコンテナを作る場合は、この環境変数を必ず設定する。

### 標準出力に何も表示されない（TTY のない実行）
- 原因: TTY がないため、Python の stdout がバッファされたまま表示されない。
- 対処: base イメージに `PYTHONUNBUFFERED=1` を設定済み。

### GUI のウィンドウが開かない（`cannot open display`）
- `DISPLAY` の値がホストと一致しているか確認する（`echo $DISPLAY`）。compose はホストの値を引き継ぐ。
- `~/.Xauthority` は読み取り専用でマウントしてあるので、通常は `xhost` を実行する必要はない。
- それでも開かない場合は、`make xhost` で現在のユーザーにだけ X サーバへの接続を許可する。

### 初回起動が数分かかる
- シェーダとメッシュのキャッシュを作っているため。キャッシュは名前付きボリューム `sim-*-cache` に保存されるので、2 回目以降は速くなる。
- `docker compose down -v` を実行するとキャッシュも消える。

## メモリ

### GR00T サーバが CUDA OOM で落ちる
- VRAM 12GB の構成では、sim と GR00T を同時に動かすと VRAM が限界に近い（ピーク約 11.4GB / 12.2GB）。
- 対処:
  - `tools/gr00t_server.py` は、推論では使わない Qwen3 の `lm_head` を `Identity` に差し替えている。これで巨大な logits 用のメモリを確保しないようにしている。
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定して、メモリの断片化を抑えている。
  - ブラウザなど GPU を使うアプリを閉じる。現在の使用量は `nvidia-smi` で確認できる。

### LeKiwi のテレオペや再生で CUDA OOM になる
- 例: `SIM_DEVICE=cuda` で LeKiwi を動かした
- 原因: Loft シーンを GPU 物理で動かすと、sim だけで VRAM を約 10GB 使う。replay はさらに、エピソード全体を GPU に読み込む。
- 対処: 既定の `SIM_DEVICE=cpu TEX_BUDGET=0.2` なら約 5〜6GB に収まる。`.env` で変えていないか、`make config` で確認する。GPU で記録した古いデータは `SIM_DEVICE=cuda` でないと正しく再生できないが、記録が大きいと OOM になる。その場合は CPU で記録し直す。ブラウザなど GPU を使うアプリは閉じる。
- 詳細: [decisions.md](decisions.md) の「LeKiwi は物理演算を CPU にし、テクスチャの予算を絞る」

### 推論中に sim のプロセスが何も出力せずに消える
- 原因: ホストの RAM 不足で、カーネルの OOM killer に強制終了されている。Kit は RAM を約 11GB 使う。
- 確認: `journalctl -k | grep -i "out of memory"`
- 実例: 別プロジェクトの `colcon build -j28` と同時に実行したときに発生した。
- 対処: 重いビルドと並行して実行しない。

## GR00T

### `ValueError`（gripper の形状が `(16,)` と `(16,1)` で一致しない）
- 原因: leisaac のクライアントと Isaac-GR00T のバージョンが合っていない。
- 対処: [versions.md](versions.md) にある組み合わせで固定する。単体での確認は `docker compose --profile inference exec gr00t python /opt/tools/probe_gr00t.py` で行える。

### `gr00t-up` が healthy にならない
- モデルの読み込みに 1〜2 分かかる。healthcheck の `start_period` は 180 秒にしてある。
- ログは `make gr00t-logs` で確認する。
- ポリシーの置き場所が `data/policies/leisaac-pick-orange-v0` と異なる場合は、`POLICY_DIR` で指定する。

## デバイス

### ゲームパッドが `Operation not permitted` で開けない
- 原因: `/dev/input` を bind マウントしても、デバイスの cgroup で許可されていない。
- 対処: compose で `device_cgroup_rules: ['c 13:* rmw']` を設定し、`input` グループを `group_add` で追加している。
- 注意: `/dev/input/js0` がゲームパッドとは限らない。例えば ASRock の LED Controller が js0 になることがある。`cat /proc/bus/input/devices` で確認する。

### Leader アームのシリアルが見えない
- `make teleop DEVICE=so101leader LEADER_PORT=/dev/ttyACMx` で実行する。`compose.leader.yaml` が自動で重ねられ、デバイスがコンテナに渡される。
- `LEADER_PORT` は、ホスト側のデバイス名（`ls /dev/ttyACM*` で確認）。
- キャリブレーションは `data/calibration/` に保存される。

## 運用

### Kit がハングして終了しない
- 例外の後、Kit が SIGTERM を無視して残ることがある。
- 対処: `make stop` で、実行中の sim のコンテナごと強制終了する。sim は常駐させず、コマンドごとに使い捨てで実行しているため、コンテナを消せば状態も残らない。

### `data/` 配下に root 所有のファイルがある
- 旧構成ではコンテナが root で動いていたため、その頃に作ったファイルは root 所有になっている。
- 現在はホストと同じ UID のユーザー `dev` で動く。
- 読むだけなら問題ない。書き込みが必要な場合は、`sudo chown -R $(id -u):$(id -g) data/` を実行する。
