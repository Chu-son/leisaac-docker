# LeKiwi の学習と推論（Phase 4 の判断材料）

2026-10-25 時点の調査。leisaac は v0.4.0（`1651c32`）。上流の main はそれ以降、ドキュメントの変更（`24d3bcd`）しかない。

## 上流で対応しているもの・していないもの

| 段階 | 上流の対応 | 根拠 |
|---|---|---|
| テレオペと記録（HDF5） | 対応 | `make teleop-lekiwi`。動作確認済み |
| 再生 | 対応 | `make replay-lekiwi`。動作確認済み |
| LeRobot 形式への変換 | 対応（要 `lerobot`） | `scripts/convert/isaaclab2lerobot.py`（v2）と `isaaclab2lerobotv3.py` は、環境の `build_lerobot_frame` を使う汎用の実装で、LeKiwi の環境定義にもこの関数がある。ただし、sim の環境には `lerobot` が入っていない（leisaac の extra `lerobot==0.4.2`） |
| 記録時に直接 LeRobot 形式で保存 | 対応（要 `lerobot`） | teleop の `--use_lerobot_recorder` |
| GR00T N1.5 / N1.6 での推論 | **未対応** | `policy_inference.py` が `task_type == "so101leader"` 以外では `ValueError` を出す |
| LeRobot / OpenPI のポリシーでの推論 | **未対応** | クライアント（`service_policy_clients.py`）の観測と action の組み立てが、`so101leader` 用しかない |
| GR00T の data config | **ない** | Isaac-GR00T `4af2b62` の data config は `so100`、`so100_dualcam` などで、LeKiwi 用はない |

env の action の構成は操作デバイスで変わる（`lekiwi-keyboard` では 11 次元で、`arm_action` 6、`gripper_action` 2、`wheel_action` 3）。LeRobot 形式に変換すると、state と action はどちらも「アームの関節位置 6 + 台車の速度 3」になる（`build_lerobot_frame`）。ただし、state のアームの取り出し方に不具合があると考えられる（sim の関節の並びは台車が先頭なのに、`joint_pos[:-3]` をアームとして扱っている。sweep.md）。

## 自前で作る必要があるもの

上流のコードは変更せず、`tools/` のラッパーで足す方針（`tools/gr00t_server.py` と同じ方針）。

1. **変換環境**: `lerobot` を入れた環境。sim の uv プロジェクトに extra を足すと、isaacsim のピンとぶつからないかを `make lock` で確認する必要がある。ぶつかる場合は、変換専用の 3 つ目の target にする。
2. **ポリシー用の定義**: GR00T を使う場合は、LeKiwi 用の data config と `modality.json`（state と action はどちらもアーム 6 + 台車 3、カメラは front と wrist）。
3. **推論クライアント**: LeKiwi の観測（関節位置と台車の速度）を送り、返ってきた action を 11 次元の env action に戻すクライアント。`policy_inference.py` は使えないので、ラッパーとして作る。

## 学習の選択肢（VRAM 12GB）

| 案 | 内容 | 12GB での見込み | 備考 |
|---|---|---|---|
| A | GR00T N1.5 を LoRA で微調整（`--lora_rank`、action head のみ、`batch_size` を小さく） | 未検証。推論だけで約 6GB 使うため、学習はかなり厳しい | 推論サーバと評価の仕組みは、今のものをほぼ流用できる |
| B | GR00T N1.5 をクラウドの GPU で微調整し、推論だけローカルで行う | 推論は約 6GB で、sim（約 5.5GB）と合わせて収まる見込み | 学習データのアップロードと、クラウドの費用がかかる |
| C | LeRobot の小さいポリシー（ACT など）をローカルで学習する | ACT は小さく、12GB でも学習できる見込み | 推論サーバ（LeRobot）とクライアントを新しく作る。GR00T の資産は使わない |

- どの案でも、まずデータを集める必要がある（数十エピソード程度）。CPU 物理（既定の設定）で記録したデータは、CPU 物理で評価する。
- 判断はユーザーが行う（計画の Phase 4）。
- 掃き掃除タスクでの具体的な進め方は [sweep.md](sweep.md) にまとめている。
