# 手作業タスク

自動化できない作業と、人が目で見て確認する項目の一覧です。

| ID | 内容 | 必要になる場面 | 状態 |
|---|---|---|---|
| M1 | ホストに NVIDIA ドライバ（580 系で確認）と nvidia-container-toolkit を入れる | すべて | ✅ |
| M2 | Lightwheel Loft シーンを [lightwheel.ai](https://lightwheel.ai/assets-type-content?type=LeRobotAssets) から入手し、`assets/scenes/lightwheel_loft/LW_Loft.usd` に置く。`lekiwi.usd` は `make assets` で取得できる。置いたあとは `make assets GROUP=lekiwi` で検出される | LeKiwi のタスク | 未 |
| M3 | ゲームパッドを接続し、`cat /proc/bus/input/devices` で js 番号を確認する | `DEVICE=gamepad` | 未接続 |
| M4 | SO101 Leader を接続してキャリブレーションする（`make teleop DEVICE=so101leader LEADER_PORT=/dev/ttyACMx`）。結果は `data/calibration/` に保存される | `DEVICE=so101leader` | 未接続 |
| M5 | GUI でテレオペと記録（`make teleop DATASET=<新しい名前>`）、リプレイ（`make replay DATASET=<同じ名前>`）を目で見て確認する | チュートリアル | 未確認 |
| M6 | 推論の前に、GPU を使うアプリ（ブラウザなど）を閉じる。重いビルドを同時に走らせない | `make eval` / `make check` | 毎回 |
| M7 | 旧構成で作った root 所有のファイル（`data/datasets/demo.hdf5`、`data/policies`）の所有者を、必要に応じて変更する（`sudo chown -R $(id -u):$(id -g) data/`） | data/ に書き込むとき | 任意 |
| M8 | 学習済みポリシー `leisaac-pick-orange-v0` は `make assets` が取得する（手作業は不要） | 推論 | ✅ |

## キーボードでのテレオペ操作（SO101）

- キーの割り当ては、テレオペを起動したときにコンソールへ表示される。
- `B` で操作を開始する。
- `N` で成功として記録し、エピソードを終える。
- `R` でリセットする（そのエピソードは記録しない）。
