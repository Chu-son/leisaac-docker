SHELL := /bin/bash

TASK    ?= LeIsaac-SO101-PickOrange-v0
DEVICE  ?= keyboard
DATASET ?= demo
HEADLESS ?= 1
EVAL_ROUNDS ?= 10
INSTRUCTION ?= Pick up the orange and place it on the plate
POLICY_DIR ?= /workspace/data/policies/leisaac-pick-orange-v0
LEADER_PORT ?= /dev/ttyACM0

DATASET_FILE := /workspace/data/datasets/$(DATASET).hdf5
SIM := docker compose exec sim python
# キーボード/ゲームパッドで記録したデータの再生には --task_type が必要
TASK_TYPE_ARG := $(if $(filter keyboard gamepad lekiwi-keyboard lekiwi-gamepad,$(DEVICE)),--task_type=$(DEVICE))

.PHONY: help build up up-leader down bash-sim bash-brain xhost list-envs teleop replay \
        teleop-lekiwi download-assets policy-server eval

help:
	@echo "利用可能なコマンド (変数: TASK DEVICE DATASET HEADLESS EVAL_ROUNDS LEADER_PORT):"
	@echo "  make build             - sim / brain をビルド"
	@echo "  make up                - sim をバックグラウンド起動"
	@echo "  make up-leader         - SO101 Leader ($(LEADER_PORT)) を渡して sim を起動"
	@echo "  make xhost             - コンテナ(root)からの X11 表示を許可 (GUI を使う前に実行)"
	@echo "  make list-envs         - 登録済みタスク一覧"
	@echo "  make teleop            - テレオペ + 記録 (TASK=$(TASK) DEVICE=$(DEVICE))"
	@echo "  make replay            - 記録データの再生"
	@echo "  make teleop-lekiwi     - LeKiwi のテレオペ (DEVICE=lekiwi-keyboard 等)"
	@echo "  make download-assets   - lekiwi.usd を取得 (Loft シーンは手動配置)"
	@echo "  make policy-server     - brain で GR00T N1.5 推論サーバ起動 (5555)"
	@echo "  make eval              - sim から閉ループ推論 (HEADLESS=0 で GUI)"
	@echo "  make bash-sim / bash-brain"

build:
	docker compose build

up:
	docker compose up -d sim

up-leader:
	LEADER_PORT=$(LEADER_PORT) docker compose -f compose.yaml -f compose.leader.yaml up -d sim

down:
	docker compose down

bash-sim:
	docker compose exec sim bash

bash-brain:
	docker compose run --rm brain bash

xhost:
	xhost +SI:localuser:root

list-envs: up
	$(SIM) scripts/environments/list_envs.py

teleop: up
	$(SIM) scripts/environments/teleoperation/teleop_se3_agent.py \
		--task=$(TASK) --teleop_device=$(DEVICE) --num_envs=1 --device=cuda --enable_cameras \
		$(if $(filter so101leader lekiwi-leader,$(DEVICE)),--port=/dev/ttyACM0) \
		--record --dataset_file=$(DATASET_FILE)

replay: up
	$(SIM) scripts/environments/teleoperation/replay.py \
		--task=$(TASK) --num_envs=1 --device=cuda --enable_cameras \
		--replay_mode=action --dataset_file=$(DATASET_FILE) $(TASK_TYPE_ARG)

teleop-lekiwi:
	$(MAKE) teleop TASK=LeIsaac-LeKiwi-CleanupTrash-v0 DEVICE=$(if $(filter keyboard,$(DEVICE)),lekiwi-keyboard,$(DEVICE)) DATASET=lekiwi_$(DATASET)

download-assets:
	mkdir -p assets/robots assets/scenes
	test -f assets/robots/lekiwi.usd || curl -fL -o assets/robots/lekiwi.usd \
		https://huggingface.co/LightwheelAI/leisaac_env/resolve/main/assets/robots/lekiwi.usd
	@test -f assets/scenes/lightwheel_loft/LW_Loft.usd || \
		echo "[手動] lightwheel_loft は https://lightwheel.ai/assets-type-content?type=LeRobotAssets から入手し assets/scenes/lightwheel_loft/LW_Loft.usd に配置"

policy-server:
	docker compose run --rm brain python /workspace/code/gr00t_server.py --server \
		--model_path $(POLICY_DIR) --embodiment_tag new_embodiment --data_config so100_dualcam --port 5555

eval: up
	$(SIM) scripts/evaluation/policy_inference.py \
		--task=$(TASK) --eval_rounds=$(EVAL_ROUNDS) --policy_type=gr00tn1.5 \
		--policy_host=localhost --policy_port=5555 --policy_timeout_ms=15000 --policy_action_horizon=16 \
		--policy_language_instruction="$(INSTRUCTION)" --device=cuda --enable_cameras $(if $(HEADLESS),--headless)
