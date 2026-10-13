SHELL := /bin/bash

TASK        ?= LeIsaac-SO101-PickOrange-v0
DEVICE      ?= keyboard
DATASET     ?= demo
HEADLESS    ?= 1
EVAL_ROUNDS ?= 10
INSTRUCTION ?= Pick up the orange and place it on the plate
LEADER_PORT ?= /dev/ttyACM0

# コンテナをホストと同じ UID/GID で作る。input グループはゲームパッド用
export HOST_UID  := $(shell id -u)
export HOST_GID  := $(shell id -g)
export INPUT_GID := $(shell getent group input | cut -d: -f3)

COMPOSE      := docker compose
SIM          := $(COMPOSE) exec sim python
DATASET_FILE := /workspace/data/datasets/$(DATASET).hdf5
# キーボード/ゲームパッドで記録したデータの再生には --task_type が必要
TASK_TYPE_ARG := $(if $(filter keyboard gamepad lekiwi-keyboard lekiwi-gamepad,$(DEVICE)),--task_type=$(DEVICE))

.PHONY: help build up up-leader down shell-sim shell-gr00t xhost list-envs teleop replay teleop-lekiwi \
        download-assets gr00t-up gr00t-down gr00t-logs eval check lock

help:
	@echo "使い方: make <target> [TASK=... DEVICE=... DATASET=... HEADLESS=0|1 EVAL_ROUNDS=N]"
	@echo ""
	@echo "  build            sim / gr00t イメージをビルド"
	@echo "  up / down        sim コンテナの起動 / 全停止"
	@echo "  up-leader        SO101 Leader ($(LEADER_PORT)) を渡して sim を起動"
	@echo "  list-envs        登録済みタスク一覧"
	@echo "  teleop           テレオペ + 記録 (TASK=$(TASK) DEVICE=$(DEVICE) -> data/datasets/$(DATASET).hdf5)"
	@echo "  replay           記録データの再生"
	@echo "  teleop-lekiwi    LeKiwi のテレオペ (DEVICE=keyboard|gamepad|lekiwi-leader)"
	@echo "  download-assets  lekiwi.usd を取得 (Loft シーンは手動配置)"
	@echo "  gr00t-up/-down   GR00T N1.5 推論サーバの起動 (healthy まで待つ) / 停止"
	@echo "  eval             閉ループ推論 (HEADLESS=0 で GUI 表示)"
	@echo "  check            動作確認一式 (タスク一覧 / サーバ疎通 / 推論 1 ラウンド / VRAM ピーク)"
	@echo "  lock             ビルド済みイメージから docker/lock/*.txt を再生成"
	@echo "  shell-sim / shell-gr00t / xhost"

build:
	$(COMPOSE) --profile inference build

up:
	$(COMPOSE) up -d sim

up-leader:
	LEADER_PORT=$(LEADER_PORT) $(COMPOSE) -f compose.yaml -f compose.leader.yaml up -d sim

down:
	$(COMPOSE) --profile inference down

shell-sim: up
	$(COMPOSE) exec sim bash

shell-gr00t:
	$(COMPOSE) --profile inference run --rm --no-deps gr00t bash

# 通常は不要 (~/.Xauthority をマウントしているため)。GUI が開かないときのみ
xhost:
	xhost +SI:localuser:$$(id -un)

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
	$(MAKE) teleop TASK=LeIsaac-LeKiwi-CleanupTrash-v0 DATASET=lekiwi_$(DATASET) \
		DEVICE=$(if $(filter keyboard gamepad,$(DEVICE)),lekiwi-$(DEVICE),$(DEVICE))

download-assets:
	mkdir -p assets/robots assets/scenes
	test -f assets/robots/lekiwi.usd || curl -fL -o assets/robots/lekiwi.usd \
		https://huggingface.co/LightwheelAI/leisaac_env/resolve/main/assets/robots/lekiwi.usd
	@test -f assets/scenes/lightwheel_loft/LW_Loft.usd || \
		echo "[手動] lightwheel_loft は https://lightwheel.ai/assets-type-content?type=LeRobotAssets から入手し assets/scenes/lightwheel_loft/LW_Loft.usd に配置"

gr00t-up:
	$(COMPOSE) --profile inference up -d --wait gr00t

gr00t-down:
	$(COMPOSE) --profile inference stop gr00t

gr00t-logs:
	$(COMPOSE) --profile inference logs -f gr00t

eval: up gr00t-up
	$(SIM) scripts/evaluation/policy_inference.py \
		--task=$(TASK) --eval_rounds=$(EVAL_ROUNDS) --policy_type=gr00tn1.5 \
		--policy_host=localhost --policy_port=5555 --policy_timeout_ms=15000 --policy_action_horizon=16 \
		--policy_language_instruction="$(INSTRUCTION)" --device=cuda --enable_cameras $(if $(filter 1,$(HEADLESS)),--headless)

check:
	./tools/check.sh

lock:
	$(COMPOSE) run --rm --no-deps -T sim uv pip freeze --exclude-editable > docker/lock/sim.txt.new
	$(COMPOSE) --profile inference run --rm --no-deps -T gr00t uv pip freeze --exclude-editable > docker/lock/gr00t.txt.new
	@for t in sim gr00t; do \
		{ printf '# make lock で生成 (%s target)。手で編集しない。\n' $$t; grep -vE '^(flash[-_]attn|torch|torchvision) ' docker/lock/$$t.txt.new | grep -v '^#'; } > docker/lock/$$t.txt; \
		rm docker/lock/$$t.txt.new; done
	@wc -l docker/lock/*.txt
