SHELL := /bin/bash

# マシンごとの設定は .env に書く (.env.example を参照。git には含めない)。
# 優先順位: make のコマンドライン > 環境変数 > .env > 下の既定値。compose も同じ .env を読む
-include .env

TASK        ?= LeIsaac-SO101-PickOrange-v0
DEVICE      ?= keyboard
DATASET     ?= demo
RESUME      ?= 0
HEADLESS    ?= 1
EVAL_ROUNDS ?= 10
INSTRUCTION ?= Pick up the orange and place it on the plate
LEADER_PORT ?= /dev/ttyACM0

# コンテナをホストと同じ UID/GID で作る。input グループはゲームパッド用
export HOST_UID  := $(shell id -u)
export HOST_GID  := $(shell id -g)
export INPUT_GID := $(shell getent group input | cut -d: -f3)
export LEADER_PORT

# SO101 Leader を使うときだけ、シリアルデバイスを渡す設定を重ねる
LEADER_DEVICES := so101leader lekiwi-leader
COMPOSE        := docker compose -f compose.yaml $(if $(filter $(LEADER_DEVICES),$(DEVICE)),-f compose.leader.yaml)

# sim はコマンドごとに使い捨てのコンテナで実行する (コンテナ内に状態を残さない)。
# 端末から実行したときだけ TTY を割り当てる
TTY_FLAG     := $(if $(shell test -t 0 && echo y),,-T)
SIM_RUN       = $(COMPOSE) run --rm $(TTY_FLAG) --name $(1) sim python
DATASET_PATH := data/datasets/$(DATASET).hdf5
DATASET_FILE := /workspace/$(DATASET_PATH)
# LeKiwi (Loft) は既定の設定だと VRAM を約 10GB 使い、12GB では OOM になる。
# LOW_VRAM=1 のとき、LOW_VRAM_TASKS は物理演算を CPU にし、テクスチャストリーミングの予算を絞る
# (約 6GB に収まる。docs/decisions.md)。VRAM に余裕がある環境では LOW_VRAM=0 にする。
# 記録と再生で物理演算のデバイスを揃えること (結果が一致しない)
LOW_VRAM       ?= 1
LOW_VRAM_TASKS ?= LeIsaac-LeKiwi-CleanupTrash-v0
LOW_VRAM_ON     = $(and $(filter 1,$(LOW_VRAM)),$(filter $(LOW_VRAM_TASKS),$(TASK)))
SIM_DEVICE ?= $(if $(LOW_VRAM_ON),cpu,cuda)
TEX_BUDGET ?= $(if $(LOW_VRAM_ON),0.2)
SIM_ARGS    = --device=$(SIM_DEVICE) \
	$(if $(TEX_BUDGET),--kit_args="--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=$(TEX_BUDGET)")
# キーボード/ゲームパッドで記録したデータの再生には --task_type が必要
TASK_TYPE_ARG := $(if $(filter keyboard gamepad lekiwi-keyboard lekiwi-gamepad,$(DEVICE)),--task_type=$(DEVICE))

.PHONY: help config build down stop shell-sim shell-gr00t xhost list-envs teleop replay teleop-lekiwi \
        assets gr00t-up gr00t-down gr00t-logs eval check lock lock-check

help:
	@echo "使い方: make <target> [TASK=... DEVICE=... DATASET=... RESUME=1 HEADLESS=0|1 EVAL_ROUNDS=N]"
	@echo ""
	@echo "  build            sim / gr00t イメージをビルド"
	@echo "  down / stop      全サービスの停止 / 実行中の sim コマンドの強制終了"
	@echo "  list-envs        登録済みタスク一覧"
	@echo "  teleop           テレオペ + 記録 (TASK=$(TASK) DEVICE=$(DEVICE) -> $(DATASET_PATH)。追記は RESUME=1)"
	@echo "                   SO101 Leader を使うときは DEVICE=so101leader LEADER_PORT=$(LEADER_PORT)"
	@echo "  replay           記録データの再生"
	@echo "  teleop-lekiwi    LeKiwi のテレオペ (DEVICE=keyboard|gamepad|lekiwi-leader)"
	@echo "                   LOW_VRAM=1 (既定) では SIM_DEVICE=cpu TEX_BUDGET=0.2。.env の LOW_VRAM=0 で無効"
	@echo "  assets           アセットと学習済みポリシーの取得・検証 (GROUP=pick-orange|lekiwi。Loft は手動配置)"
	@echo "  gr00t-up/-down   GR00T N1.5 推論サーバの起動 (healthy まで待つ) / 停止"
	@echo "  eval             閉ループ推論 (HEADLESS=0 で GUI 表示)"
	@echo "  config           実際に使われる設定の表示 (.env とコマンドラインを反映)"
	@echo "  check            動作確認一式 (タスク一覧 / サーバ疎通 / 推論 1 ラウンド / VRAM ピーク)"
	@echo "  lock / lock-check  docker/envs/*/uv.lock の更新 / 最新であることの確認"
	@echo "  shell-sim / shell-gr00t / xhost"

# 実際に使われる設定を表示する (.env とコマンドラインの指定を反映した値)
config:
	@echo "TASK=$(TASK)  DEVICE=$(DEVICE)  DATASET=$(DATASET)  RESUME=$(RESUME)"
	@echo "HEADLESS=$(HEADLESS)  EVAL_ROUNDS=$(EVAL_ROUNDS)  LEADER_PORT=$(LEADER_PORT)"
	@echo "LOW_VRAM=$(LOW_VRAM)  LOW_VRAM_TASKS=$(LOW_VRAM_TASKS)"
	@echo "SIM_DEVICE=$(SIM_DEVICE)  TEX_BUDGET=$(if $(TEX_BUDGET),$(TEX_BUDGET),(Kit の既定 0.6))"
	@echo "POLICY_DIR=$(or $(POLICY_DIR),(compose の既定))  GR00T_DATA_CONFIG=$(or $(GR00T_DATA_CONFIG),(compose の既定))"
	@echo ".env: $(if $(wildcard .env),あり,なし)"

build:
	$(COMPOSE) --profile inference build

down:
	$(COMPOSE) --profile inference down

# Kit は異常終了後に SIGTERM を無視して残ることがあるため、コンテナごと強制終了する
stop:
	-docker rm -f leisaac-teleop leisaac-replay leisaac-eval leisaac-list-envs leisaac-shell >/dev/null 2>&1

shell-sim:
	$(COMPOSE) run --rm $(TTY_FLAG) --name leisaac-shell sim bash

shell-gr00t:
	$(COMPOSE) --profile inference run --rm --no-deps $(TTY_FLAG) gr00t bash

# 通常は不要 (~/.Xauthority をマウントしているため)。GUI が開かないときのみ
xhost:
	xhost +SI:localuser:$$(id -un)

list-envs:
	$(call SIM_RUN,leisaac-list-envs) scripts/environments/list_envs.py

# leisaac は既存のデータセットがあると停止する。上書きも暗黙の追記もしないよう、先に確認する
teleop:
	@if [ "$(RESUME)" = 1 ]; then \
		[ -e "$(DATASET_PATH)" ] || { echo "[エラー] $(DATASET_PATH) がないため RESUME=1 は使えません"; exit 1; }; \
	elif [ -e "$(DATASET_PATH)" ]; then \
		echo "[エラー] $(DATASET_PATH) は既にあります。別の名前 (DATASET=...) を指定するか、追記するなら RESUME=1 を付けてください"; exit 1; \
	fi
	$(call SIM_RUN,leisaac-teleop) scripts/environments/teleoperation/teleop_se3_agent.py \
		--task=$(TASK) --teleop_device=$(DEVICE) --num_envs=1 $(SIM_ARGS) --enable_cameras \
		$(if $(filter $(LEADER_DEVICES),$(DEVICE)),--port=/dev/ttyACM0) \
		--record --dataset_file=$(DATASET_FILE) $(if $(filter 1,$(RESUME)),--resume)

replay:
	$(call SIM_RUN,leisaac-replay) scripts/environments/teleoperation/replay.py \
		--task=$(TASK) --num_envs=1 $(SIM_ARGS) --enable_cameras \
		--replay_mode=action --dataset_file=$(DATASET_FILE) $(TASK_TYPE_ARG)

teleop-lekiwi:
	$(MAKE) teleop TASK=LeIsaac-LeKiwi-CleanupTrash-v0 DATASET=lekiwi_$(DATASET) \
		DEVICE=$(if $(filter keyboard gamepad,$(DEVICE)),lekiwi-$(DEVICE),$(DEVICE))

# assets.json の外部リソース (USD アセット、学習済みポリシー) を取得して検証する。GROUP=pick-orange のように絞れる
assets:
	python3 tools/fetch_assets.py $(foreach g,$(GROUP),--group $(g))

gr00t-up:
	$(COMPOSE) --profile inference up -d --wait gr00t

gr00t-down:
	$(COMPOSE) --profile inference stop gr00t

gr00t-logs:
	$(COMPOSE) --profile inference logs -f gr00t

eval: gr00t-up
	$(call SIM_RUN,leisaac-eval) scripts/evaluation/policy_inference.py \
		--task=$(TASK) --eval_rounds=$(EVAL_ROUNDS) --policy_type=gr00tn1.5 \
		--policy_host=localhost --policy_port=5555 --policy_timeout_ms=15000 --policy_action_horizon=16 \
		--policy_language_instruction="$(INSTRUCTION)" $(SIM_ARGS) --enable_cameras $(if $(filter 1,$(HEADLESS)),--headless)

check:
	./tools/check.sh

# docker/envs/*/uv.lock を生成・更新する (UPGRADE=1 で全パッケージを最新に更新)。
# 取得元の URL とハッシュが記録されるので、再ビルドしても同じ内容になる
lock:
	$(COMPOSE) --profile tools run --rm lock-sim uv lock $(if $(filter 1,$(UPGRADE)),--upgrade)
	$(COMPOSE) --profile tools run --rm lock-gr00t uv lock $(if $(filter 1,$(UPGRADE)),--upgrade)
	@wc -l docker/envs/*/uv.lock

# lock が pyproject.toml と、固定した ref のソースに対して最新であることを確認する
lock-check:
	$(COMPOSE) --profile tools run --rm lock-sim uv lock --check
	$(COMPOSE) --profile tools run --rm lock-gr00t uv lock --check
