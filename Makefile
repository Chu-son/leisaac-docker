SHELL := /bin/bash

# 設定の入り口は .env (マシンごと。.env.example を参照。git には含めない) とコマンドラインの 2 つだけ。
# 優先順位: コマンドライン > .env > 下の既定値。シェルの環境変数では指定しない (.env の値が優先される)
-include .env

# --- 対象と記録
TASK        ?= LeIsaac-SO101-PickOrange-v0
DEVICE      ?= keyboard
DATASET     ?= demo
RESUME      ?= 0
FORCE       ?= 0
LEADER_PORT ?= /dev/ttyACM0

# --- sim の設定 (すべてのタスクで共通)。既定は VRAM 12GB 向けで、物理演算を CPU にし、
# テクスチャストリーミングの予算 (VRAM に対する割合) を絞る。VRAM に余裕があれば .env で
# SIM_DEVICE=cuda、TEX_BUDGET= (空にすると Kit の既定 0.6) にする (docs/decisions.md)
SIM_DEVICE  ?= cpu
TEX_BUDGET  ?= 0.2

# --- 推論 (eval)。POLICY_DIR / GR00T_DATA_CONFIG は gr00t サーバの設定で、既定値は compose.yaml にある
HEADLESS          ?= 1
EVAL_ROUNDS       ?= 10
INSTRUCTION       ?= Pick up the orange and place it on the plate
POLICY_HOST       ?= localhost
POLICY_TIMEOUT_MS ?= 15000
ACTION_HORIZON    ?= 16

# --- その他
GROUP   ?=
UPGRADE ?= 0

# compose が make と同じ値を使うよう、compose.yaml が参照する変数を渡す (空なら compose.yaml の既定値)
export LEADER_PORT POLICY_DIR GR00T_DATA_CONFIG

# コンテナをホストと同じ UID/GID で作る。input グループはゲームパッド用
export HOST_UID  := $(shell id -u)
export HOST_GID  := $(shell id -g)
export INPUT_GID := $(shell getent group input | cut -d: -f3)

# SO101 Leader を使うときだけ、シリアルデバイスを渡す設定を重ねる
LEADER_DEVICES := so101leader lekiwi-leader
COMPOSE        := docker compose -f compose.yaml $(if $(filter $(LEADER_DEVICES),$(DEVICE)),-f compose.leader.yaml)

# sim はコマンドごとに使い捨てのコンテナで実行する (コンテナ内に状態を残さない)。
# 端末から実行したときだけ TTY を割り当てる
TTY_FLAG     := $(if $(shell test -t 0 && echo y),,-T)
SIM_RUN       = $(COMPOSE) run --rm $(TTY_FLAG) --name $(1) sim python
SIM_ARGS      = --device=$(SIM_DEVICE) \
	$(if $(TEX_BUDGET),--kit_args="--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=$(TEX_BUDGET)")
DATASET_PATH := data/datasets/$(DATASET).hdf5
DATASET_FILE := /workspace/$(DATASET_PATH)

# 記録時の sim 設定をデータセットの隣に残し、追記と再生のときに照合する。
# 物理演算のデバイスが違うと結果が一致しないため、SIM_DEVICE が違えば止める (FORCE=1 で続行)
SIM_RECORD := data/datasets/$(DATASET).sim.env
define check_sim_record
	@if [ ! -e "$(SIM_RECORD)" ]; then \
		echo "[警告] $(SIM_RECORD) がないため、記録時の sim 設定を確認できません (この仕組みより前の記録)"; \
	else \
		rec=$$(sed -n 's/^SIM_DEVICE=//p' "$(SIM_RECORD)"); \
		if [ "$$rec" != "$(SIM_DEVICE)" ]; then \
			if [ "$(FORCE)" = 1 ]; then \
				echo "[警告] SIM_DEVICE=$$rec で記録されたデータを SIM_DEVICE=$(SIM_DEVICE) で実行します (FORCE=1)"; \
			else \
				echo "[エラー] $(DATASET) は SIM_DEVICE=$$rec で記録されています (現在は $(SIM_DEVICE))。物理演算の結果が一致しないため止めます。"; \
				echo "         SIM_DEVICE=$$rec を指定するか、承知のうえで続けるなら FORCE=1 を付けてください"; \
				exit 1; \
			fi; \
		fi; \
	fi
endef

# キーボード/ゲームパッドで記録したデータの再生には --task_type が必要
TASK_TYPE_ARG := $(if $(filter keyboard gamepad lekiwi-keyboard lekiwi-gamepad,$(DEVICE)),--task_type=$(DEVICE))

.PHONY: help config build down stop shell-sim shell-gr00t xhost list-envs smoke teleop replay teleop-lekiwi replay-lekiwi \
        assets gr00t-up gr00t-down gr00t-logs eval check lock lock-check

help:
	@echo "使い方: make <target> [変数=値 ...]  変数は .env にも書ける (.env.example)。現在の値は make config"
	@echo ""
	@echo "  build            sim / gr00t イメージをビルド"
	@echo "  down / stop      全サービスの停止 / 実行中の sim コマンドの強制終了"
	@echo "  list-envs        登録済みタスク一覧"
	@echo "  smoke            TASK をヘッドレスで作り、ゼロ action で数ステップ動かす (起動確認)"
	@echo "  teleop           テレオペ + 記録 (TASK=$(TASK) DEVICE=$(DEVICE) -> $(DATASET_PATH)。追記は RESUME=1)"
	@echo "                   SO101 Leader を使うときは DEVICE=so101leader LEADER_PORT=$(LEADER_PORT)"
	@echo "  replay           記録データの再生 (記録時と SIM_DEVICE が違えば止める。FORCE=1 で続行)"
	@echo "  teleop-lekiwi / replay-lekiwi  TASK=LeKiwi で teleop / replay (DEVICE=keyboard|gamepad|lekiwi-leader)"
	@echo "  assets           アセットと学習済みポリシーの取得・検証 (GROUP=pick-orange|lekiwi)"
	@echo "  gr00t-up/-down   GR00T N1.5 推論サーバの起動 (healthy まで待つ) / 停止"
	@echo "  eval             閉ループ推論 (HEADLESS=0 で GUI 表示)"
	@echo "  config           実際に使われる設定の表示 (.env とコマンドラインを反映)"
	@echo "  check            動作確認一式 (タスク一覧 / サーバ疎通 / 推論 1 ラウンド / VRAM ピーク)"
	@echo "  lock / lock-check  docker/envs/*/uv.lock の更新 / 最新であることの確認"
	@echo "  shell-sim / shell-gr00t / xhost"

# 実際に使われる設定を表示する (.env とコマンドラインの指定を反映した値)
config:
	@echo ".env: $(if $(wildcard .env),あり,なし)"
	@echo "[対象と記録] TASK=$(TASK) DEVICE=$(DEVICE) DATASET=$(DATASET) RESUME=$(RESUME) FORCE=$(FORCE) LEADER_PORT=$(LEADER_PORT)"
	@echo "[sim]        SIM_DEVICE=$(SIM_DEVICE) TEX_BUDGET=$(or $(TEX_BUDGET),(Kit の既定 0.6))"
	@echo "[推論]       HEADLESS=$(HEADLESS) EVAL_ROUNDS=$(EVAL_ROUNDS) POLICY_HOST=$(POLICY_HOST) POLICY_TIMEOUT_MS=$(POLICY_TIMEOUT_MS) ACTION_HORIZON=$(ACTION_HORIZON)"
	@echo "             INSTRUCTION=$(INSTRUCTION)"
	@echo "             POLICY_DIR=$(or $(POLICY_DIR),(compose.yaml の既定)) GR00T_DATA_CONFIG=$(or $(GR00T_DATA_CONFIG),(compose.yaml の既定))"
	@echo "[その他]     GROUP=$(or $(GROUP),(すべて)) UPGRADE=$(UPGRADE)"

build:
	$(COMPOSE) --profile inference build

down:
	$(COMPOSE) --profile inference down

# Kit は異常終了後に SIGTERM を無視して残ることがあるため、コンテナごと強制終了する
stop:
	-docker rm -f leisaac-teleop leisaac-replay leisaac-eval leisaac-list-envs leisaac-smoke leisaac-shell >/dev/null 2>&1

shell-sim:
	$(COMPOSE) run --rm $(TTY_FLAG) --name leisaac-shell sim bash

shell-gr00t:
	$(COMPOSE) --profile inference run --rm --no-deps $(TTY_FLAG) gr00t bash

# 通常は不要 (~/.Xauthority をマウントしているため)。GUI が開かないときのみ
xhost:
	xhost +SI:localuser:$$(id -un)

list-envs:
	$(call SIM_RUN,leisaac-list-envs) scripts/environments/list_envs.py

# sim の起動確認。TASK を DEVICE の action 構成で作り、ヘッドレスでゼロ action を数ステップ流す
smoke:
	$(call SIM_RUN,leisaac-smoke) /opt/tools/sim_smoke.py \
		--task=$(TASK) --teleop_device=$(DEVICE) $(SIM_ARGS) --enable_cameras --headless

# leisaac は既存のデータセットがあると停止する。上書きも暗黙の追記もしないよう、先に確認する
teleop:
	@if [ "$(RESUME)" = 1 ]; then \
		[ -e "$(DATASET_PATH)" ] || { echo "[エラー] $(DATASET_PATH) がないため RESUME=1 は使えません"; exit 1; }; \
	elif [ -e "$(DATASET_PATH)" ]; then \
		echo "[エラー] $(DATASET_PATH) は既にあります。別の名前 (DATASET=...) を指定するか、追記するなら RESUME=1 を付けてください"; exit 1; \
	fi
ifeq ($(RESUME),1)
	$(check_sim_record)
else
	@mkdir -p $(dir $(SIM_RECORD)) && printf 'SIM_DEVICE=%s\nTEX_BUDGET=%s\n' "$(SIM_DEVICE)" "$(TEX_BUDGET)" > $(SIM_RECORD)
endif
	$(call SIM_RUN,leisaac-teleop) scripts/environments/teleoperation/teleop_se3_agent.py \
		--task=$(TASK) --teleop_device=$(DEVICE) --num_envs=1 $(SIM_ARGS) --enable_cameras \
		$(if $(filter $(LEADER_DEVICES),$(DEVICE)),--port=/dev/ttyACM0) \
		--record --dataset_file=$(DATASET_FILE) $(if $(filter 1,$(RESUME)),--resume)

replay:
	$(check_sim_record)
	$(call SIM_RUN,leisaac-replay) scripts/environments/teleoperation/replay.py \
		--task=$(TASK) --num_envs=1 $(SIM_ARGS) --enable_cameras \
		--replay_mode=action --dataset_file=$(DATASET_FILE) $(TASK_TYPE_ARG)

# LeKiwi 用の別名。TASK を LeKiwi にし、keyboard / gamepad を LeKiwi 用のデバイス名に読み替えるだけ
LEKIWI_ARGS = TASK=LeIsaac-LeKiwi-CleanupTrash-v0 DEVICE=$(if $(filter keyboard gamepad,$(DEVICE)),lekiwi-$(DEVICE),$(DEVICE))

teleop-lekiwi:
	$(MAKE) teleop $(LEKIWI_ARGS)

replay-lekiwi:
	$(MAKE) replay $(LEKIWI_ARGS)

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
		--policy_host=$(POLICY_HOST) --policy_port=5555 --policy_timeout_ms=$(POLICY_TIMEOUT_MS) --policy_action_horizon=$(ACTION_HORIZON) \
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
