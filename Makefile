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
# Sweep タスクの自動生成で集めるエピソードの数 (dagger ではラウンドの数) と、sweep-replay / sweep-inspect で選ぶエピソードの番号 (空ならすべて)
NUM         ?= 50
EPISODES    ?=
# convert で変換する記録 (data/datasets/<名前>.hdf5。空白区切りで複数なら順につなぐ)。既定は DATASET
CONVERT_FROM ?= $(DATASET)
# dagger で、chunk ごとに計画器の action で動かす確率 (0 なら学習したポリシーだけで動かす)
DAGGER_BETA ?= 0

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
# Sweep タスクの評価 (sweep-eval) のポリシー。planner (計画器) か remote (推論サーバ。make policy-up)。
# remote の推論サーバは、data/train/$(TRAIN_NAME)/checkpoints/$(TRAIN_CHECKPOINT)/ を読む
EVAL_POLICY       ?= planner
TRAIN_CHECKPOINT  ?= last
# sweep-eval の乱数のシード。空なら現在時刻。同じ値にすると、ラウンドごとの配置がそろう
EVAL_SEED         ?=
# 1 なら sweep-eval がラウンドごとの動画 (videos/round_*.mp4) と overview.png を結果に残す
EVAL_VIDEOS       ?= 0

# --- 学習 (train)。出力先は data/train/$(TRAIN_NAME)/。TRAIN_STEPS / TRAIN_BATCH が空なら sweep_train.py の既定
TRAIN_POLICY ?= act
TRAIN_NAME   ?= $(DATASET)-$(TRAIN_POLICY)
TRAIN_STEPS  ?=
TRAIN_BATCH  ?=
# import-policy で取り込む、Colab の学習結果 (tar)
FILE         ?=

# --- その他
GROUP   ?=
UPGRADE ?= 0

# 推論サーバ (policy) が読むチェックポイント
POLICY_CHECKPOINT := data/train/$(TRAIN_NAME)/checkpoints/$(TRAIN_CHECKPOINT)
export POLICY_MODEL := /workspace/$(POLICY_CHECKPOINT)/pretrained_model

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
# 端末から実行したときだけ TTY を割り当てる。スクリプトは自前のタスクを登録するラッパーを通して実行する
TTY_FLAG     := $(if $(shell test -t 0 && echo y),,-T)
SIM_RUN       = $(COMPOSE) run --rm $(TTY_FLAG) --name $(1) sim python /opt/tools/leisaac_run.py
# train (LeRobot。記録の変換と学習) も、コマンドごとに使い捨てのコンテナで実行する
TRAIN_RUN     = $(COMPOSE) run --rm $(TTY_FLAG) --name $(1) train python
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

.PHONY: help config build down stop shell-sim shell-gr00t xhost list-envs smoke sweep-check sweep-eval datagen dagger sweep-replay sweep-inspect convert train colab-data import-policy policy-up policy-down policy-logs teleop replay teleop-lekiwi replay-lekiwi \
        assets gr00t-up gr00t-down gr00t-logs eval check lock lock-check

help:
	@echo "使い方: make <target> [変数=値 ...]  変数は .env にも書ける (.env.example)。現在の値は make config"
	@echo ""
	@echo "  build            sim / gr00t / train イメージをビルド"
	@echo "  down / stop      全サービスの停止 / 実行中の sim コマンドの強制終了"
	@echo "  list-envs        登録済みタスク一覧"
	@echo "  smoke            TASK をヘッドレスで作り、ゼロ action で数ステップ動かす (起動確認)"
	@echo "  sweep-check      Sweep タスクの確認 (contract と上流の一致、配置、成功判定。画像は data/sweep/check/)"
	@echo "  sweep-eval       Sweep タスクを EVAL_POLICY=$(EVAL_POLICY) (planner|remote) で EVAL_ROUNDS 回解き、成功率を測る (-> data/sweep/eval/<日時>/)"
	@echo "  datagen          Sweep タスクの自動生成。計画器が成功したエピソードを NUM=$(NUM) 個記録する (-> $(DATASET_PATH))"
	@echo "  dagger           学習したポリシー (TRAIN_NAME) で NUM 回動かし、計画器の action を正解として記録する (DAgger。-> $(DATASET_PATH))"
	@echo "  sweep-replay     datagen の記録を再生し、再現できるかを確かめる (EPISODES で選ぶ。HEADLESS=0 で GUI 表示)"
	@echo "  sweep-inspect    datagen の記録の一覧、上から見た図、動画を作る (Kit なし。-> data/sweep/inspect/$(DATASET)/)"
	@echo "  convert          datagen の記録 (CONVERT_FROM=$(CONVERT_FROM)) を LeRobot 形式に変換し、HDF5 と照合する (train イメージ。-> data/lerobot/$(DATASET)/)"
	@echo "  train            data/lerobot/$(DATASET) で TRAIN_POLICY=$(TRAIN_POLICY) (act|smolvla|groot) を学習する (-> data/train/$(TRAIN_NAME)/)"
	@echo "  colab-data       data/lerobot/$(DATASET) を 1 つの tar にする (Colab 用。-> data/colab/$(DATASET).tar。Drive へは手動で上げる)"
	@echo "  import-policy    Colab の学習結果 FILE=<tar> を data/train/$(TRAIN_NAME)/ に展開する"
	@echo "  policy-up/-down  学習したポリシーの推論サーバの起動 (healthy まで待つ。$(POLICY_CHECKPOINT)) / 停止"
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
	@echo "             NUM=$(NUM) EPISODES=$(or $(EPISODES),(すべて)) CONVERT_FROM=$(CONVERT_FROM) DAGGER_BETA=$(DAGGER_BETA)"
	@echo "[sim]        SIM_DEVICE=$(SIM_DEVICE) TEX_BUDGET=$(or $(TEX_BUDGET),(Kit の既定 0.6))"
	@echo "[推論]       HEADLESS=$(HEADLESS) EVAL_ROUNDS=$(EVAL_ROUNDS) POLICY_HOST=$(POLICY_HOST) POLICY_TIMEOUT_MS=$(POLICY_TIMEOUT_MS) ACTION_HORIZON=$(ACTION_HORIZON)"
	@echo "             INSTRUCTION=$(INSTRUCTION)"
	@echo "             POLICY_DIR=$(or $(POLICY_DIR),(compose.yaml の既定)) GR00T_DATA_CONFIG=$(or $(GR00T_DATA_CONFIG),(compose.yaml の既定))"
	@echo "             EVAL_POLICY=$(EVAL_POLICY) EVAL_SEED=$(or $(EVAL_SEED),(現在時刻)) EVAL_VIDEOS=$(EVAL_VIDEOS) TRAIN_CHECKPOINT=$(TRAIN_CHECKPOINT) (推論サーバ: $(POLICY_CHECKPOINT))"
	@echo "[学習]       TRAIN_POLICY=$(TRAIN_POLICY) TRAIN_NAME=$(TRAIN_NAME) TRAIN_STEPS=$(or $(TRAIN_STEPS),(既定)) TRAIN_BATCH=$(or $(TRAIN_BATCH),(既定)) FILE=$(or $(FILE),(なし))"
	@echo "[その他]     GROUP=$(or $(GROUP),(すべて)) UPGRADE=$(UPGRADE)"

build:
	$(COMPOSE) --profile inference build

down:
	$(COMPOSE) --profile inference down

# Kit は異常終了後に SIGTERM を無視して残ることがあるため、コンテナごと強制終了する
stop:
	-docker rm -f leisaac-teleop leisaac-replay leisaac-eval leisaac-list-envs leisaac-smoke leisaac-sweep-check leisaac-sweep-eval \
		leisaac-datagen leisaac-dagger leisaac-sweep-replay leisaac-sweep-inspect leisaac-convert leisaac-train leisaac-train-check leisaac-shell >/dev/null 2>&1

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

# Sweep タスク (docs/sweep.md) の確認。front と wrist の画像を data/sweep/check/ に保存する
sweep-check:
	$(call SIM_RUN,leisaac-sweep-check) /opt/tools/sweep_check.py $(SIM_ARGS) --enable_cameras --headless

# Sweep タスクの評価。EVAL_POLICY=planner は計画器 (特権情報を使う)、remote は学習したポリシー (推論サーバを先に起動する)
sweep-eval: $(if $(filter remote,$(EVAL_POLICY)),policy-up)
	$(call SIM_RUN,leisaac-sweep-eval) /opt/tools/sweep_eval.py --policy=$(EVAL_POLICY) --rounds=$(EVAL_ROUNDS) $(if $(EVAL_SEED),--seed=$(EVAL_SEED)) $(if $(filter 1,$(EVAL_VIDEOS)),--videos) \
		$(if $(filter remote,$(EVAL_POLICY)),--policy_host=$(POLICY_HOST) --policy_timeout_ms=$(POLICY_TIMEOUT_MS) --action_horizon=$(ACTION_HORIZON)) \
		$(SIM_ARGS) --enable_cameras $(if $(filter 1,$(HEADLESS)),--headless)

# Sweep タスクの自動生成 (sweep_eval.py --record)。成功したエピソードだけを記録する。上書きも追記もしない。
# ラウンドごとのログは data/datasets/$(DATASET).datagen/ に残す
datagen:
	@if [ -e "$(DATASET_PATH)" ]; then \
		echo "[エラー] $(DATASET_PATH) は既にあります。別の名前 (DATASET=...) を指定してください"; exit 1; \
	fi
	@mkdir -p $(dir $(SIM_RECORD)) && printf 'SIM_DEVICE=%s\nTEX_BUDGET=%s\n' "$(SIM_DEVICE)" "$(TEX_BUDGET)" > $(SIM_RECORD)
	$(call SIM_RUN,leisaac-datagen) /opt/tools/sweep_eval.py --policy=planner --record --dataset_file=$(DATASET_FILE) \
		--num_demos=$(NUM) $(SIM_ARGS) --enable_cameras $(if $(filter 1,$(HEADLESS)),--headless)

# DAgger のデータ。学習したポリシー (推論サーバ) で動かし、各フレームの計画器の action を正解として、すべてのラウンドを記録する
dagger: policy-up
	@if [ -e "$(DATASET_PATH)" ]; then \
		echo "[エラー] $(DATASET_PATH) は既にあります。別の名前 (DATASET=...) を指定してください"; exit 1; \
	fi
	@mkdir -p $(dir $(SIM_RECORD)) && printf 'SIM_DEVICE=%s\nTEX_BUDGET=%s\n' "$(SIM_DEVICE)" "$(TEX_BUDGET)" > $(SIM_RECORD)
	$(call SIM_RUN,leisaac-dagger) /opt/tools/sweep_eval.py --policy=remote --dagger --dagger_beta=$(DAGGER_BETA) --record \
		--dataset_file=$(DATASET_FILE) --rounds=$(NUM) $(if $(EVAL_SEED),--seed=$(EVAL_SEED)) $(if $(filter 1,$(EVAL_VIDEOS)),--videos) \
		--policy_host=$(POLICY_HOST) --policy_timeout_ms=$(POLICY_TIMEOUT_MS) --action_horizon=$(ACTION_HORIZON) \
		$(SIM_ARGS) --enable_cameras $(if $(filter 1,$(HEADLESS)),--headless)

# datagen の記録を action で再生し、成功と最終位置を記録と比べる (結果は data/sweep/replay/<日時>/)
sweep-replay:
	$(check_sim_record)
	$(call SIM_RUN,leisaac-sweep-replay) /opt/tools/sweep_replay.py --dataset_file=$(DATASET_FILE) \
		$(if $(EPISODES),--episodes $(EPISODES)) $(SIM_ARGS) --enable_cameras $(if $(filter 1,$(HEADLESS)),--headless)

# datagen の記録を、Kit を起動せずに確かめる。動画は EPISODES のもの (空ならすべて)
sweep-inspect:
	$(call SIM_RUN,leisaac-sweep-inspect) /opt/tools/sweep_inspect.py --dataset_file=$(DATASET_FILE) \
		--out_dir=/workspace/data/sweep/inspect/$(DATASET) $(if $(EPISODES),--videos $(EPISODES))

# datagen の記録を LeRobot 形式 (lerobot 0.4.2) に変換し、読み直して HDF5 と照合する。上書きも追記もしない
convert:
	@if [ -e "data/lerobot/$(DATASET)" ]; then \
		echo "[エラー] data/lerobot/$(DATASET) は既にあります。別の名前で変換するか、不要なら消してください"; exit 1; \
	fi
	@for d in $(CONVERT_FROM); do [ -e "data/datasets/$$d.hdf5" ] || { echo "[エラー] data/datasets/$$d.hdf5 がありません"; exit 1; }; done
	$(call TRAIN_RUN,leisaac-convert) /opt/tools/sweep_convert.py --dataset_file $(foreach d,$(CONVERT_FROM),/workspace/data/datasets/$(d).hdf5) \
		--root=/workspace/data/lerobot/$(DATASET)

# LeRobot 形式のデータでポリシーを学習する。上書きも再開もしない (docs/sweep.md の「学習」)
train:
	@if [ ! -d "data/lerobot/$(DATASET)" ]; then \
		echo "[エラー] data/lerobot/$(DATASET) がありません。先に make convert DATASET=$(DATASET) を実行してください"; exit 1; \
	fi
	@if [ -e "data/train/$(TRAIN_NAME)" ]; then \
		echo "[エラー] data/train/$(TRAIN_NAME) は既にあります。別の名前 (TRAIN_NAME=...) を指定してください"; exit 1; \
	fi
	$(call TRAIN_RUN,leisaac-train) /opt/tools/sweep_train.py --policy=$(TRAIN_POLICY) --dataset_root=/workspace/data/lerobot/$(DATASET) \
		--output_dir=/workspace/data/train/$(TRAIN_NAME) $(if $(TRAIN_STEPS),--steps=$(TRAIN_STEPS)) $(if $(TRAIN_BATCH),--batch_size=$(TRAIN_BATCH))

# Colab に渡すデータセットを 1 つの tar にする。Drive へは、ブラウザで手動で上げる (トークンを使わない)
colab-data:
	@if [ ! -d "data/lerobot/$(DATASET)" ]; then echo "[エラー] data/lerobot/$(DATASET) がありません"; exit 1; fi
	@if [ -e "data/colab/$(DATASET).tar" ]; then echo "[エラー] data/colab/$(DATASET).tar は既にあります"; exit 1; fi
	mkdir -p data/colab && tar -C data/lerobot -cf data/colab/$(DATASET).tar $(DATASET)
	@ls -lh data/colab/$(DATASET).tar
	@echo "これを Drive の MyDrive/leisaac/ に上げてください (notebooks/train_colab.ipynb)"

# Colab の学習結果 (ノートブックが作る tar。中身は sweep_train.py の output_dir) を data/train/$(TRAIN_NAME)/ に展開する。
# 既にあれば止める。sweep_train.json がない tar は、この仕組みの出力ではないので展開しない
import-policy:
	@if [ ! -f "$(FILE)" ]; then echo "[エラー] FILE=<Colab の学習結果の tar> を指定してください"; exit 1; fi
	@if [ -e "data/train/$(TRAIN_NAME)" ]; then \
		echo "[エラー] data/train/$(TRAIN_NAME) は既にあります。別の名前 (TRAIN_NAME=...) を指定してください"; exit 1; \
	fi
	@tar -tf "$(FILE)" | grep -qx './sweep_train.json' || { echo "[エラー] $(FILE) に sweep_train.json がありません"; exit 1; }
	mkdir -p data/train/$(TRAIN_NAME) && tar -C data/train/$(TRAIN_NAME) --no-same-owner -xf "$(FILE)"
	@du -sh data/train/$(TRAIN_NAME)

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

# 学習したポリシー (data/train/$(TRAIN_NAME)/) の推論サーバ。TRAIN_NAME か TRAIN_CHECKPOINT を変えると作り直す
policy-up:
	@if [ ! -f "$(POLICY_CHECKPOINT)/pretrained_model/config.json" ]; then \
		echo "[エラー] $(POLICY_CHECKPOINT)/pretrained_model がありません (TRAIN_NAME と TRAIN_CHECKPOINT を確かめてください)"; exit 1; \
	fi
	$(COMPOSE) --profile inference up -d --wait policy

policy-down:
	$(COMPOSE) --profile inference stop policy

policy-logs:
	$(COMPOSE) --profile inference logs -f policy

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
	$(COMPOSE) --profile tools run --rm lock-train uv lock $(if $(filter 1,$(UPGRADE)),--upgrade)
	@wc -l docker/envs/*/uv.lock

# lock が pyproject.toml と、固定した ref のソースに対して最新であることを確認する
lock-check:
	$(COMPOSE) --profile tools run --rm lock-sim uv lock --check
	$(COMPOSE) --profile tools run --rm lock-gr00t uv lock --check
	$(COMPOSE) --profile tools run --rm lock-train uv lock --check
