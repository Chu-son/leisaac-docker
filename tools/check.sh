#!/usr/bin/env bash
# 動作確認一式: make check
#   0. アセットとポリシーが assets.json のハッシュどおりに揃っていること
#   1. sim: LeIsaac のタスクが登録されていること
#   2. gr00t: 推論サーバが healthy になり、action の形状が leisaac クライアントと一致すること
#   3. sim + gr00t: PickOrange をヘッドレスで 1 ラウンド推論し、最後まで完走すること (成功率と VRAM ピークは記録のみ)
#      ポリシーの成功率はばらつくため合否に使わない。性能は make eval EVAL_ROUNDS=10 で測る
#   4. sim: LeKiwi をヘッドレスで作り、ゼロ action で動かせること (VRAM ピークは記録のみ)。
#      Loft シーン (make assets GROUP=lekiwi) がなければ SKIP
#   5. sim: Sweep タスクの確認 (make sweep-check。contract と上流の一致、配置、成功判定)。
#      front と wrist の画像をログに残す。LeKiwi のアセットがなければ SKIP
#   6. train: contract のキー名が lerobot の実機の LeKiwi と一致し、合成した小さな記録を変換して照合できること
# sim は make 経由で動かすので、sim の設定 (SIM_DEVICE など) は .env に従う。
# タスクとポリシーは、基準となる構成 (PickOrange) をコマンドラインで固定する
set -uo pipefail
cd "$(dirname "$0")/.."

export HOST_UID=$(id -u) HOST_GID=$(id -g) INPUT_GID=$(getent group input | cut -d: -f3)
LOG_DIR=data/check/$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG_DIR"
fail=0
REF=(TASK=LeIsaac-SO101-PickOrange-v0 "INSTRUCTION=Pick up the orange and place it on the plate"
     POLICY_DIR=/workspace/data/policies/leisaac-pick-orange-v0 GR00T_DATA_CONFIG=so100_dualcam
     POLICY_HOST=localhost ACTION_HORIZON=16)
# stdin を端末から外し、make が TTY を割り当てないようにする
mk() { make --no-print-directory "$@" < /dev/null; }
mk -s config "${REF[@]}" > "$LOG_DIR/config.log"
step() { printf '\n=== %s\n' "$*"; }
# nvidia-smi で VRAM 使用量 (MiB) を 1 秒ごとに $1 へ記録する。止めるときは kill $mon
monitor_vram() { ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$1" 2>/dev/null & mon=$!; }
ok()   { printf '  [OK] %s\n' "$*"; }
ng()   { printf '  [NG] %s\n' "$*"; fail=1; }

step "0/6 アセットとポリシー (assets.json)"
if python3 tools/fetch_assets.py --check --group pick-orange > "$LOG_DIR/assets.log" 2>&1; then
    ok "pick-orange のアセットとポリシーが揃っている"
else
    ng "足りない、または内容が異なる ($LOG_DIR/assets.log)。make assets GROUP=pick-orange で取得できる"
fi

step "1/6 sim: タスク一覧"
if mk list-envs > "$LOG_DIR/list_envs.log" 2>&1 \
   && grep -q "LeIsaac-SO101-PickOrange-v0" "$LOG_DIR/list_envs.log" \
   && grep -q "LeIsaac-LeKiwi-CleanupTrash-v0" "$LOG_DIR/list_envs.log" \
   && grep -q "LeIsaac-LeKiwi-Sweep-Floor-v0" "$LOG_DIR/list_envs.log"; then
    ok "$(grep -c '| LeIsaac-' "$LOG_DIR/list_envs.log") タスク (PickOrange / LeKiwi / Sweep を含む)"
else
    ng "タスク一覧の取得に失敗 ($LOG_DIR/list_envs.log)"
fi

step "2/6 gr00t: 推論サーバ"
if mk gr00t-up "${REF[@]}" > "$LOG_DIR/gr00t_up.log" 2>&1; then
    ok "healthy"
    if docker compose --profile inference exec -T gr00t python /opt/tools/probe_gr00t.py > "$LOG_DIR/probe.log" 2>&1; then
        ok "action 形状 $(tail -1 "$LOG_DIR/probe.log")"
    else
        ng "action 形状が不一致 ($LOG_DIR/probe.log)"
    fi
else
    ng "サーバが healthy にならない (docker compose --profile inference logs gr00t)"
fi

step "3/6 閉ループ推論 (PickOrange, ヘッドレス, 1 ラウンド)"
if [ $fail -eq 0 ]; then
    monitor_vram "$LOG_DIR/vram.log"
    timeout 900 make --no-print-directory eval "${REF[@]}" EVAL_ROUNDS=1 HEADLESS=1 < /dev/null > "$LOG_DIR/eval.log" 2>&1
    kill $mon 2>/dev/null
    # タイムアウトや異常終了のあとも Kit が残ることがあるため、コンテナごと確実に止める
    docker rm -f leisaac-eval >/dev/null 2>&1
    rate=$(grep -oE "Final success rate: [0-9.]+" "$LOG_DIR/eval.log" | awk '{print $4}')
    peak=$(sort -n "$LOG_DIR/vram.log" | tail -1)
    if [ -n "$rate" ]; then ok "完走 (成功率 $rate, 参考値) / VRAM ピーク ${peak} MiB / $(grep -o "SIM_DEVICE=[^ ]* TEX_BUDGET=[^ ]*" "$LOG_DIR/config.log")"; else ng "完走せず / VRAM ピーク ${peak} MiB ($LOG_DIR/eval.log)"; fi
else
    echo "  [SKIP] 前段が失敗したため実行しない"
fi

docker compose --profile inference stop gr00t >/dev/null 2>&1

step "4/6 sim: LeKiwi の起動 (ヘッドレス, ゼロ action)"
if ! python3 tools/fetch_assets.py --check --group lekiwi > "$LOG_DIR/assets_lekiwi.log" 2>&1; then
    echo "  [SKIP] LeKiwi のアセットがない ($LOG_DIR/assets_lekiwi.log)。make assets GROUP=lekiwi で取得できる"
else
    monitor_vram "$LOG_DIR/vram_lekiwi.log"
    timeout 900 make --no-print-directory smoke TASK=LeIsaac-LeKiwi-CleanupTrash-v0 DEVICE=lekiwi-keyboard < /dev/null > "$LOG_DIR/smoke_lekiwi.log" 2>&1
    kill $mon 2>/dev/null
    docker rm -f leisaac-smoke >/dev/null 2>&1
    peak=$(sort -n "$LOG_DIR/vram_lekiwi.log" | tail -1)
    if grep -q "SMOKE OK" "$LOG_DIR/smoke_lekiwi.log"; then
        ok "$(grep -o 'steps=.*' "$LOG_DIR/smoke_lekiwi.log") / VRAM ピーク ${peak} MiB"
    else
        ng "起動または実行に失敗 / VRAM ピーク ${peak} MiB ($LOG_DIR/smoke_lekiwi.log)"
    fi
fi

step "5/6 sim: Sweep タスクの確認 (ヘッドレス)"
if ! python3 tools/fetch_assets.py --check --group lekiwi > /dev/null 2>&1; then
    echo "  [SKIP] LeKiwi のアセットがない ($LOG_DIR/assets_lekiwi.log)"
else
    monitor_vram "$LOG_DIR/vram_sweep.log"
    timeout 900 make --no-print-directory sweep-check < /dev/null > "$LOG_DIR/sweep_check.log" 2>&1
    kill $mon 2>/dev/null
    docker rm -f leisaac-sweep-check >/dev/null 2>&1
    peak=$(sort -n "$LOG_DIR/vram_sweep.log" | tail -1)
    cp data/sweep/check/front.png "$LOG_DIR/sweep_front.png" 2>/dev/null
    cp data/sweep/check/wrist.png "$LOG_DIR/sweep_wrist.png" 2>/dev/null
    if grep -q "SWEEP CHECK OK" "$LOG_DIR/sweep_check.log"; then
        ok "$(grep -o '([0-9.]*s, .*)' "$LOG_DIR/sweep_check.log" | tail -1) / VRAM ピーク ${peak} MiB / 画像 $LOG_DIR/sweep_{front,wrist}.png"
    else
        grep "\[NG\]" "$LOG_DIR/sweep_check.log" | sed 's/^ */  /'
        ng "確認に失敗 / VRAM ピーク ${peak} MiB ($LOG_DIR/sweep_check.log)"
    fi
fi

step "6/6 train: 変換器の確認 (合成データ)"
timeout 600 docker compose run --rm -T --name leisaac-train-check train python /opt/tools/train_check.py \
    < /dev/null > "$LOG_DIR/train_check.log" 2>&1
docker rm -f leisaac-train-check >/dev/null 2>&1
if grep -q "TRAIN CHECK OK" "$LOG_DIR/train_check.log"; then
    ok "$(grep 'SWEEP CONVERT PASS' "$LOG_DIR/train_check.log" | sed 's/ -> .*//')"
else
    grep -E "\[NG\]|Error" "$LOG_DIR/train_check.log" | sed 's/^ */  /'
    ng "確認に失敗 ($LOG_DIR/train_check.log)"
fi

printf '\nログ: %s\n' "$LOG_DIR"
[ $fail -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $fail
