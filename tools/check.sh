#!/usr/bin/env bash
# 動作確認一式: make check
#   0. アセットとポリシーが assets.json のハッシュどおりに揃っていること
#   1. sim: LeIsaac のタスクが登録されていること
#   2. gr00t: 推論サーバが healthy になり、action の形状が leisaac クライアントと一致すること
#   3. sim + gr00t: PickOrange をヘッドレスで 1 ラウンド推論し、最後まで完走すること (成功率と VRAM ピークは記録のみ)
#      ポリシーの成功率はばらつくため合否に使わない。性能は make eval EVAL_ROUNDS=10 で測る
set -uo pipefail
cd "$(dirname "$0")/.."

export HOST_UID=$(id -u) HOST_GID=$(id -g) INPUT_GID=$(getent group input | cut -d: -f3)
LOG_DIR=data/check/$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG_DIR"
fail=0
step() { printf '\n=== %s\n' "$*"; }
ok()   { printf '  [OK] %s\n' "$*"; }
ng()   { printf '  [NG] %s\n' "$*"; fail=1; }

step "0/3 アセットとポリシー (assets.json)"
if python3 tools/fetch_assets.py --check --group pick-orange > "$LOG_DIR/assets.log" 2>&1; then
    ok "pick-orange のアセットとポリシーが揃っている"
else
    ng "足りない、または内容が異なる ($LOG_DIR/assets.log)。make assets GROUP=pick-orange で取得できる"
fi

step "1/3 sim: タスク一覧"
if docker compose run --rm -T --name leisaac-check-envs sim python scripts/environments/list_envs.py > "$LOG_DIR/list_envs.log" 2>&1 \
   && grep -q "LeIsaac-SO101-PickOrange-v0" "$LOG_DIR/list_envs.log" \
   && grep -q "LeIsaac-LeKiwi-CleanupTrash-v0" "$LOG_DIR/list_envs.log"; then
    ok "$(grep -c '| LeIsaac-' "$LOG_DIR/list_envs.log") タスク (PickOrange / LeKiwi を含む)"
else
    ng "タスク一覧の取得に失敗 ($LOG_DIR/list_envs.log)"
fi

step "2/3 gr00t: 推論サーバ"
if docker compose --profile inference up -d --wait gr00t > "$LOG_DIR/gr00t_up.log" 2>&1; then
    ok "healthy"
    if docker compose --profile inference exec -T gr00t python /opt/tools/probe_gr00t.py > "$LOG_DIR/probe.log" 2>&1; then
        ok "action 形状 $(tail -1 "$LOG_DIR/probe.log")"
    else
        ng "action 形状が不一致 ($LOG_DIR/probe.log)"
    fi
else
    ng "サーバが healthy にならない (docker compose --profile inference logs gr00t)"
fi

step "3/3 閉ループ推論 (PickOrange, ヘッドレス, 1 ラウンド)"
if [ $fail -eq 0 ]; then
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$LOG_DIR/vram.log" 2>/dev/null &
    mon=$!
    timeout 900 docker compose run --rm -T --name leisaac-check-eval sim python scripts/evaluation/policy_inference.py \
        --task=LeIsaac-SO101-PickOrange-v0 --eval_rounds=1 --policy_type=gr00tn1.5 \
        --policy_host=localhost --policy_port=5555 --policy_timeout_ms=15000 --policy_action_horizon=16 \
        --policy_language_instruction="Pick up the orange and place it on the plate" \
        --device=cuda --enable_cameras --headless > "$LOG_DIR/eval.log" 2>&1
    kill $mon 2>/dev/null
    # タイムアウトや異常終了のあとも Kit が残ることがあるため、コンテナごと確実に止める
    docker rm -f leisaac-check-eval >/dev/null 2>&1
    rate=$(grep -oE "Final success rate: [0-9.]+" "$LOG_DIR/eval.log" | awk '{print $4}')
    peak=$(sort -n "$LOG_DIR/vram.log" | tail -1)
    if [ -n "$rate" ]; then ok "完走 (成功率 $rate, 参考値) / VRAM ピーク ${peak} MiB"; else ng "完走せず / VRAM ピーク ${peak} MiB ($LOG_DIR/eval.log)"; fi
else
    echo "  [SKIP] 前段が失敗したため実行しない"
fi

docker compose --profile inference stop gr00t >/dev/null 2>&1
printf '\nログ: %s\n' "$LOG_DIR"
[ $fail -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $fail
