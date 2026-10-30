"""Sweep タスクの評価 (make sweep-eval)。ポリシーに N ラウンド解かせて、成功率を測る

    python /opt/tools/leisaac_run.py /opt/tools/sweep_eval.py --policy planner --rounds 30 --device cpu --enable_cameras --headless
ポリシーは reset() と act(obs) -> [T, 9] (contract の LeRobot 形式の action) を持つ。T 個の action を、
それぞれ contract.SIM_STEPS_PER_FRAME step ずつ続ける。学習したポリシーも計画器も、同じ経路で評価する。
    planner: 計画器 (leisaac_ext.sweep_planner)。sim の真値を読む
    (remote: ACT や GR00T のサーバにつなぐ。P6 で足す)
成功は終了条件 success (10 個中 8 個以上が目標領域に入る)。時間切れ (25 秒) は失敗。
結果は out_dir の rounds.jsonl (ラウンドごと) と summary.json に残す。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--policy", default="planner", choices=["planner"])
parser.add_argument("--rounds", type=int, default=10)
parser.add_argument("--seed", type=int, default=None, help="乱数のシード (省略時は現在時刻)")
parser.add_argument("--randomize", default=None, help="追加のランダム化の項目 (カンマ区切り。all ですべて。省略時はタスクの既定)")
parser.add_argument("--out_dir", default=None, help="結果の保存先 (省略時は /workspace/data/sweep/eval/<日時>)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import json
import math
import os
import time

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
from leisaac_ext import contract, sweep_floor, sweep_mdp, sweep_planner


class PlannerPolicy:
    """計画器を、ポリシーのインタフェースで包む。観測の代わりに sim の真値を読む"""

    def __init__(self, env):
        self.env = env
        self.poses = sweep_planner.calibrate_arm(env)
        self.planner = sweep_planner.SweepPlanner(self.poses)

    def reset(self) -> None:
        self.planner.reset()

    def act(self, obs) -> np.ndarray:
        action = self.planner.act(sweep_planner.read_state(self.env))
        joint_pos = obs["joint_pos"][0].cpu().numpy().astype(np.float64)
        return contract.to_lerobot(joint_pos, np.zeros_like(joint_pos), action)[1][None]

    def info(self) -> dict:
        return {"passes": self.planner.passes, "phase": self.planner.phase}


seed = args.seed if args.seed is not None else int(time.time())
out_dir = args.out_dir or f"/workspace/data/sweep/eval/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(out_dir, exist_ok=True)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
cfg.recorders = None
cfg.seed = seed
if args.randomize is not None:
    names = sweep_floor.RANDOMIZATIONS if args.randomize == "all" else [n for n in args.randomize.split(",") if n]
    cfg.set_randomization(names)
env = gym.make(args.task, cfg=cfg).unwrapped
# 計画器の IK (calibrate_arm) は関節の状態を書き換える。各ラウンドの最初の reset で元に戻る
env.reset()
policy = PlannerPolicy(env)
print(f"policy={args.policy} seed={seed} brush tip (base frame)={policy.poses.tip.round(3).tolist()}", flush=True)

results = []
start = time.time()
with open(f"{out_dir}/rounds.jsonl", "w") as log:
    for r in range(args.rounds):
        obs, _ = env.reset()
        policy.reset()
        t0, steps, success, done = time.time(), 0, False, False
        count = int(sweep_mdp.debris_in_target(env)[0].sum())
        # 台車の中心から最も近いゴミまでの距離の、ラウンド中の最小値。台車の外形の半径
        # (sweep_planner.BASE_RADIUS) より小さければ、ゴミに当たったか、台車の下にくぐらせている
        base_min_dist = math.inf
        while not done:
            chunk = policy.act(obs["policy"])
            for a in chunk:
                joint_pos = obs["policy"]["joint_pos"][0].cpu().numpy().astype(np.float64)
                sim_action = torch.tensor(contract.from_lerobot_action(a, joint_pos), device=env.device)[None]
                for _ in range(contract.SIM_STEPS_PER_FRAME):
                    # step の後は自動でリセットされるので、終わる前の個数を残しておく
                    count = int(sweep_mdp.debris_in_target(env)[0].sum())
                    state = sweep_planner.read_state(env)
                    base_min_dist = min(base_min_dist, float(np.linalg.norm(state.debris - state.base, axis=-1).min()))
                    obs, _, terminated, truncated, _ = env.step(sim_action)
                    steps += 1
                    if terminated[0] or truncated[0]:
                        success, done = bool(env.termination_manager.get_term("success")[0]), True
                        break
                if done:
                    break
        if success:
            count = max(count, sweep_mdp.SUCCESS_COUNT)
        result = {"round": r, "success": success, "in_target": count, "sim_time": round(steps * env.step_dt, 2),
                  "base_min_dist": round(base_min_dist, 3), "wall_time": round(time.time() - t0, 1), **policy.info()}
        results.append(result)
        log.write(json.dumps(result) + "\n")
        log.flush()
        n_ok = sum(x["success"] for x in results)
        print(f"round {r}: {'success' if success else 'failure'} in_target={count} sim_time={result['sim_time']}s "
              f"base_min_dist={result['base_min_dist']:.3f}m passes={result['passes']} phase={result['phase']} ({n_ok}/{len(results)})", flush=True)

n, n_ok = len(results), sum(x["success"] for x in results)
# 成功率の 95% 信頼区間 (Wilson)
p, z = n_ok / max(n, 1), 1.96
center = (p + z**2 / (2 * n)) / (1 + z**2 / n) if n else 0.0
half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n) if n else 0.0
summary = {"task": args.task, "policy": args.policy, "seed": seed, "rounds": n, "success": n_ok,
           "success_rate": round(p, 3), "ci95": [round(center - half, 3), round(center + half, 3)],
           "mean_in_target": round(float(np.mean([x["in_target"] for x in results])), 2) if n else 0.0,
           "mean_sim_time_success": round(float(np.mean([x["sim_time"] for x in results if x["success"]])), 2) if n_ok else None,
           # 台車がゴミに当たった (台車の中心から最も近いゴミまでが、外形の半径未満の) ラウンドの数
           "base_contact_rounds": sum(x["base_min_dist"] < sweep_planner.BASE_RADIUS for x in results),
           "wall_time": round(time.time() - start, 1), "device": args.device}
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"SWEEP EVAL {n_ok}/{n} ({p:.0%}, 95% CI {summary['ci95'][0]:.0%}-{summary['ci95'][1]:.0%}) "
      f"policy={args.policy} seed={seed} -> {out_dir}", flush=True)
env.close()
app.close()
