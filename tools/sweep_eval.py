"""Sweep タスクの評価 (make sweep-eval) と自動生成 (make datagen)。ポリシーに N ラウンド解かせて、成功率を測る

    python /opt/tools/leisaac_run.py /opt/tools/sweep_eval.py --policy planner --rounds 30 --device cpu --enable_cameras --headless
    python /opt/tools/leisaac_run.py /opt/tools/sweep_eval.py --record --dataset_file /workspace/data/datasets/sweep.hdf5 --num_demos 50 ...
ポリシーは reset() と act(obs) -> [T, 9] (contract の LeRobot 形式の action) を持つ。T 個の action を、
それぞれ contract.SIM_STEPS_PER_FRAME step ずつ続ける。学習したポリシーも計画器も、同じ経路で評価する。
    planner: 計画器 (leisaac_ext.sweep_planner)。sim の真値を読む
    (remote: ACT や GR00T のサーバにつなぐ。P6 で足す)
成功は終了条件 success (10 個中 8 個以上が目標領域に入る)。時間切れ (25 秒) は失敗。
結果は out_dir の rounds.jsonl (ラウンドごと) と summary.json に残す。
--record を付けると、成功したエピソードだけを HDF5 に記録する (isaaclab の RecorderManager。上流の teleop と同じ形式)。
成功が num_demos 個になるか、ラウンドが rounds (既定は num_demos の 2 倍 + 10) に達したら終わる。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--policy", default="planner", choices=["planner"])
parser.add_argument("--rounds", type=int, default=None, help="ラウンド数 (既定は 10。--record では上限で、既定は num_demos の 2 倍 + 10)")
parser.add_argument("--seed", type=int, default=None, help="乱数のシード (省略時は現在時刻)")
parser.add_argument("--randomize", default=None, help="追加のランダム化の項目 (カンマ区切り。all ですべて。省略時はタスクの既定)")
parser.add_argument("--out_dir", default=None,
                    help="結果の保存先 (省略時は /workspace/data/sweep/eval/<日時>。--record では <dataset_file から .hdf5 を除いたもの>.datagen)")
parser.add_argument("--record", action="store_true", help="成功したエピソードを dataset_file に記録する (自動生成)")
parser.add_argument("--dataset_file", default=None, help="記録先の HDF5 (--record。既にあれば止める)")
parser.add_argument("--num_demos", type=int, default=50, help="記録する成功エピソードの数 (--record)")
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
from isaaclab.managers import DatasetExportMode
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
if args.record:
    assert args.dataset_file and args.dataset_file.endswith(".hdf5"), "--record には --dataset_file (.hdf5) が要る"
    # isaaclab の HDF5 の書き出しは、既存のファイルを上書きする
    assert not os.path.exists(args.dataset_file), f"{args.dataset_file} は既にある"
    dataset_stem = os.path.splitext(args.dataset_file)[0]
    out_dir = args.out_dir or f"{dataset_stem}.datagen"
    max_rounds = args.rounds if args.rounds is not None else 2 * args.num_demos + 10
else:
    out_dir = args.out_dir or f"/workspace/data/sweep/eval/{time.strftime('%Y%m%d-%H%M%S')}"
    max_rounds = args.rounds if args.rounds is not None else 10
os.makedirs(out_dir, exist_ok=True)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
if args.record:
    # タスクの既定の記録項目 (上流の ActionStateRecorderManagerCfg: 初期状態、各 step の action、観測、状態)。
    # 終了条件 success が成り立って終わったエピソードだけを書き出す
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    cfg.recorders.dataset_export_dir_path = os.path.dirname(args.dataset_file)
    cfg.recorders.dataset_filename = os.path.basename(dataset_stem)
else:
    cfg.recorders = None
cfg.seed = seed
if args.randomize is not None:
    names = sweep_floor.RANDOMIZATIONS if args.randomize == "all" else [n for n in args.randomize.split(",") if n]
    cfg.set_randomization(names)
env = gym.make(args.task, cfg=cfg).unwrapped
# 計画器の IK (calibrate_arm) は関節の状態を書き換える。その後の reset で元に戻る
env.reset()
policy = PlannerPolicy(env)
print(f"policy={args.policy} seed={seed} brush tip (base frame)={policy.poses.tip.round(3).tolist()}", flush=True)
# 2 ラウンド目からは、前のラウンドの終わりの自動リセットが返した観測から始める。env.reset() を重ねると、
# 記録中のエピソード (初期状態だけ) が、前のラウンドの success の値 (リセットでは消えない) で書き出される
obs, _ = env.reset()

results = []
start = time.time()
n_ok = 0
with open(f"{out_dir}/rounds.jsonl", "w") as log:
    for r in range(max_rounds):
        if args.record and n_ok >= args.num_demos:
            break
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
        n_ok += success
        if args.record:
            # 記録したエピソードの番号 (HDF5 の demo_<番号>)。失敗したラウンドは記録しない
            exported = env.recorder_manager.exported_successful_episode_count
            assert exported == n_ok, f"記録したエピソードの数 ({exported}) が成功の数 ({n_ok}) と合わない"
            result["episode"] = n_ok - 1 if success else None
        results.append(result)
        log.write(json.dumps(result) + "\n")
        log.flush()
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
if args.record:
    summary.update(dataset_file=args.dataset_file, num_demos=n_ok,
                   dataset_bytes=os.path.getsize(args.dataset_file) if os.path.exists(args.dataset_file) else 0)
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"SWEEP {'DATAGEN' if args.record else 'EVAL'} {n_ok}/{n} ({p:.0%}, 95% CI {summary['ci95'][0]:.0%}-{summary['ci95'][1]:.0%}) "
      f"policy={args.policy} seed={seed} -> {out_dir}", flush=True)
env.close()
app.close()
