"""Sweep タスクの記録 (make datagen) を action で再生し、記録どおりに再現できるかを確かめる (make sweep-replay)

    python /opt/tools/leisaac_run.py /opt/tools/sweep_replay.py --dataset_file /workspace/data/datasets/sweep.hdf5 --episodes 0 3 --device cpu --enable_cameras --headless
エピソードごとに、記録した初期状態に戻し (reset_to)、記録した sim の action を順に流す。流し終えたときの、
目標領域に入ったゴミの個数 (8 個以上で成功) と、ゴミの最終位置の記録とのずれを調べる。
ヘッドレスでなければ、実時間の速さで GUI に表示する。
見た目のランダム化 (ゴミの色、照明、カメラの位置) は状態に含まれないので、再生の見た目は記録と違う (物理には影響しない)。
結果は out_dir の replay.jsonl に残す。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--dataset_file", required=True)
parser.add_argument("--episodes", type=int, nargs="*", default=[], help="再生するエピソードの番号 (demo_<番号>。省略時はすべて)")
parser.add_argument("--out_dir", default=None, help="結果の保存先 (省略時は /workspace/data/sweep/replay/<日時>)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import json
import os
import time

import gymnasium as gym
import torch
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
from leisaac_ext import sweep_mdp

out_dir = args.out_dir or f"/workspace/data/sweep/replay/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(out_dir, exist_ok=True)
handler = HDF5DatasetFileHandler()
handler.open(args.dataset_file)
episodes = args.episodes or list(range(handler.get_num_episodes()))

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
# 終了条件を外し、記録した action を最後まで流す (上流の replay と同じ)
cfg.recorders = None
cfg.terminations = {}
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
env_ids = torch.tensor([0], device=env.device)
origin = env.scene.env_origins[0]


def in_target(debris_pos: torch.Tensor, target_pos: torch.Tensor) -> int:
    """sweep_mdp.debris_in_target と同じ判定を、env の原点からの位置 (debris [N, 3]、target [3]) で行う"""
    inside = ((debris_pos[:, :2] - target_pos[:2]).abs() < sweep_mdp.TARGET_HALF).all(dim=-1)
    return int((inside & (debris_pos[:, 2] < sweep_mdp.IN_TARGET_MAX_Z)).sum())


results = []
with open(f"{out_dir}/replay.jsonl", "w") as log:
    for i in episodes:
        episode = handler.load_episode(f"demo_{i}", env.device)
        # DAgger の記録 (sweep_eval.py --dagger) の actions は計画器の正解なので、実際に動かした action で再生する
        actions = episode.data.get("executed_actions", episode.data["actions"])
        env.reset_to(episode.get_initial_state(), env_ids, is_relative=True)
        t0 = time.time()
        for k, action in enumerate(actions):
            env.step(action[None])
            if not args.headless:
                time.sleep(max(0.0, t0 + (k + 1) * env.step_dt - time.time()))
        # 記録の状態は env の原点からの値 (is_relative)。最後の要素が、最後の action の後の状態
        recorded = episode.data["states"]["rigid_object"]
        rec_debris = torch.stack([recorded[n]["root_pose"][-1, :3] for n in sweep_mdp.DEBRIS_NAMES])
        rec_target = recorded[sweep_mdp.TARGET_NAME]["root_pose"][-1, :3]
        debris = torch.stack([env.scene[n].data.root_pos_w[0] for n in sweep_mdp.DEBRIS_NAMES]) - origin
        target = env.scene[sweep_mdp.TARGET_NAME].data.root_pos_w[0] - origin
        count, rec_count = in_target(debris, target), in_target(rec_debris, rec_target)
        error = float((debris[:, :2] - rec_debris[:, :2]).norm(dim=-1).max())
        result = {"episode": i, "success": count >= sweep_mdp.SUCCESS_COUNT, "in_target": count, "recorded_in_target": rec_count,
                  "max_debris_error": round(error, 4), "steps": len(actions)}
        results.append(result)
        log.write(json.dumps(result) + "\n")
        log.flush()
        n_ok = sum(x["success"] for x in results)
        print(f"episode {i}: {'success' if result['success'] else 'failure'} in_target={count} (recorded {rec_count}) "
              f"max_debris_error={error * 100:.2f}cm steps={len(actions)} ({n_ok}/{len(results)})", flush=True)

n, n_ok = len(results), sum(x["success"] for x in results)
max_error = max((x["max_debris_error"] for x in results), default=0.0)
print(f"SWEEP REPLAY {n_ok}/{n} succeeded, max_debris_error={max_error * 100:.2f}cm -> {out_dir}", flush=True)
env.close()
app.close()
