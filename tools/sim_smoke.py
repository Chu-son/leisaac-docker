"""sim の起動確認: タスクを作り、ゼロ action で数ステップ動かす (make smoke / make check)

teleop と同じく --teleop_device で action の構成を決める。最後に SMOKE OK を出力する。
    python /opt/tools/sim_smoke.py --task LeIsaac-LeKiwi-CleanupTrash-v0 --teleop_device lekiwi-keyboard \
        --device cpu --enable_cameras --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--teleop_device", required=True)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import time

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401  (タスクを登録する)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device(args.teleop_device)
cfg.recorders = None
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()

action = torch.zeros(env.action_space.shape, device=env.device)
start = time.time()
for _ in range(args.steps):
    env.step(action)
print(f"SMOKE OK task={args.task} device={args.device} steps={args.steps} ({time.time() - start:.1f}s)", flush=True)

env.close()
app.close()
