"""Sweep タスクの確認 (make sweep-check / make check)。問題がなければ最後に SWEEP CHECK OK を出力する

    1. 関節の並びが contract.SIM_JOINT_NAMES と一致する
    2. front と wrist の画像が contract.IMAGE_SHAPE で、out_dir に保存する (ゴミと目標領域が写るかを目で確かめる)
    3. 決めた action を流し、contract.to_lerobot が上流の変換 (convert_leisaac_action_to_lerobot と
       user_vel_state / user_vel_action) と一致する。台車が指令の速度で動く。速さ (steps/s) を測る
    4. リセットのたびに、ゴミが重ならずに置かれ、目標領域に入っていない。ゴミと目標領域が front の画角に入る。
       置いた後に動き出さない。リセット直後の画像が新しい場面を写している
    5. 目標領域にゴミを 7 個置くと成功にならず、8 個で成功 (終了条件 success) になる
    python /opt/tools/leisaac_run.py /opt/tools/sweep_check.py --device cpu --enable_cameras --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--resets", type=int, default=5)
parser.add_argument("--randomize", default=None, help="追加のランダム化の項目 (カンマ区切り。all ですべて。省略時はタスクの既定)")
parser.add_argument("--out_dir", default="/workspace/data/sweep/check")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import os
import resource
import time

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot
from PIL import Image

import leisaac  # noqa: F401
from leisaac_ext import contract, sweep_floor, sweep_mdp

failures = []


def check(ok: bool, msg: str) -> None:
    print(f"  [{'OK' if ok else 'NG'}] {msg}", flush=True)
    if not ok:
        failures.append(msg)


cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
cfg.recorders = None
if args.randomize is not None:
    names = sweep_floor.ALL_RANDOMIZATION if args.randomize == "all" else [n for n in args.randomize.split(",") if n]
    cfg.set_randomization(names)
env = gym.make(args.task, cfg=cfg).unwrapped
active = sorted(n for n, t in vars(cfg.events).items() if t is not None)
print(f"events: {active}", flush=True)
obs, _ = env.reset()
robot = env.scene["robot"]
dev = env.device

print("1. 関節の並び", flush=True)
check(tuple(robot.joint_names) == contract.SIM_JOINT_NAMES, f"joint_names={robot.joint_names}")

print("2. カメラ", flush=True)
hold = torch.zeros(1, 9, device=dev)
hold[0, 5] = contract.GRIPPER_HOLD_RAD
for _ in range(10):  # 最初の数 step は描画が追いつかない
    obs, *_ = env.step(hold)
os.makedirs(args.out_dir, exist_ok=True)
for cam in contract.CAMERAS:
    img = obs["policy"][cam][0].cpu().numpy()
    check(img.shape == contract.IMAGE_SHAPE and img.max() > 0, f"{cam}: shape={img.shape}, max={img.max()}")
    Image.fromarray(img.astype(np.uint8)).save(f"{args.out_dir}/{cam}.png")
print(f"  画像: {args.out_dir}/{{{','.join(contract.CAMERAS)}}}.png", flush=True)

print("3. contract と上流の変換の一致", flush=True)


def scripted_action(k: int) -> torch.Tensor:
    """アームをゆっくり振り、台車を回しながら動かす (sim の action の単位)"""
    t = k / 60.0
    a = torch.zeros(1, 9, device=dev)
    a[0, :5] = 0.3 * torch.sin(torch.tensor([0.5, 0.7, 0.9, 1.1, 1.3]) * t).to(dev)
    a[0, 5] = contract.GRIPPER_HOLD_RAD
    a[0, 6:9] = torch.tensor([0.1 * np.cos(t), 0.1 * np.sin(t), 0.4], device=dev)
    return a


to_np = lambda x: x[0].cpu().numpy().astype(np.float64)


def upstream_base(v: np.ndarray) -> np.ndarray:
    """上流の台車の値 (theta は rad/s。絶対値が 1e-4 未満は 0 に丸めてある) を、contract の単位にする"""
    return np.array([v[0], v[1], np.degrees(v[2])])


def round_like_upstream(v: np.ndarray) -> np.ndarray:
    """上流と同じく、絶対値が 1e-4 未満の値を 0 にする (theta は rad/s で判定する)"""
    return np.where(np.abs([v[0], v[1], np.radians(v[2])]) < 1e-4, 0.0, v)


err = {k: 0.0 for k in ("state.arm", "state.base", "action.arm", "action.base", "upstream.state.arm")}
start = time.time()
a_prev = None
track = np.zeros(3)
for k in range(args.steps):
    a = scripted_action(k)
    pre = obs["policy"]
    jp, jv, a_np = to_np(pre["joint_pos"]), to_np(pre["joint_vel"]), to_np(a)
    state, act = contract.to_lerobot(jp, jv, a_np)
    if k >= 10:  # 動き出しの数 step を除く
        # 台車の仮想の関節の速度が、前の step の指令 (ワールド座標系) に追従している
        track = np.maximum(track, np.abs(jv[contract._BASE_IDX] - a_prev[6:9]))
    a_prev = a_np
    up_arm = convert_leisaac_action_to_lerobot(pre["joint_pos"][:, contract._ARM_IDX].cpu()).squeeze(0)
    err["state.arm"] = max(err["state.arm"], np.abs(state[:6] - up_arm).max())
    # 上流の user_vel_state は theta を rad/s で返す
    err["state.base"] = max(err["state.base"], np.abs(round_like_upstream(state[6:]) - upstream_base(to_np(pre["user_vel_state"]))).max())
    # 上流の state のアーム (joint_pos[:-3]) は台車の関節を含む (docs/sweep.md)。ずれの大きさだけ記録する
    up_state_arm = convert_leisaac_action_to_lerobot(pre["joint_pos"][:, :-3].cpu()).squeeze(0)
    err["upstream.state.arm"] = max(err["upstream.state.arm"], np.abs(state[:6] - up_state_arm).max())

    up_act_arm = convert_leisaac_action_to_lerobot(a[:, :6].cpu()).squeeze(0)
    err["action.arm"] = max(err["action.arm"], np.abs(act[:6] - up_act_arm).max())
    obs, _, terminated, truncated, _ = env.step(a)
    # 上流の user_vel_action は step の後の観測で、そのときの向きで変換する。同じ向きで比べて、式の一致を確かめる
    theta_next = to_np(obs["policy"]["joint_pos"])[contract._BASE_IDX[2]]
    ours = contract.base_world_to_robot(a_np[6:9], np.asarray(theta_next))
    err["action.base"] = max(err["action.base"], np.abs(round_like_upstream(ours) - upstream_base(to_np(obs["policy"]["user_vel_action"]))).max())
    if terminated.any() or truncated.any():
        check(False, f"step {k} で終了した (terminated={terminated.tolist()}, truncated={truncated.tolist()})")
        break
elapsed = time.time() - start
for key in ("state.arm", "state.base", "action.arm", "action.base"):
    check(err[key] < 1e-3, f"{key}: 最大誤差 {err[key]:.2e}")
# 回転は、アームの反動などで指令より数 % 遅れる (docs/sweep.md)
check(track[0] < 1e-2 and track[1] < 1e-2 and track[2] < 5e-2, f"台車の速度の、指令からのずれ: 最大 x={track[0]:.2e} y={track[1]:.2e} theta={track[2]:.2e}")
pose = to_np(obs["policy"]["joint_pos"])[contract._BASE_IDX]
print(f"  台車の姿勢: x={pose[0]:+.3f}m y={pose[1]:+.3f}m theta={pose[2]:+.3f}rad", flush=True)
print(f"  (参考) 上流の state のアームとの差: 最大 {err['upstream.state.arm']:.1f} (台車の関節が混ざるため)", flush=True)

print("4. 配置のリセット", flush=True)
names = sweep_mdp.DEBRIS_NAMES
reset_time = 0.0
for r in range(args.resets):
    # リセットの前に台車を回し、視点を大きく変えておく (古い場面の画像と見分けるため)
    turn = hold.clone()
    turn[0, 8] = 2.0
    for _ in range(30):
        obs, *_ = env.step(turn)
    before = obs["policy"]["front"][0].float()
    t0 = time.time()
    obs, _ = env.reset()
    reset_time += time.time() - t0
    if r < 3:  # ランダム化の効き目を目で確かめる
        Image.fromarray(obs["policy"]["front"][0].cpu().numpy().astype(np.uint8)).save(f"{args.out_dir}/front_reset{r}.png")
    pos = torch.stack([env.scene[n].data.root_pos_w[0] for n in names])
    gap = torch.cdist(pos[:, :2], pos[:, :2]) + torch.eye(len(names), device=dev)
    first = obs["policy"]["front"][0].float()
    for _ in range(30):
        obs, *_ = env.step(hold)
    # 静止した場面なので、リセット直後の画像は 0.5 秒後の画像に近い (古い場面のままなら、リセット前の画像に近くなる)。
    # front カメラは床が大半を写し、リセットの前後で変わるのはゴミと目標領域の 1% 前後の画素だけなので、
    # 画像全体ではなく、リセット前から大きく (いずれかの色が 40 以上) 変わった画素の中で、リセット直後の画像が
    # まだ違っている割合を見る (古い場面のままなら 1 に近い)。1 回の描画ではノイズ除去が収束しきらないので 0 にはならない
    settled = obs["policy"]["front"][0].float()
    changed = (settled - before).abs().amax(dim=-1) > 40
    stale = ((settled - first).abs().amax(dim=-1) > 40)[changed].float().mean().item() if changed.any() else 1.0
    changed_pct = changed.float().mean().item() * 100
    speed = max(env.scene[n].data.root_lin_vel_w[0].norm().item() for n in names)
    moved = (torch.stack([env.scene[n].data.root_pos_w[0] for n in names]) - pos).norm(dim=-1).max().item()
    count = sweep_mdp.debris_in_target(env)[0].sum().item()
    # ゴミすべてと目標領域の四隅が、front カメラから見て水平の画角に入っている
    cam_xy = env.scene["front"].data.pos_w[0, :2]
    corners = torch.tensor([[a, b] for a in (-1, 1) for b in (-1, 1)], device=dev) * sweep_mdp.TARGET_HALF
    pts = torch.cat([pos[:, :2], env.scene[sweep_mdp.TARGET_NAME].data.root_pos_w[0, :2] + corners]) - cam_xy
    view = torch.atan2(pts[:, 0].abs(), pts[:, 1]).max().item()
    check(gap.min().item() > 0.035 and count == 0 and moved < 0.005 and changed_pct > 0.1 and stale < 0.5
          and view < sweep_floor.FRONT_HALF_FOV,
          f"reset {r}: 最も近いゴミの間隔 {gap.min().item() * 100:.1f}cm, 目標領域の中 {count} 個, "
          f"front から見た最大の方位 {np.degrees(view):.1f}° (半画角 {np.degrees(sweep_floor.FRONT_HALF_FOV):.1f}°), "
          f"0.5 秒後の移動 {moved * 1000:.1f}mm, 速さ {speed:.3f}m/s, "
          f"リセット前から変わった画素 {changed_pct:.1f}% のうち、リセット直後にまだ違う割合 {stale:.2f}")

print(f"  リセット 1 回 {reset_time / max(args.resets, 1):.2f}s", flush=True)

print("5. 成功の判定", flush=True)
obs, _ = env.reset()
target_xy = env.scene[sweep_mdp.TARGET_NAME].data.root_pos_w[0, :2]
offsets = torch.tensor([[x, y] for x in (-0.1, -0.035, 0.035, 0.1) for y in (-0.06, 0.06)], device=dev)
results = []
for n_in in (sweep_mdp.SUCCESS_COUNT - 1, sweep_mdp.SUCCESS_COUNT):
    for i in range(n_in):
        obj = env.scene[names[i]]
        p = torch.cat([target_xy + offsets[i], torch.tensor([obj.cfg.spawn.size[2] / 2], device=dev)])
        obj.write_root_pose_to_sim(torch.cat([p, torch.tensor([1.0, 0, 0, 0], device=dev)]).view(1, 7))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    obs, _, terminated, _, _ = env.step(hold)
    results.append(bool(terminated[0]))
check(results == [False, True], f"目標領域に 7 個 -> 終了 {results[0]}, 8 個 -> 終了 {results[1]}")

ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"steps/s={args.steps / elapsed:.1f} RAM peak={ram_gb:.1f}GB", flush=True)
print(f"SWEEP CHECK {'OK' if not failures else 'NG'} task={args.task} device={args.device} "
      f"steps={args.steps} ({elapsed:.1f}s, {args.steps / elapsed:.1f} steps/s, RAM peak {ram_gb:.1f}GB)", flush=True)
env.close()
app.close()
