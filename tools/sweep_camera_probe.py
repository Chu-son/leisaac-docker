"""Sweep タスクのカメラの取り付けの候補を、同じ場面で描き比べる (make sweep-camera-probe)。docs/sweep.md の「カメラ」

    python /opt/tools/leisaac_run.py /opt/tools/sweep_camera_probe.py --seeds 1 2 --device cpu --enable_cameras --headless
配置ごとに、計画器で動かしながら 3 つの場面を撮る。start: 移動を始めるとき (アームは移動の姿勢)。travel: 移動の 1.5 秒後。
push: 最初の掃きで押し始めて 0.3 秒後 (関節の状態を直接書き換えても、step するまで描画に反映されないため、実際に動かす)。候補のカメラ (FRONT、WRIST) は、env を作る前に追加のセンサとして足し、場面ごとに全候補の
画像を、候補を行に並べた 1 枚の画像 (out_dir/<カメラ>.png) にする (env を作った後に取り付けの姿勢や焦点距離を
USD で書き換えても、描画には反映されなかった)。カメラの向きのランダム化は切る。
front は水平からの仰角、wrist は今の向きからの、画像の横軸まわりの回転。
最後に、作った後の書き換え (sweep_mdp.randomize_camera_offset と同じ方法) が描画に効くかも調べて出力する。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
parser.add_argument("--out_dir", default="/workspace/data/sweep/camera_probe")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import math
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
from leisaac_ext import contract, sweep_floor, sweep_planner, sweep_video

# front: (名前, 親リンク (base) からの位置 (m)、水平からの仰角 (度、上が正)、焦点距離 (mm。横の画角は 36.83mm の撮像面で決まる))
FRONT = [
    ("now z.025 +10 f36.5", (0.0, 0.13, 0.025), 10, 36.5),
    ("z.025 -10 f24", (0.0, 0.13, 0.025), -10, 24.0),
    ("z.20 -30 f24", (0.0, 0.13, 0.20), -30, 24.0),
    ("z.20 -30 f18", (0.0, 0.13, 0.20), -30, 18.0),
    ("z.35 -40 f24", (0.0, 0.13, 0.35), -40, 24.0),
    ("z.35 -40 f18", (0.0, 0.13, 0.35), -40, 18.0),
]
# wrist: (名前, 今の取り付けの向きからの、画像の横軸まわりの回転 (度)、焦点距離 (mm))
WRIST = [
    ("now f36.5", 0, 36.5),
    ("+30 f24", 30, 24.0),
    ("-30 f24", -30, 24.0),
    ("+60 f24", 60, 24.0),
    ("-60 f24", -60, 24.0),
]

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
cfg.recorders = None
cfg.set_randomization(tuple(n for n in sweep_floor.DEFAULT_RANDOMIZATION if n != "camera"))


def rot_x_tuple(deg: float) -> tuple:
    a = math.radians(deg) / 2
    return (math.cos(a), math.sin(a), 0.0, 0.0)


def mul(q1, q2) -> tuple:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


base_front, base_wrist = cfg.scene.front, cfg.scene.wrist
for k, (name, pos, pitch, focal) in enumerate(FRONT):
    setattr(cfg.scene, f"probe_front_{k}", base_front.replace(
        prim_path=base_front.prim_path.replace("front_camera", f"probe_front_{k}"),
        offset=base_front.offset.replace(pos=pos, rot=rot_x_tuple(90 + pitch)),
        spawn=base_front.spawn.replace(focal_length=focal)))
for k, (name, delta, focal) in enumerate(WRIST):
    setattr(cfg.scene, f"probe_wrist_{k}", base_wrist.replace(
        prim_path=base_wrist.prim_path.replace("wrist_camera", f"probe_wrist_{k}"),
        offset=base_wrist.offset.replace(rot=mul(base_wrist.offset.rot, rot_x_tuple(delta))),
        spawn=base_wrist.spawn.replace(focal_length=focal)))
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
poses = sweep_planner.calibrate_arm(env)
planner = sweep_planner.SweepPlanner(poses)
robot = env.scene["robot"]
dev = env.device
ids = torch.tensor([0], device=dev)
arm_idx = [robot.joint_names.index(j) for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
base_idx = [robot.joint_names.index(j) for j in ("base_x", "base_y", "base_theta")]
front, wrist = env.scene["front"], env.scene["wrist"]
front_rot0 = torch.tensor(front.cfg.offset.rot, device=dev)
wrist_pos0 = torch.tensor(wrist.cfg.offset.pos, device=dev)
wrist_rot0 = torch.tensor(wrist.cfg.offset.rot, device=dev)


def rot_x(deg: float) -> torch.Tensor:
    a = math.radians(deg) / 2
    return torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0], device=dev)


def set_camera(cam, pos: torch.Tensor, quat: torch.Tensor, focal: float | None) -> None:
    cam._view.set_local_poses(translations=pos[None], orientations=quat[None], indices=ids)
    prim = env.scene.stage.GetPrimAtPath(cam.cfg.prim_path.replace(env.scene.env_regex_ns, env.scene.env_prim_paths[0]))
    if focal is not None:
        prim.GetAttribute("focalLength").Set(focal)


def grab(cam) -> np.ndarray:
    for _ in range(3):
        env.sim.render()
    cam.update(0.0, force_recompute=True)
    return cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    return img


def step(action: np.ndarray, n: int = contract.SIM_STEPS_PER_FRAME) -> None:
    for _ in range(n):
        env.step(torch.tensor(action, device=dev, dtype=torch.float32)[None])


def hold_action() -> np.ndarray:
    """今のアームの関節位置を保ち、台車を止める action"""
    arm = robot.data.joint_pos[0, arm_idx].cpu().numpy()
    return np.concatenate([arm, [contract.GRIPPER_HOLD_RAD], np.zeros(3)])


os.makedirs(args.out_dir, exist_ok=True)
SCENES = ("start", "travel", "push")
rows = {"front": [[] for _ in FRONT], "wrist": [[] for _ in WRIST]}
scene_names = []
for seed in args.seeds:
    env.seed(seed)
    env.reset()
    planner.reset()
    start_state = sweep_planner.read_state(env)
    path = []
    # 計画器で動かし、移動を始めるとき (start)、移動の 1.5 秒後 (travel)、押し始めて 0.3 秒後 (push) に撮る
    taken = set()
    for _ in range(int(20 / env.step_dt / contract.SIM_STEPS_PER_FRAME)):
        state = sweep_planner.read_state(env)
        path.append(state.base)
        phase, timer = planner.phase, planner.timer
        scene = ("start" if phase == "travel" and "start" not in taken else
                 "travel" if phase == "travel" and timer >= 1.5 and "travel" not in taken else
                 "push" if phase == "push" and timer >= 0.3 and "push" not in taken else None)
        if scene:
            taken.add(scene)
            scene_names.append(f"seed{seed}-{scene}")
            for k, (name, *_) in enumerate(FRONT):
                rows["front"][k].append(label(grab(env.scene[f"probe_front_{k}"]), f"{name} | seed{seed} {scene}"))
            for k, (name, *_) in enumerate(WRIST):
                rows["wrist"][k].append(label(grab(env.scene[f"probe_wrist_{k}"]), f"{name} | seed{seed} {scene}"))
        if taken == set(SCENES):
            break
        step(planner.act(state))
    assert taken == set(SCENES), f"seed {seed}: {sorted(taken)} だけ撮れた"
    # 上から見た図 (どの場面かの確認用)
    cv2.imwrite(f"{args.out_dir}/map_seed{seed}.png", cv2.cvtColor(sweep_video.draw_map(
        state.debris, start_state.debris, np.zeros(len(state.debris), bool), state.target, np.array(path), state.theta,
        f"seed{seed}"), cv2.COLOR_RGB2BGR))

for cam, cam_rows in rows.items():
    grid = np.concatenate([np.concatenate(r, axis=1) for r in cam_rows], axis=0)
    cv2.imwrite(f"{args.out_dir}/{cam}.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
# env を作った後の書き換え (sweep_mdp.randomize_camera_offset と同じ方法) が、step の後の描画に効くか。
# ロボットを止めたまま比べる (平均の画素の差。基準は、何も変えずに step した 2 枚の差)
hold = hold_action()
step(hold, 10)
a = grab(front).astype(np.float32)
step(hold)
a2 = grab(front).astype(np.float32)
set_camera(front, torch.tensor(front.cfg.offset.pos, device=dev), rot_x(90 - 30), None)
step(hold)
b = grab(front).astype(np.float32)
set_camera(front, torch.tensor(front.cfg.offset.pos, device=dev), front_rot0, 18.0)
step(hold)
c = grab(front).astype(np.float32)
print(f"runtime change after step: baseline diff={np.abs(a2 - a).mean():.2f} pose diff={np.abs(b - a2).mean():.2f} "
      f"focal diff={np.abs(c - a2).mean():.2f}", flush=True)
print(f"SWEEP CAMERA PROBE scenes={scene_names} -> {args.out_dir}", flush=True)
env.close()
app.close()
