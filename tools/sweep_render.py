"""Sweep タスクの評価の状態の記録を、俯瞰のカメラを足して描き直し、紹介用の動画にする (make sweep-render)

    python /opt/tools/leisaac_run.py /opt/tools/sweep_render.py --eval_dir /workspace/data/sweep/eval/<日時> --rounds 0 9 --device cpu --enable_cameras --headless
物理は進めず、記録したフレームごとの状態 (台車と関節、ゴミの位置と姿勢) を書き戻して描く。sweep_replay.py と違い、
action を流し直さないので、評価のときの動きがそのまま再現される。見た目のランダム化 (照明、ゴミの色、カメラの取り付け) も記録どおりに戻す。
俯瞰のカメラは、ラウンド全体の台車の通り道、ゴミ、目標領域が入るように、ラウンドごとに台車の初めの位置の側の斜め上に置く。
動画は 1920x1080 (16:9)、30fps。左に俯瞰 (1440x1080)、右に縦 3 段で front、wrist、上から見た図 (各 480x360)。
out_dir/round_<番号>_<success|failure>.mp4 に書く。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--eval_dir", required=True, help="評価の out_dir (states/round_<番号>.npz を描き直す)")
parser.add_argument("--rounds", type=int, nargs="*", default=[], help="描き直すラウンドの番号 (省略時はすべて)")
parser.add_argument("--title", default="", help="俯瞰の左上に出す見出し (例: ACT, exp 10)")
parser.add_argument("--out_dir", default=None, help="保存先 (省略時は eval_dir/render)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import glob
import math
import os
import subprocess

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
from leisaac_ext import contract, sweep_log, sweep_mdp, sweep_video

OUT_W, OUT_H = 1920, 1080
MAIN_W = OUT_H * 4 // 3  # 俯瞰 1440x1080
SIDE_W, SIDE_H = OUT_W - MAIN_W, OUT_H // 3  # 右の 3 段 480x360
# 俯瞰のカメラ: 焦点距離 24mm、撮像面 36x27mm (横の画角約 74°)。水平から 50° 見下ろす
FOCAL, APERTURE_W, APERTURE_H = 24.0, 36.0, 27.0
PITCH = math.radians(50)
MARGIN = 0.15  # 通り道とゴミの外側に足す余白 (m)
FILL = 0.9  # 画角に対して範囲が占める割合の目安 (斜めから見るので 1 より小さくしても入る)
# 状態を書き戻した後の step の回数。カメラの出力は 1 回分遅れるので、2 回進めてから読む (大きく飛んだ後も描画が追いつく)
STEPS_PER_FRAME = 2

paths = sorted(glob.glob(f"{args.eval_dir}/states/round_*.npz"))
assert paths, f"{args.eval_dir}/states/round_*.npz がない (状態の記録より前の評価)"
rounds = args.rounds or [int(os.path.basename(p)[6:9]) for p in paths]
out_dir = args.out_dir or f"{args.eval_dir}/render"
os.makedirs(out_dir, exist_ok=True)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
cfg.recorders = None
cfg.terminations = {}
front_cfg = cfg.scene.front
cfg.scene.overview = front_cfg.replace(
    prim_path="{ENV_REGEX_NS}/OverviewCamera",
    offset=front_cfg.offset.replace(pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    width=MAIN_W, height=OUT_H,
    spawn=front_cfg.spawn.replace(focal_length=FOCAL, horizontal_aperture=APERTURE_W, vertical_aperture=APERTURE_H))
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
dev = env.device
origin = env.scene.env_origins[0]
robot = env.scene["robot"]
run = sweep_log.load_run(args.eval_dir)
assert run["robot"]["joint_names"] == robot.joint_names, "記録と今の env で関節の並びが違う"
overview = env.scene["overview"]


def place_overview(base_xy: np.ndarray, debris_xy: np.ndarray, target_xy: np.ndarray) -> None:
    """ラウンド全体 (台車の通り道 [T, 2]、ゴミ [T*N, 2]、目標領域) が入るように、俯瞰のカメラを手前 (-y) の斜め上に置く"""
    pts = np.concatenate([base_xy, debris_xy, target_xy[None]])
    lo, hi = pts.min(0) - MARGIN, pts.max(0) + MARGIN
    center = (lo + hi) / 2
    # 横の幅と奥行きが、それぞれ横と縦の画角に入る距離 (奥行きは斜めから見て縮むが、余裕として縮めずに使う)
    dist = max((hi[0] - lo[0]) / 2 / (APERTURE_W / 2 / FOCAL), (hi[1] - lo[1]) / 2 / (APERTURE_H / 2 / FOCAL)) * FILL
    eye = [center[0], center[1] - dist * math.cos(PITCH), dist * math.sin(PITCH)]
    overview.set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32, device=dev) + origin,
                                       torch.tensor([[*center, 0.0]], dtype=torch.float32, device=dev) + origin)


def put_state(joint_pos: torch.Tensor, debris_pos: torch.Tensor, debris_quat: torch.Tensor) -> None:
    """記録した 1 フレームの状態を書き戻す (速度は 0、関節の目標は今の位置)"""
    zeros = torch.zeros_like(joint_pos)[None]
    robot.write_joint_state_to_sim(joint_pos[None], zeros)
    robot.set_joint_position_target(joint_pos[None])
    robot.set_joint_velocity_target(zeros)
    for i, name in enumerate(sweep_mdp.DEBRIS_NAMES):
        env.scene[name].write_root_pose_to_sim(torch.cat([debris_pos[i] + origin, debris_quat[i]])[None])
        env.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    env.scene.write_data_to_sim()


def grab(name: str) -> np.ndarray:
    env.scene[name].update(0.0, force_recompute=True)
    return env.scene[name].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)


def yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def text(img: np.ndarray, s: str, org: tuple, scale: float, thickness: int = 2) -> None:
    """白い縁取りの黒文字"""
    for color, th in (((255, 255, 255), thickness + 3), ((20, 20, 20), thickness)):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def side_panel(img: np.ndarray, title: str) -> np.ndarray:
    """右の段 (480x360)。縦横比を保って縮め、余白は図の背景色で埋める"""
    h, w = img.shape[:2]
    s = min(SIDE_W / w, SIDE_H / h)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    panel = np.full((SIDE_H, SIDE_W, 3), 245, np.uint8)
    y0, x0 = (SIDE_H - img.shape[0]) // 2, (SIDE_W - img.shape[1]) // 2
    panel[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    text(panel, title, (10, 28), 0.7)
    cv2.rectangle(panel, (0, 0), (SIDE_W - 1, SIDE_H - 1), (200, 200, 200), 1)
    return panel


def render_round(r: int) -> str:
    data = sweep_log.load_round(f"{args.eval_dir}/states/round_{r:03d}.npz")
    sweep_log.apply_setup(env, data)  # 俯瞰のカメラは記録にないので、取り付けの姿勢は戻さない (place_overview で置く)
    jp = torch.as_tensor(data["frame/joint_pos"], device=dev)
    dpos = torch.as_tensor(data["frame/debris_pos"], device=dev)
    dquat = torch.as_tensor(data["frame/debris_quat"], device=dev)
    base_xy, debris_xy = data["frame/base_pos"][:, :2], data["frame/debris_pos"][..., :2]
    target_xy = data["setup/init/rigid_object/target/root_pose"][:2]
    place_overview(base_xy, debris_xy.reshape(-1, 2), target_xy)
    success = bool(data["meta/success"])
    path = f"{out_dir}/round_{r:03d}_{'success' if success else 'failure'}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
           "-r", str(contract.FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = len(jp)
    for t in range(n):
        for _ in range(STEPS_PER_FRAME):
            put_state(jp[t], dpos[t], dquat[t])
            env.sim.step(render=True)
            env.scene.update(env.physics_dt)
        main = grab("overview").copy()
        inside = data["frame/inside"][t]
        count = int(inside.sum())
        lines = [s for s in (args.title, f"Round {r}   t = {data['frame/time'][t]:4.1f} s",
                             f"In target: {count} / {len(inside)}") if s]
        if t == n - 1:
            lines.append("SUCCESS" if success else "FAILURE (time limit)")
        for k, s in enumerate(lines):
            text(main, s, (24, 52 + 46 * k), 1.2)
        map_img = sweep_video.draw_map(debris_xy[t], debris_xy[0], inside, target_xy, base_xy[: t + 1],
                                       yaw(data["frame/base_quat"][t]), "")
        side = [side_panel(grab("front"), "Front camera"), side_panel(grab("wrist"), "Wrist camera"),
                side_panel(map_img, "Top view (path)")]
        frame = np.concatenate([main, np.concatenate(side, axis=0)], axis=1)
        proc.stdin.write(frame.tobytes())
    # 最後のフレームを 1.5 秒止めて、結果を見せる
    for _ in range(contract.FPS * 3 // 2):
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, f"ffmpeg failed: {path}"
    return path


for r in rounds:
    path = render_round(r)
    print(f"round {r}: -> {path}", flush=True)
print(f"SWEEP RENDER {len(rounds)} rounds -> {out_dir}", flush=True)
env.close()
app.close()
