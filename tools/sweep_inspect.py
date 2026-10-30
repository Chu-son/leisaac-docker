"""Sweep タスクの記録 (make datagen の HDF5) の中身を確かめる (make sweep-inspect)。Kit は起動しない

    python /opt/tools/sweep_inspect.py --dataset_file /workspace/data/datasets/sweep.hdf5 --out_dir /workspace/data/sweep/inspect/sweep
- エピソードごとの長さ、成功の印、目標領域に入ったゴミの個数 (最初と最後) を表で出す。
  自動生成のログ (<dataset_file から .hdf5 を除いたもの>.datagen/rounds.jsonl) があれば、ラウンドの番号なども並べる
- overview.png: エピソードごとの上から見た図 (最初と最後のゴミ、目標領域、台車の通り道) を並べる
- episode_<番号>.mp4: front、wrist、上から見た図を横に並べた動画 (30fps、実時間)。--videos で選ぶ (省略時はすべて)
"""
import argparse
import json
import os
import subprocess

import cv2
import h5py
import numpy as np

from leisaac_ext import contract

# sweep_mdp.TARGET_HALF と IN_TARGET_MAX_Z と同じ値 (sweep_mdp は isaaclab を読み込むので、Kit なしでは import できない)
TARGET_HALF = 0.15
IN_TARGET_MAX_Z = 0.05
SUCCESS_COUNT = 8
BASE_RADIUS = 0.165
# 上から見た図の範囲 (env の原点からの m) と縮尺。配置の範囲 (sweep_floor) がすべて入る
MAP_X = (-1.2, 1.2)
MAP_Y = (-0.4, 2.6)
MAP_SCALE = 80  # ピクセル/m
MAP_SIZE = (round((MAP_Y[1] - MAP_Y[0]) * MAP_SCALE), round((MAP_X[1] - MAP_X[0]) * MAP_SCALE))  # (高さ, 幅)
_BASE_IDX = [contract.SIM_JOINT_NAMES.index(j) for j in ("base_x", "base_y", "base_theta")]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset_file", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--videos", type=int, nargs="*", default=None, help="動画にするエピソードの番号 (省略時はすべて。空なら作らない)")
args = parser.parse_args()


def load(group: h5py.Group) -> dict:
    """エピソードの、上から見た図に使う系列 (env の原点からの位置) を読む"""
    objects = group["states/rigid_object"]
    names = sorted((n for n in objects if n.startswith("debris_")), key=lambda n: int(n.split("_")[1]))
    joints = group["states/articulation/robot/joint_position"][:]
    root = group["states/articulation/robot/root_pose"][:, :2]
    return {
        "debris": np.stack([objects[n]["root_pose"][:, :3] for n in names], axis=1),  # [T, N, 3]
        "target": objects["target/root_pose"][:, :2],  # [T, 2]
        # 台車は仮想の関節 (base_x、base_y はワールドの軸に沿う並進、base_theta は回転) で動く
        "base": root + joints[:, _BASE_IDX[:2]],
        "theta": joints[:, _BASE_IDX[2]],
    }


def in_target(debris: np.ndarray, target: np.ndarray) -> np.ndarray:
    """sweep_mdp.debris_in_target と同じ判定 (debris [..., N, 3]、target [..., 2]) -> [..., N]"""
    inside = (np.abs(debris[..., :2] - target[..., None, :]) < TARGET_HALF).all(axis=-1)
    return inside & (debris[..., 2] < IN_TARGET_MAX_Z)


def to_px(xy: np.ndarray) -> np.ndarray:
    """env の原点からの位置 (m) -> 図のピクセル (x は右、y は上)"""
    return np.stack([(xy[..., 0] - MAP_X[0]) * MAP_SCALE, (MAP_Y[1] - xy[..., 1]) * MAP_SCALE], axis=-1).round().astype(np.int32)


def draw_map(seq: dict, t: int, label: str) -> np.ndarray:
    """step t の上から見た図 (RGB)。目標領域は青、ゴミは目標領域の中なら緑、外なら赤、最初の位置は灰色の点、台車の通り道は黄色"""
    img = np.full((*MAP_SIZE, 3), 245, np.uint8)
    for v in np.arange(np.ceil(MAP_X[0] * 2) / 2, MAP_X[1], 0.5):
        cv2.line(img, tuple(to_px(np.array([v, MAP_Y[0]]))), tuple(to_px(np.array([v, MAP_Y[1]]))), (225, 225, 225), 1)
    for v in np.arange(np.ceil(MAP_Y[0] * 2) / 2, MAP_Y[1], 0.5):
        cv2.line(img, tuple(to_px(np.array([MAP_X[0], v]))), tuple(to_px(np.array([MAP_X[1], v]))), (225, 225, 225), 1)
    lo, hi = to_px(seq["target"][t] + [-TARGET_HALF, TARGET_HALF]), to_px(seq["target"][t] + [TARGET_HALF, -TARGET_HALF])
    cv2.rectangle(img, tuple(lo), tuple(hi), (150, 180, 255), -1)
    cv2.polylines(img, [to_px(seq["base"][: t + 1])], False, (230, 190, 0), 1)
    center = to_px(seq["base"][t])
    cv2.circle(img, tuple(center), round(BASE_RADIUS * MAP_SCALE), (120, 120, 120), 1)
    heading = np.array([-np.sin(seq["theta"][t]), np.cos(seq["theta"][t])]) * BASE_RADIUS
    cv2.line(img, tuple(center), tuple(to_px(seq["base"][t] + heading)), (120, 120, 120), 1)
    for p in to_px(seq["debris"][0, :, :2]):
        cv2.circle(img, tuple(p), 1, (160, 160, 160), -1)
    inside = in_target(seq["debris"][t], seq["target"][t])
    for p, ok in zip(to_px(seq["debris"][t, :, :2]), inside):
        cv2.rectangle(img, tuple(p - 2), tuple(p + 2), (0, 160, 0) if ok else (220, 0, 0), -1)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def write_video(path: str, frames) -> None:
    """RGB のフレームを H.264 の mp4 にする (30fps、2 倍に拡大)"""
    h, w = MAP_SIZE[0], contract.IMAGE_SHAPE[1] * 2 + MAP_SIZE[1]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(contract.FPS),
           "-i", "-", "-vf", "scale=iw*2:ih*2:flags=neighbor", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, f"ffmpeg failed: {path}"


os.makedirs(args.out_dir, exist_ok=True)
rounds_file = f"{os.path.splitext(args.dataset_file)[0]}.datagen/rounds.jsonl"
rounds = {}
if os.path.exists(rounds_file):
    with open(rounds_file) as f:
        rounds = {r["episode"]: r for r in map(json.loads, f) if r.get("episode") is not None}

with h5py.File(args.dataset_file, "r") as f:
    data = f["data"]
    num = len(data)
    names = [f"demo_{i}" for i in range(num)]
    assert set(names) == set(data), f"エピソードの名前が demo_0 〜 demo_{num - 1} になっていない: {sorted(data)}"
    print(f"{args.dataset_file}: {num} episodes, {data.attrs['total']} steps, {os.path.getsize(args.dataset_file) / 1e9:.2f} GB")
    print("episode  steps  time[s]  success  in_target(start->end)  round  passes  base_min_dist[m]")
    maps, problems = [], []
    videos = range(num) if args.videos is None else args.videos
    for i, name in enumerate(names):
        group = data[name]
        seq = load(group)
        steps = len(group["actions"])
        count = in_target(seq["debris"], seq["target"]).sum(axis=-1)
        r = rounds.get(i, {})
        print(f"{i:7d}  {steps:5d}  {steps / 60:7.2f}  {str(group.attrs.get('success')):7s}  {count[0]:9d} -> {count[-1]:<9d}  "
              f"{r.get('round', '-'):>5}  {r.get('passes', '-'):>6}  {r.get('base_min_dist', '-'):>16}")
        for cam in contract.CAMERAS:
            if group[f"obs/{cam}"].shape != (steps, *contract.IMAGE_SHAPE):
                problems.append(f"episode {i}: obs/{cam} の形が {group[f'obs/{cam}'].shape}")
        if not group.attrs.get("success") or count[-1] < SUCCESS_COUNT:
            problems.append(f"episode {i}: 成功していない (success={group.attrs.get('success')}、最後の個数 {count[-1]})")
        maps.append(draw_map(seq, steps - 1, f"#{i} {steps / 60:.1f}s in={count[-1]}"))
        if i in videos:
            # gzip の塊ごとに展開されるので、1 フレームずつではなく、まとめて読む
            front, wrist = group["obs/front"][:], group["obs/wrist"][:]
            frames = (np.concatenate([front[t], wrist[t], draw_map(seq, t, f"#{i} t={t / 60:4.1f}s in={count[t]}")], axis=1)
                      for t in contract.frame_indices(steps))
            write_video(f"{args.out_dir}/episode_{i:03d}.mp4", frames)

cols = 10
rows = [np.concatenate(maps[k : k + cols] + [np.full_like(maps[0], 255)] * (cols - len(maps[k : k + cols])), axis=1)
        for k in range(0, len(maps), cols)]
cv2.imwrite(f"{args.out_dir}/overview.png", cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
for p in problems:
    print(f"[NG] {p}")
print(f"SWEEP INSPECT {num - len({p.split(':')[0] for p in problems})}/{num} ok -> {args.out_dir} "
      f"(overview.png{', episode_*.mp4' if videos else ''})")
