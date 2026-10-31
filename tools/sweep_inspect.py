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

import cv2
import h5py
import numpy as np

from leisaac_ext import contract
from leisaac_ext.sweep_video import TARGET_HALF, VideoWriter, draw_map, overview

# sweep_mdp.IN_TARGET_MAX_Z と SUCCESS_COUNT と同じ値 (sweep_mdp は isaaclab を読み込むので、Kit なしでは import できない)
IN_TARGET_MAX_Z = 0.05
SUCCESS_COUNT = 8
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


def map_at(seq: dict, t: int, label: str) -> np.ndarray:
    """step t の上から見た図"""
    inside = in_target(seq["debris"][t], seq["target"][t])
    return draw_map(seq["debris"][t, :, :2], seq["debris"][0, :, :2], inside, seq["target"][t], seq["base"][: t + 1],
                    seq["theta"][t], label)


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
        # DAgger の記録 (executed_actions がある) は、失敗したラウンドも含む
        if "executed_actions" not in group and (not group.attrs.get("success") or count[-1] < SUCCESS_COUNT):
            problems.append(f"episode {i}: 成功していない (success={group.attrs.get('success')}、最後の個数 {count[-1]})")
        maps.append(map_at(seq, steps - 1, f"#{i} {steps / 60:.1f}s in={count[-1]}"))
        if i in videos:
            # gzip の塊ごとに展開されるので、1 フレームずつではなく、まとめて読む
            front, wrist = group["obs/front"][:], group["obs/wrist"][:]
            video = VideoWriter(f"{args.out_dir}/episode_{i:03d}.mp4")
            for t in contract.frame_indices(steps):
                video.write(front[t], wrist[t], map_at(seq, t, f"#{i} t={t / 60:4.1f}s in={count[t]}"))
            video.close()

cv2.imwrite(f"{args.out_dir}/overview.png", cv2.cvtColor(overview(maps), cv2.COLOR_RGB2BGR))
for p in problems:
    print(f"[NG] {p}")
print(f"SWEEP INSPECT {num - len({p.split(':')[0] for p in problems})}/{num} ok -> {args.out_dir} "
      f"(overview.png{', episode_*.mp4' if videos else ''})")
