"""Sweep タスクの記録 (make datagen の HDF5) を LeRobot 形式 (lerobot 0.4.2、v3.0) に変換する (make convert)。train のイメージで実行する

    python /opt/tools/sweep_convert.py --dataset_file /workspace/data/datasets/sweep.hdf5 --root /workspace/data/lerobot/sweep
- 60Hz の記録から 1 step おきに取り出し (contract.frame_indices、30fps)、contract.to_lerobot で state と action にする。
  画像は front と wrist の 2 台で、動画 (lerobot の既定の AV1) にする
- 変換した後、LeRobotDataset で読み直し、すべてのフレームを HDF5 と比べる。state と action は完全に一致すること、
  画像は動画の圧縮による差 (画素の値の平均絶対誤差) が MAX_IMAGE_MAE 以下であること。
  隣り合うフレームの差は圧縮の差より小さいことがあるので、エピソードごとに、同じ番号のフレームとの差が、
  前後に 1 フレームずらしたときの差より小さいことも確かめる (フレームのずれの検出)
- 変換元と結果は root の convert.json に残す
- --dataset_file に複数の HDF5 を並べると、順につないで 1 つのデータセットにする (例: 計画器のデータと DAgger のデータ)。
  エピソードの作り方は属性 source (sweep_eval.py が残す。planner、dart、dagger) で見分ける。source のない古い記録は、
  executed_actions があれば dagger、なければ planner とみなす。dagger のエピソードは actions が計画器の正解なので、
  失敗したラウンドも使う。それ以外 (planner、dart) は成功したものだけのはず
- 指示文 (task) は、エピソードの属性 instruction (S2 の言語のタスク) を使う。属性がなければ contract.INSTRUCTION
- --goal は、エピソードの属性 goal_color (指示した色) を one-hot (contract.goal_onehot) にして、contract.GOAL_KEY に入れる
  (S2g の目標の番号を入れた ACT。docs/sweep.md の「今後」)。すべてのエピソードに goal_color があること
- 障害物のタスク (S3a) で、箱に当たったエピソード (属性 obstacle_moved が contract.OBSTACLE_MOVED 以上) は使わない
- --start_copies K は、すべてのエピソードの最初の START_S 秒を、短いエピソードとして K 回ずつ後ろに足す。
  目標で行き先が分かれるのはエピソードの始めだけなので、その場面を学習で多く見せる (S2g の 2 回目)
"""
import argparse
import contextlib
import json
import os
import time

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from leisaac_ext import contract

# 動画の圧縮による画素の値 (0〜255) の平均絶対誤差の上限。AV1 (crf 30) の実測は 1〜2 程度 (docs/sweep.md)
MAX_IMAGE_MAE = 4.0
# --start_copies で複製する、エピソードの始めの長さ (秒)。台車は約 1 秒で動き出す。1 秒の時点の action chunk
# (ACT は 100 フレーム) が収まる長さにする
START_S = 4.0

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset_file", required=True, nargs="+", help="変換する HDF5 (複数なら順につなぐ)")
parser.add_argument("--root", required=True, help="LeRobot 形式の出力先 (既にあれば止める)")
parser.add_argument("--repo_id", default=None, help="データセットの名前 (省略時は local/<root の最後の名前>)")
parser.add_argument("--goal", action="store_true", help="指示した色の one-hot を contract.GOAL_KEY に入れる")
parser.add_argument("--start_copies", type=int, default=0, help="エピソードの最初の START_S 秒を、短いエピソードとして足す回数")
args = parser.parse_args()
repo_id = args.repo_id or f"local/{os.path.basename(os.path.normpath(args.root))}"
assert not os.path.exists(args.root), f"{args.root} は既にあります (上書きも追記もしない)"

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (len(contract.STATE_NAMES),), "names": list(contract.STATE_NAMES)},
    "action": {"dtype": "float32", "shape": (len(contract.ACTION_NAMES),), "names": list(contract.ACTION_NAMES)},
    **{f"observation.images.{cam}": {"dtype": "video", "shape": contract.IMAGE_SHAPE, "names": ["height", "width", "channels"]}
       for cam in contract.CAMERAS},
}
if args.goal:
    FEATURES[contract.GOAL_KEY] = {"dtype": "float32", "shape": (len(contract.GOAL_COLORS),), "names": list(contract.GOAL_COLORS)}


def episode_source(group: h5py.Group) -> str:
    """エピソードの作り方 (planner、dart、dagger)"""
    return str(group.attrs.get("source", "dagger" if "executed_actions" in group else "planner"))


def load_episode(group: h5py.Group, frames: int | None = None) -> dict:
    """エピソードを 30fps のフレームにする。値は LeRobot 形式 (state、action は [T, 9]、画像は [T, 240, 320, 3] uint8)。
    frames を渡すと、最初の frames フレームだけにする"""
    idx = contract.frame_indices(len(group["actions"]))[:frames]
    end = int(idx[-1]) + 1
    state, action = contract.to_lerobot(group["obs/joint_pos"][:end][idx], group["obs/joint_vel"][:end][idx],
                                        group["actions"][:end][idx])
    # gzip の塊ごとに展開されるので、1 フレームずつではなく、まとめて読む
    images = {cam: group[f"obs/{cam}"][:end][idx] for cam in contract.CAMERAS}
    episode = {"observation.state": state, "action": action, **{f"observation.images.{cam}": images[cam] for cam in contract.CAMERAS}}
    if args.goal:
        episode[contract.GOAL_KEY] = np.tile(contract.goal_onehot(str(group.attrs["goal_color"])), (len(idx), 1))
    return episode


t0 = time.time()
with contextlib.ExitStack() as stack:
    groups = []  # すべてのファイルのエピソード (h5py.Group) を順に並べたもの
    skipped_obstacle = 0  # 箱に当たったので使わないエピソードの数
    for path in args.dataset_file:
        data = stack.enter_context(h5py.File(path, "r"))["data"]
        names = [f"demo_{i}" for i in range(len(data))]
        assert set(names) == set(data), f"{path}: エピソードの名前が demo_0 〜 demo_{len(data) - 1} になっていない: {sorted(data)}"
        sources = {n: episode_source(data[n]) for n in names}
        failed = [n for n in names if not data[n].attrs.get("success") and sources[n] != "dagger"]
        assert not failed, f"{path}: 成功していないエピソードがある (datagen は成功したものだけを記録する): {failed[:5]}"
        if args.goal:
            missing = [n for n in names if "goal_color" not in data[n].attrs]
            assert not missing, f"{path}: --goal なのに goal_color のないエピソードがある: {missing[:5]}"
        hit_box = [n for n in names if float(data[n].attrs.get("obstacle_moved", 0.0)) >= contract.OBSTACLE_MOVED]
        if hit_box:
            print(f"{path}: 箱に当たった {len(hit_box)} エピソードを使わない: {hit_box[:10]}", flush=True)
        skipped_obstacle += len(hit_box)
        groups += [data[n] for n in names if n not in hit_box]
        counts = {s: list(sources.values()).count(s) for s in sorted(set(sources.values()))}
        instructions = {str(data[n].attrs["instruction"]) for n in names if "instruction" in data[n].attrs}
        print(f"{path}: {len(names)} episodes {counts} ({sum('disturb' in data[n].attrs for n in names)} with disturbance, "
              f"{len(instructions)} distinct instructions)", flush=True)

    # 書き出すエピソード (記録のエピソードと、使うフレームの数。None はすべて)。--start_copies の複製は後ろに並べる
    segments = [(g, None) for g in groups] + [(g, round(START_S * contract.FPS)) for _ in range(args.start_copies) for g in groups]
    if args.start_copies:
        print(f"start copies: {args.start_copies} x {len(groups)} episodes x {round(START_S * contract.FPS)} frames", flush=True)

    # robot_type は、実機の LeKiwi を lerobot で記録したときと同じ名前 (LeKiwiClient.name)
    dataset = LeRobotDataset.create(repo_id, contract.FPS, FEATURES, root=args.root, robot_type="lekiwi_client",
                                    image_writer_threads=8)
    for i, (group, frames) in enumerate(segments):
        episode = load_episode(group, frames)
        # 指示文。言語のタスク (S2) はエピソードごとに違う。属性のない記録は固定の指示文
        task = str(group.attrs.get("instruction", contract.INSTRUCTION))
        for t in range(len(episode["action"])):
            dataset.add_frame({**{k: v[t] for k, v in episode.items()}, "task": task})
        dataset.save_episode()
        print(f"episode {i}: {len(episode['action'])} frames ({time.time() - t0:.0f}s)", flush=True)
    dataset.finalize()
    convert_s = time.time() - t0

    # 読み直して、すべてのフレームを HDF5 と比べる
    loaded = LeRobotDataset(repo_id, root=args.root)
    assert loaded.num_episodes == len(segments), f"エピソードの数が違う: {loaded.num_episodes} != {len(segments)}"
    problems, image_mae = [], {cam: [] for cam in contract.CAMERAS}
    for i, (group, frames) in enumerate(segments):
        episode = load_episode(group, frames)
        start, end = loaded.meta.episodes["dataset_from_index"][i], loaded.meta.episodes["dataset_to_index"][i]
        if end - start != len(episode["action"]):
            problems.append(f"episode {i}: フレームの数が違う ({end - start} != {len(episode['action'])})")
            continue
        decoded = {cam: [] for cam in contract.CAMERAS}
        for t in range(end - start):
            frame = loaded[start + t]
            for key in ("observation.state", "action") + ((contract.GOAL_KEY,) if args.goal else ()):
                if not np.array_equal(frame[key].numpy(), episode[key][t]):
                    problems.append(f"episode {i} frame {t}: {key} が一致しない")
            for cam in contract.CAMERAS:
                # LeRobotDataset は画像を [C, H, W]、0〜1 の float で返す
                decoded[cam].append((frame[f"observation.images.{cam}"] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy())
        shift_mae = {}
        for cam in contract.CAMERAS:
            dec, ref = np.stack(decoded[cam]).astype(np.int16), episode[f"observation.images.{cam}"].astype(np.int16)
            mae = np.abs(dec - ref).mean(axis=(1, 2, 3))
            image_mae[cam].extend(mae)
            # 復号したフレーム t を、HDF5 のフレーム t + s と比べる
            shift_mae[cam] = {s: float(np.abs(dec[max(0, -s) : len(dec) - max(0, s)] - ref[max(0, s) : len(ref) - max(0, -s)]).mean())
                              for s in (-1, 0, 1)}
            if shift_mae[cam][0] >= min(shift_mae[cam][-1], shift_mae[cam][1]):
                problems.append(f"episode {i}: {cam} のフレームがずれている (ずらしたときの差 {shift_mae[cam]})")
        print(f"verify episode {i}: " + " ".join(f"{cam}_mae={v[0]:.2f} (shift -1: {v[-1]:.2f}, +1: {v[1]:.2f})"
                                                for cam, v in shift_mae.items()), flush=True)
    for cam in contract.CAMERAS:
        if max(image_mae[cam]) > MAX_IMAGE_MAE:
            problems.append(f"{cam}: 画像の平均絶対誤差の最大が {max(image_mae[cam]):.2f} (上限 {MAX_IMAGE_MAE})")

summary = {
    "dataset_file": args.dataset_file, "root": args.root, "goal": args.goal, "start_copies": args.start_copies, "skipped_obstacle": skipped_obstacle, "repo_id": repo_id, "episodes": loaded.num_episodes,
    "frames": loaded.num_frames, "fps": contract.FPS, "convert_s": round(convert_s), "total_s": round(time.time() - t0),
    "image_mae": {cam: {"mean": round(float(np.mean(v)), 3), "max": round(float(np.max(v)), 3)} for cam, v in image_mae.items()},
    "problems": problems,
}
with open(f"{args.root}/convert.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
for p in problems[:20]:
    print(f"[NG] {p}")
print(f"SWEEP CONVERT {'PASS' if not problems else 'FAIL'} {summary['episodes']} episodes, {summary['frames']} frames, "
      f"image_mae(max) " + " ".join(f"{cam}={v['max']}" for cam, v in summary["image_mae"].items()) + f" -> {args.root}")
