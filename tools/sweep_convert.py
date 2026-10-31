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
  datagen のエピソードは成功したものだけのはず。DAgger のエピソード (executed_actions がある。sweep_eval.py --dagger) は、
  actions が計画器の正解なので、失敗したラウンドも使う
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset_file", required=True, nargs="+", help="変換する HDF5 (複数なら順につなぐ)")
parser.add_argument("--root", required=True, help="LeRobot 形式の出力先 (既にあれば止める)")
parser.add_argument("--repo_id", default=None, help="データセットの名前 (省略時は local/<root の最後の名前>)")
args = parser.parse_args()
repo_id = args.repo_id or f"local/{os.path.basename(os.path.normpath(args.root))}"
assert not os.path.exists(args.root), f"{args.root} は既にあります (上書きも追記もしない)"

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (len(contract.STATE_NAMES),), "names": list(contract.STATE_NAMES)},
    "action": {"dtype": "float32", "shape": (len(contract.ACTION_NAMES),), "names": list(contract.ACTION_NAMES)},
    **{f"observation.images.{cam}": {"dtype": "video", "shape": contract.IMAGE_SHAPE, "names": ["height", "width", "channels"]}
       for cam in contract.CAMERAS},
}


def load_episode(group: h5py.Group) -> dict:
    """エピソードを 30fps のフレームにする。値は LeRobot 形式 (state、action は [T, 9]、画像は [T, 240, 320, 3] uint8)"""
    idx = contract.frame_indices(len(group["actions"]))
    state, action = contract.to_lerobot(group["obs/joint_pos"][:][idx], group["obs/joint_vel"][:][idx], group["actions"][:][idx])
    # gzip の塊ごとに展開されるので、1 フレームずつではなく、まとめて読む
    images = {cam: group[f"obs/{cam}"][:][idx] for cam in contract.CAMERAS}
    return {"observation.state": state, "action": action, **{f"observation.images.{cam}": images[cam] for cam in contract.CAMERAS}}


t0 = time.time()
with contextlib.ExitStack() as stack:
    groups = []  # すべてのファイルのエピソード (h5py.Group) を順に並べたもの
    for path in args.dataset_file:
        data = stack.enter_context(h5py.File(path, "r"))["data"]
        names = [f"demo_{i}" for i in range(len(data))]
        assert set(names) == set(data), f"{path}: エピソードの名前が demo_0 〜 demo_{len(data) - 1} になっていない: {sorted(data)}"
        failed = [n for n in names if not data[n].attrs.get("success") and "executed_actions" not in data[n]]
        assert not failed, f"{path}: 成功していないエピソードがある (datagen は成功したものだけを記録する): {failed[:5]}"
        groups += [data[n] for n in names]
        print(f"{path}: {len(names)} episodes ({sum('executed_actions' in data[n] for n in names)} from DAgger)", flush=True)

    # robot_type は、実機の LeKiwi を lerobot で記録したときと同じ名前 (LeKiwiClient.name)
    dataset = LeRobotDataset.create(repo_id, contract.FPS, FEATURES, root=args.root, robot_type="lekiwi_client",
                                    image_writer_threads=8)
    for i, group in enumerate(groups):
        episode = load_episode(group)
        for t in range(len(episode["action"])):
            dataset.add_frame({**{k: v[t] for k, v in episode.items()}, "task": contract.INSTRUCTION})
        dataset.save_episode()
        print(f"episode {i}: {len(episode['action'])} frames ({time.time() - t0:.0f}s)", flush=True)
    dataset.finalize()
    convert_s = time.time() - t0

    # 読み直して、すべてのフレームを HDF5 と比べる
    loaded = LeRobotDataset(repo_id, root=args.root)
    assert loaded.num_episodes == len(groups), f"エピソードの数が違う: {loaded.num_episodes} != {len(groups)}"
    problems, image_mae = [], {cam: [] for cam in contract.CAMERAS}
    for i, group in enumerate(groups):
        episode = load_episode(group)
        start, end = loaded.meta.episodes["dataset_from_index"][i], loaded.meta.episodes["dataset_to_index"][i]
        if end - start != len(episode["action"]):
            problems.append(f"episode {i}: フレームの数が違う ({end - start} != {len(episode['action'])})")
            continue
        decoded = {cam: [] for cam in contract.CAMERAS}
        for t in range(end - start):
            frame = loaded[start + t]
            for key in ("observation.state", "action"):
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
    "dataset_file": args.dataset_file, "root": args.root, "repo_id": repo_id, "episodes": loaded.num_episodes,
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
