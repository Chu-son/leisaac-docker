"""train のイメージの確認 (make check の 6 番目)。sim もユーザーのデータも使わない

    python /opt/tools/train_check.py
- contract のキー名とカメラが、lerobot 0.4.2 の実機の LeKiwi (LeKiwiClient) と一致すること
- 合成した小さな HDF5 (datagen と同じ構成) を sweep_convert.py で変換し、照合が通ること (SWEEP CONVERT PASS)
"""
import subprocess
import sys
import tempfile

import h5py
import numpy as np
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

from leisaac_ext import contract

robot = LeKiwiClient(LeKiwiClientConfig(remote_ip="127.0.0.1"))
assert tuple(robot._state_ft) == contract.STATE_NAMES, f"state のキー名が実機と違う: {list(robot._state_ft)}"
assert tuple(robot.action_features) == contract.ACTION_NAMES, f"action のキー名が実機と違う: {list(robot.action_features)}"
assert tuple(robot._cameras_ft) == contract.CAMERAS, f"カメラの名前が実機と違う: {list(robot._cameras_ft)}"
print(f"contract matches lerobot {robot.name}: {len(contract.STATE_NAMES)} state/action names, cameras {contract.CAMERAS}")


def synthetic_images(steps: int, seed: int) -> np.ndarray:
    """フレームごとに動く四角を描いた画像 [steps, 240, 320, 3] (隣り合うフレームが違うので、フレームのずれを検出できる)"""
    h, w, _ = contract.IMAGE_SHAPE
    yy, xx = np.mgrid[:h, :w]
    base = np.stack([xx * 255 // w, yy * 255 // h, np.full((h, w), 64 + 32 * seed)], axis=-1).astype(np.uint8)
    images = np.repeat(base[None], steps, axis=0)
    for t in range(steps):
        x = 20 + 3 * (t // contract.SIM_STEPS_PER_FRAME)
        images[t, 80:160, x : x + 60] = (240, 240, 240)
    return images


rng = np.random.default_rng(0)
with tempfile.TemporaryDirectory() as tmp:
    with h5py.File(f"{tmp}/synthetic.hdf5", "w") as f:
        data = f.create_group("data")
        for i, steps in enumerate((60, 41)):
            g = data.create_group(f"demo_{i}")
            g.attrs["success"] = True
            g.attrs["num_samples"] = steps
            g["actions"] = rng.uniform(-1, 1, (steps, 9)).astype(np.float32)
            g["obs/joint_pos"] = rng.uniform(-1, 1, (steps, 9)).astype(np.float32)
            g["obs/joint_vel"] = rng.uniform(-1, 1, (steps, 9)).astype(np.float32)
            for k, cam in enumerate(contract.CAMERAS):
                g[f"obs/{cam}"] = synthetic_images(steps, 2 * i + k)
        data.attrs["total"] = 101
    out = subprocess.run([sys.executable, "/opt/tools/sweep_convert.py", "--dataset_file", f"{tmp}/synthetic.hdf5",
                          "--root", f"{tmp}/lerobot"], capture_output=True, text=True)
    lines = [line for line in out.stdout.splitlines() if line.startswith(("SWEEP CONVERT", "[NG]", "verify"))]
    print("\n".join(lines))
    if out.returncode != 0:
        print(out.stderr[-3000:])
    assert out.returncode == 0 and any(line.startswith("SWEEP CONVERT PASS") for line in lines), "変換器の確認に失敗"
print("TRAIN CHECK OK")
