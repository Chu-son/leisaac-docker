"""train のイメージの確認 (make check の 6 番目)。sim もユーザーのデータも使わない

    python /opt/tools/train_check.py
- contract のキー名とカメラが、lerobot 0.4.2 の実機の LeKiwi (LeKiwiClient) と一致すること
- 学習するポリシー (ACT、SmolVLA、GR00T) を import でき、flash-attn (GR00T が使う) がこの GPU で動くこと
- 合成した小さな HDF5 (datagen と同じ構成) を sweep_convert.py で変換し、照合が通ること (SWEEP CONVERT PASS)
- 変換したデータで、sweep_train.py が ACT を数 step 学習し、チェックポイントを残すこと (SWEEP TRAIN OK)
- そのチェックポイントを policy_server.py が読み、観測を送ると action の系列を返すこと (POLICY SERVER OK)
"""
import os
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
import torch
from flash_attn import flash_attn_func
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
from lerobot.policies.groot.modeling_groot import GrootPolicy  # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

from leisaac_ext import contract, remote

robot = LeKiwiClient(LeKiwiClientConfig(remote_ip="127.0.0.1"))
assert tuple(robot._state_ft) == contract.STATE_NAMES, f"state のキー名が実機と違う: {list(robot._state_ft)}"
assert tuple(robot.action_features) == contract.ACTION_NAMES, f"action のキー名が実機と違う: {list(robot.action_features)}"
assert tuple(robot._cameras_ft) == contract.CAMERAS, f"カメラの名前が実機と違う: {list(robot._cameras_ft)}"
print(f"contract matches lerobot {robot.name}: {len(contract.STATE_NAMES)} state/action names, cameras {contract.CAMERAS}")

q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
assert flash_attn_func(q, q, q).shape == q.shape
print(f"policies import OK (act, smolvla, groot), flash-attn runs on {torch.cuda.get_device_name()}")


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

    out = subprocess.run([sys.executable, "/opt/tools/sweep_train.py", "--policy", "act", "--dataset_root", f"{tmp}/lerobot",
                          "--output_dir", f"{tmp}/train", "--steps", "4", "--batch_size", "2", "--num_workers", "0"],
                         capture_output=True, text=True)
    lines = [line for line in out.stdout.splitlines() if line.startswith("SWEEP TRAIN")]
    print("\n".join(lines))
    if out.returncode != 0:
        print(out.stderr[-3000:])
    checkpoint = f"{tmp}/train/checkpoints/last/pretrained_model/model.safetensors"
    assert out.returncode == 0 and os.path.isfile(checkpoint), "学習の確認に失敗 (チェックポイントがない)"

    # ネットワークは host なので、動いているかもしれない推論サーバ (make policy-up) とは別のポートを使う
    port = remote.PORT + 100
    server = subprocess.Popen([sys.executable, "/opt/tools/policy_server.py", "--port", str(port),
                               "--policy_path", os.path.dirname(checkpoint)],
                              stdout=open(f"{tmp}/policy_server.log", "w"), stderr=subprocess.STDOUT)
    server_ok = False
    try:
        client = remote.PolicyClient(port=port, timeout_ms=120000)
        info = client.call("ping")[0]
        client.call("reset")
        h, w, _ = contract.IMAGE_SHAPE
        arrays = {"observation.state": rng.uniform(-50, 50, len(contract.STATE_NAMES)).astype(np.float32),
                  **{f"observation.images.{cam}": rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for cam in contract.CAMERAS}}
        t = time.time()
        header, action = client.call("act", arrays, task=contract.INSTRUCTION, robot_type="lekiwi_client", horizon=8)
        action = action["action"]
        assert action.shape == (8, len(contract.ACTION_NAMES)) and np.isfinite(action).all(), f"action が不正: {action.shape}"
        print(f"POLICY SERVER OK {info['policy_type']} checkpoint {info['checkpoint']}: action {list(action.shape)} "
              f"in {(time.time() - t) * 1000:.0f}ms")
        server_ok = True
    finally:
        server.terminate()
        server.wait(timeout=30)
        if not server_ok:
            print(open(f"{tmp}/policy_server.log").read()[-3000:])
print("TRAIN CHECK OK")
