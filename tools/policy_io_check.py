"""学習と推論で、Diffusion のモデルに入る値がそろっているかを確かめる (make policy-smoke)。train のイメージで、推論サーバを起動してから実行する

    python /opt/tools/policy_io_check.py --policy_path <pretrained_model> --dataset_root <LeRobot 形式のデータセット>
- データセットの同じ 2 フレームについて、(1) 学習と同じ読み出しの結果をモデルに直接入れた action と、(2) 元の形 (uint8 の画像) に戻して
  推論サーバに送った action を、同じシードで比べる。フレームの順、カメラの順、正規化のどれかが違えば一致しない
- 対照として、2 フレームの順を入れ替えて送ると結果が変わることも確かめる (比較が入力の違いを見分けられること)
- n_obs_steps が 1 のポリシーは、学習と推論で履歴の扱いが変わらないので対象外
"""
import argparse
import os
import sys

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES

from leisaac_ext import remote

TOLERANCE = 1e-3  # action の単位 (contract) での許容差
SEED = 123

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy_path", required=True)
parser.add_argument("--dataset_root", required=True)
parser.add_argument("--index", type=int, default=1000, help="データセットのフレームの番号")
args = parser.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.policy_path)
if cfg.n_obs_steps == 1:
    print(f"POLICY IO CHECK SKIP {cfg.type} (n_obs_steps=1)")
    sys.exit(0)
assert cfg.type == "diffusion", f"{cfg.type} の直接の呼び方はまだ書いていない"

repo_id = f"local/{os.path.basename(os.path.normpath(args.dataset_root))}"
meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
dataset = LeRobotDataset(repo_id, root=args.dataset_root, delta_timestamps=resolve_delta_timestamps(cfg, meta))
keys = ["observation.state", *cfg.image_features]
item = dataset[args.index]
print({k: list(item[k].shape) for k in keys})

# 学習と同じ読み出しの結果を、モデルに直接入れる
cfg.device = "cuda"
policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg).to("cuda").eval()
preprocessor, postprocessor = make_pre_post_processors(
    cfg, pretrained_path=args.policy_path, preprocessor_overrides={"device_processor": {"device": "cuda"}})
with torch.inference_mode():
    batch = preprocessor({k: item[k][None] for k in keys})
    batch = {**batch, OBS_IMAGES: torch.stack([batch[k] for k in cfg.image_features], dim=-4)}
    torch.manual_seed(SEED)
    direct = postprocessor(policy.diffusion.generate_actions(batch))[0].float().cpu().numpy()

# 同じフレームを、評価と同じ形で推論サーバに送る
arrays = {"observation.state": item["observation.state"].numpy().astype(np.float32)}
for k in cfg.image_features:
    arrays[k] = (item[k].permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
client = remote.PolicyClient("localhost", remote.PORT, 60000)


def served(frames: dict[str, np.ndarray]) -> np.ndarray:
    client.call("reset", seed=SEED)
    header, out = client.call("act", frames, task="", robot_type="lekiwi_client", horizon=cfg.n_action_steps)
    return out["action"]


gap = float(np.abs(served(arrays) - direct).max())
swapped = float(np.abs(served({k: v[::-1] for k, v in arrays.items()}) - direct).max())
ok = gap < TOLERANCE and swapped > gap
print(f"POLICY IO CHECK {'PASS' if ok else 'FAIL'} gap={gap:.6f} (許容 {TOLERANCE}) swapped_frames_gap={swapped:.6f}")
sys.exit(0 if ok else 1)
