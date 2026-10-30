"""lerobot で学習したポリシーの推論サーバ (make policy-up)。train のイメージで実行する

    python /opt/tools/policy_server.py --policy_path /workspace/data/train/sweep_floor-act/checkpoints/last/pretrained_model
- チェックポイント (sweep_train.py の出力の pretrained_model) から、ポリシーと前処理・後処理 (正規化、カメラ名の読み替え、
  言語のトークン化) を読む。前処理と後処理は学習のときに保存されたもので、学習と同じ変換になる
- 通信は leisaac_ext.remote (ZMQ の REQ/REP、ポートは remote.PORT)。観測と action は contract の LeRobot 形式で、
  act の応答は predict_action_chunk の先頭 horizon 個 (後処理で contract の単位に戻したもの)
- --ping は、動いているサーバに ping を送り、応答があれば終了コード 0 で終わる (compose の healthcheck)
"""
import argparse
import os
import sys
import time

import numpy as np

from leisaac_ext import remote

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy_path", help="チェックポイントの pretrained_model")
parser.add_argument("--port", type=int, default=remote.PORT)
parser.add_argument("--device", default="cuda")
parser.add_argument("--ping", action="store_true", help="動いているサーバに ping を送るだけ")
args = parser.parse_args()

if args.ping:
    try:
        print(remote.PolicyClient(port=args.port, timeout_ms=5000).call("ping")[0])
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ping に失敗: {e}")
    sys.exit(0)

import torch
import zmq
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference

assert args.policy_path and os.path.isfile(f"{args.policy_path}/config.json"), f"{args.policy_path} はチェックポイントではない"
t0 = time.time()
cfg = PreTrainedConfig.from_pretrained(args.policy_path)
# GR00T の前処理と後処理は、読み込むときにデータセットの統計を渡す必要がある (lerobot 0.4.2 の factory)。P6f で扱う
assert cfg.type != "groot", "GR00T はまだ扱えない (docs/sweep.md の P6f)"
cfg.device = args.device
device = torch.device(args.device)
policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg)
policy.to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    cfg, pretrained_path=args.policy_path, preprocessor_overrides={"device_processor": {"device": args.device}}
)
state_shape = list(cfg.input_features["observation.state"].shape)
info = {
    "policy_type": cfg.type,
    "policy_path": args.policy_path,
    # checkpoints/last は最後のチェックポイント (checkpoints/<step>) へのリンク
    "checkpoint": os.path.basename(os.path.realpath(os.path.dirname(args.policy_path))),
    "chunk_size": getattr(cfg, "chunk_size", None),
    "n_action_steps": getattr(cfg, "n_action_steps", None),
    "load_s": round(time.time() - t0, 1),
    "vram_mib": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
}


def act(header: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    assert list(arrays["observation.state"].shape) == state_shape, f"state の形が {state_shape} ではない"
    t = time.time()
    # prepare_observation_for_inference は配列を書き換えるので、複製を渡す (frombuffer の配列は書き込めない)
    obs = {k: v.copy() for k, v in arrays.items()}
    with torch.inference_mode():
        batch = preprocessor(prepare_observation_for_inference(obs, device, header.get("task"), header.get("robot_type")))
        chunk = policy.predict_action_chunk(batch)[:, : header["horizon"]]
        action = postprocessor(chunk)
    action = action[0].float().cpu().numpy()
    return {"infer_ms": round((time.time() - t) * 1000, 1)}, {"action": action}


socket = zmq.Context.instance().socket(zmq.REP)
socket.bind(f"tcp://*:{args.port}")
print(f"[policy_server] {info} listening on :{args.port}", flush=True)
while True:
    frames = socket.recv_multipart()
    try:
        header, arrays = remote.decode(frames)
        if header["cmd"] == "ping":
            reply = ({**info, "vram_mib": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}, {})
        elif header["cmd"] == "reset":
            policy.reset()
            reply = ({}, {})
        elif header["cmd"] == "act":
            reply = act(header, arrays)
        else:
            raise ValueError(f"未知の cmd: {header['cmd']}")
    except Exception as e:  # noqa: BLE001  REQ/REP は必ず応答しないと詰まるので、エラーも応答で返す
        reply = ({"error": f"{type(e).__name__}: {e}"}, {})
    socket.send_multipart(remote.encode(*reply))
