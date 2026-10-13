"""GR00T 推論サーバの疎通確認。

    python /opt/tools/probe_gr00t.py --ping      # 応答があれば exit 0 (healthcheck 用)
    python /opt/tools/probe_gr00t.py             # ダミー観測で get_action し、action の形状を表示

leisaac の Gr00tServicePolicyClient は action.gripper を (horizon, 1) として結合するため、
ここで (16, 1) 以外が返る場合はサーバ (Isaac-GR00T) とクライアント (leisaac) の組み合わせが合っていない。
"""
import argparse
import sys

import numpy as np
from gr00t.eval.service import ExternalRobotInferenceClient

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--ping", action="store_true")
args = parser.parse_args()

client = ExternalRobotInferenceClient(host=args.host, port=args.port)
if args.ping:
    sys.exit(0 if client.ping() else 1)

obs = {
    "video.front": np.zeros((1, 480, 640, 3), np.uint8),
    "video.wrist": np.zeros((1, 480, 640, 3), np.uint8),
    "state.single_arm": np.zeros((1, 5)),
    "state.gripper": np.zeros((1, 1)),
    "annotation.human.task_description": ["Pick up the orange and place it on the plate"],
}
action = client.get_action(obs)
shapes = {k: getattr(v, "shape", None) for k, v in action.items()}
print(shapes)
sys.exit(0 if shapes.get("action.gripper") == (16, 1) else 1)
