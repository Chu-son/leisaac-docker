"""Sweep タスクの LeRobot 形式のデータ (make convert) でポリシーを学習する (make train)。lerobot 0.4.2 の lerobot-train を呼ぶ薄いラッパー
Docker の外 (Colab。notebooks/train_colab.ipynb) でも、train の環境 (docker/envs/train) があれば同じように動く

    python /opt/tools/sweep_train.py --policy act --dataset_root /workspace/data/lerobot/sweep_floor --output_dir /workspace/data/train/sweep_floor-act
- ポリシーごとの既定 (PRESETS) に、共通の設定 (ローカルのデータセット、Hub に上げない、wandb を使わない) を足して lerobot-train を実行する
- --steps と --batch_size で既定を変えられる。-- の後ろは、そのまま lerobot-train に渡す (例: -- --policy.chunk_size=50)
- output_dir が既にあれば止める (lerobot-train も止める)。実行したコマンドと結果を output_dir の sweep_train.json に残す
"""
import argparse
import json
import os
import subprocess
import sys
import time

# steps と batch_size は lerobot の例に合わせた出発点 (P6 で見直す。docs/sweep.md)。
# pretrained は事前学習済みのモデルから始めるもの。rename はデータセットのカメラ名を、そのモデルの名前に読み替える
PRESETS = {
    "act": {"type": "act", "steps": 100_000, "batch_size": 8},
    # 事前学習時のカメラは camera1〜3。使わない camera3 は飛ばされる (empty_cameras=0)
    "smolvla": {"pretrained": "lerobot/smolvla_base", "steps": 20_000, "batch_size": 64,
                "rename": {"observation.images.front": "observation.images.camera1",
                           "observation.images.wrist": "observation.images.camera2"}},
    # 基盤モデルは nvidia/GR00T-N1.5-3B。既定では LLM と視覚エンコーダを凍結し、projector と action head を学習する
    "groot": {"type": "groot", "steps": 20_000, "batch_size": 32},
}


def policy_args(preset: dict, dataset_root: str) -> list[str]:
    """ポリシーを選ぶ引数。事前学習済みのモデルは入出力の名前と形を自分の設定に持っていて (smolvla_base は state と
    action が 6 次元)、lerobot はそれをデータセットに合わせない。action が 6 次元のままだと、推論で台車の速度が切り捨てられる。
    そこで、データセットの形 (meta/info.json) で上書きする。dict の上書きは足し込みなので、使わない名前は設定に残る"""
    if "type" in preset:
        return [f"--policy.type={preset['type']}"]
    with open(f"{dataset_root}/meta/info.json") as f:
        features = json.load(f)["features"]
    rename = preset.get("rename", {})
    inputs = {"observation.state": {"type": "STATE", "shape": features["observation.state"]["shape"]}}
    for key, ft in features.items():
        if key.startswith("observation.images."):
            h, w, c = ft["shape"]
            inputs[rename.get(key, key)] = {"type": "VISUAL", "shape": [c, h, w]}
    outputs = {"action": {"type": "ACTION", "shape": features["action"]["shape"]}}
    return [f"--policy.path={preset['pretrained']}", f"--rename_map={json.dumps(rename)}",
            f"--policy.input_features={json.dumps(inputs)}", f"--policy.output_features={json.dumps(outputs)}"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, choices=sorted(PRESETS))
parser.add_argument("--dataset_root", required=True, help="LeRobot 形式のデータセット (make convert の出力)")
parser.add_argument("--output_dir", required=True, help="学習の出力先 (既にあれば止める)")
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--save_freq", type=int, default=None, help="チェックポイントを残す間隔 (省略時は steps の 1/5)")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("extra", nargs="*", help="-- の後ろに書いた引数は、そのまま lerobot-train に渡す")
args = parser.parse_args()
assert os.path.isfile(f"{args.dataset_root}/meta/info.json"), f"{args.dataset_root} は LeRobot 形式のデータセットではない"
assert not os.path.exists(args.output_dir), f"{args.output_dir} は既にあります (上書きも再開もしない)"

preset = PRESETS[args.policy]
steps = args.steps or preset["steps"]
cmd = [
    sys.executable, "-m", "lerobot.scripts.lerobot_train",
    *policy_args(preset, args.dataset_root),
    f"--dataset.repo_id=local/{os.path.basename(os.path.normpath(args.dataset_root))}",
    f"--dataset.root={args.dataset_root}",
    f"--output_dir={args.output_dir}",
    f"--job_name={os.path.basename(os.path.normpath(args.output_dir))}",
    f"--steps={steps}",
    f"--batch_size={args.batch_size or preset['batch_size']}",
    f"--save_freq={args.save_freq or max(1, steps // 5)}",
    f"--log_freq={min(200, max(1, steps // 10))}",
    f"--num_workers={args.num_workers}",
    "--policy.device=cuda",
    "--policy.push_to_hub=false",
    "--wandb.enable=false",
    *args.extra,
]
print("+ " + " ".join(cmd[1:]), flush=True)
t0 = time.time()
returncode = subprocess.run(cmd).returncode
summary = {"policy": args.policy, "dataset_root": args.dataset_root, "command": cmd[1:], "returncode": returncode,
           "elapsed_s": round(time.time() - t0)}
if os.path.isdir(args.output_dir):
    with open(f"{args.output_dir}/sweep_train.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"SWEEP TRAIN {'OK' if returncode == 0 else 'FAIL'} {args.policy} {steps} steps in {summary['elapsed_s']}s -> {args.output_dir}")
sys.exit(returncode)
