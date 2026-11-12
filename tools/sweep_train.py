"""Sweep タスクの LeRobot 形式のデータ (make convert) でポリシーを学習する (make train)。lerobot 0.4.2 の lerobot-train を呼ぶ薄いラッパー
Docker の外 (Colab。notebooks/train_colab.ipynb) でも、train の環境 (docker/envs/train) があれば同じように動く

    python /opt/tools/sweep_train.py --policy act --dataset_root /workspace/data/lerobot/sweep_floor --output_dir /workspace/data/train/sweep_floor-act
- ポリシーごとの既定 (PRESETS) に、共通の設定 (ローカルのデータセット、Hub に上げない、wandb を使わない) を足して lerobot-train を実行する
- --steps と --batch_size で既定を変えられる。-- の後ろは、そのまま lerobot-train に渡す (例: -- --policy.chunk_size=50)
- output_dir が既にあれば止める (lerobot-train も止める)。実行したコマンドと結果を output_dir の sweep_train.json に残す
- --resume_from に学習済みのチェックポイント (<学習>/checkpoints/<step>) を与えると、それを新しい output_dir に写し、
  その続きから --steps まで学習する (lerobot の --resume。optimizer と乱数の状態も戻す)。元の学習は変えない。
  ポリシーやデータセットや batch はチェックポイントの train_config.json のものを使う
- --resume は、途中で止まった学習 (Colab の切断など) を、同じ output_dir の最後のチェックポイントから、最初に決めた
  steps まで続ける。学習率の計画も最初の設定のままなので、止まらずに学習したのと同じになる
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# steps と batch_size は lerobot の例に合わせた出発点 (P6 で見直す。docs/sweep.md)。
# pretrained は事前学習済みのモデルから始めるもの。rename はデータセットのカメラ名を、そのモデルの名前に読み替える。
# args はこのポリシーにだけ渡す lerobot-train の引数。ポリシーを足すときは、ここに 1 項目足す (docs/sweep.md の「比較するポリシーを足す」)
PRESETS = {
    "act": {"type": "act", "steps": 100_000, "batch_size": 8},
    # 事前学習時のカメラは camera1〜3。使わない camera3 は飛ばされる (empty_cameras=0)
    "smolvla": {"pretrained": "lerobot/smolvla_base", "steps": 20_000, "batch_size": 64,
                "rename": {"observation.images.front": "observation.images.camera1",
                           "observation.images.wrist": "observation.images.camera2"}},
    # 基盤モデルは nvidia/GR00T-N1.5-3B。既定では LLM と視覚エンコーダを凍結し、projector と action head を学習する
    "groot": {"type": "groot", "steps": 20_000, "batch_size": 32},
    # 画像は [240, 320]。既定の crop (84, 84) では大半が捨てられるので、9 割を残す。eval は中央で切る
    "diffusion": {"type": "diffusion", "steps": 100_000, "batch_size": 32, "args": ["--policy.crop_shape=[216,288]"]},
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
parser.add_argument("--seed", type=int, default=None, help="学習の乱数のシード (省略時は lerobot の既定の 1000)")
parser.add_argument("--resume_from", default=None, help="続きから学習するチェックポイント (<学習>/checkpoints/<step>)。新しい output_dir に写してから再開する")
parser.add_argument("--resume", action="store_true", help="途中で止まった学習を、同じ output_dir の最後のチェックポイントから続ける")
parser.add_argument("extra", nargs="*", help="-- の後ろに書いた引数は、そのまま lerobot-train に渡す")
args = parser.parse_args()
assert os.path.isfile(f"{args.dataset_root}/meta/info.json"), f"{args.dataset_root} は LeRobot 形式のデータセットではない"
if args.resume:
    assert not args.resume_from, "--resume と --resume_from は一緒に使えない"
    assert os.path.isfile(f"{args.output_dir}/checkpoints/last/pretrained_model/train_config.json"), \
        f"{args.output_dir}/checkpoints/last がない (再開できるチェックポイントがない)"
else:
    assert not os.path.exists(args.output_dir), f"{args.output_dir} は既にあります (上書きしない。続きは --resume か --resume_from)"

preset = PRESETS[args.policy]
steps = args.steps or preset["steps"]
previous = None
if args.resume:
    with open(f"{args.output_dir}/checkpoints/last/pretrained_model/train_config.json") as f:
        saved = json.load(f)
    assert args.steps in (None, saved["steps"]), "--resume では steps を変えない (延ばすなら --resume_from で新しい output_dir に)"
    if os.path.isfile(f"{args.output_dir}/sweep_train.json"):
        with open(f"{args.output_dir}/sweep_train.json") as f:
            previous = json.load(f)
    steps = saved["steps"]
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--resume=true",
        f"--config_path={args.output_dir}/checkpoints/last/pretrained_model/train_config.json",
        *args.extra,
    ]
elif args.resume_from:
    src = os.path.normpath(args.resume_from)
    config = f"{src}/pretrained_model/train_config.json"
    assert os.path.isfile(config), f"{config} がありません"
    with open(config) as f:
        saved = json.load(f)
    assert saved["policy"]["type"] == preset.get("type", saved["policy"]["type"]), "--policy がチェックポイントと違います"
    assert os.path.normpath(saved["dataset"]["root"]) == os.path.normpath(args.dataset_root), "--dataset_root がチェックポイントと違います"
    assert args.batch_size is None, "再開では batch を変えない (チェックポイントのものを使う)"
    with open(f"{src}/training_state/training_step.json") as f:
        done = json.load(f)["step"]
    assert args.steps and args.steps > done, f"--steps はチェックポイントの step ({done}) より大きくする"
    ckpt = f"{args.output_dir}/checkpoints/{os.path.basename(src)}"
    shutil.copytree(src, ckpt, symlinks=False)
    os.symlink(os.path.basename(src), f"{args.output_dir}/checkpoints/last")
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--resume=true",
        f"--config_path={ckpt}/pretrained_model/train_config.json",
        f"--output_dir={args.output_dir}",
        f"--job_name={os.path.basename(os.path.normpath(args.output_dir))}",
        f"--steps={steps}",
        f"--save_freq={args.save_freq or max(1, (steps - done) // 2)}",
        *args.extra,
    ]
else:
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        *policy_args(preset, args.dataset_root),
        *preset.get("args", []),
        f"--dataset.repo_id=local/{os.path.basename(os.path.normpath(args.dataset_root))}",
        f"--dataset.root={args.dataset_root}",
        f"--output_dir={args.output_dir}",
        f"--job_name={os.path.basename(os.path.normpath(args.output_dir))}",
        f"--steps={steps}",
        f"--batch_size={args.batch_size or preset['batch_size']}",
        f"--save_freq={args.save_freq or max(1, steps // 5)}",
        f"--log_freq={min(200, max(1, steps // 10))}",
        f"--num_workers={args.num_workers}",
        *([f"--seed={args.seed}"] if args.seed is not None else []),
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        *args.extra,
    ]
print("+ " + " ".join(cmd[1:]), flush=True)
t0 = time.time()
returncode = subprocess.run(cmd).returncode
summary = {"policy": args.policy, "dataset_root": args.dataset_root, "resume_from": args.resume_from, "resume": args.resume, "previous": previous, "command": cmd[1:], "returncode": returncode,
           "elapsed_s": round(time.time() - t0)}
if os.path.isdir(args.output_dir):
    with open(f"{args.output_dir}/sweep_train.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"SWEEP TRAIN {'OK' if returncode == 0 else 'FAIL'} {args.policy} {steps} steps in {summary['elapsed_s']}s -> {args.output_dir}")
sys.exit(returncode)
