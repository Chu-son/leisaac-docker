"""Sweep タスクの記録 (make datagen) か評価の状態の記録 (make sweep-eval) を action で再生し、記録どおりに再現できるかを確かめる (make sweep-replay)

    python /opt/tools/leisaac_run.py /opt/tools/sweep_replay.py --dataset_file /workspace/data/datasets/sweep.hdf5 --episodes 0 3 --device cpu --enable_cameras --headless
エピソードごとに、記録した初期状態に戻し (reset_to)、記録した sim の action を順に流す。流し終えたときの、
目標領域に入ったゴミの個数 (8 個以上で成功) と、ゴミの最終位置の記録とのずれを調べる。
ヘッドレスでなければ、実時間の速さで GUI に表示する。
見た目のランダム化 (ゴミの色、照明、カメラの位置) は状態に含まれないので、再生の見た目は記録と違う (物理には影響しない)。
--eval_dir (評価の out_dir) を渡すと、states/round_<番号>.npz (leisaac_ext.sweep_log) を再生する。ランダム化の値も
記録どおりに戻すので、--videos で、評価のときと同じ見た目の動画を作り直せる。ゴミの位置は、フレームごとに記録と比べる。
結果は out_dir の replay.jsonl に残す。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--dataset_file", default=None, help="datagen の HDF5")
parser.add_argument("--eval_dir", default=None, help="評価の out_dir (states/round_<番号>.npz を再生する)")
parser.add_argument("--episodes", type=int, nargs="*", default=[],
                    help="再生するエピソード (demo_<番号>) か、--eval_dir ではラウンドの番号 (省略時はすべて)")
parser.add_argument("--videos", action="store_true", help="--eval_dir: 再生の動画を out_dir/videos に残す")
parser.add_argument("--out_dir", default=None, help="結果の保存先 (省略時は /workspace/data/sweep/replay/<日時>)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import json
import os
import time

import glob

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
from leisaac_ext import contract, sweep_log, sweep_mdp, sweep_video

assert (args.dataset_file is None) != (args.eval_dir is None), "--dataset_file か --eval_dir のどちらか 1 つを渡す"
out_dir = args.out_dir or f"/workspace/data/sweep/replay/{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(out_dir, exist_ok=True)
if args.dataset_file:
    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset_file)
    episodes = args.episodes or list(range(handler.get_num_episodes()))
else:
    paths = sorted(glob.glob(f"{args.eval_dir}/states/round_*.npz"))
    assert paths, f"{args.eval_dir}/states/round_*.npz がない (状態の記録より前の評価)"
    episodes = args.episodes or [int(os.path.basename(p)[6:9]) for p in paths]

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
# 終了条件を外し、記録した action を最後まで流す (上流の replay と同じ)
cfg.recorders = None
cfg.terminations = {}
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
env_ids = torch.tensor([0], device=env.device)
origin = env.scene.env_origins[0]


def in_target(debris_pos: torch.Tensor, target_pos: torch.Tensor) -> int:
    """sweep_mdp.debris_in_target と同じ判定を、env の原点からの位置 (debris [N, 3]、target [3]) で行う"""
    inside = ((debris_pos[:, :2] - target_pos[:2]).abs() < sweep_mdp.TARGET_HALF).all(dim=-1)
    return int((inside & (debris_pos[:, 2] < sweep_mdp.IN_TARGET_MAX_Z)).sum())


RERENDERS = 3


def replay_round(r: int) -> dict:
    """評価の状態の記録 (states/round_<r>.npz) を再生する。フレームごとに、ゴミの位置を記録と比べる"""
    data = sweep_log.load_round(f"{args.eval_dir}/states/round_{r:03d}.npz")
    sweep_log.apply_setup(env, data)
    actions = torch.as_tensor(data["frame/action"], device=env.device)
    rec_debris = torch.as_tensor(data["frame/debris_pos"], device=env.device)
    target = torch.as_tensor(data["setup/init/rigid_object/target/root_pose"][:3], device=env.device)
    if args.videos:
        os.makedirs(f"{out_dir}/videos", exist_ok=True)
        video = sweep_video.VideoWriter(f"{out_dir}/videos/round_{r:03d}.mp4")
        start_xy, base_path = data["frame/debris_pos"][0, :, :2], []
    # 最初のフレームの画像は、記録の見た目に戻した後で描き直す (カメラは reset_to の観測で更新済みなので、強制する)
    # 数回描く。それでも最初のフレームには、RTX の描画が前の場面を引きずった像が残ることがある (2 フレーム目からは記録どおり)
    for _ in range(RERENDERS):
        env.sim.render()
    for cam in sweep_log.camera_names(env):
        env.scene[cam].update(0.0, force_recompute=True)
    obs = env.observation_manager.compute()
    error = 0.0
    for t, action in enumerate(actions):
        debris = torch.stack([env.scene[n].data.root_pos_w[0] for n in sweep_mdp.DEBRIS_NAMES]) - origin
        error = max(error, float((debris[:, :2] - rec_debris[t, :, :2]).norm(dim=-1).max()))
        if args.videos:
            robot = env.scene["robot"]
            base = robot.data.body_pos_w[0, robot.body_names.index("base"), :2] - origin[:2]
            base_path.append(base.cpu().numpy())
            theta = float(robot.data.joint_pos[0, robot.joint_names.index("base_theta")])
            inside = data["frame/inside"][t]
            label = f"#{r} replay t={data['frame/time'][t]:4.1f}s in={int(inside.sum())}"
            map_img = sweep_video.draw_map(debris[:, :2].cpu().numpy(), start_xy, inside, target[:2].cpu().numpy(),
                                           np.array(base_path), theta, label)
            video.write(*(obs["policy"][cam][0].cpu().numpy() for cam in contract.CAMERAS), map_img)
        for _ in range(contract.SIM_STEPS_PER_FRAME):
            obs, *_ = env.step(action[None])
    if args.videos:
        video.close()
    # 最後のフレームの action の後の状態は記録にないので、個数は最後のフレームの記録と、再生の同じ時点で比べる
    count = in_target(debris, target)
    rec_count = in_target(rec_debris[-1], target)
    return {"round": r, "recorded_success": bool(data["meta/success"]), "in_target_last_frame": count,
            "recorded_in_target_last_frame": rec_count, "max_debris_error": round(error, 4), "frames": len(actions)}


results = []
with open(f"{out_dir}/replay.jsonl", "w") as log:
    for i in episodes:
        if args.eval_dir:
            result = replay_round(i)
            results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(f"round {i}: in_target(last frame)={result['in_target_last_frame']} (recorded {result['recorded_in_target_last_frame']}) "
                  f"max_debris_error={result['max_debris_error'] * 100:.2f}cm frames={result['frames']}", flush=True)
            continue
        episode = handler.load_episode(f"demo_{i}", env.device)
        # DAgger の記録 (sweep_eval.py --dagger) の actions は計画器の正解なので、実際に動かした action で再生する
        actions = episode.data.get("executed_actions", episode.data["actions"])
        env.reset_to(episode.get_initial_state(), env_ids, is_relative=True)
        t0 = time.time()
        for k, action in enumerate(actions):
            env.step(action[None])
            if not args.headless:
                time.sleep(max(0.0, t0 + (k + 1) * env.step_dt - time.time()))
        # 記録の状態は env の原点からの値 (is_relative)。最後の要素が、最後の action の後の状態
        recorded = episode.data["states"]["rigid_object"]
        rec_debris = torch.stack([recorded[n]["root_pose"][-1, :3] for n in sweep_mdp.DEBRIS_NAMES])
        rec_target = recorded[sweep_mdp.TARGET_NAME]["root_pose"][-1, :3]
        debris = torch.stack([env.scene[n].data.root_pos_w[0] for n in sweep_mdp.DEBRIS_NAMES]) - origin
        target = env.scene[sweep_mdp.TARGET_NAME].data.root_pos_w[0] - origin
        count, rec_count = in_target(debris, target), in_target(rec_debris, rec_target)
        error = float((debris[:, :2] - rec_debris[:, :2]).norm(dim=-1).max())
        result = {"episode": i, "success": count >= sweep_mdp.SUCCESS_COUNT, "in_target": count, "recorded_in_target": rec_count,
                  "max_debris_error": round(error, 4), "steps": len(actions)}
        results.append(result)
        log.write(json.dumps(result) + "\n")
        log.flush()
        n_ok = sum(x["success"] for x in results)
        print(f"episode {i}: {'success' if result['success'] else 'failure'} in_target={count} (recorded {rec_count}) "
              f"max_debris_error={error * 100:.2f}cm steps={len(actions)} ({n_ok}/{len(results)})", flush=True)

max_error = max((x["max_debris_error"] for x in results), default=0.0)
if args.eval_dir:
    n_match = sum(x["in_target_last_frame"] == x["recorded_in_target_last_frame"] for x in results)
    print(f"SWEEP REPLAY {n_match}/{len(results)} rounds matched, max_debris_error={max_error * 100:.2f}cm -> {out_dir}", flush=True)
    env.close()
    app.close()
    raise SystemExit(0)
n, n_ok = len(results), sum(x["success"] for x in results)
print(f"SWEEP REPLAY {n_ok}/{n} succeeded, max_debris_error={max_error * 100:.2f}cm -> {out_dir}", flush=True)
env.close()
app.close()
