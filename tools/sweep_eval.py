"""Sweep タスクの評価 (make sweep-eval) と自動生成 (make datagen)。ポリシーに N ラウンド解かせて、成功率を測る

    python /opt/tools/leisaac_run.py /opt/tools/sweep_eval.py --policy planner --rounds 30 --device cpu --enable_cameras --headless
    python /opt/tools/leisaac_run.py /opt/tools/sweep_eval.py --record --dataset_file /workspace/data/datasets/sweep.hdf5 --num_demos 50 ...
ポリシーは reset() と act(obs) -> [T, 9] (contract の LeRobot 形式の action) を持つ。T 個の action を、
それぞれ contract.SIM_STEPS_PER_FRAME step ずつ続ける。学習したポリシーも計画器も、同じ経路で評価する。
    planner: 計画器 (leisaac_ext.sweep_planner)。sim の真値を読む
    remote:  推論サーバ (tools/policy_server.py。make policy-up) につなぐ。観測 (カメラと関節の状態) だけを送り、
             action の系列の先頭 action_horizon 個を受け取る (leisaac_ext.remote)
成功は終了条件 success (10 個中 8 個以上が目標領域に入る)。時間切れ (タスクの既定は 40 秒。--time_limit で変える) は失敗。
結果は out_dir の rounds.jsonl (ラウンドごと) と summary.json に残す。view_fraction は、押し終わっていないゴミが
front カメラの水平の画角に入っていたフレームの割合 (ゴミを見失わずに動けているかの目安)。
状態の記録 (leisaac_ext.sweep_log) も常に残す: run.json (構成)、env_cfg.yaml、states/round_<番号>.npz (ラウンドの初めの
状態とランダム化の値、30fps のフレームごとの状態と action)。sweep_replay.py --eval_dir で再生でき、動画も作り直せる。
--videos を付けると、ラウンドごとの動画 (videos/round_<番号>_<success|failure>.mp4。front、wrist、上から見た図を
並べた 30fps の実時間) と、全ラウンドの終わりの図を並べた overview.png も残す (leisaac_ext.sweep_video)。
--record を付けると、成功したエピソードだけを HDF5 に記録する (isaaclab の RecorderManager。上流の teleop と同じ形式)。
成功が num_demos 個になるか、ラウンドが rounds (既定は num_demos の 2 倍 + 10) に達したら終わる。
--dagger (--policy remote と --record と一緒に使う) は DAgger のデータを作る。学習したポリシーの action で動かし
(chunk ごとに確率 dagger_beta で計画器の action で動かす)、各フレームで計画器が出した action を正解として
actions に記録する。実際に動かした action は executed_actions に残す。成功も失敗も、すべてのラウンドを記録する。
計画器は台車の速度の指令を自分の前の指令から加速度の上限をかけて作るので、--dagger では毎フレーム、
実際の台車の速度に合わせ直す (ポリシーが動かした後の状態から、どう立て直すかを正解にする)。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LeIsaac-LeKiwi-Sweep-Floor-v0")
parser.add_argument("--policy", default="planner", choices=["planner", "remote"])
parser.add_argument("--policy_host", default="localhost", help="推論サーバ (--policy remote)")
parser.add_argument("--policy_port", type=int, default=5556, help="推論サーバのポート (leisaac_ext.remote.PORT)")
parser.add_argument("--policy_timeout_ms", type=int, default=15000)
parser.add_argument("--action_horizon", type=int, default=16, help="1 回の推論で実行する action の数 (--policy remote)")
parser.add_argument("--rounds", type=int, default=None, help="ラウンド数 (既定は 10。--record では上限で、既定は num_demos の 2 倍 + 10)")
parser.add_argument("--seed", type=int, default=None, help="乱数のシード (省略時は現在時刻)")
parser.add_argument("--randomize", default=None, help="追加のランダム化の項目 (カンマ区切り。all ですべて。省略時はタスクの既定)")
parser.add_argument("--out_dir", default=None,
                    help="結果の保存先 (省略時は /workspace/data/sweep/eval/<日時>。--record では <dataset_file から .hdf5 を除いたもの>.datagen)")
parser.add_argument("--record", action="store_true", help="成功したエピソードを dataset_file に記録する (自動生成)")
parser.add_argument("--dataset_file", default=None, help="記録先の HDF5 (--record。既にあれば止める)")
parser.add_argument("--num_demos", type=int, default=50, help="記録する成功エピソードの数 (--record)")
parser.add_argument("--dagger", action="store_true", help="DAgger のデータを作る (--policy remote と --record と一緒に使う)")
parser.add_argument("--dagger_beta", type=float, default=0.0, help="--dagger で、chunk ごとに計画器の action で動かす確率")
parser.add_argument("--time_limit", type=float, default=None, help="1 ラウンドの制限時間 (秒。省略時はタスクの既定の 40 秒)")
parser.add_argument("--videos", action="store_true", help="ラウンドごとの動画と overview.png を out_dir に残す")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(vars(args)).app

import json
import math
import os
import time

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs.mdp.recorders.recorders_cfg import PreStepActionsRecorderCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401
from leisaac_ext import contract, remote, sweep_floor, sweep_log, sweep_mdp, sweep_planner, sweep_video


class PlannerPolicy:
    """計画器を、ポリシーのインタフェースで包む。観測の代わりに sim の真値を読む"""

    def __init__(self, env):
        self.env = env
        self.poses = sweep_planner.calibrate_arm(env)
        self.planner = sweep_planner.SweepPlanner(self.poses)

    def reset(self) -> None:
        self.planner.reset()

    def act(self, obs) -> np.ndarray:
        action = self.planner.act(sweep_planner.read_state(self.env))
        joint_pos = obs["joint_pos"][0].cpu().numpy().astype(np.float64)
        return contract.to_lerobot(joint_pos, np.zeros_like(joint_pos), action)[1][None]

    def info(self) -> dict:
        return {"passes": self.planner.passes, "phase": self.planner.phase}


class RemotePolicy:
    """推論サーバのポリシー。学習データと同じ形 (contract.to_lerobot の state と、[240, 320, 3] uint8 の画像) で観測を送る"""

    def __init__(self, host: str, port: int, timeout_ms: int, horizon: int):
        self.client = remote.PolicyClient(host, port, timeout_ms)
        self.server = self.client.call("ping")[0]
        self.horizon = horizon

    def reset(self) -> None:
        self.client.call("reset")
        self.infer_ms = []

    def act(self, obs) -> np.ndarray:
        joint_pos = obs["joint_pos"][0].cpu().numpy().astype(np.float64)
        joint_vel = obs["joint_vel"][0].cpu().numpy().astype(np.float64)
        state, _ = contract.to_lerobot(joint_pos, joint_vel, np.zeros(len(contract.ACTION_NAMES)))
        arrays = {"observation.state": state}
        for cam in contract.CAMERAS:
            image = obs[cam][0].cpu().numpy()
            assert image.shape == contract.IMAGE_SHAPE and image.dtype == np.uint8, f"{cam}: {image.shape} {image.dtype}"
            arrays[f"observation.images.{cam}"] = image
        header, out = self.client.call("act", arrays, task=contract.INSTRUCTION, robot_type="lekiwi_client",
                                       horizon=self.horizon)
        self.infer_ms.append(header["infer_ms"])
        return out["action"]

    def info(self) -> dict:
        return {"chunks": len(self.infer_ms), "infer_ms": round(float(np.mean(self.infer_ms)), 1) if self.infer_ms else None}


class ExpertActionsRecorder(RecorderTerm):
    """--dagger: actions に、実際に動かした action ではなく、計画器の action (env.expert_action) を記録する"""

    def record_pre_step(self):
        return "actions", self._env.expert_action


class ExecutedActionsRecorder(RecorderTerm):
    """--dagger: 実際に動かした action を executed_actions に記録する"""

    def record_pre_step(self):
        return "executed_actions", self._env.action_manager.action


_BASE_IDX = [contract.SIM_JOINT_NAMES.index(j) for j in ("base_x", "base_y", "base_theta")]


def pending_in_view(state: sweep_planner.SweepState) -> bool | None:
    """押し終わっていない (目標領域の外の) ゴミのどれかが、front カメラの水平の画角に入っているか (なければ None)。
    上下の画角と、穂に隠れるかどうかは見ない"""
    pending = ~sweep_planner.in_target(state.debris, state.target)
    if not pending.any():
        return None
    cam_x, cam_y = sweep_floor.FRONT_CAMERA_XY
    camera = state.base + cam_x * sweep_planner.right_of(state.theta) + cam_y * sweep_planner.forward_of(state.theta)
    bearing = [sweep_planner.wrap(sweep_planner.heading_of(d - camera) - state.theta) for d in state.debris[pending]]
    return min(abs(b) for b in bearing) < sweep_floor.FRONT_HALF_FOV
seed = args.seed if args.seed is not None else int(time.time())
if args.dagger:
    assert args.policy == "remote" and args.record, "--dagger は --policy remote と --record と一緒に使う"
if args.record:
    assert args.dataset_file and args.dataset_file.endswith(".hdf5"), "--record には --dataset_file (.hdf5) が要る"
    # isaaclab の HDF5 の書き出しは、既存のファイルを上書きする
    assert not os.path.exists(args.dataset_file), f"{args.dataset_file} は既にある"
    dataset_stem = os.path.splitext(args.dataset_file)[0]
    out_dir = args.out_dir or f"{dataset_stem}.datagen"
    max_rounds = args.rounds if args.rounds is not None else (10 if args.dagger else 2 * args.num_demos + 10)
else:
    out_dir = args.out_dir or f"/workspace/data/sweep/eval/{time.strftime('%Y%m%d-%H%M%S')}"
    max_rounds = args.rounds if args.rounds is not None else 10
os.makedirs(out_dir, exist_ok=True)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
cfg.use_teleop_device("lekiwi-leader")
if args.record:
    # タスクの既定の記録項目 (上流の ActionStateRecorderManagerCfg: 初期状態、各 step の action、観測、状態)。
    # 終了条件 success が成り立って終わったエピソードだけを書き出す
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    cfg.recorders.dataset_export_dir_path = os.path.dirname(args.dataset_file)
    cfg.recorders.dataset_filename = os.path.basename(dataset_stem)
    if args.dagger:
        cfg.recorders.record_pre_step_actions = PreStepActionsRecorderCfg(class_type=ExpertActionsRecorder)
        cfg.recorders.record_pre_step_executed_actions = PreStepActionsRecorderCfg(class_type=ExecutedActionsRecorder)
else:
    cfg.recorders = None
cfg.seed = seed
if args.time_limit is not None:
    cfg.episode_length_s = args.time_limit
if args.randomize is not None:
    names = sweep_floor.RANDOMIZATIONS if args.randomize == "all" else [n for n in args.randomize.split(",") if n]
    cfg.set_randomization(names)
env = gym.make(args.task, cfg=cfg).unwrapped
# 計画器の IK (calibrate_arm) は関節の状態を書き換える。その後の reset で元に戻る
env.reset()
if args.policy == "planner":
    policy = PlannerPolicy(env)
    print(f"policy=planner seed={seed} brush tip (base frame)={policy.poses.tip.round(3).tolist()}", flush=True)
else:
    policy = RemotePolicy(args.policy_host, args.policy_port, args.policy_timeout_ms, args.action_horizon)
    print(f"policy=remote seed={seed} action_horizon={args.action_horizon} server={policy.server}", flush=True)
expert = PlannerPolicy(env) if args.dagger else None
policy_info = {"type": args.policy}
if args.policy == "remote":
    policy_info.update(server=policy.server, action_horizon=args.action_horizon)
for name, p in (("planner", policy if args.policy == "planner" else None), ("expert", expert)):
    if p is not None:
        policy_info[name] = {"arm_heights": list(p.poses.heights), "arm_q": p.poses.q.round(6).tolist(),
                             "brush_tip": p.poses.tip.round(6).tolist(), "arm_travel": p.poses.travel.round(6).tolist(),
                             "params": vars(p.planner.params), "phases": list(sweep_planner.PHASES)}
sweep_log.write_run(out_dir, env, {"task": args.task, "seed": seed, "args": vars(args), "policy": policy_info,
                                   "contract": {"fps": contract.FPS, "sim_steps_per_frame": contract.SIM_STEPS_PER_FRAME,
                                                "sim_joint_names": list(contract.SIM_JOINT_NAMES),
                                                # sim の action (frame/action) の並び (lekiwi-leader の構成)
                                                "sim_action": [*contract.ARM_JOINTS, "base_vx_world", "base_vy_world", "base_vtheta"],
                                                "action_names": list(contract.ACTION_NAMES), "cameras": list(contract.CAMERAS),
                                                "instruction": contract.INSTRUCTION},
                                   "task_constants": {"target_half": sweep_mdp.TARGET_HALF, "success_count": sweep_mdp.SUCCESS_COUNT,
                                                      "in_target_max_z": sweep_mdp.IN_TARGET_MAX_Z}})
round_log = sweep_log.RoundLog(env)
rng = np.random.default_rng(seed)
# 2 ラウンド目からは、前のラウンドの終わりの自動リセットが返した観測から始める。env.reset() を重ねると、
# 記録中のエピソード (初期状態だけ) が、前のラウンドの success の値 (リセットでは消えない) で書き出される
obs, _ = env.reset()
if args.dagger:
    # 失敗したラウンドも書き出す。ここまでの reset の、初期状態だけのエピソードは書き出さないように、
    # 成功だけを書き出す設定で env を作り、ここで切り替える (書き出すかは、書き出すときの設定で決まる)
    env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    # 書き出さなかったエピソードも、書き出した数に数えられるので、ここまでの数を差し引く
    exported_before = env.recorder_manager.exported_successful_episode_count + env.recorder_manager.exported_failed_episode_count

results = []
start = time.time()
n_ok = 0
maps = []
if args.videos:
    os.makedirs(f"{out_dir}/videos", exist_ok=True)
with open(f"{out_dir}/rounds.jsonl", "w") as log:
    for r in range(max_rounds):
        if args.record and not args.dagger and n_ok >= args.num_demos:
            break
        policy.reset()
        if expert:
            expert.reset()
        t0, steps, success, done = time.time(), 0, False, False
        count = int(sweep_mdp.debris_in_target(env)[0].sum())
        # 台車の中心から最も近いゴミまでの距離の、ラウンド中の最小値。台車の外形の半径
        # (sweep_planner.BASE_RADIUS) より小さければ、ゴミに当たったか、台車の下にくぐらせている
        base_min_dist = math.inf
        # 押し終わっていないゴミが front カメラの画角に入っていたフレームの数と、押し終わっていないゴミがあったフレームの数
        view_frames, pending_frames = 0, 0
        round_log.begin()
        if args.videos:
            video = sweep_video.VideoWriter(f"{out_dir}/videos/round_{r:03d}.mp4")
            debris_start, base_path = sweep_planner.read_state(env).debris, []
        while not done:
            chunk = policy.act(obs["policy"])
            use_expert = expert is not None and rng.random() < args.dagger_beta
            for k, a in enumerate(chunk):
                joint_pos = obs["policy"]["joint_pos"][0].cpu().numpy().astype(np.float64)
                sim_action = torch.tensor(contract.from_lerobot_action(a, joint_pos), device=env.device)[None]
                if expert:
                    # 計画器の速度の指令を、実際の台車の速度 (ワールド座標系) に合わせてから、正解の action を求める
                    expert.planner.vel = obs["policy"]["joint_vel"][0].cpu().numpy().astype(np.float64)[_BASE_IDX]
                    expert_action = torch.tensor(expert.planner.act(sweep_planner.read_state(env)), dtype=torch.float32,
                                                 device=env.device)[None]
                    env.expert_action = expert_action
                    if use_expert:
                        sim_action = expert_action
                for sub in range(contract.SIM_STEPS_PER_FRAME):
                    # step の後は自動でリセットされるので、終わる前の個数を残しておく
                    inside = sweep_mdp.debris_in_target(env)[0]
                    count = int(inside.sum())
                    state = sweep_planner.read_state(env)
                    base_min_dist = min(base_min_dist, float(np.linalg.norm(state.debris - state.base, axis=-1).min()))
                    if steps % contract.SIM_STEPS_PER_FRAME == 0:
                        in_view = pending_in_view(state)
                        if in_view is not None:
                            view_frames += in_view
                            pending_frames += 1
                        extra = {"expert_action": env.expert_action[0]} if expert else {}
                        if args.policy == "planner":  # 計画器の段階 (sweep_planner.PHASES の番号)
                            extra["planner_phase"] = np.int8(sweep_planner.PHASES.index(policy.planner.phase))
                        # in_view: 押し終わっていないゴミが front の画角に入っているか (1/0。押し終わっていなければ -1)
                        round_log.add(steps, sim_action[0], inside=inside, in_view=np.int8(-1 if in_view is None else in_view),
                                      chunk_start=k == 0 and sub == 0, **extra)
                    if args.videos and steps % contract.SIM_STEPS_PER_FRAME == 0:
                        # 観測 (カメラ) は、この step を実行する前の状態のもの。図も同じ時点の真値で描く
                        base_path.append(state.base)
                        last = (state.debris, debris_start, inside.cpu().numpy(), state.target, np.array(base_path), state.theta)
                        map_img = sweep_video.draw_map(*last, f"#{r} t={steps * env.step_dt:4.1f}s in={count}")
                        video.write(*(obs["policy"][cam][0].cpu().numpy() for cam in contract.CAMERAS), map_img)
                    obs, _, terminated, truncated, _ = env.step(sim_action)
                    steps += 1
                    if terminated[0] or truncated[0]:
                        success, done = bool(env.termination_manager.get_term("success")[0]), True
                        break
                if done:
                    break
        if success:
            count = max(count, sweep_mdp.SUCCESS_COUNT)
        if args.videos:
            video.close()
            os.rename(video.path, f"{out_dir}/videos/round_{r:03d}_{'success' if success else 'failure'}.mp4")
            # 最後に描いたフレームの状態に、ラウンドの結果を書き添える (自動リセットの後は、終わりの状態を読めない)
            maps.append(sweep_video.draw_map(*last, f"#{r} {'S' if success else 'F'} {steps * env.step_dt:.1f}s in={count}"))
        round_log.save(f"{out_dir}/states/round_{r:03d}.npz",
                       {"round": r, "success": success, "in_target": count, "steps": steps, "seed": seed})
        result = {"round": r, "success": success, "in_target": count, "sim_time": round(steps * env.step_dt, 2),
                  "base_min_dist": round(base_min_dist, 3), "view_fraction": round(view_frames / max(pending_frames, 1), 3),
                  "states": f"states/round_{r:03d}.npz", "wall_time": round(time.time() - t0, 1), **policy.info()}
        n_ok += success
        if args.dagger:
            # すべてのラウンドを記録する (HDF5 の demo_<ラウンドの番号>)
            exported = (env.recorder_manager.exported_successful_episode_count
                        + env.recorder_manager.exported_failed_episode_count - exported_before)
            assert exported == r + 1, f"記録したエピソードの数 ({exported}) がラウンドの数 ({r + 1}) と合わない"
            result["episode"] = r
        elif args.record:
            # 記録したエピソードの番号 (HDF5 の demo_<番号>)。失敗したラウンドは記録しない
            exported = env.recorder_manager.exported_successful_episode_count
            assert exported == n_ok, f"記録したエピソードの数 ({exported}) が成功の数 ({n_ok}) と合わない"
            result["episode"] = n_ok - 1 if success else None
        results.append(result)
        log.write(json.dumps(result) + "\n")
        log.flush()
        print(f"round {r}: {'success' if success else 'failure'} in_target={count} sim_time={result['sim_time']}s "
              f"base_min_dist={result['base_min_dist']:.3f}m view={result['view_fraction']:.2f} " + " ".join(f"{k}={v}" for k, v in policy.info().items()) + f" ({n_ok}/{len(results)})", flush=True)

n, n_ok = len(results), sum(x["success"] for x in results)
# 成功率の 95% 信頼区間 (Wilson)
p, z = n_ok / max(n, 1), 1.96
center = (p + z**2 / (2 * n)) / (1 + z**2 / n) if n else 0.0
half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n) if n else 0.0
summary = {"task": args.task, "policy": args.policy, "seed": seed, "rounds": n, "success": n_ok,
           "success_rate": round(p, 3), "ci95": [round(center - half, 3), round(center + half, 3)],
           "mean_in_target": round(float(np.mean([x["in_target"] for x in results])), 2) if n else 0.0,
           "mean_sim_time_success": round(float(np.mean([x["sim_time"] for x in results if x["success"]])), 2) if n_ok else None,
           # 台車がゴミに当たった (台車の中心から最も近いゴミまでが、外形の半径未満の) ラウンドの数
           "base_contact_rounds": sum(x["base_min_dist"] < sweep_planner.BASE_RADIUS for x in results),
           # 押し終わっていないゴミが front カメラの画角に入っていたフレームの割合 (ラウンドの平均)
           "mean_view_fraction": round(float(np.mean([x["view_fraction"] for x in results])), 3) if n else None,
           "time_limit": cfg.episode_length_s, "wall_time": round(time.time() - start, 1), "device": args.device}
if args.policy == "remote":
    # ping は、ここまでの推論サーバの VRAM のピークも返す
    summary.update(action_horizon=args.action_horizon, server=policy.client.call("ping")[0])
if args.record:
    summary.update(dataset_file=args.dataset_file, num_demos=n if args.dagger else n_ok,
                   dataset_bytes=os.path.getsize(args.dataset_file) if os.path.exists(args.dataset_file) else 0)
if args.dagger:
    summary.update(dagger=True, dagger_beta=args.dagger_beta)
if maps:
    cv2.imwrite(f"{out_dir}/overview.png", cv2.cvtColor(sweep_video.overview(maps), cv2.COLOR_RGB2BGR))
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"SWEEP {'DAGGER' if args.dagger else 'DATAGEN' if args.record else 'EVAL'} {n_ok}/{n} ({p:.0%}, 95% CI {summary['ci95'][0]:.0%}-{summary['ci95'][1]:.0%}) "
      f"policy={args.policy} seed={seed} -> {out_dir}", flush=True)
env.close()
app.close()
