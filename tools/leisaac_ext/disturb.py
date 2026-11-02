"""評価の外乱 (sweep_eval.py --disturb)。docs/sweep.md の「外乱」

1 ラウンドに 1 種類だけ加える。sweep_eval がフレームの境目 (action を決めた後、記録と env.step の前) で step を呼ぶ。
    push:  ラウンドの PUSH_TIME 秒のどこかで 1 回、台車の位置を横に PUSH_SHIFT、向きを PUSH_YAW だけずらす
           (仮想の関節 base_x、base_y、base_theta に書き込む)。ずらした先で台車か穂がゴミに重なるなら、次のフレームに延ばす。
    spill: 目標領域に入ったゴミが初めて SPILL_TRIGGER 個以上になったフレームで、そのうち SPILL_COUNT 個を目標領域の縁の
           外 SPILL_DIST へ移す。移す先は front カメラの水平の画角の中で、台車、穂、ほかのゴミから離れた位置。
乱数は numpy (np.random.default_rng([seed, ラウンド, 0])) だけを使う。torch の乱数を使わないので、配置は外乱なしと同じになる。
目標領域、ゴミ、カメラの位置は引数で受け取り、タスクの定数には依存しない。
"""
import math

import numpy as np
import torch

KINDS = ("push", "spill")
PUSH_TIME = (2.0, 7.0)  # s (計画器や ACT は早いと 8 秒で終わるので、それより前に押す)
PUSH_SHIFT = (0.10, 0.15)  # m (台車の左右の向き)
PUSH_YAW = (math.radians(5), math.radians(15))  # rad (向きはランダム)
SPILL_TRIGGER = 6
SPILL_COUNT = 2
SPILL_DIST = (0.05, 0.15)  # 目標領域の縁から外へ (m)
# 重なりを避ける距離 (m): 台車の中心から、穂の中心から、ほかのゴミから
BASE_CLEAR = 0.165 + 0.04
HEAD_CLEAR = 0.15
DEBRIS_CLEAR = 0.04
VIEW_MARGIN = 0.05  # front の水平の半画角から引く余裕 (rad)
SPILL_TRIES = 200


def _right(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def _forward(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta)])


class RoundDisturbance:
    """1 ラウンドの外乱。step をフレームごとに呼ぶ。起きたことは info に残る"""

    def __init__(self, kind: str, seed: int, round_index: int, fps: int):
        assert kind in KINDS, kind
        self.kind = kind
        self.rng = np.random.default_rng([seed, round_index, 0])
        self.done = False
        self.info: dict = {"disturb": kind, "disturb_applied": False}
        if kind == "push":
            self.push_frame = int(self.rng.uniform(*PUSH_TIME) * fps)
            side = self.rng.choice([-1.0, 1.0])
            self.push_shift = side * self.rng.uniform(*PUSH_SHIFT)
            self.push_yaw = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(*PUSH_YAW)

    def step(self, frame: int, ctx: "Context") -> None:
        if self.done:
            return
        if self.kind == "push" and frame >= self.push_frame:
            if self._push(ctx):
                self.done = True
                self.info.update(disturb_applied=True, disturb_frame=frame, push_shift=round(float(self.push_shift), 4),
                                 push_yaw=round(float(self.push_yaw), 4))
        elif self.kind == "spill" and int(ctx.inside().sum()) >= SPILL_TRIGGER:
            self.done = True
            moved = self._spill(ctx)
            self.info.update(disturb_frame=frame, disturb_applied=moved is not None)
            if moved is not None:
                self.info.update(spill_ids=[int(i) for i, _ in moved],
                                 spill_xy=[[round(float(v), 4) for v in xy] for _, xy in moved])

    def _push(self, ctx: "Context") -> bool:
        base, theta = ctx.base()
        new_base = base + self.push_shift * _right(theta)
        new_theta = theta + self.push_yaw
        # 穂は台車と一緒に動く (台車の座標系での位置は変わらない)
        head_local = ctx.head() - base
        rot = np.array([[math.cos(self.push_yaw), -math.sin(self.push_yaw)], [math.sin(self.push_yaw), math.cos(self.push_yaw)]])
        new_head = new_base + rot @ head_local
        debris = ctx.debris()
        if (np.linalg.norm(debris - new_base, axis=-1) < BASE_CLEAR).any() or (np.linalg.norm(debris - new_head, axis=-1) < HEAD_CLEAR).any():
            return False
        ctx.set_base(new_base, new_theta)
        return True

    def _spill(self, ctx: "Context") -> list | None:
        inside = np.flatnonzero(ctx.inside())
        ids = self.rng.choice(inside, size=SPILL_COUNT, replace=False)
        base, theta = ctx.base()
        head = ctx.head()
        camera = base + ctx.camera_xy[0] * _right(theta) + ctx.camera_xy[1] * _forward(theta)
        target, half = ctx.target_xy, ctx.target_half
        keep = [p for i, p in enumerate(ctx.debris()) if i not in ids]
        placed = []
        for i in ids:
            for _ in range(SPILL_TRIES):
                # 目標領域の 4 辺のどれかの外側
                side = self.rng.integers(4)
                along = self.rng.uniform(-half, half)
                out = half + self.rng.uniform(*SPILL_DIST)
                offset = [np.array([out, along]), np.array([-out, along]), np.array([along, out]), np.array([along, -out])][side]
                p = target + offset
                d = p - camera
                bearing = abs(math.atan2(-(d @ _right(theta)), d @ _forward(theta)))
                others = np.array(keep + [q for _, q in placed]) if keep or placed else np.zeros((0, 2))
                if (bearing < ctx.half_fov - VIEW_MARGIN and np.linalg.norm(p - base) > BASE_CLEAR
                        and np.linalg.norm(p - head) > HEAD_CLEAR
                        and (len(others) == 0 or np.linalg.norm(others - p, axis=-1).min() > DEBRIS_CLEAR)):
                    placed.append((i, p))
                    break
            else:
                return None
        for i, p in placed:
            ctx.set_debris(i, p)
        return placed


class Context:
    """外乱が読む状態と、書き込む口 (env の 1 つの環境)。位置は env の原点からの相対値 (m)"""

    def __init__(self, env, debris_names, target_name: str, target_half: float, camera_xy, half_fov: float,
                 head_center_local, env_id: int = 0):
        self.env, self.e = env, env_id
        self.debris_names, self.target_name = list(debris_names), target_name
        self.target_half, self.camera_xy, self.half_fov = target_half, camera_xy, half_fov
        self.robot = env.scene["robot"]
        self.i_base = self.robot.body_names.index("base")
        self.i_broom = self.robot.body_names.index("broom")
        self.j_base = [self.robot.joint_names.index(j) for j in ("base_x", "base_y", "base_theta")]
        self.head_center_local = torch.tensor(head_center_local, dtype=torch.float32, device=env.device)
        self.origin = env.scene.env_origins[env_id]
        self.target_xy = None

    def begin_round(self) -> None:
        self.target_xy = self._np(self.env.scene[self.target_name].data.root_pos_w[self.e, :2] - self.origin[:2])

    @staticmethod
    def _np(x) -> np.ndarray:
        return x.detach().cpu().numpy().astype(np.float64)

    def base(self) -> tuple[np.ndarray, float]:
        joint_pos = self.robot.data.joint_pos[self.e]
        base = self._np(self.robot.data.body_pos_w[self.e, self.i_base, :2] - self.origin[:2])
        return base, float(joint_pos[self.j_base[2]])

    def head(self) -> np.ndarray:
        from isaaclab.utils.math import quat_apply

        pos = self.robot.data.body_pos_w[self.e, self.i_broom]
        quat = self.robot.data.body_quat_w[self.e, self.i_broom]
        return self._np((pos + quat_apply(quat, self.head_center_local))[:2] - self.origin[:2])

    def debris(self) -> np.ndarray:
        return np.stack([self._np(self.env.scene[n].data.root_pos_w[self.e, :2] - self.origin[:2]) for n in self.debris_names])

    def inside(self) -> np.ndarray:
        return (np.abs(self.debris() - self.target_xy) < self.target_half).all(axis=-1)

    def set_base(self, base_xy: np.ndarray, theta: float) -> None:
        """台車の位置と向きを書き込む。仮想の関節の値は、今の値に、台車の中心の移動量を足して決める"""
        cur_base, cur_theta = self.base()
        joint_pos = self.robot.data.joint_pos[self.e].clone()
        joint_vel = self.robot.data.joint_vel[self.e].clone()
        delta = torch.tensor([*(base_xy - cur_base), theta - cur_theta], dtype=joint_pos.dtype, device=joint_pos.device)
        joint_pos[self.j_base] += delta
        joint_vel[self.j_base] = 0.0
        ids = torch.tensor([self.e], device=self.env.device)
        self.robot.write_joint_state_to_sim(joint_pos[None], joint_vel[None], env_ids=ids)

    def set_debris(self, i: int, xy: np.ndarray) -> None:
        obj = self.env.scene[self.debris_names[i]]
        ids = torch.tensor([self.e], device=self.env.device)
        pos = self.origin + torch.tensor([*xy, obj.cfg.spawn.size[2] / 2], dtype=torch.float32, device=self.env.device)
        obj.write_root_pose_to_sim(torch.cat([pos, obj.data.root_quat_w[self.e]])[None], env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device), env_ids=ids)
