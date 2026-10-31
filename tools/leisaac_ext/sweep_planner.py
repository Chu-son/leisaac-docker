"""Sweep の計画器 (特権情報を使う古典的な手法)。docs/sweep.md の「計画器」

1 回の掃き: アームを移動の姿勢にして台車を開始位置へ動かし、穂を床に押し当てて、台車をまっすぐ前に進め、
穂先の前のゴミを目標領域まで押す。穂を上げて、次の掃きを選ぶ。これを繰り返す。
- 移動の間は、学習するポリシーが front カメラでゴミを見失わないようにする (docs/sweep_experiments.md の実験 6)。
  アームは穂が front カメラの床を隠さない姿勢 (既定の姿勢) にし、台車は押し終わっていないゴミの方を向けたまま動かす
  (台車は全方向に動けるので、向きと通り道は独立に決められる)。
- 入力は SweepState (ゴミの位置、目標領域の位置、台車の姿勢、アームの関節位置) だけで、シーンに依存しない。
  位置は env の原点からの相対値 (m)。台車の向き theta は base_theta (0 で +y を向く。前は (-sin, cos)、右は (cos, sin))。
- 出力は sim の action (lekiwi-leader の構成、9 次元。台車の速度はワールド座標系)。
- アームは、あらかじめ IK で求めた姿勢 (穂の高さごと) の間を補間するだけ。掃く向きは台車の向きで決まる。
- 障害物 (家具など) は扱わない。避けるのはゴミだけ (P7 で足す)。
"""
import heapq
import math
from dataclasses import dataclass

import numpy as np
import torch

from . import contract, sweep_mdp
from .broom import BRUSH_TIP, HEAD_SIZE

# 台車の外形 (車輪を含む) が収まる円の半径 (m)
BASE_RADIUS = 0.165
# 穂を下ろした姿勢: 穂先を台車の中心から BRUSH_FORWARD (m) 前に置き、穂の長さの向きを床から BRUSH_ALPHA 傾ける。
# 穂の幅の向きは台車の左右にそろえる (前に押すとゴミが横に逃げない)。穂先は台車の中心線より右にずれる (約 15cm。
# 肩の位置と箒の握りの向きによる。中心線の上に置く解はない)
BRUSH_FORWARD = 0.50
BRUSH_ALPHA = math.radians(45)
# 穂の高さ (m、床から)。最初が押し当てた姿勢 (床より少し下を狙い、アームの柔らかさで押し付ける)、最後が上げた姿勢
BRUSH_HEIGHTS = (-0.01, 0.02, 0.05, 0.08)
BRUSH_HALF_WIDTH = HEAD_SIZE[1] / 2
DEBRIS_HALF = 0.015  # ゴミの半分の大きさ (最大)
# 速さの上限 (上流のテレオペの最大と同じ 0.3m/s、90°/s) と、加速度の上限
TRAVEL_SPEED = 0.3
PUSH_SPEED = 0.2
TURN_SPEED = math.pi / 2
LINEAR_ACCEL = 0.8
ANGULAR_ACCEL = 4.0
# アームの動きの時間 (s)
ARM_INIT_TIME = 1.0
ARM_MOVE_TIME = 0.5
# 移動の姿勢と、穂を上げた姿勢の間。穂を上げた後は台車を動かしながら移動の姿勢へ移し、開始位置の LOOK_NEAR 手前から戻す
ARM_TRAVEL_TIME = 1.0
# 移動の間、台車をゴミの方へ向ける。開始位置までの距離が LOOK_FAR より近くなったら掃く向きへ切り替え始め、LOOK_NEAR で切り替え終わる
LOOK_FAR = 0.4
LOOK_NEAR = 0.15


def forward_of(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta)])


def right_of(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def heading_of(direction: np.ndarray) -> float:
    """前が direction を向く台車の向き theta"""
    return math.atan2(-direction[0], direction[1])


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def in_target(points: np.ndarray, target: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """points [N, 2] が目標領域の内側 (縁から margin 以上内側) にあるか (sweep_mdp.debris_in_target の xy の条件)"""
    return (np.abs(points - target) < sweep_mdp.TARGET_HALF - margin).all(axis=-1)


@dataclass
class SweepState:
    debris: np.ndarray  # [N, 2] ゴミの位置
    target: np.ndarray  # [2] 目標領域の中心
    base: np.ndarray  # [2] 台車の中心
    theta: float  # 台車の向き
    arm: np.ndarray  # [5] アームの関節位置 (rad、action の並び)


def read_state(env, env_id: int = 0) -> SweepState:
    """sim の真値から SweepState を作る"""
    origin = env.scene.env_origins[env_id, :2]
    robot = env.scene["robot"]
    debris = torch.stack([env.scene[n].data.root_pos_w[env_id, :2] for n in sweep_mdp.DEBRIS_NAMES]) - origin
    target = env.scene[sweep_mdp.TARGET_NAME].data.root_pos_w[env_id, :2] - origin
    base = robot.data.body_pos_w[env_id, robot.body_names.index("base"), :2] - origin
    joint_pos = robot.data.joint_pos[env_id]
    arm = joint_pos[[robot.joint_names.index(j) for j in contract.ARM_JOINTS[:5]]]
    theta = joint_pos[robot.joint_names.index("base_theta")]
    to_np = lambda x: x.cpu().numpy().astype(np.float64)
    return SweepState(to_np(debris), to_np(target), to_np(base), float(theta), to_np(arm))


@dataclass
class ArmPoses:
    """穂の高さごとのアームの関節位置 (calibrate_arm で求める) と、移動の姿勢"""

    heights: tuple[float, ...]
    q: np.ndarray  # [len(heights), 5] (rad)
    tip: np.ndarray  # [2] 押し当てた姿勢の穂先の位置 (台車の座標系。x が右、y が前)
    travel: np.ndarray  # [5] 移動の姿勢 (rad)。穂が front カメラの視野の上の端にあり、床を隠さない

    def at(self, s: float) -> np.ndarray:
        """s=0 (押し当てた姿勢) から s=1 (上げた姿勢) までの関節位置。高さの順に区分線形に補間する"""
        x = min(max(s, 0.0), 1.0) * (len(self.heights) - 1)
        i = min(int(x), len(self.heights) - 2)
        return self.q[i] + (self.q[i + 1] - self.q[i]) * (x - i)


def calibrate_arm(env, forward: float = BRUSH_FORWARD, alpha: float = BRUSH_ALPHA, heights=BRUSH_HEIGHTS) -> ArmPoses:
    """穂先を台車の前 forward、高さ h、傾き alpha に置くアームの関節位置を、高さごとに数値 IK で求める

    関節の状態を直接書き換えて順運動学を調べるので、エピソードの外で呼び、後で env.reset() する。
    条件は 5 つ (穂先の前後の位置と高さ、穂の幅の向きが水平かつ台車の左右、穂の傾き) で、関節も 5 つ。
    穂先の左右の位置は条件にしない (計画器が台車の位置で合わせる)。
    """
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    robot = env.scene["robot"]
    dev = env.device
    arm = [robot.joint_names.index(j) for j in contract.ARM_JOINTS[:5]]
    i_broom, i_base = robot.body_names.index("broom"), robot.body_names.index("base")
    lo, hi = robot.data.soft_joint_pos_limits[0, arm].unbind(-1)
    env_ids = torch.tensor([0], device=dev)
    floor_z = env.scene.env_origins[0, 2]
    tip_local = torch.tensor(BRUSH_TIP, device=dev)
    width_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)
    length_axis = torch.tensor([0.0, 0.0, -1.0], device=dev)

    def fk(q_arm: torch.Tensor):
        """台車の座標系での穂先の位置 (z は床からの高さ)、穂の幅の向き、穂の長さの向き"""
        q = robot.data.default_joint_pos[:1].clone()
        q[0, arm] = q_arm
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        p_base, q_base = robot.data.body_pos_w[0, i_base], robot.data.body_quat_w[0, i_base]
        p, quat = robot.data.body_pos_w[0, i_broom], robot.data.body_quat_w[0, i_broom]
        tip_w = quat_apply(quat, tip_local) + p
        tip = quat_apply_inverse(q_base, tip_w - p_base)
        tip[2] = tip_w[2] - floor_z
        return tip, quat_apply_inverse(q_base, quat_apply(quat, width_axis)), quat_apply_inverse(q_base, quat_apply(quat, length_axis))

    def residual(q_arm: torch.Tensor, h: float) -> torch.Tensor:
        tip, e, u = fk(q_arm)
        return torch.stack([tip[1] - forward, tip[2] - h, e[2], e[1], u[2] + math.sin(alpha)])

    # 初期値は、穂を床に当てた姿勢の近く (2026-10-30 に sim で調べた解)
    q = torch.tensor([0.58, -0.7, 0.86, 0.5, 0.4], device=dev)
    solutions = []
    for h in heights:
        for _ in range(100):
            r = residual(q, h)
            if r.norm() < 1e-5:
                break
            jac = torch.zeros(5, 5, device=dev)
            for j in range(5):
                dq = torch.zeros(5, device=dev)
                dq[j] = 1e-3
                jac[:, j] = (residual(q + dq, h) - r) / 1e-3
            step = torch.linalg.solve(jac.T @ jac + 1e-4 * torch.eye(5, device=dev), -jac.T @ r)
            q = torch.clamp(q + step.clamp(-0.2, 0.2), lo, hi)
        err = residual(q, h).norm().item()
        if err > 1e-3:
            raise RuntimeError(f"brush pose not reachable: forward={forward} height={h} alpha={alpha:.2f} (residual {err:.1e})")
        solutions.append(q.clone())
    tip, _, _ = fk(solutions[0])
    to_np = lambda x: x.cpu().numpy().astype(np.float64)
    return ArmPoses(tuple(heights), to_np(torch.stack(solutions)), to_np(tip[:2]), to_np(robot.data.default_joint_pos[0, arm]))


@dataclass
class Pass:
    """1 回の掃き。穂先の中心の経路は target + s * direction + lateral * normal (s は start から end まで)"""

    direction: np.ndarray  # 掃く向き (単位ベクトル)
    lateral: float  # 目標領域の中心を通る掃く向きの線からの、横のずれ (normal の向きが正)
    start: float  # 穂を下ろす位置 (目標領域の中心から、掃く向きに測った距離。手前は負)
    end: float  # 穂を止める位置
    covered: int  # この掃きで目標領域に入る見込みのゴミの数

    @property
    def normal(self) -> np.ndarray:
        """掃く向きの右"""
        return np.array([self.direction[1], -self.direction[0]])

    def tip_at(self, target: np.ndarray, s: float) -> np.ndarray:
        return target + s * self.direction + self.lateral * self.normal


@dataclass
class PassParams:
    end: float = -0.03  # 穂を止める位置。ゴミは穂先の 2〜5cm 前にたまるので、目標領域の中心の少し手前で止める
    gap: float = 0.05  # 穂を下ろす位置と、最も手前のゴミとの間隔
    cover_half: float = 0.085  # 穂の中心からこの範囲のゴミは、まっすぐ押される (穂の幅の半分より内側)
    end_half: float = 0.12  # 目標領域の中心線からこの範囲のゴミは、押した後に目標領域に入る (半分の幅 0.15 より内側)
    done_margin: float = 0.02  # 目標領域の縁からこれ以上内側のゴミは、押し終わったものとする
    base_clearance: float = 0.03  # 台車の外形とゴミの間に空ける距離
    laterals: tuple[float, ...] = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)


def choose_pass(state: SweepState, tip: np.ndarray, params: PassParams = PassParams()) -> Pass | None:
    """次の掃きを選ぶ (押し終わっていないゴミがなければ None)

    候補は、押し終わっていないゴミそれぞれから目標領域の中心への向きと、横のずれ params.laterals の組。
    台車がゴミを轢かない候補のうち、目標領域に入る見込みのゴミが最も多く、掃く距離が最も短いものを選ぶ。
    tip は押し当てた姿勢の穂先の位置 (台車の座標系、ArmPoses.tip)。
    """
    t = state.target
    rel = state.debris - t
    pending = ~in_target(state.debris, t, params.done_margin)
    radius = BASE_RADIUS + DEBRIS_HALF + params.base_clearance
    best, best_key = None, None
    for i in np.flatnonzero(pending):
        dist = np.linalg.norm(rel[i])
        if dist < 1e-6:
            continue
        d = -rel[i] / dist
        n = np.array([d[1], -d[0]])
        along, lat = rel @ d, rel @ n
        for c in params.laterals:
            # 穂に触れるゴミ (穂を下ろす位置より先にあり、止める位置より手前)
            band = (np.abs(lat - c) < BRUSH_HALF_WIDTH + DEBRIS_HALF) & (along < params.end)
            pushed = band & (np.abs(lat - c) < params.cover_half)
            covered = pushed & pending & (np.abs(lat) < params.end_half)
            start = along[band].min() - params.gap
            # 掃く間の台車の中心は、穂先から横に -tip[0]、前後に -tip[1] の位置。まっすぐ押されないゴミを轢かないか。
            # 穂の端をかすめるだけのゴミ (band の縁) は押されずに残り、台車の通り道 (穂の左) に入ることがある
            base_lat = c - tip[0]
            hit = ~pushed & (np.abs(lat - base_lat) < radius) & (along > start - tip[1] - radius) & (along < params.end - tip[1] + radius)
            key = (not hit.any(), int(covered.sum()), -(params.end - start))
            if best_key is None or key > best_key:
                best, best_key = Pass(d, c, start, params.end, int(covered.sum())), key
    return best


def _segment_distance(a: np.ndarray, b: np.ndarray, points: np.ndarray) -> np.ndarray:
    """線分 ab と points [N, 2] の距離"""
    ab = b - a
    s = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + s[:, None] * ab), axis=-1)


def plan_route(start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, clearance: float, ring_points: int = 24) -> list[np.ndarray]:
    """台車の中心を start から goal へ動かす経由点の列 (最後が goal)。obstacles から clearance 以内を通らない

    まっすぐ行けなければ、obstacles を囲む円の上の点を経由する最短の経路を選ぶ。start のすぐ近くにある
    obstacles (すでに clearance 以内) は、start から出る線分では無視する。経路がなければ、まっすぐ goal へ向かう。
    """
    near_start = np.linalg.norm(obstacles - start, axis=-1) < clearance

    def clear(a, b, from_start):
        obs = obstacles[~near_start] if from_start else obstacles
        return len(obs) == 0 or _segment_distance(a, b, obs).min() >= clearance

    if len(obstacles) == 0 or clear(start, goal, True):
        return [goal]
    center = obstacles.mean(axis=0)
    radius = np.linalg.norm(obstacles - center, axis=-1).max() + clearance * 1.3
    angles = np.linspace(0, 2 * math.pi, ring_points, endpoint=False)
    nodes = [start, goal] + [center + radius * np.array([math.cos(a), math.sin(a)]) for a in angles]
    dist = {0: 0.0}
    prev = {}
    queue = [(0.0, 0)]
    while queue:
        cost, k = heapq.heappop(queue)
        if k == 1:
            break
        if cost > dist.get(k, math.inf):
            continue
        for j in range(1, len(nodes)):
            if j != k and clear(nodes[k], nodes[j], k == 0):
                c = cost + np.linalg.norm(nodes[j] - nodes[k])
                if c < dist.get(j, math.inf):
                    dist[j], prev[j] = c, k
                    heapq.heappush(queue, (c, j))
    if 1 not in prev:
        return [goal]
    route, k = [], 1
    while k != 0:
        route.append(nodes[k])
        k = prev[k]
    return route[::-1]


class SweepPlanner:
    """掃きを繰り返す状態機械。act を 1 フレーム (dt 秒) ごとに呼ぶ

    状態: init (アームを移動の姿勢へ) -> plan -> travel (台車を開始位置へ) -> lower -> push -> raise -> plan ...
    押し終わっていないゴミがなくなったら idle (止まって、ときどき計画し直す)。
    """

    def __init__(self, poses: ArmPoses, params: PassParams = PassParams(), dt: float = 1.0 / contract.FPS):
        self.poses, self.params, self.dt = poses, params, dt
        self.reset()

    def reset(self) -> None:
        self.phase = "init"
        self.timer = 0.0
        self.passes = 0
        self.current: Pass | None = None
        self.route: list[np.ndarray] = []
        self.goal_theta = 0.0
        self.arm_from: np.ndarray | None = None
        self.arm_t = 0.0  # アームの経路 (_arm_at) の上の位置 (s)。0 が移動の姿勢
        self.vel = np.zeros(3)  # 前のフレームの台車の速度の指令 (ワールド座標系の x, y と theta)

    def _tip(self, state: SweepState) -> np.ndarray:
        """押し当てた姿勢の穂先の、env の座標系での位置"""
        return state.base + self.poses.tip[0] * right_of(state.theta) + self.poses.tip[1] * forward_of(state.theta)

    def _base_for_tip(self, tip: np.ndarray, theta: float) -> np.ndarray:
        return tip - self.poses.tip[0] * right_of(theta) - self.poses.tip[1] * forward_of(theta)

    def _limit(self, vel: np.ndarray, speed: float) -> np.ndarray:
        """速さの上限と加速度の上限をかける"""
        v = vel.copy()
        norm = np.linalg.norm(v[:2])
        if norm > speed:
            v[:2] *= speed / norm
        v[2] = np.clip(v[2], -TURN_SPEED, TURN_SPEED)
        dv = v - self.vel
        norm = np.linalg.norm(dv[:2])
        if norm > LINEAR_ACCEL * self.dt:
            dv[:2] *= LINEAR_ACCEL * self.dt / norm
        dv[2] = np.clip(dv[2], -ANGULAR_ACCEL * self.dt, ANGULAR_ACCEL * self.dt)
        self.vel = self.vel + dv
        return self.vel

    def _goto(self, state: SweepState, goal: np.ndarray, theta: float, speed: float) -> tuple[np.ndarray, float, float]:
        """台車を goal と theta へ向かわせる速度 (比例制御)。位置と向きの残りの誤差も返す"""
        err = goal - state.base
        err_theta = wrap(theta - state.theta)
        # 止まれる速さ (加速度の上限で減速しきれる速さ) を超えない
        dist = np.linalg.norm(err)
        stop_speed = math.sqrt(2 * LINEAR_ACCEL * 0.8 * dist)
        cmd = np.array([*(err * 3.0), err_theta * 3.0])
        return self._limit(cmd, min(speed, stop_speed + 0.02)), dist, abs(err_theta)

    def _travel_theta(self, state: SweepState) -> float:
        """移動の間の台車の向き。開始位置から遠いうちは、押し終わっていないゴミの重心の方を向く"""
        pending = ~in_target(state.debris, state.target, self.params.done_margin)
        if not pending.any():
            return self.goal_theta
        look = heading_of(state.debris[pending].mean(axis=0) - state.base)
        dist = np.linalg.norm(self.route[-1] - state.base)
        w = min(max((dist - LOOK_NEAR) / (LOOK_FAR - LOOK_NEAR), 0.0), 1.0)
        return self.goal_theta + w * wrap(look - self.goal_theta)

    def _arm_at(self, t: float) -> np.ndarray:
        """移動の姿勢 (t=0) から、穂を上げた姿勢 (t=ARM_TRAVEL_TIME) を経て、押し当てた姿勢
        (t=ARM_TRAVEL_TIME + ARM_MOVE_TIME) までの関節位置"""
        if t < ARM_TRAVEL_TIME:
            return self.poses.travel + (self.poses.at(1.0) - self.poses.travel) * (t / ARM_TRAVEL_TIME)
        return self.poses.at(1.0 - (t - ARM_TRAVEL_TIME) / ARM_MOVE_TIME)

    def _start_phase(self, phase: str) -> None:
        self.phase, self.timer = phase, 0.0

    def act(self, state: SweepState) -> np.ndarray:
        """次のフレームの sim の action (9 次元)"""
        self.timer += self.dt
        vel = np.zeros(3)
        if self.arm_from is None:
            self.arm_from = state.arm.copy()
        if self.phase == "travel" and len(self.route) == 1 and np.linalg.norm(self.route[0] - state.base) < LOOK_NEAR:
            # 開始位置の近く (掃く向きを向いた後) では、穂を上げた姿勢へ移しておく
            self.arm_t = min(self.arm_t + self.dt, ARM_TRAVEL_TIME)
        elif self.phase not in ("init", "lower", "push", "raise"):
            self.arm_t = max(self.arm_t - self.dt, 0.0)
        arm = self._arm_at(self.arm_t)

        if self.phase == "init":
            a = min(self.timer / ARM_INIT_TIME, 1.0)
            arm = self.arm_from + (self.poses.travel - self.arm_from) * a
            vel = self._limit(np.zeros(3), 0.0)
            if a >= 1.0:
                self._start_phase("plan")
        elif self.phase in ("plan", "idle"):
            vel = self._limit(np.zeros(3), 0.0)
            if self.phase == "plan" or self.timer > 0.5:
                self.current = choose_pass(state, self.poses.tip, self.params)
                if self.current is None:
                    self._start_phase("idle")
                else:
                    p = self.current
                    self.goal_theta = heading_of(p.direction)
                    goal = self._base_for_tip(p.tip_at(state.target, p.start), self.goal_theta)
                    clearance = BASE_RADIUS + DEBRIS_HALF + self.params.base_clearance
                    self.route = plan_route(state.base, goal, state.debris, clearance)
                    self._start_phase("travel")
        elif self.phase == "travel":
            final = len(self.route) == 1
            vel, dist, err_theta = self._goto(state, self.route[0], self._travel_theta(state), TRAVEL_SPEED)
            if not final and dist < 0.05:
                self.route.pop(0)
            elif final and dist < 0.01 and err_theta < 0.02 and np.abs(self.vel).max() < 0.05:
                self._start_phase("lower")
            elif self.timer > 15.0:  # 着かない (押されたゴミが邪魔をしているなど)。計画し直す
                self._start_phase("plan")
        elif self.phase == "lower":
            self.arm_t = min(self.arm_t + self.dt, ARM_TRAVEL_TIME + ARM_MOVE_TIME)
            arm = self._arm_at(self.arm_t)
            vel, _, _ = self._goto(state, self.route[-1], self.goal_theta, 0.05)
            if self.arm_t >= ARM_TRAVEL_TIME + ARM_MOVE_TIME:
                self._start_phase("push")
        elif self.phase == "push":
            p = self.current
            arm = self._arm_at(self.arm_t)
            tip = self._tip(state) - state.target
            along, lat = tip @ p.direction, tip @ p.normal
            remaining = p.end - along
            speed = min(PUSH_SPEED, 1.5 * remaining + 0.03)
            cmd = speed * p.direction - 2.0 * (lat - p.lateral) * p.normal
            vel = self._limit(np.array([*cmd, 3.0 * wrap(self.goal_theta - state.theta)]), PUSH_SPEED)
            if remaining <= 0.005 or self.timer > (p.end - p.start) / PUSH_SPEED * 2 + 2.0:
                self.passes += 1
                self._start_phase("raise")
        elif self.phase == "raise":
            self.arm_t = max(self.arm_t - self.dt, ARM_TRAVEL_TIME)
            arm = self._arm_at(self.arm_t)
            vel = self._limit(np.zeros(3), 0.0)
            if self.arm_t <= ARM_TRAVEL_TIME:
                self._start_phase("plan")

        return np.concatenate([arm, [contract.GRIPPER_HOLD_RAD], vel])
