"""Sweep の計画器 (特権情報を使う古典的な手法)。docs/sweep.md の「計画器」

1 回の掃き: アームを移動の姿勢にして台車を開始位置へ動かし、穂を床に押し当てて、台車をまっすぐ前に進め、
穂先の前のゴミを目標領域まで押す。穂を上げ、掃いた跡に沿って下がりながら残りのゴミの方へ向きを変え、
次の掃きを選ぶ。これを繰り返す。
- 押した後の後退は、押した直後に通った跡 (ゴミを押し出した後で空いている) を戻るだけにする。押した後は、こぼれた
  ゴミが台車の横や後ろにあり、front カメラに写らない。下がりながら向きを変えて、それを画面に入れる
  (docs/sweep_experiments.md の実験 10)。
- 開始位置へは、押し終わっていないゴミの重心を向いたまま、重心のまわりを、半径を保つか縮めながら回り込み
  (orbit_route。台車から見て横か前へ進む)、開始位置の APPROACH 手前から前へ進んで入る。後退するのは、押した後に
  掃いた跡を戻るときだけにする。回り込む道がゴミに近すぎるときは、ゴミを囲む円を経由する (plan_route)。
- 移動の間は、学習するポリシーが front カメラでゴミを見失わないようにする (docs/sweep_experiments.md の実験 6)。
  アームは穂が front カメラの床を隠さない姿勢 (既定の姿勢) にし、台車は押し終わっていないゴミの方を向けたまま動かす
  (台車は全方向に動けるので、向きと通り道は独立に決められる)。
- 入力は SweepState (ゴミの位置、目標領域の位置、台車の姿勢、アームの関節位置) だけで、シーンに依存しない。
  位置は env の原点からの相対値 (m)。台車の向き theta は base_theta (0 で +y を向く。前は (-sin, cos)、右は (cos, sin))。
- 出力は sim の action (lekiwi-leader の構成、9 次元。台車の速度はワールド座標系)。
- アームは、あらかじめ IK で求めた姿勢 (穂の高さごと) の間を補間するだけ。掃く向きは台車の向きで決まる。
- 障害物 (S3a の箱) は、外形の上の点 (SweepState.obstacles) として扱う。経路は箱の点もゴミと同じように避け、
  台車、アーム、穂、押したゴミの通り道が箱にかかる掃きは選ばない (選べる掃きがなければ、最後の手段として使う)。
  まっすぐ目標領域へ押せないときは、箱の横の中継点 (choose_via) へ押してから、目標領域へ押す (2 段の掃き)。
  障害物がなければ、計算は障害物を扱う前と同じになる。
"""
import heapq
import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import contract, sweep_mdp
from .broom import BRUSH_TIP, HEAD_SIZE

# 障害物の箱の名前 (sweep_floor.OBSTACLE_NAME。sweep_floor は isaaclab の設定を読み込むので、ここでは import しない)
OBSTACLE_NAME = "obstacle"
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
# 押した後に、掃いた跡に沿って下がる距離の範囲 (m) と速さ (m/s)。下がる間に、残りのゴミの方へ向きを変える。
# 距離は、残りのゴミの重心から、次の開始位置の手前の点 (APPROACH) と同じだけ離れるまで (回り込みで外へ出ないため)
RETREAT_MIN = 0.15
RETREAT_MAX = 0.6
RETREAT_SPEED = 0.2
# ゴミのまわりを回り込むときの経由点の角度の間隔
ORBIT_STEP = math.radians(15)
# 開始位置へは、掃く向きの APPROACH (m) 手前を経由し、そこから前へ進んで入る (後退して入らない)
APPROACH = 0.15
# 掃きの通り道 (台車、アーム、穂、押したゴミ) と箱の外形の間に空ける距離 (m)
OBSTACLE_MARGIN = 0.03
# 中継点へ押す掃き (2 段の掃きの 1 段目) の余裕。ACT は 1 段目で、計画器より遠くまでまっすぐ押して箱の角に当てるので、
# この掃きだけ広げる (すべての掃きで広げると、終わり近くの掃きが悪くなった。docs/sweep_experiments.md の S3a′)
VIA_MARGIN = 0.08
# 中継点 (choose_via) の候補は、箱の中心から、箱の外接円の半径にこれらを足した距離の円の上の VIA_DIRECTIONS 個の点
VIA_DISTANCES = (0.3, 0.4, 0.5)
VIA_DIRECTIONS = 16
# 中継点に集めたゴミの広がり (押し終わっていないゴミの重心からの位置を、この倍率に縮めて中継点に置いたものとみなす)
VIA_SPREAD = 0.5
# 1 段目の押しすぎに強くする (S3a′。手本は、真似したポリシーの誤差より広い余裕を持つ):
# - 押す向きの延長: ゴミの重心から中継点への線を、中継点の先へ VIA_RAY_LENGTH (m) 延ばしても、箱の外形から
#   VIA_RAY_CLEARANCE 以上離れている中継点だけを使う (押しすぎても、ゴミは箱から離れる方へ行く)
VIA_RAY_LENGTH = 0.5
VIA_RAY_CLEARANCE = 0.12
# - 中継点は、全体の長さが最短のものの VIA_LENGTH_SLACK 倍までの候補のうち、通り道と箱の最小距離が最大のもの
VIA_LENGTH_SLACK = 1.3
# - 1 段目は、中継点の中心の VIA_END (m) 手前で止め、VIA_PUSH_SPEED (m/s) で押す
VIA_END = -0.10
VIA_PUSH_SPEED = 0.12
# 箱に当たらない掃きも中継点もないとき、箱に最も近いゴミを、箱から離れる向きに AWAY_DISTANCE (m) 押す
AWAY_DISTANCE = 0.3
# 状態機械の段階 (状態の記録では、この並びの番号で残す)
PHASES = ("init", "plan", "travel", "lower", "push", "raise", "retreat", "idle")


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
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # [M, 2] 箱の外形の上の点 (なければ 0 個)


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
    obstacles = np.zeros((0, 2))
    if OBSTACLE_NAME in env.scene.rigid_objects:
        box = env.scene[OBSTACLE_NAME]
        quat = box.data.root_quat_w[env_id]  # (w, x, y, z)。箱は床の上で、z 軸まわりにだけ回る
        yaw = 2 * torch.atan2(quat[3], quat[0])
        half = torch.tensor(box.cfg.spawn.size[:2], device=quat.device) / 2
        obstacles = to_np(sweep_mdp.box_outline(box.data.root_pos_w[env_id, :2] - origin, yaw, half, per_side=8))
    return SweepState(to_np(debris), to_np(target), to_np(base), float(theta), to_np(arm), obstacles)


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
    goal: np.ndarray | None = None  # 押す先の正方形の中心 (目標領域か、中継点)。経路は goal を基準にする
    blocked: bool = False  # 通り道が障害物にかかる
    hit: bool = False  # 台車が、まっすぐ押されないゴミを轢く見込み
    relay: bool = False  # 中継点へ押す掃き (2 段の掃きの 1 段目。押す速さは VIA_PUSH_SPEED)

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


def choose_pass(state: SweepState, tip: np.ndarray, params: PassParams = PassParams(), goal: np.ndarray | None = None,
                done: np.ndarray | None = None, box_first: bool = False, margin: float = OBSTACLE_MARGIN) -> Pass | None:
    """次の掃きを選ぶ (押し終わっていないゴミがなければ None)

    候補は、押し終わっていないゴミそれぞれから目標領域の中心への向きと、横のずれ params.laterals の組。
    台車がゴミを轢かない候補のうち、目標領域に入る見込みのゴミが最も多く、掃く距離が最も短いものを選ぶ。
    tip は押し当てた姿勢の穂先の位置 (台車の座標系、ArmPoses.tip)。
    障害物があれば、通り道 (台車の後ろの端から、押したゴミがたまる穂先の前まで、台車の左の端から穂の右の端までの
    長方形) が箱の外形の点にかかる候補を blocked とする。箱にかからずゴミも轢かない候補を最も先にし、次にゴミを轢かない
    候補 (箱にかかるなら、SweepPlanner._choose が中継点へ押す) を先にする。box_first なら、箱にかからない候補を先にする
    (中継点がないとき。ゴミを轢いても、箱は押さない)。
    goal を与えると、目標領域の代わりに goal (中継点) を中心とする同じ大きさの正方形へ押す。done (ゴミごと) が真の
    ゴミは、押し終わったものとする。margin は、通り道と箱の間に空ける距離 (中継点へ押す掃きは VIA_MARGIN)。
    """
    t = state.target if goal is None else goal
    rel = state.debris - t
    pending = ~in_target(state.debris, t, params.done_margin)
    if done is not None:
        pending &= ~done
    obstacles = state.obstacles - t
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
            # 開始位置の手前の経由点 (APPROACH) から入る区間も含める
            hit = (~pushed & (np.abs(lat - base_lat) < radius) & (along > start - tip[1] - APPROACH - radius)
                   & (along < params.end - tip[1] + radius))
            blocked = False
            if len(obstacles):
                o_along, o_lat = obstacles @ d, obstacles @ n
                blocked = bool(((o_lat > base_lat - BASE_RADIUS - margin)
                                & (o_lat < c + BRUSH_HALF_WIDTH + margin)
                                & (o_along > start - tip[1] - APPROACH - BASE_RADIUS - margin)
                                & (o_along < params.end + 2 * DEBRIS_HALF + 0.05 + margin)).any())
            # 箱がなければ前と同じ順 (blocked は常に偽)
            if box_first:
                key = (not blocked, not hit.any(), int(covered.sum()), -(params.end - start))
            else:
                key = (not (blocked or hit.any()), not hit.any(), not blocked, int(covered.sum()), -(params.end - start))
            if best_key is None or key > best_key:
                best, best_key = Pass(d, c, start, params.end, int(covered.sum()), t.copy(), blocked, bool(hit.any())), key
    return best


def relay_params(params: PassParams) -> PassParams:
    """中継点へ押す掃きの設定 (中継点の中心の VIA_END 手前で止める)"""
    return replace(params, end=VIA_END)


def choose_via(state: SweepState, tip: np.ndarray, params: PassParams, pending: np.ndarray) -> np.ndarray | None:
    """箱をよけて押すための中継点 (なければ None)。候補 (VIA_DISTANCES、VIA_DIRECTIONS) のうち、次をすべて満たすもの:
    目標領域と重ならない。押し終わっていないゴミを中継点へ押す掃き (余裕 VIA_MARGIN) と、中継点に集めたゴミ
    (VIA_SPREAD) を目標領域へ押す掃きの、どちらの通り道も箱にかからない (choose_pass の blocked と同じ判定)。
    ゴミの重心から中継点への線を VIA_RAY_LENGTH 延ばしても、箱から VIA_RAY_CLEARANCE 以上離れている。
    そのうち、長さ (ゴミの重心から中継点、中継点から目標領域) が最短のものの VIA_LENGTH_SLACK 倍までで、
    ゴミの通り道 (延長を含む) と箱の最小距離が最大のもの"""
    if not len(state.obstacles) or not pending.any():
        return None
    center = state.debris[pending].mean(axis=0)
    box = state.obstacles.mean(axis=0)
    box_radius = np.linalg.norm(state.obstacles - box, axis=-1).max()
    candidates = []  # (長さ, 箱との最小距離, 中継点)
    for r in VIA_DISTANCES:
        for a in np.linspace(0, 2 * math.pi, VIA_DIRECTIONS, endpoint=False):
            via = box + (box_radius + r) * np.array([math.cos(a), math.sin(a)])
            if (np.abs(via - state.target) < 2 * sweep_mdp.TARGET_HALF).all():
                continue
            u = (via - center) / max(np.linalg.norm(via - center), 1e-6)
            ray = _segment_distance(via, via + VIA_RAY_LENGTH * u, state.obstacles).min()
            if VIA_RAY_LENGTH > 0 and ray < VIA_RAY_CLEARANCE:
                continue
            first = choose_pass(state, tip, relay_params(params), goal=via, done=~pending, box_first=True, margin=VIA_MARGIN)
            if first is None or first.blocked:
                continue
            gathered = state.debris.copy()
            gathered[pending] = via + VIA_SPREAD * (state.debris[pending] - center)
            second = choose_pass(SweepState(gathered, state.target, state.base, state.theta, state.arm, state.obstacles), tip, params,
                                 box_first=True)
            if second is None or second.blocked:
                continue
            clearance = min(_segment_distance(center, via, state.obstacles).min(), ray,
                            _segment_distance(via, state.target, state.obstacles).min())
            length = np.linalg.norm(via - center) + np.linalg.norm(state.target - via)
            candidates.append((length, clearance, via))
    if not candidates:
        return None
    shortest = min(c[0] for c in candidates)
    return max((c for c in candidates if c[0] <= VIA_LENGTH_SLACK * shortest + 1e-9), key=lambda c: (c[1], -c[0]))[2]


def choose_away(state: SweepState, tip: np.ndarray, params: PassParams, pending: np.ndarray) -> Pass | None:
    """箱に当たらない掃きも中継点もないとき: 箱に最も近い押し終わっていないゴミを、箱から離れる向き (または箱の縁に
    沿う向き) に AWAY_DISTANCE 押す掃き。どの向きも箱にかかるなら None"""
    if AWAY_DISTANCE <= 0 or not len(state.obstacles) or not pending.any():
        return None
    box = state.obstacles.mean(axis=0)
    idx = np.flatnonzero(pending)
    i = idx[np.argmin(np.linalg.norm(state.debris[idx][:, None] - state.obstacles[None], axis=-1).min(axis=1))]
    away = state.debris[i] - box
    away /= max(np.linalg.norm(away), 1e-6)
    side = np.array([-away[1], away[0]])
    near = np.linalg.norm(state.debris - state.debris[i], axis=-1) < 0.1
    for d in (away, (away + side) / math.sqrt(2), (away - side) / math.sqrt(2), side, -side):
        p = choose_pass(state, tip, params, goal=state.debris[i] + AWAY_DISTANCE * d, done=~(pending & near), box_first=True)
        if p is not None and not p.blocked:
            return p
    return None


def _segment_distance(a: np.ndarray, b: np.ndarray, points: np.ndarray) -> np.ndarray:
    """線分 ab と points [N, 2] の距離"""
    ab = b - a
    s = np.clip(((points - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + s[:, None] * ab), axis=-1)


def orbit_route(start: np.ndarray, goal: np.ndarray, center: np.ndarray, obstacles: np.ndarray, clearance: float) -> list[np.ndarray] | None:
    """center のまわりを回り込んで start から goal へ行く経由点の列 (最後が goal)。半径は start から goal の半径へ
    一定の割合で変える (center を向いたまま進めば、台車から見て横か、半径が縮むなら前へ進む)。
    obstacles から clearance 以内を通るなら None。start のすぐ近くの obstacles は、最初の区間では無視する"""
    r0, r1 = np.linalg.norm(start - center), np.linalg.norm(goal - center)
    a0 = math.atan2(*(start - center)[::-1])
    da = wrap(math.atan2(*(goal - center)[::-1]) - a0)
    n = max(1, math.ceil(abs(da) / ORBIT_STEP))
    points = [center + (r0 + (r1 - r0) * k / n) * np.array([math.cos(a0 + da * k / n), math.sin(a0 + da * k / n)]) for k in range(1, n + 1)]
    points[-1] = goal
    near_start = np.linalg.norm(obstacles - start, axis=-1) < clearance
    prev = start
    for k, p in enumerate(points):
        obs = obstacles[~near_start] if k == 0 else obstacles
        if len(obs) and _segment_distance(prev, p, obs).min() < clearance:
            return None
        prev = p
    return points


def plan_route(start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, clearance: float, ring_points: int = 24,
               min_radius: float = 0.0, ring_obstacles: np.ndarray | None = None,
               extra_nodes: tuple[np.ndarray, ...] = ()) -> list[np.ndarray]:
    """台車の中心を start から goal へ動かす経由点の列 (最後が goal)。obstacles から clearance 以内を通らない

    まっすぐ行けなければ、obstacles を囲む円の上の点を経由する最短の経路を選ぶ。start のすぐ近くにある
    obstacles (すでに clearance 以内) は、start から出る線分では無視する。経路がなければ、まっすぐ goal へ向かう。
    ring_obstacles を与えると、円は obstacles ではなく ring_obstacles を囲むものにする。extra_nodes は、ほかに
    経由してよい点 (箱のまわりの円の点など)。
    """
    near_start = np.linalg.norm(obstacles - start, axis=-1) < clearance

    def clear(a, b, from_start):
        obs = obstacles[~near_start] if from_start else obstacles
        return len(obs) == 0 or _segment_distance(a, b, obs).min() >= clearance

    if len(obstacles) == 0 or clear(start, goal, True):
        return [goal]
    ring = obstacles if ring_obstacles is None else ring_obstacles
    center = ring.mean(axis=0)
    radius = max(np.linalg.norm(ring - center, axis=-1).max() + clearance * 1.3, min_radius)
    angles = np.linspace(0, 2 * math.pi, ring_points, endpoint=False)
    nodes = [start, goal] + [center + radius * np.array([math.cos(a), math.sin(a)]) for a in angles] + list(extra_nodes)
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

    状態: init (アームを移動の姿勢へ) -> plan -> travel (台車を開始位置へ) -> lower -> push -> raise -> retreat -> plan ...
    押し終わっていないゴミがなくなったら idle (止まって、ときどき計画し直す)。
    """

    def __init__(self, poses: ArmPoses, params: PassParams = PassParams(), dt: float = 1.0 / contract.FPS, overshoot: float = 0.0):
        # overshoot: 中継点へ押す掃きを、決めた位置よりこれだけ (m) 先まで押す。学習したポリシーの押しすぎをまねて、
        # 計画器の手本が押しすぎに強いかを評価するためだけに使う (sweep_eval.py --via_overshoot。datagen では使わない)
        self.poses, self.params, self.dt, self.overshoot = poses, params, dt, overshoot
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
        self.retreat_goal: np.ndarray | None = None
        self.via: np.ndarray | None = None  # 箱をよけて押す中継点 (2 段の掃きの途中のとき)

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

    def _look_theta(self, state: SweepState) -> float | None:
        """押し終わっていないゴミの重心の方を向く台車の向き (なければ None)"""
        pending = ~in_target(state.debris, state.target, self.params.done_margin)
        if not pending.any():
            return None
        return heading_of(state.debris[pending].mean(axis=0) - state.base)

    def _retreat_goal(self, state: SweepState) -> np.ndarray:
        """押した後に下がる先。掃いた跡 (今の掃く向きの逆) に沿って、残りのゴミの重心から、次の掃きの開始位置の
        APPROACH 手前の点と同じだけ離れるまで (RETREAT_MIN から RETREAT_MAX)"""
        back = -self.current.direction
        nxt = choose_pass(state, self.poses.tip, self.params)
        if nxt is None:
            return state.base + RETREAT_MIN * back
        pending = ~in_target(state.debris, state.target, self.params.done_margin)
        center = state.debris[pending].mean(axis=0)
        theta = heading_of(nxt.direction)
        approach = self._base_for_tip(nxt.tip_at(state.target, nxt.start), theta) - APPROACH * nxt.direction
        want = np.linalg.norm(approach - center)
        for d in np.arange(RETREAT_MIN, RETREAT_MAX + 1e-6, 0.05):
            if np.linalg.norm(state.base + d * back - center) >= want:
                return state.base + d * back
        return state.base + RETREAT_MAX * back

    def _arm_at(self, t: float) -> np.ndarray:
        """移動の姿勢 (t=0) から、穂を上げた姿勢 (t=ARM_TRAVEL_TIME) を経て、押し当てた姿勢
        (t=ARM_TRAVEL_TIME + ARM_MOVE_TIME) までの関節位置"""
        if t < ARM_TRAVEL_TIME:
            return self.poses.travel + (self.poses.at(1.0) - self.poses.travel) * (t / ARM_TRAVEL_TIME)
        return self.poses.at(1.0 - (t - ARM_TRAVEL_TIME) / ARM_MOVE_TIME)

    def _choose(self, state: SweepState) -> Pass | None:
        """次の掃き。目標領域へまっすぐ押す掃きの通り道が箱にかかるなら、中継点へ押す掃き (2 段の掃きの 1 段目) にする。
        中継点は、2 段の掃きを始めたときに決め、まっすぐ押せるようになるまで変えない"""
        p = choose_pass(state, self.poses.tip, self.params)
        if p is None or not p.blocked:
            self.via = None
            return p
        pending = ~in_target(state.debris, state.target, self.params.done_margin)
        if self.via is None:
            self.via = choose_via(state, self.poses.tip, self.params, pending)
        if self.via is None:
            return self._fallback(state, pending)
        # 中継点に集めたゴミは、1 段目では押し終わったものとする。すべて集まったら、中継点から目標領域へ押す
        done = ~pending | in_target(state.debris, self.via, self.params.done_margin)
        q = choose_pass(state, self.poses.tip, relay_params(self.params), goal=self.via, done=done, box_first=True, margin=VIA_MARGIN)
        if q is None:
            # 中継点に集まった。目標領域へ押す掃き (通り道は空いているはず) を、中継点を忘れて選び直す
            self.via = None
            return p if not p.blocked else self._fallback(state, pending)
        if q.blocked:
            return self._fallback(state, pending)
        q.relay = True
        return q

    def _fallback(self, state: SweepState, pending: np.ndarray) -> Pass | None:
        """まっすぐ押す掃きが箱にかかり、中継点へも押せないとき。箱にかからない掃き (ゴミを轢いてもよい) があればそれ、
        なければ、箱の近くのゴミを箱から離す掃き (choose_away)。どれもなければ、箱にかかる掃き"""
        p = choose_pass(state, self.poses.tip, self.params, box_first=True)
        if p is None or not p.blocked:
            return p
        return choose_away(state, self.poses.tip, self.params, pending) or p

    def _obstacle_nodes(self, state: SweepState, clearance: float) -> tuple[np.ndarray, ...]:
        """箱のまわりを回る経由点 (plan_route の extra_nodes)"""
        if not len(state.obstacles):
            return ()
        box = state.obstacles.mean(axis=0)
        radius = np.linalg.norm(state.obstacles - box, axis=-1).max() + clearance * 1.3
        return tuple(box + radius * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False))

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
                self.current = self._choose(state)
                if self.current is None:
                    self._start_phase("idle")
                else:
                    p = self.current
                    self.goal_theta = heading_of(p.direction)
                    goal = self._base_for_tip(p.tip_at(p.goal, p.start), self.goal_theta)
                    clearance = BASE_RADIUS + DEBRIS_HALF + self.params.base_clearance
                    # 掃く向きの APPROACH 手前を経由する。回り込む円は、その点より外側にとる (円から内向きに進んで入る)
                    approach = goal - APPROACH * p.direction
                    pending = ~in_target(state.debris, state.target, self.params.done_margin)
                    center = state.debris[pending].mean(axis=0)
                    obstacles = np.concatenate([state.debris, state.obstacles])
                    route = orbit_route(state.base, approach, center, obstacles, clearance)
                    if route is None:
                        ring = np.linalg.norm(approach - state.debris.mean(axis=0)) + 0.05
                        route = plan_route(state.base, approach, obstacles, clearance, min_radius=ring,
                                           ring_obstacles=state.debris, extra_nodes=self._obstacle_nodes(state, clearance))
                    self.route = route + [goal]
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
            tip = self._tip(state) - p.goal
            along, lat = tip @ p.direction, tip @ p.normal
            # 押しすぎの評価 (overshoot) では、中継点へ押す掃きを、決めた位置より先まで押す
            end = p.end + (self.overshoot if p.relay else 0.0)
            remaining = end - along
            push_speed = VIA_PUSH_SPEED if p.relay else PUSH_SPEED
            speed = min(push_speed, 1.5 * remaining + 0.03)
            cmd = speed * p.direction - 2.0 * (lat - p.lateral) * p.normal
            vel = self._limit(np.array([*cmd, 3.0 * wrap(self.goal_theta - state.theta)]), push_speed)
            if remaining <= 0.005 or self.timer > (end - p.start) / push_speed * 2 + 2.0:
                self.passes += 1
                self._start_phase("raise")
        elif self.phase == "raise":
            self.arm_t = max(self.arm_t - self.dt, ARM_TRAVEL_TIME)
            arm = self._arm_at(self.arm_t)
            vel = self._limit(np.zeros(3), 0.0)
            if self.arm_t <= ARM_TRAVEL_TIME:
                self.retreat_goal = self._retreat_goal(state)
                self._start_phase("retreat")
        elif self.phase == "retreat":
            # 掃いた跡に沿って下がりながら、残りのゴミの方へ向きを変える
            look = self._look_theta(state)
            theta = state.theta if look is None else look
            vel, dist, err_theta = self._goto(state, self.retreat_goal, theta, RETREAT_SPEED)
            if (dist < 0.02 and err_theta < 0.1) or self.timer > 4.0:
                self._start_phase("plan")

        return np.concatenate([arm, [contract.GRIPPER_HOLD_RAD], vel])
