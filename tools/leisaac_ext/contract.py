"""Sweep タスクのデータの約束事 (contract)。sim、変換、学習、推論で共通に使う

isaaclab も lerobot も import しない (numpy だけ)。sim と train の両方のイメージで読み込める。
キー名と単位は実機の LeRobot LeKiwi (lerobot 0.4.2 の robots/lekiwi) に合わせる:
    アーム: arm_<関節名>.pos。モータの正規化値 (RANGE_M100_100、グリッパは RANGE_0_100)
    台車:   x.vel (前進, m/s)、y.vel (左, m/s)、theta.vel (反時計回り, deg/s)。ロボット基準
sim の env の action (lekiwi-leader の構成、9 次元) との対応:
    [0:5] アームの関節位置 (rad)、[5] グリッパ (rad)、[6:9] 台車の速度 (x と y はワールド座標系、θ は rad/s)
sim の観測の joint_pos と joint_vel は、articulation の関節の並び (SIM_JOINT_NAMES) で、台車が先に来る。
詳細と根拠は docs/sweep.md。
"""
import numpy as np

FPS = 30
# sim は 60Hz で 1 step 進む。ポリシーと計画器は 30Hz で action を決め、同じ action を 2 step 続ける
SIM_STEPS_PER_FRAME = 2

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STATE_NAMES = tuple(f"arm_{j}.pos" for j in ARM_JOINTS) + ("x.vel", "y.vel", "theta.vel")
ACTION_NAMES = STATE_NAMES
CAMERAS = ("front", "wrist")
IMAGE_SHAPE = (240, 320, 3)  # (高さ, 幅, チャンネル)
INSTRUCTION = "Sweep the debris into the marked square on the floor"

# sim の articulation の関節の並び (観測の joint_pos と joint_vel の 9 次元)。PhysX が決める並びで、
# action の並び (アームが先) とは違う。一致は sim の中のテストで確かめる (P2)
SIM_JOINT_NAMES = ("base_x", "base_y", "base_theta") + ARM_JOINTS
_ARM_IDX = [SIM_JOINT_NAMES.index(j) for j in ARM_JOINTS]
_BASE_IDX = [SIM_JOINT_NAMES.index(j) for j in ("base_x", "base_y", "base_theta")]

# 箒を握ったときのグリッパの角度 (rad)。0 で、厚さ 12mm の柄がどちらの指にも触れない (broom.py)
GRIPPER_HOLD_RAD = 0.0

# 上流の leisaac.assets.robots.lerobot と同じ値 (度)。一致は sim の中のテストで確かめる (P2)
_USD_JOINT_LIMITS = np.array(
    [(-110.0, 110.0), (-100.0, 100.0), (-100.0, 90.0), (-95.0, 95.0), (-160.0, 160.0), (-10.0, 100.0)]
)
_MOTOR_LIMITS = np.array(
    [(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (0.0, 100.0)]
)


def arm_rad_to_motor(rad: np.ndarray) -> np.ndarray:
    """アームの関節位置 [..., 6] (rad) をモータの正規化値にする"""
    deg = np.degrees(rad)
    lo, hi = _USD_JOINT_LIMITS[:, 0], _USD_JOINT_LIMITS[:, 1]
    mlo, mhi = _MOTOR_LIMITS[:, 0], _MOTOR_LIMITS[:, 1]
    return (deg - lo) / (hi - lo) * (mhi - mlo) + mlo


def arm_motor_to_rad(motor: np.ndarray) -> np.ndarray:
    """arm_rad_to_motor の逆変換"""
    lo, hi = _USD_JOINT_LIMITS[:, 0], _USD_JOINT_LIMITS[:, 1]
    mlo, mhi = _MOTOR_LIMITS[:, 0], _MOTOR_LIMITS[:, 1]
    return np.radians((motor - mlo) / (mhi - mlo) * (hi - lo) + lo)


def base_world_to_robot(vel: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """台車の速度 [..., 3] (sim の座標系、θ は rad/s) をロボット基準の [x.vel, y.vel, theta.vel] にする

    sim の台車は、本体の +y が前、-x が左 (上流の convert_lekiwi_wheel_action_env2robot と同じ)。
    theta は台車のワールド座標系での向き (base_theta の関節位置)。
    """
    c, s = np.cos(theta), np.sin(theta)
    dx_body = c * vel[..., 0] + s * vel[..., 1]
    dy_body = -s * vel[..., 0] + c * vel[..., 1]
    return np.stack([dy_body, -dx_body, np.degrees(vel[..., 2])], axis=-1)


def base_robot_to_world(vel: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """base_world_to_robot の逆変換"""
    c, s = np.cos(theta), np.sin(theta)
    dx_body, dy_body = -vel[..., 1], vel[..., 0]
    dx = c * dx_body - s * dy_body
    dy = s * dx_body + c * dy_body
    return np.stack([dx, dy, np.radians(vel[..., 2])], axis=-1)


def to_lerobot(joint_pos: np.ndarray, joint_vel: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sim の 1 step 分 (または [N, ...] の系列) を、LeRobot 形式の (state, action) にする

    joint_pos と joint_vel は step の前の観測 (SIM_JOINT_NAMES の並び)、action はその step で env に渡した値。
    台車の action は、渡した値そのものを観測時の向きで変換する。上流の build_lerobot_frame は
    1 step 後の観測 (user_vel_action) を使うので、ラベルが 1 フレーム遅れる。ここではその遅れがない。
    """
    theta = joint_pos[..., _BASE_IDX[2]]
    state = np.concatenate(
        [arm_rad_to_motor(joint_pos[..., _ARM_IDX]), base_world_to_robot(joint_vel[..., _BASE_IDX], theta)], axis=-1
    )
    act = np.concatenate([arm_rad_to_motor(action[..., :6]), base_world_to_robot(action[..., 6:9], theta)], axis=-1)
    return state.astype(np.float32), act.astype(np.float32)


def from_lerobot_action(action: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
    """LeRobot 形式の action [..., 9] を、sim の env に渡す action にする (to_lerobot の action の逆)"""
    theta = joint_pos[..., _BASE_IDX[2]]
    return np.concatenate(
        [arm_motor_to_rad(action[..., :6]), base_robot_to_world(action[..., 6:9], theta)], axis=-1
    ).astype(np.float32)


def frame_indices(num_steps: int) -> np.ndarray:
    """60Hz で記録した step のうち、30fps の学習データに使う step の番号"""
    return np.arange(0, num_steps, SIM_STEPS_PER_FRAME)
