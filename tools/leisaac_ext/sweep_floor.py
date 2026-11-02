"""床のゴミを目標領域へ掃き集めるタスク LeIsaac-LeKiwi-Sweep-Floor-v0 (docs/sweep.md)

床 (直方体。外部のアセットを使わない)、箒付きの LeKiwi、ゴミ debris_0..9、目標領域 target (見た目だけ) からなる。
成功は、10 個中 8 個以上のゴミの中心が目標領域に入ったとき (終了条件 success)。
ランダム化は、ゴミと目標領域の配置 (常に有効) と、set_randomization で選ぶ追加の項目 (RANDOMIZATIONS)。
"""
import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.tasks.template import LeKiwiObservationsCfg, LeKiwiTaskEnvCfg, LeKiwiTaskSceneCfg, LeKiwiTerminationsCfg

from . import contract, sweep_mdp
from .broom import LEKIWI_BROOM_CFG

FLOOR_SIZE = (8.0, 8.0, 0.05)
FLOOR_COLOR = (0.55, 0.52, 0.48)
FLOOR_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5)
# ゴミの大きさ (立方体の 1 辺、m) と質量 (kg) は、番号ごとに 2〜3cm、5〜10g に振り分ける
DEBRIS_SIZES = tuple(0.02 + 0.01 * i / (sweep_mdp.NUM_DEBRIS - 1) for i in range(sweep_mdp.NUM_DEBRIS))
DEBRIS_MASSES = tuple(0.005 + 0.005 * i / (sweep_mdp.NUM_DEBRIS - 1) for i in range(sweep_mdp.NUM_DEBRIS))
# ゴミの色 (目標領域の青と紛れない色)
DEBRIS_COLORS = (
    (0.55, 0.35, 0.20), (0.85, 0.20, 0.15), (0.20, 0.55, 0.20), (0.90, 0.75, 0.15), (0.30, 0.30, 0.30),
    (0.65, 0.45, 0.30), (0.80, 0.45, 0.10), (0.45, 0.60, 0.15), (0.70, 0.15, 0.40), (0.15, 0.15, 0.15),
)
DEBRIS_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4)
TARGET_COLOR = (0.10, 0.35, 0.85)
TARGET_THICKNESS = 0.002
# 照明 (DomeLight)。色は上流のテンプレートと同じ。強さは、上流の 3000 (Loft の室内向け) では床だけのシーンが白飛びするので下げる
LIGHT_COLOR = (0.75, 0.75, 0.75)
LIGHT_INTENSITY = 1000.0

# 配置の範囲 (env の原点から、m)。ロボットは原点で +y を向き、箒の穂先は約 65cm 前にある。
# この範囲から、ゴミと目標領域の両方が front カメラの画角に入る配置だけを選ぶ (sweep_mdp.reset_layout)
CLUSTER_RANGE = {"x": (-0.4, 0.4), "y": (0.9, 1.5)}
TARGET_RANGE = {"x": (-0.9, 0.9), "y": (0.5, 2.2)}
TARGET_DISTANCE = (0.45, 0.7)  # ゴミのまとまりの中心から目標領域の中心まで

CAMERA_WIDTH, CAMERA_HEIGHT = contract.IMAGE_SHAPE[1], contract.IMAGE_SHAPE[0]


def _rot_x(deg: float) -> tuple[float, float, float, float]:
    """x 軸まわりの回転の四元数 (w, x, y, z)"""
    a = math.radians(deg) / 2
    return (math.cos(a), math.sin(a), 0.0, 0.0)


def _quat_mul(q1, q2) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


# カメラの取り付け (2026-11-01 に見直し。docs/sweep.md の「カメラの見直し」。実機はこれに合わせて付け直す)。
# 上流の front は台車の前の床から約 6cm で水平より 10° 上を向き、wrist は箒の柄しか写らなかった。
# front: 台車 (base) の前 0.13m、上 0.20m (床から約 0.235m) に置き、水平より 30° 下を向ける (opengl の向きで x 軸まわり 60°)
# wrist: 上流の向きから、画像の横軸まわりに 45° 回し、穂先とその前の床を写す
# どちらも焦点距離 24mm (撮像面の幅 36.83mm で、水平の画角は約 75°)
_UPSTREAM = LeKiwiTaskSceneCfg(num_envs=1, env_spacing=1.0)
FRONT_POS = (0.0, 0.13, 0.20)
FRONT_PITCH_DEG = -30.0
WRIST_TILT_DEG = 45.0
FOCAL_LENGTH = 24.0
FRONT_ROT = _rot_x(90.0 + FRONT_PITCH_DEG)
WRIST_ROT = _quat_mul(_UPSTREAM.wrist.offset.rot, _rot_x(WRIST_TILT_DEG))
# front カメラの位置 (ロボットが原点にいるとき) と、水平の半画角 (rad)
FRONT_CAMERA_XY = FRONT_POS[:2]
FRONT_HALF_FOV = math.atan(_UPSTREAM.front.spawn.horizontal_aperture / 2 / FOCAL_LENGTH)
# 配置で使う方位の上限 (rad)。見直し前の front の半画角 (約 26.8°) から、カメラの向きのランダム化 (±0.03rad) と余裕の分を
# 引いた値のまま変えない (配置の分布をカメラの見直しの前とそろえ、同じシードで同じ配置にする)
LAYOUT_MAX_BEARING = math.atan(_UPSTREAM.front.spawn.horizontal_aperture / 2 / _UPSTREAM.front.spawn.focal_length) - 0.05


def _debris_cfg(i: int) -> RigidObjectCfg:
    size = DEBRIS_SIZES[i]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{sweep_mdp.DEBRIS_NAMES[i]}",
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=DEBRIS_MASSES[i]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEBRIS_COLORS[i]),
            physics_material=DEBRIS_MATERIAL,
        ),
        # 実際の位置はリセットのたびに sweep_mdp.reset_layout が決める
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1 * i - 0.45, 1.0, size / 2)),
    )


@configclass
class SweepFloorSceneCfg(LeKiwiTaskSceneCfg):
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(
            size=FLOOR_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=FLOOR_COLOR),
            physics_material=FLOOR_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -FLOOR_SIZE[2] / 2)),
    )

    # 上流の CleanupTrash と同じ高さに置く (z=0 では車輪が床にめり込み、台車が動かない)
    robot = LEKIWI_BROOM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", init_state=LEKIWI_BROOM_CFG.init_state.replace(pos=(0.0, 0.0, 0.035))
    )

    # 見た目だけの正方形。剛体にして、位置を状態として記録・復元できるようにする。衝突も重力もないので動かない。
    # kinematic にしないのは、上流のリセット (reset_scene_to_default) が速度を書き込み、PhysX がエラーを出すため
    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(2 * sweep_mdp.TARGET_HALF, 2 * sweep_mdp.TARGET_HALF, TARGET_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TARGET_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.5, TARGET_THICKNESS / 2)),
    )

    def __post_init__(self):
        for i, name in enumerate(sweep_mdp.DEBRIS_NAMES):
            setattr(self, name, _debris_cfg(i))
        self.light.spawn = self.light.spawn.replace(intensity=LIGHT_INTENSITY, color=LIGHT_COLOR)
        for camera in (self.front, self.wrist):
            camera.width, camera.height = CAMERA_WIDTH, CAMERA_HEIGHT
            camera.spawn = camera.spawn.replace(focal_length=FOCAL_LENGTH)
        self.front.offset = self.front.offset.replace(pos=FRONT_POS, rot=FRONT_ROT)
        self.wrist.offset = self.wrist.offset.replace(rot=WRIST_ROT)


@configclass
class SweepFloorTerminationsCfg(LeKiwiTerminationsCfg):
    success = DoneTerm(func=sweep_mdp.sweep_success, params={"min_count": sweep_mdp.SUCCESS_COUNT})


def _randomization_terms(name: str) -> dict[str, EventTerm]:
    """追加のランダム化の項目 name のイベント (event の名前 -> EventTerm)"""
    debris = [SceneEntityCfg(n) for n in sweep_mdp.DEBRIS_NAMES]
    if name == "debris_mass":
        return {
            f"rand_mass_{c.name}": EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="reset",
                params={"asset_cfg": c, "mass_distribution_params": (0.005, 0.010), "operation": "abs"},
            )
            for c in debris
        }
    if name == "debris_friction":
        # 床の摩擦の違いを、ゴミの側の摩擦で代用する (PhysX は既定で 2 つの物体の摩擦の平均を使う)
        return {
            f"rand_friction_{c.name}": EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="reset",
                params={
                    "asset_cfg": c,
                    "static_friction_range": (0.3, 0.8),
                    "dynamic_friction_range": (0.25, 0.7),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                    "make_consistent": True,
                },
            )
            for c in debris
        }
    if name == "debris_color":
        # 青 (目標領域の色) を避ける
        return {
            f"rand_color_{c.name}": EventTerm(
                func=sweep_mdp.randomize_color,
                mode="reset",
                params={"asset_cfg": c, "color_range": {"r": (0.1, 0.9), "g": (0.1, 0.8), "b": (0.05, 0.35)}},
            )
            for c in debris
        }
    if name == "light":
        return {
            "rand_light": EventTerm(
                func=sweep_mdp.randomize_light,
                mode="reset",
                params={
                    "intensity_range": (0.6 * LIGHT_INTENSITY, 1.4 * LIGHT_INTENSITY),
                    "tint": 0.1,
                    "base_color": LIGHT_COLOR,
                },
            )
        }
    if name == "light_wide":
        # 配置の外の評価用 (学習データには使わない)。light より広い範囲
        return {
            "rand_light": EventTerm(
                func=sweep_mdp.randomize_light,
                mode="reset",
                params={
                    "intensity_range": (0.3 * LIGHT_INTENSITY, 2.0 * LIGHT_INTENSITY),
                    "tint": 0.25,
                    "base_color": LIGHT_COLOR,
                },
            )
        }
    if name == "floor_color":
        # 配置の外の評価用 (学習データには使わない)。学習データの床は FLOOR_COLOR の 1 色
        return {
            "rand_floor_color": EventTerm(
                func=sweep_mdp.randomize_floor_color,
                mode="reset",
                params={"gray_range": (0.25, 0.85), "warmth": 0.15},
            )
        }
    if name == "camera":
        pose_range = {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005),
                      "roll": (-0.03, 0.03), "pitch": (-0.03, 0.03), "yaw": (-0.03, 0.03)}  # m, rad
        return {
            f"rand_camera_{cam}": EventTerm(
                func=sweep_mdp.randomize_camera_offset,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg(cam), "pose_range": pose_range},
            )
            for cam in contract.CAMERAS
        }
    raise ValueError(f"unknown randomization: {name}")


# 追加のランダム化の項目と、既定で有効にするもの。どれもコストはほぼない (docs/sweep.md)。
# 質量と摩擦は、記録される状態 (初期状態) に含まれず、replay で再現できなくなるので、既定では使わない。
# light_wide と floor_color は配置の外の評価用 (docs/sweep.md の「評価の配置」)。light_wide は light と同じイベントを置き換える
RANDOMIZATIONS = ("debris_mass", "debris_friction", "debris_color", "light", "camera", "light_wide", "floor_color")
DEFAULT_RANDOMIZATION = ("debris_color", "light", "camera")
# --randomize all で有効にするもの (light_wide は light と同時に使えないので除く)
ALL_RANDOMIZATION = tuple(n for n in RANDOMIZATIONS if n != "light_wide")


EPISODE_LENGTH_S = 40.0


@configclass
class SweepFloorEnvCfg(LeKiwiTaskEnvCfg):
    scene: SweepFloorSceneCfg = SweepFloorSceneCfg(num_envs=1, env_spacing=FLOOR_SIZE[0])
    observations: LeKiwiObservationsCfg = LeKiwiObservationsCfg()
    terminations: SweepFloorTerminationsCfg = SweepFloorTerminationsCfg()
    task_description: str = contract.INSTRUCTION
    # グリッパに最も近い物体の質量に合わせて把持力を変える上流の処理。箒は固定なので使わない
    dynamic_reset_gripper_effort_limit: bool = False
    # リセットの後に描画し直す。既定 (False) では、reset が返す画像がリセットの前の場面のままになる
    rerender_on_reset: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # 1 エピソードの制限時間。計画器は後退しないので、目標領域がロボットとゴミの間にある配置では回り込みと
        # 向きの切り替えに時間がかかる (25 秒では 30 回中 3 回時間切れ。docs/sweep_experiments.md の実験 9)
        self.episode_length_s = EPISODE_LENGTH_S
        self.viewer.eye = (0.0, -1.0, 1.4)
        self.viewer.lookat = (0.0, 1.0, 0.0)

        gravity_bodies = self.events.disable_arm_gravity.params["asset_cfg"].body_names
        self.events.disable_arm_gravity.params["asset_cfg"].body_names = [*gravity_bodies, "broom"]
        self.events.reset_layout = EventTerm(
            func=sweep_mdp.reset_layout,
            mode="reset",
            params={
                "cluster_range": CLUSTER_RANGE,
                "target_range": TARGET_RANGE,
                "target_distance": TARGET_DISTANCE,
                "view_origin": FRONT_CAMERA_XY,
                "max_bearing": LAYOUT_MAX_BEARING,
            },
        )
        self.set_randomization(DEFAULT_RANDOMIZATION)

    def set_randomization(self, names) -> None:
        """追加のランダム化の項目を names (RANDOMIZATIONS の部分集合) にする。env を作る前に呼ぶ"""
        unknown = set(names) - set(RANDOMIZATIONS)
        if unknown:
            raise ValueError(f"unknown randomization: {sorted(unknown)} (choose from {RANDOMIZATIONS})")
        if {"light", "light_wide"} <= set(names):
            raise ValueError("light と light_wide は同時に使えない (同じ照明を変える)")
        terms = {}
        for name in RANDOMIZATIONS:
            for key, term in _randomization_terms(name).items():
                if name in names or key not in terms:
                    terms[key] = term if name in names else None
        for key, term in terms.items():
            setattr(self.events, key, term)
