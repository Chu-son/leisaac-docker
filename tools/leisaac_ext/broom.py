"""手箒を固定した LeKiwi (docs/sweep.md)

箒は剛体 `broom` として lekiwi.usd に足し、gripper のリンクと固定ジョイントでつなぐ (車輪と同じ作り)。
アーティキュレーションのリンクが 1 つ増えるだけで、関節の並び (contract の 9 次元) は変わらない。
寸法はグリッパのリンクの座標系で書く: 指先は -z 方向に伸び、x 方向に挟む (固定指は +x 側、可動指は -x 側)。
グリッパの関節が 0 のとき、可動指と固定指の隙間は約 13mm なので、厚さ 12mm の柄がどちらにも触れずに収まる
(contract.GRIPPER_HOLD_RAD = 0)。
"""
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from leisaac.assets.robots.lerobot import LEKIWI_CFG
from pxr import Gf, UsdGeom, UsdPhysics

from .contract import GRIPPER_HOLD_RAD

# 柄と穂の (x 厚さ, y 幅, z 長さ) (m) と質量 (kg)。合計で全長 30cm、100g
HANDLE_SIZE = (0.012, 0.025, 0.18)
HEAD_SIZE = (0.02, 0.20, 0.12)
HANDLE_MASS = 0.03
HEAD_MASS = 0.07
# 柄と穂の中心線の (x, y) と、柄の上端の z。指の間 (z = -0.05〜-0.10) で握る
AXIS_XY = (0.0015, 0.03)
HANDLE_TOP_Z = -0.055

HANDLE_CENTER = (*AXIS_XY, HANDLE_TOP_Z - HANDLE_SIZE[2] / 2)
HEAD_CENTER = (*AXIS_XY, HANDLE_TOP_Z - HANDLE_SIZE[2] - HEAD_SIZE[2] / 2)
# 穂先の中心 (計画器が狙う点)
BRUSH_TIP = (*AXIS_XY, HANDLE_TOP_Z - HANDLE_SIZE[2] - HEAD_SIZE[2])

# 穂と床・ゴミの摩擦。接触モデルは docs/sweep.md の「接触モデル」を参照
HEAD_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4)


def _mass_properties() -> tuple[float, tuple, tuple]:
    """箒全体の質量、重心、慣性 (重心まわり、主軸は座標軸) を、2 つの直方体から求める"""
    parts = [(HANDLE_MASS, HANDLE_SIZE, HANDLE_CENTER), (HEAD_MASS, HEAD_SIZE, HEAD_CENTER)]
    mass = sum(m for m, _, _ in parts)
    com = tuple(sum(m * c[i] for m, _, c in parts) / mass for i in range(3))
    inertia = [0.0, 0.0, 0.0]
    for m, (sx, sy, sz), c in parts:
        d = [c[i] - com[i] for i in range(3)]
        own = (m * (sy**2 + sz**2) / 12, m * (sx**2 + sz**2) / 12, m * (sx**2 + sy**2) / 12)
        for i in range(3):
            # 平行軸の定理 (重心からのずれ d の、軸 i に垂直な成分)
            inertia[i] += own[i] + m * sum(d[k] ** 2 for k in range(3) if k != i)
    return mass, com, tuple(inertia)


@clone
def spawn_lekiwi_with_broom(prim_path: str, cfg: sim_utils.UsdFileCfg, *args, **kwargs):
    """lekiwi.usd を読み込み、gripper に箒を固定する (UsdFileCfg の func に指定する)"""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage()
    gripper = stage.GetPrimAtPath(f"{prim_path}/gripper")
    broom_path = f"{prim_path}/broom"

    # gripper と同じ姿勢に置く (どちらも prim_path の直下なので、ローカルの変換をそのまま写す)
    broom = UsdGeom.Xform.Define(stage, broom_path)
    for op in UsdGeom.Xformable(gripper).GetOrderedXformOps():
        broom.AddXformOp(op.GetOpType(), op.GetPrecision()).Set(op.Get())

    UsdPhysics.RigidBodyAPI.Apply(broom.GetPrim())
    mass, com, inertia = _mass_properties()
    mass_api = UsdPhysics.MassAPI.Apply(broom.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    for name, size, center, color, material in [
        ("handle", HANDLE_SIZE, HANDLE_CENTER, (0.55, 0.35, 0.2), None),
        ("head", HEAD_SIZE, HEAD_CENTER, (0.8, 0.7, 0.4), HEAD_MATERIAL),
    ]:
        part = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=material,
        )
        part.func(f"{broom_path}/{name}", part, translation=center)

    joint = UsdPhysics.FixedJoint.Define(stage, f"{broom_path}/FixedJoint")
    joint.CreateBody0Rel().SetTargets([gripper.GetPath()])
    joint.CreateBody1Rel().SetTargets([broom.GetPath()])
    return prim


LEKIWI_BROOM_CFG = LEKIWI_CFG.replace(
    spawn=LEKIWI_CFG.spawn.replace(func=spawn_lekiwi_with_broom),
    init_state=LEKIWI_CFG.init_state.replace(
        joint_pos={**LEKIWI_CFG.init_state.joint_pos, "gripper": GRIPPER_HOLD_RAD}
    ),
)
