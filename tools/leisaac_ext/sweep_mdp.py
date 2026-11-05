"""Sweep タスクの MDP の部品 (配置のイベント、ランダム化、指標、終了条件)。docs/sweep.md

位置はすべて env の原点からの相対値 (m)。ロボットは原点にいて、前は +y。
"""
import math

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from pxr import Gf, Usd, UsdShade

NUM_DEBRIS = 10
DEBRIS_NAMES = tuple(f"debris_{i}" for i in range(NUM_DEBRIS))
TARGET_NAME = "target"
# 目標領域 (30cm 角の正方形、向きはワールドの軸にそろえる) の半分の幅と、成功に必要な個数
TARGET_HALF = 0.15
SUCCESS_COUNT = 8
# これより高いゴミは、目標領域の上にあっても数えない (箒に乗っている、跳ねているなど)
IN_TARGET_MAX_Z = 0.05
# 障害物の配置の型 (reset_layout の obstacle)
OBSTACLE_MODES = ("between", "side")
# 台車と穂の外形の代わりに使う線分 (原点のロボットの中心から、穂先の少し前まで。穂先は右に約 15cm、前に約 65cm)
ROBOT_SEGMENT = ((0.0, 0.0), (0.15, 0.75))


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def _bearing(points: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    """view から見た points [..., 2] の方位の、+y (ロボットの前) からの角度の大きさ (rad)。後ろ側は pi 以上にする"""
    d = points - view
    angle = torch.atan2(d[..., 0].abs(), d[..., 1])
    return torch.where(d[..., 1] > 0, angle, angle + math.pi)


def reset_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cluster_range: dict[str, tuple[float, float]],
    target_range: dict[str, tuple[float, float]],
    target_distance: tuple[float, float],
    view_origin: tuple[float, float],
    max_bearing: float,
    grid: tuple[int, int] = (4, 3),
    pitch: float = 0.08,
    jitter: float = 0.015,
    num_candidates: int = 256,
    target_clearance: float | None = None,
    min_debris_y: float | None = None,
    distractors: tuple[str, ...] = (),
    square_colors: tuple[tuple[float, float, float], ...] = (),
    square_separation: float = 0.4,
    obstacle: str | None = None,
    obstacle_modes: tuple[str, ...] = OBSTACLE_MODES,
    debris_clearance: float = 0.12,
    square_clearance: float = 0.05,
    robot_clearance: float = 0.2,
):
    """ゴミのまとまりと目標領域を、ランダムな位置に置く (mode="reset")

    ゴミは、まとまりの中心のまわりの格子 (grid の個数、間隔 pitch。格子全体もランダムに回す) から NUM_DEBRIS 個の
    升目を選び、jitter だけずらして置く。pitch は、隣のゴミと重ならない間隔にする (3cm 角で、ずれが ±1.5cm なら 8cm)。
    目標領域の中心は target_range の中で、まとまりの中心から target_distance の距離に置く。
    さらに、ゴミすべてと目標領域の四隅が、view_origin (front カメラの位置) から見て ±max_bearing 以内に入るようにする
    (最初の画像に両方が写る)。配置の候補をまとめて引き、条件を満たす最初のものを使う (なければ最も条件に近いもの)。
    target_clearance を与えると、ゴミが目標領域の縁から外へこれ以上離れていることも条件にする。min_debris_y を与えると、
    ゴミの y がこれ以上であること (ロボットの前、穂より先) も条件にする。既定 (None) では条件に入れず、配置は変わらない。
    使った配置の違反の値 (0 なら条件をすべて満たす) は env.sweep_layout_violation に残す。

    distractors (正方形の名前) を与えると、目標領域と同じ大きさの正方形を足す (S2 の言語で目標を選ぶタスク)。正方形は
    目標領域と合わせて 1 + len(distractors) 個で、どれも target_distance と画角の条件を満たし、中心どうしを
    square_separation 以上離す (target_clearance は、すべての正方形に効かせる)。位置と色 (square_colors の並べ替え) を
    先に決め、最後に、どの正方形を目標領域 (target) にするかを選ぶ。env.sweep_goal_override (整数) があればその番号、
    なければ一様に選ぶ。同じ配置で番号だけを変えると、見た目は同じで、目標領域の役割だけが入れ替わる。
    選んだ結果は env.sweep_squares (目標領域にした正方形の番号 goal と、target、distractors の順の色の番号 colors) に残す。
    正方形の乱数は、ほかの乱数を引いた後に引くので、distractors がなければ配置は変わらない。

    obstacle (箱の名前) を与えると、箱を 1 つ置く (S3a の障害物のタスク)。型 (obstacle_modes から一様に 1 つ) は
    between (ゴミのまとまりの中心と目標領域の中心を結ぶ線の 40〜60% の位置に、長い辺を線に直交させて置く。回り込みが要る)
    と side (まとまりのまわりに置き、線から離す)。箱の外形は、ゴミから debris_clearance、目標領域の縁から square_clearance、
    台車と穂 (原点から穂先の前まで) から robot_clearance 以上離し、四隅が画角の条件を満たすようにする。
    型と大きさは env.sweep_obstacle に残す。obstacle がなければ箱の乱数を引かないので、配置は変わらない。
    """
    device, k = env.device, num_candidates
    nx, ny = grid
    ix, iy = torch.meshgrid(torch.arange(nx, device=device), torch.arange(ny, device=device), indexing="ij")
    slots = torch.stack([(ix.flatten() - (nx - 1) / 2) * pitch, (iy.flatten() - (ny - 1) / 2) * pitch], dim=-1)
    corners = torch.tensor([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)], device=device) * TARGET_HALF
    view = torch.tensor(view_origin, device=device)
    zero_vel = torch.zeros(1, 6, device=device)
    zero = torch.zeros(1, device=device)

    for env_id in env_ids.tolist():
        center = torch.stack([_uniform(*cluster_range["x"], k, device), _uniform(*cluster_range["y"], k, device)], -1)
        target_xy = torch.stack([_uniform(*target_range["x"], k, device), _uniform(*target_range["y"], k, device)], -1)
        phi = _uniform(0.0, 2 * math.pi, k, device)
        rot = torch.stack([torch.stack([phi.cos(), -phi.sin()], -1), torch.stack([phi.sin(), phi.cos()], -1)], -2)
        chosen = slots[torch.rand(k, len(slots), device=device).argsort(dim=-1)[:, :NUM_DEBRIS]]  # [k, N, 2]
        offsets = chosen @ rot.transpose(-1, -2) + (torch.rand(k, NUM_DEBRIS, 2, device=device) * 2 - 1) * jitter
        debris_xy = center[:, None] + offsets
        # 正方形の中心 [k, S, 2]。先頭が target_xy (distractors がなければ S = 1)
        squares = target_xy[:, None]
        if distractors:
            extra = torch.stack([_uniform(*target_range["x"], k * len(distractors), device),
                                 _uniform(*target_range["y"], k * len(distractors), device)], -1)
            squares = torch.cat([squares, extra.view(k, len(distractors), 2)], dim=1)

        dist = (squares - center[:, None]).norm(dim=-1)  # [k, S]
        lo, hi = target_distance
        square_corners = (squares[:, :, None] + corners).flatten(1, 2)
        bearing = torch.cat([_bearing(debris_xy, view), _bearing(square_corners, view)], dim=1)
        violation = ((lo - dist).clamp(min=0) + (dist - hi).clamp(min=0)).sum(dim=1) \
            + (bearing.amax(dim=1) - max_bearing).clamp(min=0)
        if distractors:
            gap = (squares[:, :, None] - squares[:, None]).norm(dim=-1)
            gap = gap + torch.eye(squares.shape[1], device=device) * square_separation
            violation = violation + (square_separation - gap).clamp(min=0).sum(dim=(1, 2)) / 2
        if target_clearance is not None:
            # ゴミから正方形の縁までの、外への距離 (中にあれば負) の最小
            edge = ((debris_xy[:, None] - squares[:, :, None]).abs().amax(dim=-1) - TARGET_HALF).amin(dim=(1, 2))
            violation = violation + (target_clearance - edge).clamp(min=0)
        if min_debris_y is not None:
            violation = violation + (min_debris_y - debris_xy[..., 1].amin(dim=1)).clamp(min=0)
        if obstacle is not None:
            mode = obstacle_modes[int(torch.randint(len(obstacle_modes), (1,), device=device))]
            box_xy, box_yaw = _obstacle_candidates(mode, center, target_xy, k, device)
            half = torch.tensor(env.scene[obstacle].cfg.spawn.size[:2], device=device) / 2
            violation = violation + _obstacle_violation(
                box_xy, box_yaw, half, debris_xy, squares, center, target_xy, view, max_bearing, mode,
                debris_clearance, square_clearance, robot_clearance)
        best = violation.argmin()
        env.sweep_layout_violation = float(violation[best])

        origin = env.scene.env_origins[env_id]
        ids = torch.tensor([env_id], device=device)
        yaws = _uniform(0.0, 2 * math.pi, NUM_DEBRIS, device)
        for name, xy, yaw in zip(DEBRIS_NAMES, debris_xy[best], yaws):
            obj = env.scene[name]
            pos = origin + torch.cat([xy, zero + obj.cfg.spawn.size[2] / 2])
            quat = quat_from_euler_xyz(zero, zero, yaw.view(1))
            obj.write_root_pose_to_sim(torch.cat([pos.view(1, 3), quat], -1), env_ids=ids)
            obj.write_root_velocity_to_sim(zero_vel, env_ids=ids)

        names, goal = [TARGET_NAME], 0
        if distractors:
            colors = torch.randperm(len(square_colors), device=device)[: squares.shape[1]].tolist()
            override = getattr(env, "sweep_goal_override", None)
            goal = int(override) if override is not None else int(torch.randint(squares.shape[1], (1,), device=device))
            # 正方形 goal を目標領域にし、残りを順に distractors に割り当てる
            order = [goal] + [i for i in range(squares.shape[1]) if i != goal]
            names = [TARGET_NAME, *distractors]
            env.sweep_squares = {"goal": goal, "colors": [colors[i] for i in order]}
            for name, i in zip(names, order):
                set_color(env, name, env_id, square_colors[colors[i]])
            squares = squares[:, order]
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        for name, xy in zip(names, squares[best]):
            obj = env.scene[name]
            pos = origin + torch.cat([xy, zero + obj.cfg.spawn.size[2] / 2])
            obj.write_root_pose_to_sim(torch.cat([pos.view(1, 3), quat], -1), env_ids=ids)
        if obstacle is not None:
            obj = env.scene[obstacle]
            pos = origin + torch.cat([box_xy[best], zero + obj.cfg.spawn.size[2] / 2])
            quat = quat_from_euler_xyz(zero, zero, box_yaw[best].view(1))
            obj.write_root_pose_to_sim(torch.cat([pos.view(1, 3), quat], -1), env_ids=ids)
            obj.write_root_velocity_to_sim(zero_vel, env_ids=ids)
            env.sweep_obstacle = {"mode": mode, "size": list(obj.cfg.spawn.size)}



def box_outline(xy: torch.Tensor, yaw: torch.Tensor, half: torch.Tensor, per_side: int = 6) -> torch.Tensor:
    """箱の外形 (床に投影した長方形の周) の上の点 [..., 4 * per_side, 2]。xy [..., 2]、yaw [...]、half [2] (半分の大きさ)"""
    t = torch.linspace(-1, 1, per_side + 1, device=xy.device)[:-1]
    one = torch.ones_like(t)
    local = torch.cat([torch.stack([t, -one], -1), torch.stack([one, t], -1), torch.stack([-t, one], -1),
                       torch.stack([-one, -t], -1)]) * half  # [4 * per_side, 2]
    c, s = yaw.cos()[..., None], yaw.sin()[..., None]
    x = local[:, 0] * c - local[:, 1] * s
    y = local[:, 0] * s + local[:, 1] * c
    return xy[..., None, :] + torch.stack([x, y], -1)


def _to_box_frame(points: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """points [k, P, 2] を、箱 (xy [k, 2]、yaw [k]) の座標系に移す"""
    d = points - xy[:, None]
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    return torch.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)


def _box_distance(points: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """points [k, P, 2] から箱の外形までの距離 [k, P] (中にあれば 0)"""
    local = _to_box_frame(points, xy, yaw).abs() - half
    return local.clamp(min=0).norm(dim=-1)


def _segment_point_distance(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """線分 ab (a、b [k, 2]) と点 p [k, P, 2] の距離 [k, P]"""
    ab = (b - a)[:, None]
    s = (((p - a[:, None]) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)).clamp(0, 1)
    return (p - (a[:, None] + s[..., None] * ab)).norm(dim=-1)


def _obstacle_candidates(mode: str, center: torch.Tensor, target_xy: torch.Tensor, k: int, device):
    """箱の中心 [k, 2] と向き [k] の候補"""
    d = target_xy - center
    heading = torch.atan2(d[:, 1], d[:, 0])
    if mode == "between":
        f = _uniform(0.4, 0.6, k, device)
        n = torch.stack([-d[:, 1], d[:, 0]], -1) / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        xy = center + f[:, None] * d + _uniform(-0.05, 0.05, k, device)[:, None] * n
        # 長い辺 (x) を線に直交させる
        yaw = heading + math.pi / 2 + _uniform(-0.3, 0.3, k, device)
    elif mode == "side":
        xy = center + torch.stack([_uniform(-0.8, 0.8, k, device), _uniform(-0.8, 0.8, k, device)], -1)
        yaw = _uniform(0.0, math.pi, k, device)
    else:
        raise ValueError(f"unknown obstacle mode: {mode}")
    return xy, yaw


def _obstacle_violation(box_xy, box_yaw, half, debris_xy, squares, center, target_xy, view, max_bearing, mode,
                        debris_clearance, square_clearance, robot_clearance) -> torch.Tensor:
    """箱の配置の条件の違反の値 [k] (0 なら満たす)"""
    outline = box_outline(box_xy, box_yaw, half)  # [k, P, 2]
    v = (debris_clearance - _box_distance(debris_xy, box_xy, box_yaw, half).amin(dim=1)).clamp(min=0)
    # 正方形の縁からの距離 (箱の外形の点が正方形の外にあり、正方形の四隅と中心が箱の外にあること)
    edge = (outline[:, None] - squares[:, :, None]).abs().amax(dim=-1) - TARGET_HALF  # [k, S, P]
    v = v + (square_clearance - edge.amin(dim=(1, 2))).clamp(min=0)
    corners = torch.tensor([[sx, sy] for sx in (-1, 0, 1) for sy in (-1, 0, 1)], device=box_xy.device) * TARGET_HALF
    square_points = (squares[:, :, None] + corners).flatten(1, 2)
    v = v + (square_clearance - _box_distance(square_points, box_xy, box_yaw, half).amin(dim=1)).clamp(min=0)
    a, b = (torch.tensor(p, device=box_xy.device).expand(len(box_xy), 2) for p in ROBOT_SEGMENT)
    v = v + (robot_clearance - _segment_point_distance(a, b, outline).amin(dim=1)).clamp(min=0)
    v = v + (_bearing(outline, view).amax(dim=1) - max_bearing).clamp(min=0)
    if mode == "side":
        # まとまりと目標領域を結ぶ線から離す (回り込みが要らない)
        v = v + (0.15 - _segment_point_distance(center, target_xy, outline).amin(dim=1)).clamp(min=0)
    return v


def randomize_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    tint: float,
    base_color: tuple[float, float, float],
    light_name: str = "Light",
):
    """シーンの照明 (DomeLight) の強さと色を変える (mode="reset")。色は base_color の各成分を ±tint の割合で変える"""
    for env_id in env_ids.tolist():
        prim = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[env_id]}/{light_name}")
        prim.GetAttribute("inputs:intensity").Set(_uniform(*intensity_range, 1, "cpu").item())
        scale = 1 + (torch.rand(3) * 2 - 1) * tint
        prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*[min(1.0, c * s.item()) for c, s in zip(base_color, scale)]))


def randomize_floor_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gray_range: tuple[float, float],
    warmth: float,
    floor_name: str = "Floor",
):
    """床の色 (PreviewSurface の diffuseColor) を変える (mode="reset")。灰色の明るさを gray_range から選び、
    赤を最大 warmth の割合で強め、青を最大 2 * warmth の割合で弱める (目標領域の青と紛れないように、青には寄せない)"""
    for env_id in env_ids.tolist():
        root = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[env_id]}/{floor_name}")
        assert root.IsValid(), f"prim not found: {floor_name} (env {env_id})"
        v = _uniform(*gray_range, 1, "cpu").item()
        w = torch.rand(1).item() * warmth
        color = Gf.Vec3f(min(1.0, v * (1 + w)), v, v * (1 - 2 * w))
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdShade.Shader):
                UsdShade.Shader(prim).GetInput("diffuseColor").Set(color)


def set_color(env: ManagerBasedEnv, name: str, env_id: int, color) -> None:
    """物体 name の見た目の色 (PreviewSurface の diffuseColor) を color (r, g, b) にする"""
    asset = env.scene[name]
    # シーンを作った後の prim_path は、{ENV_REGEX_NS} が env_regex_ns (/World/envs/env_.*) に置き換わっている
    root = env.scene.stage.GetPrimAtPath(asset.cfg.prim_path.replace(env.scene.env_regex_ns, env.scene.env_prim_paths[env_id]))
    assert root.IsValid(), f"prim not found: {asset.cfg.prim_path} (env {env_id})"
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdShade.Shader):
            UsdShade.Shader(prim).GetInput("diffuseColor").Set(Gf.Vec3f(*[float(c) for c in color]))


def randomize_color(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, color_range: dict[str, tuple[float, float]]
):
    """物体の見た目の色 (PreviewSurface の diffuseColor) を変える (mode="reset")

    上流の randomize_visual_color (replicator) は、項目ごとの乱数が同じ系列になり、ゴミがすべて同じ色になる。
    """
    for env_id in env_ids.tolist():
        set_color(env, asset_cfg.name, env_id, [_uniform(*color_range[c], 1, "cpu").item() for c in "rgb"])


def randomize_camera_offset(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, pose_range: dict[str, tuple[float, float]]
):
    """カメラの取り付け位置と向きを、設定値 (cfg.offset) のまわりでずらす (mode="reset")

    上流の randomize_camera_uniform は現在の姿勢に乱数を足すので、リセットのたびにずれが累積する。
    ここでは設定値を基準にする。cfg.offset は convention="opengl" の、親 (base か gripper) に対する姿勢。
    """
    cam = env.scene[asset_cfg.name]
    assert cam.cfg.offset.convention == "opengl", cam.cfg.offset.convention
    n, device = len(env_ids), env.device
    ranges = [pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    d = torch.stack([_uniform(lo, hi, n, device) for lo, hi in ranges], dim=-1)
    pos = torch.tensor(cam.cfg.offset.pos, device=device).repeat(n, 1) + d[:, :3]
    rot = quat_mul(torch.tensor(cam.cfg.offset.rot, device=device).repeat(n, 1), quat_from_euler_xyz(d[:, 3], d[:, 4], d[:, 5]))
    cam._view.set_local_poses(translations=pos, orientations=rot, indices=env_ids)


def debris_in_target(env: ManagerBasedEnv, square: str = TARGET_NAME) -> torch.Tensor:
    """各ゴミの中心が目標領域 (square を与えれば、その正方形) に入っているか [num_envs, NUM_DEBRIS] (bool)"""
    target_xy = env.scene[square].data.root_pos_w[:, :2]
    pos = torch.stack([env.scene[name].data.root_pos_w for name in DEBRIS_NAMES], dim=1)
    inside = ((pos[..., :2] - target_xy[:, None]).abs() < TARGET_HALF).all(dim=-1)
    return inside & (pos[..., 2] - env.scene.env_origins[:, None, 2] < IN_TARGET_MAX_Z)


def sweep_success(env: ManagerBasedEnv, min_count: int = SUCCESS_COUNT) -> torch.Tensor:
    """目標領域に入ったゴミが min_count 個以上 (終了条件 success)"""
    return debris_in_target(env).sum(dim=-1) >= min_count
