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
):
    """ゴミのまとまりと目標領域を、ランダムな位置に置く (mode="reset")

    ゴミは、まとまりの中心のまわりの格子 (grid の個数、間隔 pitch。格子全体もランダムに回す) から NUM_DEBRIS 個の
    升目を選び、jitter だけずらして置く。pitch は、隣のゴミと重ならない間隔にする (3cm 角で、ずれが ±1.5cm なら 8cm)。
    目標領域の中心は target_range の中で、まとまりの中心から target_distance の距離に置く。
    さらに、ゴミすべてと目標領域の四隅が、view_origin (front カメラの位置) から見て ±max_bearing 以内に入るようにする
    (最初の画像に両方が写る)。配置の候補をまとめて引き、条件を満たす最初のものを使う (なければ最も条件に近いもの)。
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

        dist = (target_xy - center).norm(dim=-1)
        lo, hi = target_distance
        bearing = torch.cat([_bearing(debris_xy, view), _bearing(target_xy[:, None] + corners, view)], dim=1)
        violation = (lo - dist).clamp(min=0) + (dist - hi).clamp(min=0) + (bearing.amax(dim=1) - max_bearing).clamp(min=0)
        best = violation.argmin()

        origin = env.scene.env_origins[env_id]
        ids = torch.tensor([env_id], device=device)
        yaws = _uniform(0.0, 2 * math.pi, NUM_DEBRIS, device)
        for name, xy, yaw in zip(DEBRIS_NAMES, debris_xy[best], yaws):
            obj = env.scene[name]
            pos = origin + torch.cat([xy, zero + obj.cfg.spawn.size[2] / 2])
            quat = quat_from_euler_xyz(zero, zero, yaw.view(1))
            obj.write_root_pose_to_sim(torch.cat([pos.view(1, 3), quat], -1), env_ids=ids)
            obj.write_root_velocity_to_sim(zero_vel, env_ids=ids)

        target = env.scene[TARGET_NAME]
        pos = origin + torch.cat([target_xy[best], zero + target.cfg.spawn.size[2] / 2])
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        target.write_root_pose_to_sim(torch.cat([pos.view(1, 3), quat], -1), env_ids=ids)


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


def randomize_color(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, color_range: dict[str, tuple[float, float]]
):
    """物体の見た目の色 (PreviewSurface の diffuseColor) を変える (mode="reset")

    上流の randomize_visual_color (replicator) は、項目ごとの乱数が同じ系列になり、ゴミがすべて同じ色になる。
    """
    asset = env.scene[asset_cfg.name]
    for env_id in env_ids.tolist():
        # シーンを作った後の prim_path は、{ENV_REGEX_NS} が env_regex_ns (/World/envs/env_.*) に置き換わっている
        root = env.scene.stage.GetPrimAtPath(asset.cfg.prim_path.replace(env.scene.env_regex_ns, env.scene.env_prim_paths[env_id]))
        assert root.IsValid(), f"prim not found: {asset.cfg.prim_path} (env {env_id})"
        color = Gf.Vec3f(*[_uniform(*color_range[c], 1, "cpu").item() for c in "rgb"])
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdShade.Shader):
                UsdShade.Shader(prim).GetInput("diffuseColor").Set(color)


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


def debris_in_target(env: ManagerBasedEnv) -> torch.Tensor:
    """各ゴミの中心が目標領域に入っているか [num_envs, NUM_DEBRIS] (bool)"""
    target_xy = env.scene[TARGET_NAME].data.root_pos_w[:, :2]
    pos = torch.stack([env.scene[name].data.root_pos_w for name in DEBRIS_NAMES], dim=1)
    inside = ((pos[..., :2] - target_xy[:, None]).abs() < TARGET_HALF).all(dim=-1)
    return inside & (pos[..., 2] - env.scene.env_origins[:, None, 2] < IN_TARGET_MAX_Z)


def sweep_success(env: ManagerBasedEnv, min_count: int = SUCCESS_COUNT) -> torch.Tensor:
    """目標領域に入ったゴミが min_count 個以上 (終了条件 success)"""
    return debris_in_target(env).sum(dim=-1) >= min_count
