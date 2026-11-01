"""Sweep の評価の状態の記録 (sweep_eval.py が書き、sweep_replay.py と解析が読む)。docs/sweep.md の「状態の記録」

out_dir に次を残す。
    run.json                 実行全体の構成 (バージョン、引数、ロボット、カメラ、ゴミ、ポリシー)。run_info
    env_cfg.yaml             env の設定のすべて (isaaclab の dump_yaml)
    states/round_<番号>.npz  ラウンドごとの状態 (np.savez_compressed)。RoundLog
npz のキー:
    setup/...   ラウンドの初めの状態。setup/init/<種類>/<名前>/<項目> は scene.get_state(is_relative=True)
                (env.reset_to にそのまま渡せる)。見た目と物理のランダム化の値 (照明、ゴミの色・質量・摩擦、
                カメラの取り付けの姿勢) も入る。capture_setup / apply_setup
    frame/...   30fps のフレームごとの値 (先頭の次元がフレーム)。観測を取った時点の状態と、その後に実行した action。
                位置は env の原点から (m)、姿勢は四元数 (w, x, y, z)。カメラの姿勢は opengl の向き (-z が視線)
    meta/...    ラウンドの番号と結果
最後のフレームの action の後の状態 (ラウンドの終わり) は、自動リセットで消えるので残らない。
読むだけなら isaaclab は要らない (load_round、load_run)。書く関数は env を受け取り、isaaclab は関数の中で import する。
"""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time

import numpy as np

FORMAT_VERSION = 1


def _np(x) -> np.ndarray:
    """コピーした numpy 配列。CPU の tensor の .numpy() は元とメモリを共有し、isaaclab のデータのバッファ
    (joint_pos など) は同じ場所に上書きされるので、コピーしないと後の値に変わる"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x, copy=True)


def _flatten(prefix: str, nested: dict, out: dict) -> dict:
    for k, v in nested.items():
        if isinstance(v, dict):
            _flatten(f"{prefix}/{k}", v, out)
        else:
            out[f"{prefix}/{k}"] = v
    return out


def unflatten(arrays: dict, prefix: str) -> dict:
    """load_round の結果から、prefix の下を入れ子の dict に戻す (setup/init を env.reset_to に渡すときなど)"""
    nested: dict = {}
    for key, value in arrays.items():
        if not key.startswith(prefix + "/"):
            continue
        node = nested
        *parents, leaf = key[len(prefix) + 1 :].split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return nested


# ---- 実行全体の構成 ----

def _versions() -> dict:
    out = {}
    for pkg in ("isaacsim", "isaaclab", "leisaac", "torch", "numpy"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    for name, path in (("leisaac_rev", "/workspace/leisaac"), ("isaaclab_rev", "/workspace/leisaac/dependencies/IsaacLab")):
        try:
            out[name] = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                                       check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            out[name] = None
    # このリポジトリの commit (Makefile が渡す。-dirty は未コミットの変更あり) と、イメージの中の tools の内容のハッシュ
    out["source_rev"] = os.environ.get("SOURCE_REV") or None
    tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digest = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(tools_dir)):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                path = os.path.join(root, f)
                digest.update(os.path.relpath(path, tools_dir).encode())
                digest.update(open(path, "rb").read())
    out["tools_sha256"] = digest.hexdigest()
    return out


def _camera_parent(cam) -> str:
    """カメラが付いているリンクの名前 (prim_path の 1 つ上。front は base、wrist は gripper)"""
    return cam.cfg.prim_path.rstrip("/").split("/")[-2]


def camera_names(env) -> list[str]:
    from isaaclab.sensors import TiledCamera

    return sorted(name for name, s in env.scene.sensors.items() if isinstance(s, TiledCamera))


def run_info(env, extra: dict) -> dict:
    """run.json の中身。extra (引数、シード、ポリシーなど) を足す"""
    robot = env.scene["robot"]
    cameras = {}
    for name in camera_names(env):
        cam = env.scene[name]
        spawn = cam.cfg.spawn
        cameras[name] = {
            "prim_path": cam.cfg.prim_path,
            "parent_body": _camera_parent(cam),
            "offset_pos": list(cam.cfg.offset.pos),
            "offset_rot_wxyz": list(cam.cfg.offset.rot),
            "offset_convention": cam.cfg.offset.convention,
            "width": cam.cfg.width,
            "height": cam.cfg.height,
            "update_period": cam.cfg.update_period,
            "focal_length": spawn.focal_length,
            "horizontal_aperture": spawn.horizontal_aperture,
            "vertical_aperture": spawn.vertical_aperture,
            "clipping_range": list(spawn.clipping_range),
            "intrinsic_matrix": _np(cam.data.intrinsic_matrices[0]).round(4).tolist(),
        }
    debris = {}
    for name in sorted(n for n in env.scene.rigid_objects if n.startswith("debris_")):
        spawn = env.scene[name].cfg.spawn
        debris[name] = {"size": list(spawn.size), "mass": spawn.mass_props.mass if spawn.mass_props else None}
    info = {
        "format_version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "versions": _versions(),
        "sim": {"physics_dt": env.physics_dt, "step_dt": env.step_dt, "decimation": env.cfg.decimation,
                "episode_length_s": env.cfg.episode_length_s, "device": env.device},
        "robot": {
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "default_joint_pos": _np(robot.data.default_joint_pos[0]).round(6).tolist(),
            "joint_stiffness": _np(robot.data.joint_stiffness[0]).round(6).tolist(),
            "joint_damping": _np(robot.data.joint_damping[0]).round(6).tolist(),
            "soft_joint_pos_limits": _np(robot.data.soft_joint_pos_limits[0]).round(6).tolist(),
        },
        "cameras": cameras,
        "debris": debris,
    }
    info.update(extra)
    return info


def write_run(out_dir: str, env, extra: dict) -> None:
    """run.json と env_cfg.yaml を書く"""
    from isaaclab.utils.io import dump_yaml

    with open(f"{out_dir}/run.json", "w") as f:
        json.dump(run_info(env, extra), f, indent=2, default=str)
    dump_yaml(f"{out_dir}/env_cfg.yaml", env.cfg)


def load_run(out_dir: str) -> dict:
    with open(f"{out_dir}/run.json") as f:
        return json.load(f)


# ---- ラウンドの初めの状態 (ランダム化の値を含む) ----

def _light_prim(env, env_id: int = 0):
    return env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[env_id]}/Light")


def _shaders(env, name: str, env_id: int = 0):
    from pxr import Usd, UsdShade

    asset = env.scene[name]
    root = env.scene.stage.GetPrimAtPath(asset.cfg.prim_path.replace(env.scene.env_regex_ns, env.scene.env_prim_paths[env_id]))
    return [UsdShade.Shader(p) for p in Usd.PrimRange(root) if p.IsA(UsdShade.Shader)]


def _debris_names(env) -> list[str]:
    return sorted((n for n in env.scene.rigid_objects if n.startswith("debris_")), key=lambda n: int(n.split("_")[1]))


def capture_setup(env, env_id: int = 0) -> dict[str, np.ndarray]:
    """ラウンドの初めの状態 (setup/...)。ゴミと目標領域の配置、ロボットの関節は init に、ランダム化の値はその他に入る"""
    out = {}
    state = env.scene.get_state(is_relative=True)
    for key, value in _flatten("setup/init", state, {}).items():
        out[key] = _np(value[env_id])
    light = _light_prim(env, env_id)
    out["setup/light/intensity"] = np.array(light.GetAttribute("inputs:intensity").Get(), np.float32)
    out["setup/light/color"] = np.array(light.GetAttribute("inputs:color").Get(), np.float32)
    names = _debris_names(env)
    out["setup/debris/color"] = np.array([_shaders(env, n, env_id)[0].GetInput("diffuseColor").Get() for n in names], np.float32)
    out["setup/debris/mass"] = np.stack([_np(env.scene[n].root_physx_view.get_masses())[env_id] for n in names])
    # [ゴミ, 形状, (静止摩擦, 動摩擦, 反発)]
    out["setup/debris/material"] = np.stack([_np(env.scene[n].root_physx_view.get_material_properties())[env_id] for n in names])
    for name in camera_names(env):
        pos, quat = env.scene[name]._view.get_local_poses()
        out[f"setup/camera/{name}/local_pos"] = _np(pos)[env_id].astype(np.float32)
        out[f"setup/camera/{name}/local_quat"] = _np(quat)[env_id].astype(np.float32)
    return out


def apply_setup(env, arrays: dict, env_id: int = 0) -> None:
    """capture_setup の状態に戻す (再生用)。env.reset_to の後に呼ぶ (reset のイベントがランダム化をやり直すため)"""
    import torch
    from pxr import Gf

    ids = torch.tensor([env_id], device=env.device)
    env.reset_to(unflatten({k: torch.as_tensor(v)[None].to(env.device) for k, v in arrays.items()}, "setup/init"),
                 ids, is_relative=True)
    light = _light_prim(env, env_id)
    light.GetAttribute("inputs:intensity").Set(float(arrays["setup/light/intensity"]))
    light.GetAttribute("inputs:color").Set(Gf.Vec3f(*arrays["setup/light/color"].tolist()))
    cpu_ids = torch.tensor([env_id])
    for i, name in enumerate(_debris_names(env)):
        for shader in _shaders(env, name, env_id):
            shader.GetInput("diffuseColor").Set(Gf.Vec3f(*arrays["setup/debris/color"][i].tolist()))
        view = env.scene[name].root_physx_view
        masses = view.get_masses()
        masses[env_id] = torch.as_tensor(arrays["setup/debris/mass"][i])
        view.set_masses(masses, cpu_ids)
        material = view.get_material_properties()
        material[env_id] = torch.as_tensor(arrays["setup/debris/material"][i])
        view.set_material_properties(material, cpu_ids)
    for name in camera_names(env):
        pos = torch.as_tensor(arrays[f"setup/camera/{name}/local_pos"], device=env.device)[None]
        quat = torch.as_tensor(arrays[f"setup/camera/{name}/local_quat"], device=env.device)[None]
        env.scene[name]._view.set_local_poses(translations=pos, orientations=quat, indices=ids)


# ---- フレームごとの値 ----

class RoundLog:
    """1 ラウンドの記録。begin でラウンドの初めの状態を取り、add をフレームごとに呼び、save で書く"""

    def __init__(self, env, env_id: int = 0):
        self.env, self.env_id = env, env_id
        robot = env.scene["robot"]
        self.i_base = robot.body_names.index("base")
        self.debris = _debris_names(env)
        self.cameras = {name: robot.body_names.index(_camera_parent(env.scene[name])) for name in camera_names(env)}

    def begin(self) -> None:
        self.setup = capture_setup(self.env, self.env_id)
        self.local = {name: (self.setup[f"setup/camera/{name}/local_pos"], self.setup[f"setup/camera/{name}/local_quat"])
                      for name in self.cameras}
        self.frames: dict[str, list] = {}

    def add(self, step: int, action, **values) -> None:
        """観測を取った時点の状態と、その後に実行する action (sim の action、9 次元)。values は呼ぶ側が足す値
        (目標領域の判定、chunk の先頭か、など)"""
        import torch
        from isaaclab.utils.math import quat_apply, quat_mul

        env, e = self.env, self.env_id
        origin = env.scene.env_origins[e]
        robot = env.scene["robot"]
        body_pos, body_quat = robot.data.body_pos_w[e], robot.data.body_quat_w[e]
        row = {
            "step": step,
            "time": step * env.physics_dt * env.cfg.decimation,
            "base_pos": body_pos[self.i_base] - origin,
            "base_quat": body_quat[self.i_base],
            "joint_pos": robot.data.joint_pos[e],
            "joint_vel": robot.data.joint_vel[e],
            "debris_pos": torch.stack([env.scene[n].data.root_pos_w[e] for n in self.debris]) - origin,
            "debris_quat": torch.stack([env.scene[n].data.root_quat_w[e] for n in self.debris]),
            "action": action,
        }
        for name, i in self.cameras.items():
            local_pos = torch.as_tensor(self.local[name][0], device=body_pos.device)
            local_quat = torch.as_tensor(self.local[name][1], device=body_pos.device)
            row[f"camera/{name}/pos"] = body_pos[i] + quat_apply(body_quat[i], local_pos) - origin
            row[f"camera/{name}/quat"] = quat_mul(body_quat[i], local_quat)
        row.update(values)
        for key, value in row.items():
            self.frames.setdefault(key, []).append(_np(value))

    def save(self, path: str, meta: dict) -> None:
        arrays = dict(self.setup)
        for key, values in self.frames.items():
            stacked = np.stack(values)
            arrays[f"frame/{key}"] = stacked.astype(np.float32) if stacked.dtype == np.float64 else stacked
        for key, value in meta.items():
            arrays[f"meta/{key}"] = np.asarray(value)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arrays)


def load_round(path: str) -> dict[str, np.ndarray]:
    """npz を読む (isaaclab は要らない)。キーは本文の docstring のとおり"""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
