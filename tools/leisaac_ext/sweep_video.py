"""Sweep タスクの上から見た図と動画 (sweep_inspect.py と sweep_eval.py の --videos で共通)。isaaclab を import しない

位置はすべて env の原点からの m (x は右、y は前)。動画は front、wrist、上から見た図を横に並べ、30fps (実時間) で書く
"""
import subprocess

import cv2
import numpy as np

from leisaac_ext import contract

# sweep_mdp.TARGET_HALF と sweep_planner.BASE_RADIUS と同じ値 (どちらも isaaclab を読み込むので、Kit なしでは import できない)
TARGET_HALF = 0.15
BASE_RADIUS = 0.165
# 上から見た図の範囲 (env の原点からの m) と縮尺。配置の範囲 (sweep_floor) がすべて入る
MAP_X = (-1.2, 1.2)
MAP_Y = (-0.4, 2.6)
MAP_SCALE = 80  # ピクセル/m
MAP_SIZE = (round((MAP_Y[1] - MAP_Y[0]) * MAP_SCALE), round((MAP_X[1] - MAP_X[0]) * MAP_SCALE))  # (高さ, 幅)


def to_px(xy: np.ndarray) -> np.ndarray:
    """env の原点からの位置 (m) -> 図のピクセル (x は右、y は上)"""
    return np.stack([(xy[..., 0] - MAP_X[0]) * MAP_SCALE, (MAP_Y[1] - xy[..., 1]) * MAP_SCALE], axis=-1).round().astype(np.int32)


def draw_map(debris: np.ndarray, debris_start: np.ndarray, inside: np.ndarray, target: np.ndarray,
             base_path: np.ndarray, theta: float, label: str) -> np.ndarray:
    """上から見た図 (RGB)。debris と debris_start は [N, 2]、inside は [N] (目標領域に入ったか)、base_path は台車の通り道 [T, 2]
    (最後が今の位置)。目標領域は青、ゴミは目標領域の中なら緑、外なら赤、最初の位置は灰色の点、台車の通り道は黄色"""
    img = np.full((*MAP_SIZE, 3), 245, np.uint8)
    for v in np.arange(np.ceil(MAP_X[0] * 2) / 2, MAP_X[1], 0.5):
        cv2.line(img, tuple(to_px(np.array([v, MAP_Y[0]]))), tuple(to_px(np.array([v, MAP_Y[1]]))), (225, 225, 225), 1)
    for v in np.arange(np.ceil(MAP_Y[0] * 2) / 2, MAP_Y[1], 0.5):
        cv2.line(img, tuple(to_px(np.array([MAP_X[0], v]))), tuple(to_px(np.array([MAP_X[1], v]))), (225, 225, 225), 1)
    lo, hi = to_px(target + [-TARGET_HALF, TARGET_HALF]), to_px(target + [TARGET_HALF, -TARGET_HALF])
    cv2.rectangle(img, tuple(lo), tuple(hi), (150, 180, 255), -1)
    cv2.polylines(img, [to_px(base_path)], False, (230, 190, 0), 1)
    center = to_px(base_path[-1])
    cv2.circle(img, tuple(center), round(BASE_RADIUS * MAP_SCALE), (120, 120, 120), 1)
    heading = np.array([-np.sin(theta), np.cos(theta)]) * BASE_RADIUS
    cv2.line(img, tuple(center), tuple(to_px(base_path[-1] + heading)), (120, 120, 120), 1)
    for p in to_px(debris_start):
        cv2.circle(img, tuple(p), 1, (160, 160, 160), -1)
    for p, ok in zip(to_px(debris), inside):
        cv2.rectangle(img, tuple(p - 2), tuple(p + 2), (0, 160, 0) if ok else (220, 0, 0), -1)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def overview(maps: list[np.ndarray], cols: int = 10) -> np.ndarray:
    """上から見た図を cols 列に並べる (RGB)"""
    rows = [np.concatenate(maps[k : k + cols] + [np.full_like(maps[0], 255)] * (cols - len(maps[k : k + cols])), axis=1)
            for k in range(0, len(maps), cols)]
    return np.concatenate(rows, axis=0)


class VideoWriter:
    """front、wrist、上から見た図を横に並べたフレームを、H.264 の mp4 に書く (30fps、2 倍に拡大)。1 フレームずつ ffmpeg に渡す"""

    def __init__(self, path: str):
        self.path = path
        h, w = MAP_SIZE[0], contract.IMAGE_SHAPE[1] * 2 + MAP_SIZE[1]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(contract.FPS), "-i", "-", "-vf", "scale=iw*2:ih*2:flags=neighbor",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, front: np.ndarray, wrist: np.ndarray, map_img: np.ndarray) -> None:
        self.proc.stdin.write(np.concatenate([front, wrist, map_img], axis=1).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        assert self.proc.wait() == 0, f"ffmpeg failed: {self.path}"
