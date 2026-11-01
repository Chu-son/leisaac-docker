"""レポートの図 (success_rate.svg) を作る。数値は docs/sweep_experiments.md の「全条件の比較」から写す。依存なし。

    python3 docs/report/make_figures.py
"""
import math
import os

# (ラベル 1 行目, 2 行目, 成功数, 採用したか)。30 ラウンド
ROWS = [
    ("1", "ACT 50", 7, True),
    ("3", "ACT 200", 17, True),
    ("4", "SmolVLA", 14, False),
    ("5", "DAgger", 12, False),
    ("6", "見失わない", 21, True),
    ("7", "DAgger", 14, False),
    ("8", "カメラ", 22, True),
    ("8'", "40 秒", 25, True),
    ("9", "後退なし", 6, False),
    ("10", "回り込み", 28, True),
]
N = 30
ADOPTED, REJECTED = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


W, H = 760, 330
L, R, T, B = 52, 16, 44, 62
pw, ph = W - L - R, H - T - B
y = lambda v: T + ph * (1 - v)  # noqa: E731
slot = pw / len(ROWS)
bw = 30
font = "font-family='Noto Sans CJK JP, sans-serif'"
out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' {font}>",
       f"<rect width='{W}' height='{H}' fill='{SURFACE}'/>"]
for v in (0, 0.25, 0.5, 0.75, 1.0):
    out.append(f"<line x1='{L}' x2='{W - R}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{AXIS if v == 0 else GRID}' stroke-width='1'/>")
    out.append(f"<text x='{L - 6}' y='{y(v) + 4:.1f}' font-size='11' fill='{MUTED}' text-anchor='end'>{int(v * 100)}%</text>")
for i, (num, name, k, adopted) in enumerate(ROWS):
    cx = L + slot * (i + 0.5)
    p = k / N
    lo, hi = wilson(k, N)
    color = ADOPTED if adopted else REJECTED
    top, base = y(p), y(0)
    r = min(4, (base - top) / 2)
    # 上の角だけ丸めた棒
    out.append(f"<path d='M{cx - bw / 2:.1f},{base:.1f} V{top + r:.1f} Q{cx - bw / 2:.1f},{top:.1f} {cx - bw / 2 + r:.1f},{top:.1f} "
               f"H{cx + bw / 2 - r:.1f} Q{cx + bw / 2:.1f},{top:.1f} {cx + bw / 2:.1f},{top + r:.1f} V{base:.1f} Z' fill='{color}'/>")
    out.append(f"<line x1='{cx:.1f}' x2='{cx:.1f}' y1='{y(lo):.1f}' y2='{y(hi):.1f}' stroke='{INK2}' stroke-width='1.5'/>")
    for v in (lo, hi):
        out.append(f"<line x1='{cx - 5:.1f}' x2='{cx + 5:.1f}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{INK2}' stroke-width='1.5'/>")
    out.append(f"<text x='{cx + bw / 2 + 2:.1f}' y='{min(top, y(hi)) - 4:.1f}' font-size='11' fill='{INK}' text-anchor='middle'>{k}/{N}</text>")
    out.append(f"<text x='{cx:.1f}' y='{H - B + 18}' font-size='12' fill='{INK}' text-anchor='middle'>{num}</text>")
    out.append(f"<text x='{cx:.1f}' y='{H - B + 34}' font-size='11' fill='{INK2}' text-anchor='middle'>{name}</text>")
out.append(f"<text x='{L}' y='18' font-size='13' fill='{INK}' font-weight='bold'>ACT などの成功率（30 ラウンド、ひげは 95% 信頼区間）</text>")
for j, (label, color) in enumerate((("採用した変更", ADOPTED), ("採らなかった変更", REJECTED))):
    lx = L + j * 130
    out.append(f"<rect x='{lx}' y='27' width='10' height='10' rx='2' fill='{color}'/>")
    out.append(f"<text x='{lx + 14}' y='36' font-size='11' fill='{INK2}'>{label}</text>")
out.append(f"<text x='{L}' y='{H - 6}' font-size='11' fill='{MUTED}'>実験の番号（本文の表）。8 までは制限時間 25 秒、8' 以降は 40 秒</text>")
out.append("</svg>")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "success_rate.svg"), "w") as f:
    f.write("\n".join(out) + "\n")
