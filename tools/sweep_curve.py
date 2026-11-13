"""学習曲線の集計 (make sweep-curve の最後に実行する)。チェックポイントごとの評価 (sweep_eval.py の出力) を表にする

    python /opt/tools/sweep_curve.py <評価のディレクトリ> ... [--within 90] [--out curve.json]
- 複数の学習の合計は、ポリシーの種類と手本 (train_config.json の dataset.root) が同じものどうしで取る
- 引数はそれぞれ、評価の結果のディレクトリ (summary.json と rounds.jsonl がある) か、それを並べた親のディレクトリ
  (make sweep-curve の出力 data/sweep/curve/<TRAIN_NAME>/)。複数の学習 (シード違い) を並べると、チェックポイントごとに合計も出す
- ラウンドは成功した時点で終わるので、制限時間を延ばした評価からも「--within 秒以内の成功」を数えられる
  (docs/sweep_experiments.md の「S3a の追加の検証」)
- 標準ライブラリだけを使う
"""
import argparse
import json
import math
import os
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def eval_dirs(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if os.path.isfile(f"{p}/summary.json"):
            out.append(p)
        elif os.path.isdir(p):
            out += sorted(f"{p}/{d}" for d in os.listdir(p) if os.path.isfile(f"{p}/{d}/summary.json"))
    return out


def local_path(path: str, d: str) -> str:
    """コンテナの /workspace/data/... を、評価のディレクトリから見た data/ の下に読み替える (ホストで実行するとき)"""
    if os.path.exists(path) or not path.startswith("/workspace/data/"):
        return path
    return os.path.abspath(d).rsplit("/sweep/", 1)[0] + path[len("/workspace/data"):]


def train_info(policy_path: str | None, checkpoint: str | None, d: str) -> tuple[str, int]:
    """学習の手本 (dataset.root の名前) と step。checkpoint が last なら、リンク先の step にする"""
    if not policy_path:
        return "?", 0
    p = local_path(policy_path, d)
    dataset = "?"
    if os.path.isfile(f"{p}/train_config.json"):
        with open(f"{p}/train_config.json") as f:
            dataset = os.path.basename(json.load(f)["dataset"]["root"].rstrip("/"))
    step = os.path.basename(os.path.realpath(os.path.dirname(p))) if checkpoint == "last" else checkpoint
    return dataset, int(step) if str(step).isdigit() else 0


def tally(d: str, within: float) -> dict:
    with open(f"{d}/summary.json") as f:
        s = json.load(f)
    with open(f"{d}/rounds.jsonl") as f:
        rows = [json.loads(l) for l in f]
    server = s.get("server") or {}
    dataset, step = train_info(server.get("policy_path"), server.get("checkpoint"), d)
    row = {"dir": d, "policy_path": server.get("policy_path"), "policy_type": server.get("policy_type"),
           "dataset": dataset, "checkpoint": step,
           "seed": s["seed"], "rounds": len(rows), "time_limit": s["time_limit"],
           "ok_within": sum(r["success"] and r["sim_time"] <= within for r in rows),
           "ok_limit": sum(r["success"] for r in rows), "modes": {}}
    for r in rows:
        m = row["modes"].setdefault(r.get("obstacle_mode") or "all", [0, 0, 0])
        m[0] += r["success"] and r["sim_time"] <= within
        m[1] += r["success"]
        m[2] += 1
    return row


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paths", nargs="+")
parser.add_argument("--within", type=float, default=90.0, help="この秒数以内の成功も数える (既定 90 秒。タスクの既定の制限時間)")
parser.add_argument("--out", default=None, help="結果の JSON の出力先")
args = parser.parse_args()

rows = [tally(d, args.within) for d in eval_dirs(args.paths)]
assert rows, "評価の結果が見つからない"
w = f"{args.within:g}s"
print(f"| 学習 | step | 評価のシード | {w} 以内 | 制限時間内 | 型ごと ({w} / 制限時間内 / 数) | 評価 |")
print("|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["policy_path"] or "", r["checkpoint"], r["seed"])):
    name = (r["policy_path"] or "?").split("/train/")[-1].split("/checkpoints/")[0]
    modes = ", ".join(f"{k} {a}/{b}/{n}" for k, (a, b, n) in sorted(r["modes"].items()))
    print(f"| {name} | {r['checkpoint']} | {r['seed']} | {r['ok_within']}/{r['rounds']} | {r['ok_limit']}/{r['rounds']}"
          f"（{r['time_limit']:g}s） | {modes} | `{os.path.basename(r['dir'])}` |")
# 合計は、ポリシーの種類と手本が同じ学習 (シード違い) どうしでだけ取る
by_ckpt = defaultdict(list)
for r in rows:
    by_ckpt[(r["policy_type"] or "?", r["dataset"], r["checkpoint"], r["seed"])].append(r)
total = []
if any(len(v) > 1 for v in by_ckpt.values()):
    print(f"\n| ポリシー | 手本 | step | 評価のシード | 学習の数 | {w} 以内の合計 (95% CI) | 制限時間内の合計 |")
    print("|---|---|---|---|---|---|---|")
    for (ptype, dataset, ckpt, seed), v in sorted(by_ckpt.items()):
        k, n, kl = sum(x["ok_within"] for x in v), sum(x["rounds"] for x in v), sum(x["ok_limit"] for x in v)
        total.append({"policy_type": ptype, "dataset": dataset, "checkpoint": ckpt, "seed": seed, "runs": len(v),
                      "ok_within": k, "ok_limit": kl, "rounds": n, "ci95_within": wilson(k, n)})
        print(f"| {ptype} | {dataset} | {ckpt} | {seed} | {len(v)} | {k}/{n} {wilson(k, n)} | {kl}/{n} |")
if args.out:
    with open(args.out, "w") as f:
        json.dump({"within": args.within, "rows": rows, "total": total}, f, indent=2, ensure_ascii=False)
