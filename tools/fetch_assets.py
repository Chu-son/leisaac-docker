#!/usr/bin/env python3
"""assets.json に書かれた外部リソース (USD アセット、学習済みポリシー) を取得・検証する。

    python3 tools/fetch_assets.py                    # 足りないものを取得し、すべて検証する (make assets)
    python3 tools/fetch_assets.py --check            # 取得せず、検証だけする (ネットワーク不要)
    python3 tools/fetch_assets.py --group pick-orange [--group ...]   # 対象の group を絞る
    python3 tools/fetch_assets.py --tree-hash DIR    # ディレクトリのハッシュを表示する (assets.json の作成用)

- 既にあるものは上書きしない。ハッシュが一致しなければ、削除せずに報告する。
- 手動で入手するもの (kind: manual) は、置かれているかだけを確認する。
- ホストの python3 (標準ライブラリのみ) で動く。
"""
import argparse
import hashlib
import http.client
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "assets.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(directory):
    """ディレクトリ内の全ファイルの (相対パス, sha256) から、1 つのハッシュを作る。"""
    h = hashlib.sha256()
    for p in sorted(x for x in Path(directory).rglob("*") if x.is_file()):
        h.update(f"{p.relative_to(directory).as_posix()}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


def download(url, dest, progress=True, attempts=4):
    """url を dest.part に保存して返す (dest への移動とハッシュの検証は呼び出し側で行う)。一時的な接続エラーは再試行する。"""
    part = Path(str(dest) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "leisaac-docker"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if progress and total and sys.stderr.isatty():
                        print(f"\r    {done >> 20} / {total >> 20} MiB", end="", file=sys.stderr)
            break
        except (OSError, http.client.HTTPException):
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    if progress and sys.stderr.isatty():
        print(file=sys.stderr)
    return part


def fetch_verified(url, dest, expected):
    part = download(url, dest)
    actual = sha256_file(part)
    if actual != expected:
        part.unlink()
        raise RuntimeError(f"ハッシュが一致しません: {url}\n    期待 {expected}\n    実際 {actual}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part.replace(dest)


def process_file(e, check_only):
    dest = ROOT / e["dest"]
    if dest.exists():
        return "ok" if sha256_file(dest) == e["sha256"] else "mismatch"
    if check_only:
        return "missing"
    fetch_verified(e["url"], dest, e["sha256"])
    return "fetched"


def process_zip(e, check_only):
    target = ROOT / e["tree"]["path"]
    if target.exists():
        return "ok" if tree_hash(target) == e["tree"]["sha256"] else "mismatch"
    if check_only:
        return "missing"
    with tempfile.TemporaryDirectory(dir=ROOT / e["dest"].split("/")[0]) as tmp:
        tmp = Path(tmp)
        archive = tmp / "archive.zip"
        fetch_verified(e["url"], archive, e["sha256"])
        out = tmp / "out"
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (out / name).resolve().is_relative_to(out.resolve()):
                    raise RuntimeError(f"zip に不正なパスがあります: {name}")
            z.extractall(out)
        extracted = out / target.name
        if tree_hash(extracted) != e["tree"]["sha256"]:
            raise RuntimeError(f"展開した内容のハッシュが一致しません: {e['dest']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(target))
    return "fetched"


def process_hf(e, check_only):
    base = ROOT / e["dest"]
    states = []
    for rel, sha in e["files"].items():
        dest = base / rel
        if dest.exists():
            states.append("ok" if sha256_file(dest) == sha else "mismatch")
        elif check_only:
            states.append("missing")
        else:
            url = f"https://huggingface.co/{e['repo']}/resolve/{e['revision']}/{rel}"
            fetch_verified(url, dest, sha)
            states.append("fetched")
    for s in ("mismatch", "missing", "fetched"):
        if s in states:
            return s
    return "ok"


def process_hf_dir(e, check_only):
    """HF リポジトリのディレクトリを、revision を固定して丸ごと取得する。

    LFS のファイルは API が返す sha256 でファイルごとに検証し、全体はツリーハッシュで検証する。
    """
    target = ROOT / e["tree"]["path"]
    if target.exists():
        return "ok" if tree_hash(target) == e["tree"]["sha256"] else "mismatch"
    if check_only:
        return "missing"
    api = f"https://huggingface.co/api/models/{e['repo']}/tree/{e['revision']}/{e['prefix']}?recursive=true"
    req = urllib.request.Request(api, headers={"User-Agent": "leisaac-docker"})
    with urllib.request.urlopen(req) as resp:
        files = [f for f in json.load(resp) if f["type"] == "file"]
    with tempfile.TemporaryDirectory(dir=ROOT / e["dest"].split("/")[0]) as tmp:
        tmp = Path(tmp)
        out = tmp / "out"

        def one(f):
            rel = f["path"][len(e["prefix"]):].lstrip("/")
            url = f"https://huggingface.co/{e['repo']}/resolve/{e['revision']}/{f['path']}"
            dest = out / rel
            part = download(url, dest, progress=False)
            lfs = (f.get("lfs") or {}).get("oid")
            if lfs and sha256_file(part) != lfs:
                raise RuntimeError(f"ハッシュが一致しません: {f['path']}")
            part.replace(dest)

        with ThreadPoolExecutor(max_workers=8) as ex:
            for _ in ex.map(one, files):
                pass
        if tree_hash(out) != e["tree"]["sha256"]:
            raise RuntimeError(f"取得した内容のツリーハッシュが一致しません: {e['tree']['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(target))
    return "fetched"


def process_manual(e, check_only):
    dest = ROOT / e["dest"]
    if not dest.exists():
        return "manual"
    if e.get("sha256"):
        return "ok" if sha256_file(dest) == e["sha256"] else "mismatch"
    return "ok"


HANDLERS = {"file": process_file, "zip": process_zip, "hf": process_hf, "hf-dir": process_hf_dir, "manual": process_manual}
LABEL = {
    "ok": "[OK]    ",
    "fetched": "[取得]  ",
    "missing": "[未取得]",
    "mismatch": "[不一致]",
    "manual": "[手動]  ",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="取得せず検証だけする")
    ap.add_argument("--group", action="append", help="対象の group (複数指定可)")
    ap.add_argument("--tree-hash", metavar="DIR", help="ディレクトリのハッシュを表示して終了する")
    args = ap.parse_args()

    if args.tree_hash:
        print(tree_hash(args.tree_hash))
        return 0

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["assets"]
    if args.group:
        entries = [e for e in entries if e["group"] in args.group]
    failed = False
    for e in entries:
        try:
            state = HANDLERS[e["kind"]](e, args.check)
        except Exception as ex:  # 1 つの失敗で他の確認を止めない
            print(f"[失敗]   {e['dest']}: {ex}")
            failed = True
            continue
        shown = e["tree"]["path"] if e["kind"] in ("zip", "hf-dir") else e["dest"]
        print(f"{LABEL[state]} {shown}  ({e['group']})")
        if state == "mismatch":
            print("         内容が期待と異なります。削除はしません。入れ替えるなら手で消してから再実行してください")
            failed = True
        elif state == "missing":
            failed = True
        elif state == "manual":
            print(f"         {e['note']}")
            failed = failed or args.check
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
