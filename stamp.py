"""Tamper-evident results: hash-chain every CSV row, hash the model and the input.

  python stamp.py verify out/        re-checks out/counts.csv against out/stamp.json
"""
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_source(src):
    p = Path(src)
    if p.is_file():
        return sha_file(p)
    h = hashlib.sha256()
    for f in sorted(x for x in p.iterdir() if x.is_file()):
        h.update(f.name.encode() + sha_file(f).encode())
    return h.hexdigest()


def chain(rows):
    """rows: list of dicts. Returns the list of per-row hashes (last = head)."""
    prev, out = "0" * 64, []
    for r in rows:
        blob = json.dumps(r, sort_keys=True, separators=(",", ":")).encode()
        prev = hashlib.sha256(prev.encode() + blob).hexdigest()
        out.append(prev)
    return out


def write(out_dir, rows, model_path, source):
    pte = sorted(Path(model_path).glob("*.pte")) if Path(model_path).is_dir() else [Path(model_path)]
    hashes = chain(rows)
    stamp = {"model_sha256": sha_file(pte[0]), "source_sha256": sha_source(source),
             "rows": len(rows), "head": hashes[-1] if hashes else "0" * 64,
             "row_hashes": hashes}
    Path(out_dir, "stamp.json").write_text(json.dumps(stamp, indent=2))
    return stamp


def verify(out_dir):
    stamp = json.loads(Path(out_dir, "stamp.json").read_text())
    with open(Path(out_dir, "counts.csv")) as f:
        rows = list(csv.DictReader(f))
    hashes = chain(rows)
    want = stamp["row_hashes"]
    if hashes != want:
        bad = next((i for i, (a, b) in enumerate(zip(hashes, want)) if a != b), min(len(hashes), len(want)))
        print(f"TAMPERED: first mismatch at data row {bad + 1} "
              f"(csv has {len(rows)} rows, stamp has {stamp['rows']}).")
        return 1
    print(f"OK: {len(rows)} rows, head {stamp['head'][:16]}…, model {stamp['model_sha256'][:12]}…")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        sys.exit(verify(sys.argv[2]))
    print(__doc__)
