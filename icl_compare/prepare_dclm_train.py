"""Build the DCLM training byte stream from the Hugging Face mirror.

    python prepare_dclm_train.py --out data/dclm_train --gbytes 4 --workers 8

Downloads ``*.jsonl.zst`` files from ``mlfoundations/dclm-baseline-1.0``,
round-robin over the 10 global shards, and writes each file's document text
(UTF-8, concatenated with no separator, like the eval bake) to
``<out>/part_<nnnnn>.bin``. It stops once ``--gbytes`` of text is written.

Train/eval disjointness: the eval corpus (``prepare_dclm_benchmark``) reads the
first file of ``local-shard_0_of_10`` in each global shard, so all of
``local-shard_0`` is excluded here.

Re-running resumes: existing parts are kept and counted.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

REPO = "mlfoundations/dclm-baseline-1.0"


def list_files(token) -> list[str]:
    """Training files, interleaved across global shards (skipping local-shard_0)."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    per_global = []
    for g in range(1, 11):
        gdir = f"global-shard_{g:02d}_of_10"
        fs = []
        for ls in range(1, 10):
            ldir = f"{gdir}/local-shard_{ls}_of_10"
            fs += sorted(x.path for x in api.list_repo_tree(REPO, repo_type="dataset",
                                                            path_in_repo=ldir)
                         if x.path.endswith(".jsonl.zst"))
            if len(fs) >= 40:
                break                       # plenty per global shard
        per_global.append(fs)
    out = []
    for i in range(max(len(f) for f in per_global)):
        out += [f[i] for f in per_global if i < len(f)]
    return out


def process(job) -> tuple[int, int, float]:
    """Download one file, write its text to part_<idx>.bin; return (idx, bytes, s)."""
    idx, path, out_dir, token = job
    import zstandard as zstd
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    dst = Path(out_dir) / f"part_{idx:05d}.bin"
    tmp_dir = Path(out_dir) / "_dl" / str(idx)
    local = hf_hub_download(REPO, path, repo_type="dataset", token=token,
                            local_dir=tmp_dir)
    n = 0
    tmp = dst.with_suffix(".tmp")
    with open(local, "rb") as fh, open(tmp, "wb") as out:
        reader = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        buf = []
        for line in reader:
            b = json.loads(line).get("text", "").encode("utf-8", errors="replace")
            buf.append(b); n += len(b)
            if len(buf) >= 2048:
                out.write(b"".join(buf)); buf = []
        out.write(b"".join(buf))
    tmp.replace(dst)
    for p in sorted(tmp_dir.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp_dir.rmdir()
    return idx, n, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/dclm_train"))
    ap.add_argument("--gbytes", type=float, required=True, help="target GB of text")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    from common import hf_token
    token = hf_token()
    args.out.mkdir(parents=True, exist_ok=True)
    target = int(args.gbytes * 1e9)
    have = {int(p.stem.split("_")[1]): p.stat().st_size for p in args.out.glob("part_*.bin")}
    total = sum(have.values())
    print(f"existing: {len(have)} parts, {total / 1e9:.2f} GB; target {target / 1e9:.2f} GB")
    if total >= target:
        return
    files = list_files(token)
    jobs = [(i, f, str(args.out), token) for i, f in enumerate(files) if i not in have]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for idx, n, dt in pool.imap_unordered(process, jobs):
            total += n
            print(f"part {idx:5d}: {n / 1e6:7.1f} MB in {dt:5.0f}s | total "
                  f"{total / 1e9:6.2f} GB | {total / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
            if total >= target:
                pool.terminate()
                break
    (args.out / "_dl").exists() and __import__("shutil").rmtree(args.out / "_dl", ignore_errors=True)
    for p in args.out.glob("*.tmp"):
        p.unlink()
    print(f"done: {total / 1e9:.2f} GB in {len(list(args.out.glob('part_*.bin')))} parts")


if __name__ == "__main__":
    main()
