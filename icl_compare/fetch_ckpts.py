"""Download a subset of the released learners into the shared checkpoint layout.

    python fetch_ckpts.py --arm sp --sizes 1M,3M,6M,24M \\
        --rounds 0,256,512,1024,2048,2816,8191 --n-seeds 4

Arms: ``sp`` (self-play ladder, ``<size>/``) and ``up`` (universal-prior
baseline, ``baselines/uniform-prior/<size>/``, sizes up to 6M). Files land in
``ckpts/<arm>/<size>/seed-<seed>/{config.json,learner_<round>.pth}``, the
layout ``icl.run`` and ``train.py --init-from`` read. Seeds are taken in
sorted order, starting with the paper's ICL seeds 40354564-67. Uses
HF_TOKEN / HUGGINGFACE_API_KEY if set.

Released rounds are 0, 256, 512, ..., 7936 and 8191 (2816 is the paper's ICL
point). The default set is ``common.SP_ROUNDS``.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from common import HF_REPO, ROOT, SP_ROUNDS, hf_token

PREFIX = {"sp": "{size}", "up": "baselines/uniform-prior/{size}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(PREFIX), default="sp")
    ap.add_argument("--sizes", default="1M,3M,6M,24M")
    ap.add_argument("--rounds", default=",".join(map(str, SP_ROUNDS)),
                    help="comma list or 'all'")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "ckpts")
    args = ap.parse_args()

    token = hf_token()
    files = HfApi(token=token).list_repo_files(HF_REPO)
    want_rounds = None if args.rounds == "all" else {int(r) for r in args.rounds.split(",")}
    for size in args.sizes.split(","):
        prefix = PREFIX[args.arm].format(size=size)
        pat = re.compile(rf"^{re.escape(prefix)}/(seed-\d+)/learner_(\d+)\.pth$")
        by_seed: dict[str, list[tuple[int, str]]] = {}
        for f in files:
            m = pat.match(f)
            if m and (want_rounds is None or int(m.group(2)) in want_rounds):
                by_seed.setdefault(m.group(1), []).append((int(m.group(2)), f))
        seeds = sorted(by_seed)[:args.n_seeds]
        if not seeds:
            print(f"[warn] no checkpoints for {args.arm}/{size} rounds={args.rounds}")
            continue
        for seed in seeds:
            dst_dir = args.out / args.arm / size / seed
            dst_dir.mkdir(parents=True, exist_ok=True)
            cfg = f"{prefix}/{seed}/config.json"
            if cfg in files and not (dst_dir / "config.json").exists():
                shutil.copyfile(hf_hub_download(HF_REPO, cfg, token=token), dst_dir / "config.json")
            got = []
            for rnd, f in sorted(by_seed[seed]):
                dst = dst_dir / f"learner_{rnd}.pth"
                if not dst.exists():
                    shutil.copyfile(hf_hub_download(HF_REPO, f, token=token), dst)
                got.append(rnd)
            missing = sorted((want_rounds or set()) - set(got))
            print(f"{args.arm}/{size}/{seed}: rounds {got}"
                  + (f"  (not released: {missing})" if missing else ""))


if __name__ == "__main__":
    main()
