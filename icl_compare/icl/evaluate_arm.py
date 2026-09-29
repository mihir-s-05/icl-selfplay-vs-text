"""Evaluate every checkpoint of one arm and size, logging ICL curves to W&B.

    python -m icl.evaluate_arm --arm sp --size 24M --n-seeds 4
    python -m icl.evaluate_arm --arm dclm --size 1M

Finds ``ckpts/<arm>/<size>/seed-*/learner_<tag>.pth`` and groups files by tag
(a self-play round, or a DCLM token count). Seeds sharing a tag are scored
together, giving single-seed and ensemble metrics. Each group is written to
``results/icl/<arm>/<size>/<tag>.json`` and skipped on re-runs.

W&B: one run per (arm, size), named ``icl-<arm>-<size>`` in group ``icl``,
with x-axis ``tokens_B`` (learner tokens seen, in billions):
  ``icl/<section>/<task>``        seed-mean accuracy of every cell (per-byte for
                                  teacher-forced cells, as in the paper)
  ``icl_exact/<section>/<task>``  whole-answer accuracy of teacher-forced cells
  ``icl_ens/...``                 the same for the seed ensemble (several seeds)
  ``summary/<suite>_<task>``      headline cells (Fig. 4 tasks at m = 32-128)
The JSONs are uploaded as an artifact at the end.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, sp_tokens, wandb_init  # noqa: E402
from icl.run import evaluate  # noqa: E402

# Universal-prior learners train on a different number of programs per round
# (figures/fig2_transfer_across_modalities/data/prior_programs_per_round.json).
UP_PROGRAMS = {"100k": 1024, "500k": 2048, "1M": 1024, "3M": 2048, "6M": 1024}

# Headline cells: (suite, section, tag), one per Fig. 4 task (+ the new ones).
HEADLINE = {
    "raw_reverse": ("v4", "palin|m=32|k=8"),
    "raw_stack": ("v4", "stack|m=32|L=4"),
    "raw_assoc": ("v3", "assoc|m=128|V=16"),
    "raw_sum": ("sweep", "sum|64|2"),
    "raw_max": ("sweep", "max|64|2"),
    "raw_min": ("sweep", "min|64|2"),
    "raw_first": ("sweep", "first|64|2"),
    "raw_last": ("sweep", "last|64|2"),
    "text_reverse": ("printable", "reverse|m=32|k=8"),
    "text_stack": ("printable", "stack|m=32|L=4"),
    "text_assoc": ("printable", "assoc|extra=112|V=16"),
    "text_sum": ("printable", "sum|m=64|k=2"),
    "text_max": ("printable", "max|m=64|k=2"),
    "text_min": ("printable", "min|m=64|k=2"),
    "text_first": ("printable", "first|m=64|k=2"),
    "text_last": ("printable", "last|m=64|k=2"),
    "text_cipher": ("printable", "cipher|m=64|k=4"),
    "word_cls": ("words", "wordcls|m=32|C=2"),
    "word_cls4": ("words", "wordcls|m=32|C=4"),
    "word_seen": ("words", "wordseen|m=32|C=4"),
    "word_rev": ("words", "wordrev|m=32|n=3"),
}
# multi-byte answers in the text suites count only when every byte is right
EXACT = ("text_sum", "text_reverse", "text_cipher", "word_rev")


def key(section: str, tag: str) -> str:
    return f"{section}/" + re.sub(r"[^A-Za-z0-9_.]+", "_", tag.replace("=", "")).strip("_")


def primary(m: dict) -> float:
    return m.get("acc_mean", m["acc"])


def seed_mean(entry: dict, field=primary) -> float:
    return float(np.mean([field(v) for v in entry["per_seed"].values()]))


def tokens_of(arm: str, size: str, tag: str, files: list[Path]) -> int:
    if arm == "sp":
        return sp_tokens(int(tag))
    if arm == "up":
        return UP_PROGRAMS[size] * 4096 * (int(tag) + 1)
    blob = torch.load(files[0], map_location="cpu", weights_only=False)
    return int(blob["tokens"])


def flatten(results: dict) -> dict:
    out = {}
    for section, cells in results.items():
        for tag, e in cells.items():
            k = key(section, tag)
            out[f"icl/{k}"] = seed_mean(e)
            if "acc_mean" in next(iter(e["per_seed"].values())):
                out[f"icl_exact/{k}"] = seed_mean(e, lambda m: m["acc"])
            if "ensemble" in e:
                out[f"icl_ens/{k}"] = primary(e["ensemble"])
    for name, (section, tag) in HEADLINE.items():
        if tag in results.get(section, {}):
            e = results[section][tag]
            exact = name in EXACT
            out[f"summary/{name}"] = seed_mean(e, (lambda m: m["acc"]) if exact else primary)
    for suite in ("raw", "text", "word"):
        vals = [v for k, v in out.items() if k.startswith(f"summary/{suite}_")]
        if vals:
            out[f"summary/{suite}_mean"] = float(np.mean(vals))
    return out


def write_json(path: Path, doc: dict) -> None:
    """Atomic write, so an interrupted run never leaves a truncated result."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, help="sp | up | dclm | sp2dclm")
    ap.add_argument("--size", required=True)
    ap.add_argument("--ckpt-root", type=Path, default=ROOT / "ckpts")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results/icl")
    ap.add_argument("--suite", default="fig4,printable")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--seed-glob", default="seed-*",
                    help="which seed dirs to score, e.g. seed-1 for an extra DCLM seed")
    ap.add_argument("--run-name", default=None, help="W&B run name (default icl-<arm>-<size>)")
    ap.add_argument("--tags", default="", help="comma list to restrict to (default all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-len", type=int, default=None, help="for quick CPU tests")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    seed_dirs = sorted((args.ckpt_root / args.arm / args.size).glob(args.seed_glob))[:args.n_seeds]
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in seed_dirs:
        for f in d.glob("learner_*.pth"):
            groups[f.stem.removeprefix("learner_")].append(f)
    if args.tags:
        groups = {t: groups[t] for t in args.tags.split(",") if t in groups}
    if not groups:
        raise SystemExit(f"no checkpoints under {args.ckpt_root / args.arm / args.size}")
    order = sorted(groups, key=lambda t: tokens_of(args.arm, args.size, t, groups[t]))

    out_dir = args.out_root / args.arm / args.size
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = wandb_init(args.wandb, out_dir / "wandb_id",
                    name=args.run_name or f"icl-{args.arm}-{args.size}",
                    group="icl", job_type="icl-eval", tags=[args.arm, args.size, "icl"],
                    config={"arm": args.arm, "size": args.size, "suite": args.suite,
                            "n_seeds": len(seed_dirs), "bf16": args.bf16,
                            "max_len": args.max_len})
    if wb:
        wb.define_metric("tokens_B")
        for pat in ("icl/*", "icl_exact/*", "icl_ens/*", "summary/*"):
            wb.define_metric(pat, step_metric="tokens_B")

    device = torch.device(args.device)
    for tag in order:
        files = sorted(groups[tag])
        tokens = tokens_of(args.arm, args.size, tag, files)
        out = out_dir / f"{tag}.json"
        doc = read_json(out) if out.exists() else None
        if doc is None:
            print(f"== {args.arm}/{args.size} {tag} ({tokens / 1e9:.3f}B tokens, "
                  f"{len(files)} seed(s))", flush=True)
            results, meta, _ = evaluate([str(f) for f in files], args.suite, device,
                                        max_len=args.max_len, bf16=args.bf16, verbose=False)
            meta.update(arm=args.arm, size=args.size, tag=tag, tokens=tokens)
            doc = {"meta": meta, "results": results, "wandb_logged": False}
            write_json(out, doc)
            print(f"   {meta['seconds']:.0f}s -> {out}", flush=True)
        if wb and not doc.get("wandb_logged"):
            flat = flatten(doc["results"])
            wb.log({"tokens_B": tokens / 1e9, **flat})
            print("   " + "  ".join(f"{k.split('/')[1]}={v:.2f}" for k, v in flat.items()
                                     if k.startswith("summary/")), flush=True)
            doc["wandb_logged"] = True
            write_json(out, doc)
    if wb:
        import wandb
        art = wandb.Artifact(re.sub(r"[^A-Za-z0-9_.-]", "-", args.run_name or
                                    f"icl-{args.arm}-{args.size}"), type="icl-results")
        art.add_dir(str(out_dir))
        wb.log_artifact(art)
        wb.finish()


if __name__ == "__main__":
    main()
