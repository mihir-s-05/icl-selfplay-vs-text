"""Flatten the experiment's results into tidy CSVs for the analysis and figures.

    python writeup/figs/extract.py

Reads ``icl_compare/results_final``, which is the Hugging Face dataset
``rihim/icl-selfplay-vs-text-results``:

    hf download rihim/icl-selfplay-vs-text-results --repo-type dataset         --local-dir icl_compare/results_final

plus the paper's own self-play DCLM scores (``self_play_pretraining/figures/
table2_scaling_exponents``), and writes into ``writeup/figs/data/``:

  headline.csv  arm,size,tokens,suite,seed,value   suite means per seed
  tasks.csv     arm,size,tokens,suite,task,seed,value   headline cells per seed
  mcurves.csv   arm,size,tokens,suite,task,m,seed,value  accuracy vs examples
  valbpb.csv    arm,size,seed,tokens,val_bpb        our DCLM / warm-start runs
  sp_bpb.csv    size,round,tokens,dclm_bpb          self-play learners, zero-shot DCLM
                                                     (the paper's scored ladder, K=1 mean)

Scoring conventions (as in the paper for the raw suite): greedy exact match;
multi-byte printable/word answers count only when every byte is right; the raw
reverse-string task uses per-position accuracy.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "icl_compare"))
from analysis.report import FAMILIES, load, val  # noqa: E402
from common import ARCH  # noqa: E402

RESULTS = PROJECT / "icl_compare/results_final/results"
LOGS = PROJECT / "icl_compare/results_final/training_logs"
PAPER_TRAJ = PROJECT / "self_play_pretraining/figures/table2_scaling_exponents/frontier_traj_perk.json"
OUT = HERE / "data"

# Headline cells: one per task, at the m the paper's Fig. 4 plots (raw) or the
# comparable m (printable, word). field = which accuracy defines "correct".
TASKS = {
    "raw": {"reverse": ("v4", "palin|m=32|k=8", "acc_mean"), "stack": ("v4", "stack|m=32|L=4", "acc"),
            "assoc": ("v3", "assoc|m=128|V=16", "acc"), "sum": ("sweep", "sum|64|2", "acc"),
            "max": ("sweep", "max|64|2", "acc"), "min": ("sweep", "min|64|2", "acc"),
            "first": ("sweep", "first|64|2", "acc"), "last": ("sweep", "last|64|2", "acc")},
    "text": {"first": ("printable", "first|m=64|k=2", "acc"), "last": ("printable", "last|m=64|k=2", "acc"),
             "max": ("printable", "max|m=64|k=2", "acc"), "min": ("printable", "min|m=64|k=2", "acc"),
             "stack": ("printable", "stack|m=32|L=4", "acc"),
             "assoc": ("printable", "assoc|extra=112|V=16", "acc"),
             "sum": ("printable", "sum|m=64|k=2", "acc"), "reverse": ("printable", "reverse|m=32|k=8", "acc"),
             "cipher": ("printable", "cipher|m=64|k=4", "acc")},
    "word": {"classify-2": ("words", "wordcls|m=32|C=2", "acc"),
             "classify-4": ("words", "wordcls|m=32|C=4", "acc"),
             "lookup-4": ("words", "wordseen|m=32|C=4", "acc"),
             "word-reverse": ("words", "wordrev|m=32|n=3", "acc")},
}
# Tasks averaged into each suite's headline score. Word-order reversal is 0 for
# every model at every size, so it is reported separately rather than diluting
# the word score.
HEADLINE = {"raw": list(TASKS["raw"]), "text": list(TASKS["text"]),
            "word": ["classify-2", "classify-4", "lookup-4"]}
SIZE_OF_RUNG = {f"d{a['d_model']}h{a['n_heads']}L{a['n_layers']}": s for s, a in ARCH.items()}


def write(name, header, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main():
    data = load(RESULTS)
    head, tasks, curves = [], [], []
    for (arm, size, tok), res in sorted(data.items()):
        per_task = {}
        for suite, spec in TASKS.items():
            for task, (sec, tag, field) in spec.items():
                e = res.get(sec, {}).get(tag)
                if e is None:
                    continue
                for seed, m in e["per_seed"].items():
                    v = val(m, field)
                    tasks.append((arm, size, tok, suite, task, seed, round(v, 5)))
                    per_task[(suite, task, seed)] = v
        for suite, names in HEADLINE.items():
            seeds = {s for (su, _, s) in per_task if su == suite}
            for seed in sorted(seeds):
                vals = [per_task.get((suite, t, seed)) for t in names]
                if all(v is not None for v in vals):
                    head.append((arm, size, tok, suite, seed, round(sum(vals) / len(vals), 5)))
        for suite, fams in FAMILIES.items():
            for task, pts in fams.items():
                for m, sec, tag, field in pts:
                    e = res.get(sec, {}).get(tag)
                    if e is None:
                        continue
                    for seed, mm in e["per_seed"].items():
                        curves.append((arm, size, tok, suite, task, m, seed, round(val(mm, field), 5)))
    write("headline.csv", ["arm", "size", "tokens", "suite", "seed", "value"], head)
    write("tasks.csv", ["arm", "size", "tokens", "suite", "task", "seed", "value"], tasks)
    write("mcurves.csv", ["arm", "size", "tokens", "suite", "task", "m", "seed", "value"], curves)

    rows = []
    for p in sorted(LOGS.glob("*/*/seed-*/log.jsonl")):
        arm, size, seed = p.parts[-4], p.parts[-3], p.parts[-2]
        for line in open(p):
            r = json.loads(line)
            if "val_bpb" in r:
                rows.append((arm, size, seed, round(r["tokens"] / 1e9, 4), round(r["val_bpb"], 4)))
    write("valbpb.csv", ["arm", "size", "seed", "tokens", "val_bpb"], rows)

    traj = json.loads(PAPER_TRAJ.read_text())
    rows = []
    for rung, d in traj.items():
        size = SIZE_OF_RUNG.get(rung)
        for rnd, r in d["rounds"].items():
            if "dclm" in r and "1" in r["dclm"]:
                rows.append((size, int(rnd), round(1536 * 4096 * (int(rnd) + 1) / 1e9, 4),
                             round(r["dclm"]["1"], 4)))
    write("sp_bpb.csv", ["size", "round", "tokens", "dclm_bpb"], sorted(rows, key=lambda x: (x[0], x[1])))


if __name__ == "__main__":
    main()
