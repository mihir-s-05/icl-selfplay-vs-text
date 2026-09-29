"""Compare an ``icl.run`` result on the paper's self-play ensemble to the shipped numbers.

    python -m icl.check_repro results/repro/sp_24M_r2816.json

Uses the ensemble metrics when the result has several seeds, else the single
seed. Each shared cell is reported as exact, within one trial (``|diff| <=
1/N``, argmax ties can flip under different kernels), or a mismatch. Exits
non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FIG = PROJECT / "self_play_pretraining" / "figures"
PAPER_FILES = {
    "sweep": FIG / "fig4_icl_across_methods/data/selfplay/icl_results.json",
    "v2": FIG / "fig5_icl_sum_behavior/icl_harness/icl_v2_results.json",
    "v3": FIG / "fig4_icl_across_methods/data/selfplay/icl_v3_results.json",
    "v4": FIG / "fig4_icl_across_methods/data/selfplay/icl_v4_results.json",
    "m0": FIG / "fig4_icl_across_methods/data/selfplay/icl_m0_results.json",
    "sum_lowm": FIG / "fig4_icl_across_methods/data/selfplay/icl_sum_lowm_results.json",
    "assoc_dict": FIG / "fig4_icl_across_methods/data/selfplay/icl_assoc_dict_results.json",
    "control": FIG / "fig5_icl_sum_behavior/icl_harness/icl_control_results.json",
}
FIELDS = ("acc", "acc_mean", "acc_decoupled", "p_correct")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result", type=Path)
    ap.add_argument("--p-tol", type=float, default=2e-3,
                    help="absolute tolerance on p_correct")
    args = ap.parse_args()
    res = json.loads(args.result.read_text())["results"]

    n_exact = n_near = n_bad = 0
    for section, cells in res.items():
        if section not in PAPER_FILES:
            continue
        paper = json.loads(PAPER_FILES[section].read_text())["results"]
        for tag, entry in cells.items():
            if tag not in paper:
                print(f"  [absent] {section}/{tag} not in paper file")
                continue
            ours = entry.get("ensemble") or next(iter(entry["per_seed"].values()))
            N = entry["n_trials"] // (2 if section == "control" else 1)
            for f in FIELDS:
                if f not in paper[tag] or f not in ours:
                    continue
                if f == "acc" and "acc_mean" in paper[tag]:
                    continue          # teacher-forced cells: the paper reports acc_mean
                d = abs(ours[f] - paper[tag][f])
                tol = args.p_tol if f == "p_correct" else 1e-9
                if d <= tol:
                    n_exact += 1
                elif f != "p_correct" and d <= 1.0 / N + 1e-9:
                    n_near += 1
                    print(f"  [1-trial] {section}/{tag} {f}: ours {ours[f]:.4f} paper {paper[tag][f]:.4f}")
                else:
                    n_bad += 1
                    print(f"  [MISMATCH] {section}/{tag} {f}: ours {ours[f]:.4f} paper {paper[tag][f]:.4f}")
    print(f"exact {n_exact}  within-one-trial {n_near}  mismatch {n_bad}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
