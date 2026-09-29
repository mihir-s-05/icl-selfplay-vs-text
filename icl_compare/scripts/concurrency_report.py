"""Summarize scripts/concurrency_test.sh: per-model share of solo speed, and the gain.

    python scripts/concurrency_report.py results/bench_concurrency.jsonl

For a concurrent group, each model runs at c_i tokens/s against its solo r_i.
Running them one after another takes sum_i(T_i / r_i) hours; concurrently,
each advances at c_i. The packing gain sum_i(c_i / r_i) is how many solo-GPUs'
worth of work the shared GPU does: above 1 means packing saves money.
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path)]
    solo = {r["size"]: r["tok_s"] for r in rows if r["tag"] == "solo"}
    groups = defaultdict(dict)
    for r in rows:
        if r["tag"] != "solo":
            groups[r["tag"]][r["size"]] = r
    print("solo: " + "  ".join(f"{s} {v / 1e6:.2f}M tok/s" for s, v in solo.items()))
    for tag, g in groups.items():
        gain = 0.0
        print(f"\n{tag}:")
        for s, r in g.items():
            share = r["tok_s"] / solo[s]
            gain += share
            print(f"  {s:>4}: {r['tok_s'] / 1e6:5.2f}M tok/s = {share:5.1%} of solo "
                  f"(peak {r['peak_gb']:.1f} GB)")
        print(f"  packing gain {gain:.2f}x -> {'saves' if gain > 1 else 'costs'} "
              f"{abs(1 - 1 / gain):.0%} of GPU-hours vs one-at-a-time")


if __name__ == "__main__":
    main()
