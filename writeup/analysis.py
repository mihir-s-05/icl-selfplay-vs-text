"""Statistics behind WRITEUP.md, printed as markdown and saved to analysis_out.md.

    python writeup/figs/extract.py && python writeup/analysis.py

Every number in the write-up comes from this output.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "figs/data"
SIZES = ["100k", "500k", "1M", "3M", "6M", "24M"]
N = {"100k": 65_728, "500k": 492_160, "1M": 984_192, "3M": 3_016_960, "6M": 6_033_664, "24M": 24_253_184}
MATCH = [1.6169, 3.2275, 6.4487, 12.8912, 17.723]      # self-play rounds 256 ... 2816 (B tokens)
FINAL = 17.723
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45}
OUT: list[str] = []


def p(line=""):
    OUT.append(line)
    print(line)


def read(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def near(t, target, tol=0.02):
    return abs(float(t) - target) <= tol * target


def seeds(rows, arm, size, tokens, **kw):
    out = {}
    for r in rows:
        if r["arm"] == arm and r["size"] == size and near(r["tokens"], tokens) and \
                all(r[k] == v for k, v in kw.items()):
            out[r["seed"]] = float(r["value"])
    return out


def ci(vals):
    v = np.array(list(vals), float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan")
    return float(v.mean()), T975[len(v) - 1] * float(v.std(ddof=1)) / math.sqrt(len(v))


def fmt_ci(vals):
    m, h = ci(vals)
    return f"{m:.3f} ± {h:.3f}" if not math.isnan(h) else f"{m:.3f}"


def fmt_seeds(d):
    return ", ".join(f"{v:.3f}" for _, v in sorted(d.items())) or "–"


def main():
    H, TK, MC = read("headline.csv"), read("tasks.csv"), read("mcurves.csv")
    VB, SB = read("valbpb.csv"), read("sp_bpb.csv")

    # ---------------------------------------------------------------- 1
    p("## A. Headline ICL at the paper's ICL point (17.7B learner tokens)")
    p("Self-play (SP) = mean ± 95% t-interval over 4 released seeds; DCLM = each seed listed; "
      "UP = universal prior, round 2816, 4 seeds. `sep` = every DCLM seed outside the SP interval.")
    for suite in ("text", "word", "raw"):
        p(f"\n**{suite} suite**\n")
        p("| size | self-play | DCLM seeds | SP − DCLM | sep | universal prior |")
        p("|---|---|---|---|---|---|")
        for size in SIZES:
            sp = seeds(H, "sp", size, FINAL, suite=suite)
            dc = seeds(H, "dclm", size, FINAL, suite=suite)
            up_tok = {"100k": 1024, "500k": 2048, "1M": 1024, "3M": 2048, "6M": 1024}.get(size)
            up = seeds(H, "up", size, up_tok * 4096 * 2817 / 1e9, suite=suite) if up_tok else {}
            m, h = ci(sp.values())
            dmean = np.mean(list(dc.values())) if dc else float("nan")
            sep = all(v < m - h or v > m + h for v in dc.values()) if dc else False
            p(f"| {size} | {fmt_ci(sp.values())} | {fmt_seeds(dc)} | {m - dmean:+.3f} | "
              f"{'yes' if sep else 'no'} | {fmt_ci(up.values()) if up else '–'} |")

    # ---------------------------------------------------------------- 2
    p("\n## B. Does the gap grow with size? Slope of (SP − DCLM) vs log10 N, sizes 1M–24M")
    rng = np.random.default_rng(0)
    for suite in ("text", "word", "raw"):
        xs, sp_s, dc_s = [], [], []
        for size in ["1M", "3M", "6M", "24M"]:
            xs.append(math.log10(N[size]))
            sp_s.append(list(seeds(H, "sp", size, FINAL, suite=suite).values()))
            dc_s.append(list(seeds(H, "dclm", size, FINAL, suite=suite).values()))
        gaps = [np.mean(a) - np.mean(b) for a, b in zip(sp_s, dc_s)]
        slope = np.polyfit(xs, gaps, 1)[0]
        boots = []
        for _ in range(4000):   # resample seeds within each size and arm
            g = [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))) for a, b in zip(sp_s, dc_s)]
            boots.append(np.polyfit(xs, g, 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p(f"- {suite}: gaps {', '.join(f'{g:+.3f}' for g in gaps)}; slope {slope:+.3f} per decade "
          f"of parameters (seed bootstrap 95% [{lo:+.3f}, {hi:+.3f}])")
    p("  (DCLM has 1 seed at 24M, so its within-size resampling is degenerate there.)")

    # ---------------------------------------------------------------- 3
    p("\n## C. Timing: headline vs learner tokens (seed means)")
    for suite in ("text", "word"):
        p(f"\n**{suite}**\n")
        p("| size | arm | " + " | ".join(f"{t:.1f}B" for t in MATCH) + " |")
        p("|---|---|" + "---|" * len(MATCH))
        for size in ["1M", "3M", "6M", "24M"]:
            for arm in ("sp", "dclm"):
                vals = [np.mean(list(seeds(H, arm, size, t, suite=suite).values()) or [np.nan]) for t in MATCH]
                p(f"| {size} | {arm} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    # ---------------------------------------------------------------- 4
    p("\n## D. Compute handicap check: SP at 6.4B (≈2.75× fewer learner tokens) vs DCLM at 17.7B")
    p("| size | suite | SP @ 6.4B | DCLM @ 17.7B |")
    p("|---|---|---|---|")
    for size in ["1M", "3M", "6M", "24M"]:
        for suite in ("text", "word"):
            sp = seeds(H, "sp", size, 6.4487, suite=suite)
            dc = seeds(H, "dclm", size, FINAL, suite=suite)
            p(f"| {size} | {suite} | {fmt_ci(sp.values())} | {fmt_seeds(dc)} |")

    # ---------------------------------------------------------------- 5
    p("\n## E. Per-task accuracy at 17.7B (seed means; SP n=4, DCLM n=2 at 1M/3M/6M else 1)")
    for size in ["1M", "6M", "24M"]:
        p(f"\n**{size}**\n")
        p("| suite | task | self-play | DCLM | SP − DCLM |")
        p("|---|---|---|---|---|")
        for suite in ("text", "word", "raw"):
            for task in sorted({r["task"] for r in TK if r["suite"] == suite}):
                sp = seeds(TK, "sp", size, FINAL, suite=suite, task=task)
                dc = seeds(TK, "dclm", size, FINAL, suite=suite, task=task)
                if sp and dc:
                    a, b = np.mean(list(sp.values())), np.mean(list(dc.values()))
                    p(f"| {suite} | {task} | {fmt_ci(sp.values())} | {b:.3f} | {a - b:+.3f} |")

    # ---------------------------------------------------------------- 6
    p("\n## F. Learning from the examples: accuracy at few vs many in-context examples (17.7B)")
    p("Gain = accuracy at the largest m minus at the smallest m scored (m=0 where available).")
    p("| size | suite | task | SP m_lo → m_hi | DCLM m_lo → m_hi |")
    p("|---|---|---|---|---|")
    curves = defaultdict(lambda: defaultdict(list))
    for r in MC:
        if near(r["tokens"], FINAL) and r["arm"] in ("sp", "dclm"):
            curves[(r["size"], r["suite"], r["task"], r["arm"])][int(r["m"])].append(float(r["value"]))
    for size in ["6M", "24M"]:
        for suite, tasks in (("text", ["first", "last", "max", "stack", "assoc"]),
                             ("word", ["classify (unseen word, 4 labels)", "lookup (seen word, 4 labels)"])):
            for task in tasks:
                cells = []
                for arm in ("sp", "dclm"):
                    c = curves.get((size, suite, task, arm))
                    if not c:
                        cells.append("–"); continue
                    ms = sorted(c)
                    lo_m, hi_m = ms[0], ms[-1]
                    cells.append(f"m={lo_m}: {np.mean(c[lo_m]):.2f} → m={hi_m}: {np.mean(c[hi_m]):.2f}")
                p(f"| {size} | {suite} | {task} | {cells[0]} | {cells[1]} |")

    # ---------------------------------------------------------------- 7
    p("\n## G. Warm starts: self-play (round 8191) → DCLM, single seed")
    p("| size | suite | SP r8191 | +0.05B | +0.1B | +0.25B | +0.5B | DCLM scratch @0.5B | DCLM scratch @17.7B |")
    p("|---|---|---|---|---|---|---|---|---|")
    for size in ["1M", "3M", "6M", "24M"]:
        for suite in ("word", "text", "raw"):
            start = np.mean(list(seeds(H, "sp", size, 51.5396, suite=suite).values()))
            warm = [np.mean(list(seeds(H, "sp2dclm", size, t, suite=suite).values()) or [np.nan])
                    for t in (0.05, 0.1, 0.25, 0.5)]
            s05 = np.mean(list(seeds(H, "dclm", size, 0.5, suite=suite).values()))
            s17 = np.mean(list(seeds(H, "dclm", size, FINAL, suite=suite).values()))
            p(f"| {size} | {suite} | {start:.3f} | " + " | ".join(f"{w:.3f}" for w in warm)
              + f" | {s05:.3f} | {s17:.3f} |")

    # ---------------------------------------------------------------- 8
    p("\n## H. Language modelling vs ICL: DCLM validation bits/byte at 17.7B")
    p("| size | DCLM (our runs, per seed) | self-play zero-shot on DCLM (paper's scores, round 2816) | text ICL: SP / DCLM |")
    p("|---|---|---|---|")
    for size in SIZES:
        last = {}
        for r in VB:                                   # final validation row of each seed
            if r["arm"] == "dclm" and r["size"] == size:
                last[r["seed"]] = float(r["val_bpb"])
        ours = sorted(last.items())
        sp = [float(r["dclm_bpb"]) for r in SB if r["size"] == size and int(r["round"]) == 2816]
        a = np.mean(list(seeds(H, "sp", size, FINAL, suite="text").values()))
        b = np.mean(list(seeds(H, "dclm", size, FINAL, suite="text").values()))
        p(f"| {size} | {', '.join(f'{v:.3f}' for _, v in ours)} | {sp[0]:.2f} | {a:.2f} / {b:.2f} |")

    # ---------------------------------------------------------------- 9
    p("\n## I. Self-play checkpoint dependence (seed means; per-seed test in the report)")
    rounds = [(256, 1.6169), (512, 3.2275), (1024, 6.4487), (2048, 12.8912), (2816, 17.723), (8191, 51.5396)]
    p("| size | suite | " + " | ".join(f"r{r}" for r, _ in rounds) + " |")
    p("|---|---|" + "---|" * len(rounds))
    for size in ["1M", "3M", "6M", "24M"]:
        for suite in ("text", "word", "raw"):
            vals = [np.mean(list(seeds(H, "sp", size, t, suite=suite).values())) for _, t in rounds]
            p(f"| {size} | {suite} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    (HERE / "analysis_out.md").write_text("\n".join(OUT), encoding="utf-8")


if __name__ == "__main__":
    main()
