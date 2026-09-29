"""ICL comparison report: m-curves, seed error bars, the per-seed self-play check.

    python analysis/report.py --results results --out results/report [--wandb]

Reads every ``<results>/icl*/<arm>/<size>/<tag>.json`` (the main suite, the
word suite, and extra DCLM seeds all merge by arm/size/token count, keyed per
seed) and writes ``report.md`` plus figures:

  1. headline means (raw / text / word suites) at matched token counts, with
     95% confidence intervals across seeds (t-interval; a single seed gets the
     binomial trial-noise interval instead, marked "n=1")
  2. scaling at the paper's ICL point (17.7B tokens)
  3. m-curves: accuracy vs number of in-context examples, per task and size
  4. per-seed check: self-play round 2816 vs round 8191
  5. warm starts (self-play -> DCLM)
  6. DCLM training health (validation BPB)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import N_NONEMB, sp_tokens  # noqa: E402
from icl.evaluate_arm import EXACT, HEADLINE, UP_PROGRAMS  # noqa: E402

SIZES = ["100k", "500k", "1M", "3M", "6M", "24M"]
MATCH_ROUNDS = (256, 512, 1024, 2048, 2816)
PAPER_T = sp_tokens(2816) / 1e9                       # 17.72B
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}
SUITE_PREFIX = {"raw": "raw_", "text": "text_", "word": "word_"}
ARM_STYLE = {"sp": ("Self-play", "#c0392b", "-"), "dclm": ("DCLM text", "#2471a3", "--"),
             "up": ("Universal prior", "#7f8c8d", ":"), "sp2dclm": ("SP→DCLM", "#8e44ad", "-.")}

SWEEP_MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
P_MS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
W_MS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
ASSOC = [(16 + e, "assoc_dict", f"assoc|extra={e}|V=16") for e in (0, 1, 2, 4, 8, 16)]
FAMILIES = {   # suite -> task -> [(m, section, tag, field)]
    "raw": {
        "reverse": [(0, "m0", "palin|m=0|k=8", "acc_mean")]
                   + [(m, "v4", f"palin|m={m}|k=8", "acc_mean") for m in (1, 2, 4, 8, 32, 128)],
        "stack": [(0, "m0", "stack|m=0|L=4", "acc")]
                 + [(m, "v4", f"stack|m={m}|L=4", "acc") for m in (1, 2, 4, 8, 32, 128)],
        "assoc": [(m, s, t, "acc") for m, s, t in ASSOC]
                 + [(m, "v3", f"assoc|m={m}|V=16", "acc") for m in (128, 512)],
        "sum": [(m, "sum_lowm", f"sum|{m}|2", "acc") for m in range(5)]
               + [(m, "sweep", f"sum|{m}|2", "acc") for m in SWEEP_MS if m >= 8],
        **{f: ([(0, "m0", f"{f}|0|2", "acc")] if f in ("max", "min") else [])
           + [(m, "sweep", f"{f}|{m}|2", "acc") for m in SWEEP_MS]
           for f in ("max", "min", "first", "last")},
    },
    "text": {
        **{f: [(m, "printable", f"{f}|m={m}|k=2", "acc") for m in P_MS]
           for f in ("first", "last", "max", "min", "sum")},
        "reverse": [(m, "printable", f"reverse|m={m}|k=8", "acc") for m in P_MS],
        "stack": [(m, "printable", f"stack|m={m}|L=4", "acc") for m in P_MS],
        "assoc": [(16 + e, "printable", f"assoc|extra={e}|V=16", "acc")
                  for e in (0, 1, 2, 4, 8, 16, 48, 112)],
        "cipher": [(m, "printable", f"cipher|m={m}|k=4", "acc") for m in P_MS],
    },
    "word": {
        "classify (unseen word, 2 labels)": [(m, "words", f"wordcls|m={m}|C=2", "acc") for m in W_MS],
        "classify (unseen word, 4 labels)": [(m, "words", f"wordcls|m={m}|C=4", "acc") for m in W_MS],
        "lookup (seen word, 4 labels)": [(m, "words", f"wordseen|m={m}|C=4", "acc") for m in W_MS],
        "reverse word order": [(m, "words", f"wordrev|m={m}|n=3", "acc") for m in W_MS],
    },
}


# ------------------------------------------------------------------ loading

def load(results: Path, smoke: bool = False) -> dict:
    """{(arm, size, tokens_B rounded): {section: {tag: {"n": N, "per_seed": {seed: m}}}}}"""
    data: dict = defaultdict(lambda: defaultdict(dict))
    for f in sorted(results.glob("icl*/*/*/*.json")):
        if ("smoke" in f.parts[-4]) != smoke:
            continue
        d = json.loads(f.read_text())
        meta = d["meta"]
        key = (meta["arm"], meta["size"], round(meta["tokens"] / 1e9, 3))
        for sec, cells in d["results"].items():
            for tag, e in cells.items():
                slot = data[key][sec].setdefault(tag, {"n": e["n_trials"], "per_seed": {}})
                for seed, m in e["per_seed"].items():
                    slot["per_seed"][seed.split("/")[0]] = m
    return data


def tokens_of(data, arm, size):
    return sorted(t for a, s, t in data if a == arm and s == size)


def at(data, arm, size, t, tol=0.02):
    ts = tokens_of(data, arm, size)
    if not ts:
        return None
    best = min(ts, key=lambda x: abs(x - t))
    return data[(arm, size, best)] if abs(best - t) <= tol * t else None


def val(m: dict, field: str) -> float:
    return m.get(field, m["acc"]) if field == "acc_mean" else m["acc"]


def headline_per_seed(res: dict, suite: str) -> dict:
    """seed -> mean over the suite's headline cells (only seeds with all cells)."""
    names = [n for n in HEADLINE if n.startswith(SUITE_PREFIX[suite])]
    per = defaultdict(list)
    for n in names:
        sec, tag = HEADLINE[n]
        e = res.get(sec, {}).get(tag)
        if e is None:
            return {}
        for seed, m in e["per_seed"].items():
            per[seed].append(m["acc"] if n in EXACT else m.get("acc_mean", m["acc"]))
    return {s: float(np.mean(v)) for s, v in per.items() if len(v) == len(names)}


def trial_se(res: dict, suite: str, seed: str) -> float:
    """Binomial standard error of one seed's headline mean (trial noise only)."""
    names = [n for n in HEADLINE if n.startswith(SUITE_PREFIX[suite])]
    var = 0.0
    for n in names:
        sec, tag = HEADLINE[n]
        e = res[sec][tag]
        p = e["per_seed"][seed]["acc"]
        var += p * (1 - p) / e["n"]
    return math.sqrt(var) / len(names)


def summarize(vals: list[float], res=None, suite=None, seed=None):
    """(mean, half-width of 95% CI, n)."""
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = float(np.mean(vals))
    if n >= 2:
        return (mean, T975[n - 1] * float(np.std(vals, ddof=1)) / math.sqrt(n), n)
    se = trial_se(res, suite, seed) if res is not None else float("nan")
    return (mean, 1.96 * se, 1)


def fmt(s) -> str:
    mean, hw, n = s
    if n == 0 or math.isnan(mean):
        return "–"
    return f"{mean:.3f} ± {hw:.3f}" + (" (n=1)" if n == 1 else f" (n={n})")


def headline(data, arm, size, t, suite):
    res = at(data, arm, size, t)
    if res is None:
        return (float("nan"), float("nan"), 0), {}
    per = headline_per_seed(res, suite)
    seeds = list(per)
    return summarize(list(per.values()), res, suite, seeds[0] if len(seeds) == 1 else None), per


# ------------------------------------------------------------------ report

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    ap.add_argument("--ckpts", type=Path, default=Path("ckpts"), help="for training logs")
    ap.add_argument("--smoke", action="store_true", help="read the *_smoke result dirs instead")
    ap.add_argument("--wandb", action="store_true", help="log figures to a W&B run 'analysis'")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = load(args.results, args.smoke)
    args.out.mkdir(parents=True, exist_ok=True)
    L: list[str] = ["# Self-play vs DCLM text pretraining: in-context learning", ""]
    L += ["Accuracy is greedy exact match. Headline = mean of the suite's headline cells "
          "(Fig. 4 tasks at m = 32–128). ± = 95% CI across seeds (t-interval); "
          "n=1 rows show binomial trial noise only.", ""]
    sizes = [s for s in SIZES if tokens_of(data, "sp", s)]
    if not sizes:
        raise SystemExit(f"no self-play results under {args.results}")

    # 1. matched tokens
    L += ["## 1. Headline ICL at matched learner tokens", ""]
    for suite in ("text", "word", "raw"):
        L += [f"### {suite} suite", "", "| size | tokens | self-play | DCLM | SP − DCLM |",
              "|---|---|---|---|---|"]
        for size in sizes:
            for r in MATCH_ROUNDS:
                t = sp_tokens(r) / 1e9
                sp, _ = headline(data, "sp", size, t, suite)
                dc, _ = headline(data, "dclm", size, t, suite)
                if dc[2] == 0:
                    continue
                L.append(f"| {size} | {t:.1f}B | {fmt(sp)} | {fmt(dc)} | {sp[0] - dc[0]:+.3f} |")
        L.append("")

    # 2. scaling
    L += ["## 2. Scaling at the paper's ICL point (17.7B tokens; UP at its round 2816)", ""]
    L += ["| size | N (non-emb) | suite | self-play | universal prior | DCLM |", "|---|---|---|---|---|---|"]
    scaling = defaultdict(dict)
    for size in sizes:
        for suite in ("text", "word", "raw"):
            row = {}
            for arm in ("sp", "up", "dclm"):
                if arm == "up":
                    ts = tokens_of(data, "up", size)
                    t_up = [t for t in ts
                            if abs(t - UP_PROGRAMS.get(size, 0) * 4096 * 2817 / 1e9) < 0.05]
                    s = headline(data, "up", size, t_up[0], suite)[0] if t_up else (float("nan"), 0, 0)
                else:
                    s = headline(data, arm, size, PAPER_T, suite)[0]
                row[arm] = s
            scaling[suite][size] = row
            L.append(f"| {size} | {N_NONEMB[size]:,} | {suite} | {fmt(row['sp'])} | "
                     f"{fmt(row['up'])} | {fmt(row['dclm'])} |")
    L.append("")

    # 3. m-curves (tables for a few sizes; figures for all)
    L += ["## 3. m-curves at 17.7B tokens (accuracy vs in-context examples)", "",
          "Each cell: seed mean (self-play n=4; DCLM n = its seeds). Figures: "
          "`m_curves_<suite>.png`.", ""]
    curves = {}
    for suite, fams in FAMILIES.items():
        for task, pts in fams.items():
            for size in sizes:
                for arm in ("sp", "dclm"):
                    res = at(data, arm, size, PAPER_T)
                    if res is None:
                        continue
                    xs, mu, hw = [], [], []
                    for m, sec, tag, field in pts:
                        e = res.get(sec, {}).get(tag)
                        if e is None:
                            continue
                        v = [val(x, field) for x in e["per_seed"].values()]
                        s = summarize(v)
                        xs.append(m); mu.append(s[0]); hw.append(0 if s[2] < 2 else s[1])
                    curves[(suite, task, size, arm)] = (xs, mu, hw)
    for suite, fams in FAMILIES.items():
        L += [f"### {suite}", ""]
        for size in [s for s in ("1M", "6M", "24M") if s in sizes]:
            L += [f"**{size}**", "", "| task | arm | " + " | ".join(f"m={m}" for m in
                  sorted({m for pts in fams.values() for m, *_ in pts})) + " |"]
            ms_all = sorted({m for pts in fams.values() for m, *_ in pts})
            L.append("|---|---|" + "---|" * len(ms_all))
            for task in fams:
                for arm in ("sp", "dclm"):
                    c = curves.get((suite, task, size, arm))
                    if not c:
                        continue
                    d = dict(zip(c[0], c[1]))
                    L.append(f"| {task} | {arm} | " + " | ".join(
                        f"{d[m]:.2f}" if m in d else "" for m in ms_all) + " |")
            L.append("")

    # 4. per-seed self-play check
    L += ["## 4. Checkpoint dependence: self-play round 2816 vs round 8191, per seed", "",
          "| size | suite | per-seed (r2816 → r8191) | seeds that drop | mean change ± 95% CI |",
          "|---|---|---|---|---|"]
    for size in sizes:
        for suite in ("text", "word", "raw"):
            a = at(data, "sp", size, sp_tokens(2816) / 1e9)
            b = at(data, "sp", size, sp_tokens(8191) / 1e9)
            if a is None or b is None:
                continue
            pa, pb = headline_per_seed(a, suite), headline_per_seed(b, suite)
            seeds = sorted(set(pa) & set(pb))
            if not seeds:
                continue
            diffs = [pb[s] - pa[s] for s in seeds]
            s = summarize(diffs)
            L.append(f"| {size} | {suite} | " + ", ".join(f"{pa[x]:.2f}→{pb[x]:.2f}" for x in seeds)
                     + f" | {sum(d < 0 for d in diffs)}/{len(seeds)} | {s[0]:+.3f} ± {s[1]:.3f} |")
    L.append("")

    # 5. warm starts
    L += ["## 5. Warm starts: self-play (round 8191) → DCLM", "",
          "| size | suite | SP r8191 (start) | +50M | +100M | +250M | +500M | DCLM scratch @0.5B |",
          "|---|---|---|---|---|---|---|---|"]
    for size in sizes:
        if not tokens_of(data, "sp2dclm", size):
            continue
        for suite in ("text", "word", "raw"):
            start = headline(data, "sp", size, sp_tokens(8191) / 1e9, suite)[0]
            cells = [headline(data, "sp2dclm", size, t, suite)[0] for t in (0.05, 0.1, 0.25, 0.5)]
            scratch = headline(data, "dclm", size, 0.5, suite)[0]
            L.append(f"| {size} | {suite} | {start[0]:.3f} | "
                     + " | ".join(f"{c[0]:.3f}" if c[2] else "–" for c in cells)
                     + (f" | {scratch[0]:.3f} |" if scratch[2] else " | – |"))
    L.append("")

    # 6. training health
    L += ["## 6. DCLM validation BPB at the end of training", "", "| size | seed | final val BPB | tokens |",
          "|---|---|---|---|"]
    for p in sorted((args.ckpts / "dclm").glob("*/seed-*/log.jsonl")):
        rows = [json.loads(l) for l in open(p) if "val_bpb" in l]
        if rows:
            L.append(f"| {p.parts[-3]} | {p.parts[-2]} | {rows[-1]['val_bpb']:.3f} | "
                     f"{rows[-1]['tokens'] / 1e9:.1f}B |")
    L.append("")
    (args.out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    figs = plots(data, curves, scaling, sizes, args.out)
    if args.wandb and figs:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from common import wandb_init
        wb = wandb_init(True, name="analysis", job_type="analysis")
        if wb:
            import wandb
            wb.log({p.stem: wandb.Image(str(p)) for p in figs})
            art = wandb.Artifact("report", type="report")
            art.add_dir(str(args.out))
            wb.log_artifact(art).wait()
            wb.finish()


def plots(data, curves, scaling, sizes, out: Path) -> list[Path]:
    import os, tempfile
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib missing; skipping figures")
        return []
    paths = []
    for suite, fams in FAMILIES.items():                    # m-curves
        tasks = list(fams)
        fig, axes = plt.subplots(len(tasks), len(sizes), figsize=(2.6 * len(sizes), 2.0 * len(tasks)),
                                 sharex=True, sharey=True, squeeze=False)
        for i, task in enumerate(tasks):
            for j, size in enumerate(sizes):
                ax = axes[i][j]
                for arm in ("sp", "dclm"):
                    c = curves.get((suite, task, size, arm))
                    if not c or not c[0]:
                        continue
                    lab, col, ls = ARM_STYLE[arm]
                    x = np.array(c[0], float) + 0.5
                    mu, hw = np.array(c[1]), np.array(c[2])
                    ax.plot(x, mu, ls, color=col, lw=1.4, marker="o", ms=2.5, label=lab)
                    ax.fill_between(x, np.clip(mu - hw, 0, 1), np.clip(mu + hw, 0, 1), color=col, alpha=0.15, lw=0)
                ax.set_xscale("log", base=2); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
                if i == 0:
                    ax.set_title(size, fontsize=10)
                if j == 0:
                    ax.set_ylabel(task, fontsize=8)
                if i == len(tasks) - 1:
                    ax.set_xlabel("m + ½ (examples)", fontsize=8)
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"{suite} suite: accuracy vs in-context examples at 17.7B tokens "
                     "(band = 95% CI over seeds)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = out / f"m_curves_{suite}.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))            # headline vs tokens
    cmap = plt.get_cmap("viridis")
    for k, suite in enumerate(("text", "word", "raw")):
        ax = axes[k]
        for j, size in enumerate(sizes):
            col = cmap(j / max(1, len(sizes) - 1))
            for arm, ls in (("sp", "-"), ("dclm", "--")):
                pts = [(t, headline(data, arm, size, t, suite)[0]) for t in tokens_of(data, arm, size) if t > 0.05]
                pts = [(t, s) for t, s in pts if s[2]]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1][0] for p in pts], ls, color=col, marker="o", ms=3,
                            label=f"{size} {'SP' if arm == 'sp' else 'DCLM'}")
        ax.set_xscale("log"); ax.set_title(f"{suite} headline"); ax.set_xlabel("learner tokens (B)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean accuracy"); axes[2].legend(fontsize=6, ncol=2)
    fig.suptitle("ICL vs training tokens (solid = self-play, dashed = DCLM)")
    fig.tight_layout(); p = out / "headline_vs_tokens.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))            # scaling
    for k, suite in enumerate(("text", "word", "raw")):
        ax = axes[k]
        for arm in ("sp", "up", "dclm"):
            lab, col, ls = ARM_STYLE[arm]
            pts = [(N_NONEMB[s], scaling[suite][s][arm]) for s in sizes if scaling[suite][s][arm][2]]
            if pts:
                x = [p[0] for p in pts]; y = [p[1][0] for p in pts]; e = [p[1][1] for p in pts]
                ax.errorbar(x, y, yerr=e, fmt=ls, color=col, marker="o", capsize=3, label=lab)
        ax.set_xscale("log"); ax.set_title(f"{suite} headline @17.7B tokens")
        ax.set_xlabel("non-embedding parameters"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean accuracy"); axes[0].legend(fontsize=8)
    fig.tight_layout(); p = out / "scaling.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    return paths


if __name__ == "__main__":
    main()
