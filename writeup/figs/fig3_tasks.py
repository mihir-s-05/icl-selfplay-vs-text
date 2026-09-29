"""Fig. 3: per-task accuracy difference (self-play − DCLM) at 17.7B tokens.

Regenerate: python writeup/figs/fig3_tasks.py
"""
import math

import numpy as np
from io_data import FINAL, rows, seeds
from orx_figstyle import BASELINE, PALETTE, TEXT, figure, save, use_style, _t95

# (suite, task, label). Tasks that every model scores 0 on (multi-digit sum,
# 8-letter reverse, cipher, word-order reversal) are left out; see the caption.
TASKS = [("text", "first", "text: output first letter"), ("text", "last", "text: output last letter"),
         ("text", "max", "text: max of 2 letters"), ("text", "min", "text: min of 2 letters"),
         ("text", "stack", "text: stack (push/pop)"), ("text", "assoc", "text: key→value recall"),
         ("word", "lookup-4", "word: label of a shown word"),
         ("word", "classify-2", "word: category, unseen word (2)"),
         ("word", "classify-4", "word: category, unseen word (4)")]
SIZES = [("6M", PALETTE["blue"], "o", -0.14), ("24M", PALETTE["orange"], "s", 0.14)]


def interval(sp, dc):
    """SP mean − DCLM mean, whiskered by the 95% t-interval over the SP seeds.

    With 1–2 DCLM seeds a two-sample (Welch) interval has ~1 degree of freedom
    and spans the whole axis, so DCLM's seed spread is shown directly instead
    (hollow markers, one per DCLM seed).
    """
    sp = np.asarray(sp)
    half = _t95(len(sp) - 1) * sp.std(ddof=1) / math.sqrt(len(sp))
    d = sp.mean() - np.mean(dc)
    return d, d - half, d + half


def main():
    use_style()
    T = rows("tasks.csv")
    fig, ax = figure(width=TEXT, ratio=0.5)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    for y, (suite, task, _) in enumerate(TASKS):
        for size, color, marker, dy in SIZES:
            sp = list(seeds(T, "sp", size, FINAL, suite=suite, task=task).values())
            dc = list(seeds(T, "dclm", size, FINAL, suite=suite, task=task).values())
            d, lo, hi = interval(sp, dc)
            ax.plot([lo, hi], [y + dy] * 2, color=color, linewidth=1.0, zorder=2)
            ax.plot([d], [y + dy], marker, color=color, zorder=3,
                    label=f"{size} (SP n={len(sp)}, DCLM n={len(dc)})" if y == 0 else None)
            if len(dc) > 1:                       # difference against each DCLM seed
                ax.scatter([np.mean(sp) - v for v in dc], [y + dy] * len(dc), s=14, marker=marker,
                           facecolor="white", edgecolor=color, linewidth=0.7, zorder=4)
    ax.axhline(5.5, color=MUTED_LINE, linewidth=0.5)
    ax.set_yticks(range(len(TASKS)), [lab for *_, lab in TASKS])
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Accuracy difference, self-play − DCLM (positive: self-play better)")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="black")
    save(fig, "writeup/figs/fig3_tasks")


MUTED_LINE = "#BFBFBF"

if __name__ == "__main__":
    main()
