"""Fig. 4: self-play warm start vs DCLM from scratch, as DCLM tokens accumulate.

Regenerate: python writeup/figs/fig4_warm.py
"""
import numpy as np
from io_data import rows, seeds
from orx_figstyle import BASELINE, PALETTE, WIDE, figure_grid, panel_labels, save, use_style

SIZES = [("3M", PALETTE["green"]), ("6M", PALETTE["blue"]), ("24M", PALETTE["orange"])]
SUITES = [("word", "Word-task accuracy"), ("text", "Printable-text accuracy"), ("raw", "Raw-byte accuracy")]
WARM = [0.05, 0.1, 0.25, 0.5]
SCRATCH = [0.1, 0.25, 0.5, 1.0, 1.6169, 3.2275, 6.4487, 12.8912, 17.723]
START_X = 0.025                                   # where the "0 DCLM tokens" point is drawn


def one(H, arm, size, t, suite):
    v = seeds(H, arm, size, t, suite=suite, seed="seed-0")
    return v.get("seed-0", np.nan)


def main():
    use_style()
    H = rows("headline.csv")
    fig, axes = figure_grid(1, 3, width=WIDE, ratio=0.34, sharey=True)
    for ax, (suite, ylab) in zip(axes, SUITES):
        for size, color in SIZES:
            start = np.mean(list(seeds(H, "sp", size, 51.5396, suite=suite).values()))
            warm = [one(H, "sp2dclm", size, t, suite) for t in WARM]
            ax.plot([START_X] + WARM, [start] + warm, "o-", color=color, markersize=3,
                    label=f"{size} self-play → DCLM")
            ax.plot(SCRATCH, [one(H, "dclm", size, t, suite) for t in SCRATCH], "--",
                    color=color, linewidth=0.9, label=f"{size} DCLM from scratch")
        ax.axvline(0.5, color=BASELINE, linewidth=0.6, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks([START_X, 0.1, 0.5, 2, 17.7], ["0", "0.1", "0.5", "2", "17.7"])
        ax.minorticks_off()
        ax.set_xlabel("DCLM tokens trained (B)")
        ax.text(0.97, 0.97, ylab, transform=ax.transAxes, ha="right", va="top", fontsize=7)
    axes[0].set_ylabel("Mean exact-match accuracy")
    axes[0].set_ylim(-0.02, 0.75)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    panel_labels(axes)
    save(fig, "writeup/figs/fig4_warm")


if __name__ == "__main__":
    main()
