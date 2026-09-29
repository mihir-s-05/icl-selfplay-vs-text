"""Fig. 2: ICL vs learner tokens, self-play vs DCLM, per model size.

Regenerate: python writeup/figs/fig2_tokens.py
"""
import numpy as np
from io_data import MATCH, rows, seeds
from orx_figstyle import PALETTE, WIDE, band, figure_grid, panel_labels, save, use_style

SIZES = ["1M", "3M", "6M", "24M"]
SUITES = [("text", "Printable-text\naccuracy"), ("word", "Word-task\naccuracy")]


def main():
    use_style()
    H = rows("headline.csv")
    fig, axes = figure_grid(2, 4, width=WIDE, ratio=0.5, sharex=True, sharey=True)
    x = np.array(MATCH)
    for r, (suite, ylab) in enumerate(SUITES):
        for c, size in enumerate(SIZES):
            ax = axes[r][c]
            sp = np.array([list(seeds(H, "sp", size, t, suite=suite).values()) for t in MATCH]).T
            band(ax, x, sp, color=PALETTE["red"], marker="o", markersize=2.5,
                 label="Self-play (4 seeds, 95% CI)")
            dc = [seeds(H, "dclm", size, t, suite=suite) for t in MATCH]
            for seed in sorted(dc[0]):
                ax.plot(x, [d[seed] for d in dc], "s--", color=PALETTE["blue"], markersize=2.5,
                        linewidth=0.9, label="DCLM text (each seed)" if seed == "seed-0" else None)
            ax.set_xscale("log")
            ax.set_xticks([2, 5, 10, 17.7], ["2", "5", "10", "17.7"])
            ax.minorticks_off()
            if r == 0:
                ax.text(0.5, 1.02, size, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
            if r == 1:
                ax.set_xlabel("Learner tokens (B)")
            if c == 0:
                ax.set_ylabel(ylab)
    axes[0][0].set_ylim(-0.02, 0.75)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    panel_labels(axes)
    save(fig, "writeup/figs/fig2_tokens")


if __name__ == "__main__":
    main()
