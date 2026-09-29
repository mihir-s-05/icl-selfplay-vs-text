"""Fig. 5: accuracy vs number of in-context examples at 17.7B tokens (6M and 24M).

Regenerate: python writeup/figs/fig5_mcurves.py
"""
import numpy as np
from io_data import as_runs, curve, rows
from orx_figstyle import PALETTE, WIDE, band, figure_grid, panel_labels, save, use_style

TASKS = [("text", "last", "text: last letter"), ("text", "stack", "text: stack"),
         ("text", "assoc", "text: key→value"),
         ("word", "classify (unseen word, 4 labels)", "word: unseen category")]
SIZES = ["6M", "24M"]


def main():
    use_style()
    M = rows("mcurves.csv")
    fig, axes = figure_grid(2, 4, width=WIDE, ratio=0.5, sharey=True, sharex="col")
    for r, size in enumerate(SIZES):
        for c, (suite, task, name) in enumerate(TASKS):
            ax = axes[r][c]
            ms, sp = as_runs(curve(M, "sp", size, suite, task))
            band(ax, ms + 0.5, sp, color=PALETTE["red"], marker="o", markersize=2,
                 label="Self-play (4 seeds, 95% CI)")
            by_m = curve(M, "dclm", size, suite, task)
            ms_d, dc = as_runs(by_m)
            for i in range(dc.shape[0]):
                ax.plot(ms_d + 0.5, dc[i], "s--", color=PALETTE["blue"], markersize=2, linewidth=0.9,
                        label="DCLM text (each seed)" if i == 0 else None)
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.5, 2.5, 16.5, 128.5], ["0", "2", "16", "128"])
            ax.minorticks_off()
            if r == 0:
                ax.text(0.5, 1.03, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
            if r == 1:
                ax.set_xlabel("In-context examples m")
            if c == 0:
                ax.set_ylabel(f"{size}: accuracy")
    axes[0][0].set_ylim(-0.02, 1.02)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    panel_labels(axes)
    save(fig, "writeup/figs/fig5_mcurves")


if __name__ == "__main__":
    main()
