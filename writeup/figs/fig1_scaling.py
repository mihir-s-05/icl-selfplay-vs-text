"""Fig. 1: ICL vs model size at 17.7B learner tokens, three task formats.

Regenerate: python writeup/figs/fig1_scaling.py
"""
import numpy as np
from io_data import FINAL, N_NONEMB, SIZES, rows, seeds, up_tokens
from orx_figstyle import BASELINE, PALETTE, WIDE, figure_grid, mean_ci, panel_labels, save, si_ticks, use_style

SUITES = [("text", "Printable-text tasks"), ("word", "Word tasks"), ("raw", "Raw-byte tasks (paper's format)")]


def main():
    use_style()
    H = rows("headline.csv")
    fig, axes = figure_grid(1, 3, width=WIDE, ratio=0.33, sharey=True)
    for ax, (suite, name) in zip(axes, SUITES):
        x_sp, m_sp, lo_sp, hi_sp = [], [], [], []
        for size in SIZES:
            v = list(seeds(H, "sp", size, FINAL, suite=suite).values())
            m, lo, hi = mean_ci(np.array(v)[:, None])
            x_sp.append(N_NONEMB[size]); m_sp.append(m[0]); lo_sp.append(lo[0]); hi_sp.append(hi[0])
        x_sp = np.array(x_sp)
        ax.fill_between(x_sp, np.clip(lo_sp, 0, 1), np.clip(hi_sp, 0, 1), color=PALETTE["red"],
                        alpha=0.18, linewidth=0)
        ax.plot(x_sp, m_sp, "o-", color=PALETTE["red"], label="Self-play (4 seeds, 95% CI)")

        xd, md = [], []
        for size in SIZES:
            v = list(seeds(H, "dclm", size, FINAL, suite=suite).values())
            ax.scatter([N_NONEMB[size]] * len(v), v, s=10, facecolor="none",
                       edgecolor=PALETTE["blue"], linewidth=0.7, zorder=3)
            xd.append(N_NONEMB[size]); md.append(np.mean(v))
        ax.plot(xd, md, "s--", color=PALETTE["blue"], markersize=3,
                label="DCLM text (seeds hollow, mean filled)")

        xu, mu = [], []
        for size in SIZES[:-1]:
            v = list(seeds(H, "up", size, up_tokens(size), suite=suite).values())
            xu.append(N_NONEMB[size]); mu.append(np.mean(v))
        ax.plot(xu, mu, "^:", color=BASELINE, markersize=3, label="Universal prior (4 seeds)")

        ax.set_xscale("log")
        ax.set_xlabel("Non-embedding parameters")
        ax.text(0.03, 0.97, name, transform=ax.transAxes, va="top", fontsize=7)
        si_ticks(ax, "x")
    axes[0].set_ylabel("Mean exact-match accuracy")
    axes[0].set_ylim(-0.02, 0.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    panel_labels(axes)
    save(fig, "writeup/figs/fig1_scaling")


if __name__ == "__main__":
    main()
