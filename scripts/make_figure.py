"""Hero figure for the README: hedging error on real crashes, trained vs untrained, with the scripted
reference players as context. Numbers are the ones printed by transfer_analyze.py / baselines.py.
Produces assets/hero-light.png and assets/hero-dark.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROWS = [  # (label, 2020 loss, 2022 loss, emphasis)
    ("Never hedges", 0.0388, 0.0293, False),
    ("Untrained model", 0.0352, 0.0281, False),
    ("Constant 50% hedge", 0.0203, 0.0149, False),
    ("Trained only in synthetic worlds", 0.0173, 0.0144, True),
]
DELTA = {1: 0.0120, 2: 0.0091}  # textbook delta hedge: drawn as a reference line, not a competing bar
ACCENT = "#2563eb"


def draw(mode: str):
    dark = mode == "dark"
    surface = "#0d1117" if dark else "#ffffff"
    ink = "#e6edf3" if dark else "#1f2328"
    muted = "#8b949e" if dark else "#6e7781"
    gray = "#3d444d" if dark else "#d0d7de"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), dpi=200, facecolor=surface)
    for ax, (col, title) in zip(axes, ((1, "Real 2020 crash"), (2, "Real 2022 crash"))):
        ax.set_facecolor(surface)
        labels = [r[0] for r in ROWS][::-1]; vals = [r[col] for r in ROWS][::-1]; emph = [r[3] for r in ROWS][::-1]
        y = range(len(ROWS))
        for yi, v, e, lab in zip(y, vals, emph, labels):
            ax.add_patch(FancyBboxPatch((0, yi - 0.28), v, 0.56, boxstyle="round,pad=0,rounding_size=0.0012",
                                        linewidth=0, facecolor=ACCENT if e else gray, mutation_aspect=1))
            ax.text(v + 0.0008, yi, f"{v:.4f}", va="center", ha="left", fontsize=9,
                    color=ink if e else muted, fontweight="bold" if e else "normal")
        ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=9.5, color=ink)
        for lab, e in zip(ax.get_yticklabels(), emph):
            lab.set_fontweight("bold" if e else "normal"); lab.set_color(ink if e else muted)
        ax.axvline(DELTA[col], color=ink, linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.75, zorder=3)
        ax.text(DELTA[col] + 0.0008, len(ROWS) - 0.45, f"textbook delta hedge {DELTA[col]:.4f}", fontsize=8, color=muted,
                ha="left", va="bottom", rotation=0)
        ax.set_xlim(0, 0.046); ax.set_ylim(-0.6, len(ROWS) - 0.1)
        ax.set_title(title, loc="left", fontsize=11.5, color=ink, fontweight="bold", pad=8)
        ax.tick_params(axis="x", colors=muted, labelsize=8, length=0); ax.tick_params(axis="y", length=0)
        ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04]); ax.set_xticklabels(["0", ".01", ".02", ".03", ".04"])
        ax.grid(axis="x", color=gray, linewidth=0.6, alpha=0.5 if dark else 0.8); ax.set_axisbelow(True)
        for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.01, 0.015, "Mean hedging error as a fraction of the option strike, 21-day windows, four US ETFs. Lower is better. "
             "Model trained 3,840 episodes on synthetic worlds only.", fontsize=7.5, color=muted)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"assets/hero-{mode}.png", facecolor=surface); plt.close(fig)


if __name__ == "__main__":
    for m in ("light", "dark"):
        draw(m)
    print("written assets/hero-light.png assets/hero-dark.png")
