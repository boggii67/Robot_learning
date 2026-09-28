"""Plot an experiment created with scripts/run_experiment.py.

    python scripts/plot.py runs/lunar-add-one
    python scripts/plot.py runs/lunar-add-one --variants dqn dqn+double      # only some variants
    python scripts/plot.py runs/lunar-add-one --metric charts/episodic_return  # training instead of eval return
    python scripts/plot.py runs/lunar-add-one --metric train/loss              # any logged metric

Writes to runs/<experiment>/plots/:
    <metric>.png          learning curves: mean over seeds, shaded 95% CI (>= 3 seeds) or min-max range
    <metric>_summary.png  final score and area under the curve per variant (dots = single seeds)
    q_estimates.png       predicted Q vs. actual discounted return (shows overestimation, see Double DQN)
and prints the numbers as a table.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (colorblind-safe in this order). A variant keeps its slot in every plot.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# 95% two-sided t critical values by number of seeds n (df = n - 1); few seeds -> wider intervals
T_CRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
GRID_POINTS = 200

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False, "lines.solid_capstyle": "round",
})


def load_metric(exp_dir, variant, metric):
    """List of (steps, values) arrays, one per seed."""
    runs = []
    for csv_path in sorted(glob.glob(os.path.join(exp_dir, variant, "s*", "progress.csv"))):
        df = pd.read_csv(csv_path)
        df = df[df["name"] == metric].sort_values("step")
        if len(df) >= 2:
            runs.append((df["step"].to_numpy(), df["value"].to_numpy()))
    return runs


def to_grid(runs, smooth):
    """Put all seeds on a common step grid (up to the shortest run) -> (grid, [n_seeds, GRID_POINTS])."""
    end = min(steps[-1] for steps, _ in runs)
    grid = np.linspace(0, end, GRID_POINTS)
    curves = []
    for steps, values in runs:
        if smooth > 1:
            values = pd.Series(values).rolling(smooth, min_periods=1).mean().to_numpy()
        curves.append(np.interp(grid, steps, values))
    return grid, np.array(curves)


def spread(x):
    """Mean and a (lower, upper) band over axis 0 (seeds).

    >= 3 seeds: 95% confidence interval of the mean.
    <  3 seeds: min-max range of the seeds (a CI from 2 seeds is so wide it says nothing).
    """
    n = x.shape[0]
    mean = x.mean(0)
    if n >= 3:
        half = T_CRIT.get(n, 1.96) * x.std(0, ddof=1) / np.sqrt(n)
        return mean, mean - half, mean + half
    return mean, x.min(0), x.max(0)


def band_label(data):
    n_min = min(len(c) for _, c in data.values())
    return "95% CI over seeds" if n_min >= 3 else "range over seeds (use >= 3 seeds for a CI)"


def step_formatter(x, _):
    return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"


def style_axis(ax, ylabel):
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(step_formatter))
    ax.tick_params(length=0)


def plot_curves(data, colors, metric, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for variant, (grid, curves) in data.items():
        mean, lo, hi = spread(curves)
        ax.fill_between(grid, lo, hi, color=colors[variant], alpha=0.1 if len(data) > 3 else 0.15, linewidth=0)
        ax.plot(grid, mean, color=colors[variant], linewidth=2, label=f"{variant} (n={len(curves)})")
    if len(data) <= 4:  # direct value labels at the line ends (in text color), skipping ones that would collide
        ends = sorted((c.mean(0)[-1], g[-1]) for g, c in data.values())
        y_range = np.subtract(*ax.get_ylim()[::-1])
        last_y = -np.inf
        for y_end, x_end in ends:
            if y_end - last_y > 0.04 * y_range:
                ax.annotate(f"{y_end:.0f}", (x_end, y_end), xytext=(6, 0), textcoords="offset points",
                            va="center", color=TEXT_2, fontsize=9)
                last_y = y_end
    style_axis(ax, metric)
    ax.set_title(title)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), handlelength=1.5,
                       title=f"shaded: {band_label(data)}", title_fontsize=8, alignment="left")
    legend.get_title().set_color(TEXT_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(data, colors, metric, out_path, final_fraction):
    variants = list(data)
    finals = {v: c[:, -max(1, int(final_fraction * GRID_POINTS)):].mean(1) for v, (_, c) in data.items()}
    aucs = {v: c.mean(1) for v, (_, c) in data.items()}  # mean over training = normalized area under curve

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(variants) + 1.6), sharey=True)
    for ax, values, title in [(axes[0], finals, f"Final score (last {final_fraction:.0%} of training)"),
                              (axes[1], aucs, "Area under curve (mean over training)")]:
        y = np.arange(len(variants))[::-1]
        for yi, v in zip(y, variants):
            mean, lo, hi = spread(values[v][:, None])
            ax.barh(yi, mean[0], height=0.5, color=colors[v], zorder=2)
            ax.errorbar(mean[0], yi, xerr=[[mean[0] - lo[0]], [hi[0] - mean[0]]], color=TEXT, capsize=3,
                        linewidth=1, zorder=3)
            ax.scatter(values[v], np.full(len(values[v]), yi), s=12, color=TEXT, alpha=0.55, zorder=4,
                       edgecolors=SURFACE, linewidths=0.8)
            ax.annotate(f"{mean[0]:.0f}", (max(hi[0], values[v].max()), yi), xytext=(6, 0),
                        textcoords="offset points", va="center", color=TEXT_2, fontsize=9)
        ax.set_yticks(y, variants)
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=TEXT_2, linewidth=0.8, zorder=1)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.text(0.01, -0.02, f"bars: mean · dots: individual seeds · whiskers: {band_label(data)}",
             color=TEXT_2, fontsize=8, va="top")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return finals, aucs


def plot_q_estimates(exp_dir, variants, colors, out_path):
    """Small multiples: one panel per variant, predicted Q (solid) vs. actual discounted return (dashed)."""
    panels = []
    for v in variants:
        q, g = load_metric(exp_dir, v, "eval/q_mean"), load_metric(exp_dir, v, "eval/discounted_return")
        if q and g:
            panels.append((v, to_grid(q, 1), to_grid(g, 1)))
    if not panels:
        return False
    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, (v, (grid_q, q), (grid_g, g)) in zip(axes.flat, panels):
        for grid, curves, style, label in [(grid_q, q, "-", "predicted Q"), (grid_g, g, "--", "actual return")]:
            mean, lo, hi = spread(curves)
            ax.fill_between(grid, lo, hi, color=colors[v], alpha=0.12, linewidth=0)
            ax.plot(grid, mean, style, color=colors[v], linewidth=2, label=label)
        ax.set_title(v, fontsize=10)
        style_axis(ax, "discounted return")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="best")
    fig.suptitle("Value estimates vs. reality (solid above dashed = overestimation)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp_dir", help="runs/<experiment>")
    p.add_argument("--metric", default="eval/return")
    p.add_argument("--variants", nargs="*", help="subset of variants to plot (default: all)")
    p.add_argument("--smooth", type=int, default=None,
                   help="rolling mean window over logged points (default: 20 for episodic returns, else 1)")
    p.add_argument("--final-fraction", type=float, default=0.1)
    args = p.parse_args()

    order_path = os.path.join(args.exp_dir, "variants.json")
    all_variants = json.load(open(order_path)) if os.path.exists(order_path) else sorted(
        d for d in os.listdir(args.exp_dir) if glob.glob(os.path.join(args.exp_dir, d, "s*", "progress.csv")))
    if len(all_variants) > len(PALETTE):
        print(f"Note: {len(all_variants)} variants but only {len(PALETTE)} distinguishable colors; "
              "plot a subset with --variants.")
    # Color follows the variant (its position in the experiment), not its position in this plot
    colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(all_variants)}
    variants = [v for v in all_variants if not args.variants or v in args.variants]
    smooth = args.smooth or (20 if args.metric == "charts/episodic_return" else 1)

    data = {}
    for v in variants:
        runs = load_metric(args.exp_dir, v, args.metric)
        if runs:
            data[v] = to_grid(runs, smooth)
        else:
            print(f"(no '{args.metric}' data for {v} yet)")
    if not data:
        raise SystemExit("Nothing to plot.")

    out_dir = os.path.join(args.exp_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.normpath(args.exp_dir))
    metric_file = args.metric.replace("/", "_")

    plot_curves(data, colors, args.metric, os.path.join(out_dir, f"{metric_file}.png"),
                f"{exp_name}: {args.metric}")
    finals, aucs = plot_summary(data, colors, args.metric, os.path.join(out_dir, f"{metric_file}_summary.png"),
                                args.final_fraction)
    has_q = plot_q_estimates(args.exp_dir, list(data), colors, os.path.join(out_dir, "q_estimates.png"))

    # Table view of the same numbers
    print(f"\n{args.metric}  (band = {band_label(data)})")
    print(f"{'variant':<34}{'seeds':>6}{'final':>10}{'band':>18}{'AUC':>10}{'band':>18}")
    for v in data:
        f_mean, f_lo, f_hi = spread(finals[v][:, None])
        a_mean, a_lo, a_hi = spread(aucs[v][:, None])
        print(f"{v:<34}{len(finals[v]):>6}{f_mean[0]:>10.1f}{f'[{f_lo[0]:.0f}, {f_hi[0]:.0f}]':>18}"
              f"{a_mean[0]:>10.1f}{f'[{a_lo[0]:.0f}, {a_hi[0]:.0f}]':>18}")
    print(f"\nPlots written to {out_dir}/: {metric_file}.png, {metric_file}_summary.png"
          + (", q_estimates.png" if has_q else ""))


if __name__ == "__main__":
    main()
