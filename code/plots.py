"""
Publication-grade plotting for RAFL results.

Every curve function accepts {method_name: [history_dict, ...]} -- a *list* of
per-round history dicts, one per seed, for each method. With one seed it plots a
single line; with more than one it plots the mean with a shaded +/-1 std band, which
is what a Q1 reviewer expects a convergence figure to show instead of a single
lucky/unlucky run. Headless-safe (Agg backend) for a RunPod/CI environment with no
display.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _stack_metric(histories, metric):
    """histories: list of history dicts for ONE method (one per seed). Truncates to
    the shortest history so partially-completed/resumed runs of differing length
    don't crash the stack -- this is expected to happen with independently
    checkpointed multi-seed Tier-2 runs."""
    min_len = min(len(h[metric]) for h in histories)
    rounds = histories[0]["round"][:min_len]
    arrs = np.array([h[metric][:min_len] for h in histories], dtype=float)
    mean = arrs.mean(axis=0)
    std = arrs.std(axis=0) if len(histories) > 1 else np.zeros_like(mean)
    return rounds, mean, std


def plot_metric(method_histories: dict, metric: str, ylabel: str, title: str, out_path,
                 lower_is_better: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for method, histories in method_histories.items():
        histories = [h for h in histories if h.get(metric) and h.get("round")]
        if not histories:
            continue
        rounds, mean, std = _stack_metric(histories, metric)
        ax.plot(rounds, mean, marker="o", markersize=3, label=method)
        if len(histories) > 1:
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title + (" (lower is fairer)" if lower_is_better else ""))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tier_accuracy(method_histories: dict, out_dir):
    """One figure per client tier (weak/medium/strong submodel accuracy), all
    methods overlaid -- shows whether an approach helps weak-tier clients at the
    expense of strong ones, which the scalar avg_submodel_acc metric hides."""
    for tier in ("weak", "medium", "strong"):
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for method, histories in method_histories.items():
            histories = [h for h in histories if h.get("round") and h.get("tier_acc")]
            if not histories:
                continue
            min_len = min(len(h["tier_acc"]) for h in histories)
            rounds = histories[0]["round"][:min_len]
            arrs = np.array([[ta.get(tier, np.nan) for ta in h["tier_acc"][:min_len]] for h in histories],
                             dtype=float)
            mean = np.nanmean(arrs, axis=0)
            std = np.nanstd(arrs, axis=0) if len(histories) > 1 else np.zeros_like(mean)
            ax.plot(rounds, mean, marker="o", markersize=3, label=method)
            if len(histories) > 1:
                ax.fill_between(rounds, mean - std, mean + std, alpha=0.15)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{tier.capitalize()}-tier submodel accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = Path(out_dir) / f"tier_accuracy_{tier}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_all_metrics(method_histories: dict, out_dir):
    """Produces the full set of per-round convergence/behavior figures used for the
    paper: global accuracy, avg submodel accuracy, per-tier accuracy, both fairness
    metrics, cumulative comm cost, cumulative energy cost, comm waste, and (where
    present) the capability-gate exclusion rate."""
    out_dir = Path(out_dir)
    plot_metric(method_histories, "global_acc", "Accuracy", "Global model accuracy",
                out_dir / "global_acc.png")
    plot_metric(method_histories, "avg_submodel_acc", "Accuracy", "Average submodel accuracy",
                out_dir / "avg_submodel_acc.png")
    plot_metric(method_histories, "selection_fairness", "F_select", "Selection fairness",
                out_dir / "selection_fairness.png")
    plot_metric(method_histories, "perf_fairness", "F_perf", "Performance fairness",
                out_dir / "perf_fairness.png", lower_is_better=True)
    plot_metric(method_histories, "comm_cost_cumulative", "Params transmitted (cumulative)",
                "Communication cost", out_dir / "comm_cost.png")
    plot_metric(method_histories, "energy_cost_cumulative", "Energy units (cumulative)",
                "Energy cost", out_dir / "energy_cost.png")
    plot_metric(method_histories, "comm_waste", "Fraction wasted", "Communication waste per round",
                out_dir / "comm_waste.png")
    plot_metric(method_histories, "gated_out_frac", "Fraction gated out", "Capability-gate exclusion rate",
                out_dir / "gated_out_frac.png")
    plot_tier_accuracy(method_histories, out_dir)


def plot_bar_with_error(labels, means, stds, ylabel, title, out_path):
    """Grouped bar chart with error bars -- for final headline-number comparisons
    (e.g. best global accuracy per method) and for the component-ablation table."""
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(variant_to_agg: dict, out_dir, metrics=None):
    """variant_to_agg: {variant_name: aggregate_summaries() output} from the
    component-ablation driver. One bar chart per metric, comparing RAFL-full against
    each leave-one-out variant with error bars across seeds."""
    metrics = metrics or [
        ("best_global_acc", "Accuracy", "Best global accuracy"),
        ("final_selection_fairness", "F_select", "Selection fairness"),
        ("final_perf_fairness", "F_perf", "Performance fairness"),
        ("final_energy_cost", "Energy units", "Final cumulative energy cost"),
    ]
    labels = list(variant_to_agg.keys())
    out_dir = Path(out_dir)
    for metric_key, ylabel, title in metrics:
        means, stds = [], []
        for variant in labels:
            entry = variant_to_agg[variant].get(metric_key, {})
            means.append(entry.get("mean", 0.0) or 0.0)
            stds.append(entry.get("std", 0.0) or 0.0)
        plot_bar_with_error(labels, means, stds, ylabel, title, out_dir / f"ablation_{metric_key}.png")
