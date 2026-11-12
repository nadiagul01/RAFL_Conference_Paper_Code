"""
Tier 1 (fast sweep) and Tier 2 (checkpointed paper-scale) experiment drivers.
See Corrected_Methodology.md Section 6.

Every sweep/ablation function below accepts a `seeds` tuple (default (42,), i.e. the
original single-seed behavior). Passing more than one seed reruns the same config
under each seed and reports mean +/- std -- required for a Q1-grade results section,
since a single run carries no information about run-to-run variance. Each function
writes, per sweep point: a per-seed per-round CSV, a long-format CSV across all
seeds, a mean+-std comparison table (CSV/Markdown/LaTeX), and the full plot set from
plots.py -- not just a JSON blob.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from rafl_unified import Config, set_seed, load_dataset, dirichlet_split
from rafl_runners import run_all_methods, RUNNERS, summarize_history
from reporting import (
    save_history_csv, save_long_format_csv, aggregate_summaries,
    comparison_table_rows, save_comparison_table,
)
from plots import plot_all_metrics, plot_ablation_bars


def build_round(cfg_kwargs):
    cfg = Config(**cfg_kwargs)
    set_seed(cfg.seed)
    trainset, testset, num_classes = load_dataset(cfg)
    train_labels = np.array(trainset.targets)
    client_indices = dirichlet_split(train_labels, cfg.num_clients, cfg.alpha, cfg.dirichlet_min_size)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return cfg, trainset, testset, test_loader, client_indices, num_classes


def _save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _run_methods_across_seeds(base_kwargs: dict, methods, seeds, overrides=None, resume=False, tag=""):
    """Runs `methods` under `base_kwargs` (+ optional per-run Config overrides, e.g.
    for ablation variants) once per seed in `seeds`, each in its own save_dir so
    checkpoints/results never collide across seeds. Returns
    (method_seed_summary, method_seed_history) both shaped {method: {seed: ...}}."""
    overrides = overrides or {}
    method_seed_summary = {m: {} for m in methods}
    method_seed_history = {m: {} for m in methods}
    root_save_dir = base_kwargs["save_dir"]
    for seed in seeds:
        print(f"\n{'-' * 70}\n{tag} seed={seed}\n{'-' * 70}")
        kwargs = {**base_kwargs, **overrides, "seed": seed, "baseline_names": tuple(methods),
                  "save_dir": f"{root_save_dir}/seed_{seed}"}
        cfg, trainset, testset, test_loader, client_indices, num_classes = build_round(kwargs)
        outputs = run_all_methods(cfg, trainset, testset, test_loader, client_indices, num_classes, resume=resume)
        for m in methods:
            method_seed_summary[m][seed] = outputs[m]["summary"]
            method_seed_history[m][seed] = outputs[m]["history"]
            save_history_csv(outputs[m]["history"], f"{root_save_dir}/seed_{seed}/{m}_history.csv")
    return method_seed_summary, method_seed_history


def _report_sweep_point(root_save_dir, method_seed_summary, method_seed_history):
    """Writes the comparison table (CSV/MD/LaTeX), the long-format all-rounds CSV,
    and the full plot set for one sweep point (one alpha value, one proportion, ...).
    Returns {method: aggregate_summaries() output} for the caller to fold into the
    sweep-level JSON summary."""
    methods = list(method_seed_summary.keys())
    method_agg = {m: aggregate_summaries(method_seed_summary[m]) for m in methods}
    rows = comparison_table_rows(method_agg)
    save_comparison_table(rows, f"{root_save_dir}/comparison_table.csv",
                           f"{root_save_dir}/comparison_table.md", f"{root_save_dir}/comparison_table.tex")
    save_long_format_csv(method_seed_history, f"{root_save_dir}/all_rounds_long.csv")
    plot_all_metrics({m: list(method_seed_history[m].values()) for m in methods}, f"{root_save_dir}/plots")
    return method_agg


# ---------------------------------------------------------------------
# TIER 1a: non-IID (alpha) sweep
# ---------------------------------------------------------------------
def run_tier1_alpha_sweep(base_kwargs: dict, alphas=(0.1, 0.3, 1.0),
                           methods=("fedavg", "heterofl", "adaptivefl", "rafl"), seeds=(42,)):
    all_results = {}
    for alpha in alphas:
        print(f"\n{'#' * 70}\nTIER1 alpha sweep: alpha={alpha} ({len(seeds)} seed(s))\n{'#' * 70}")
        alpha_save_dir = f"{base_kwargs['save_dir']}/tier1_alpha_{alpha}"
        kwargs = {**base_kwargs, "save_dir": alpha_save_dir}
        method_seed_summary, method_seed_history = _run_methods_across_seeds(
            kwargs, methods, seeds, overrides={"alpha": alpha}, resume=False,
            tag=f"TIER1 alpha={alpha}")
        method_agg = _report_sweep_point(alpha_save_dir, method_seed_summary, method_seed_history)
        all_results[f"alpha_{alpha}"] = method_agg
        _save_json(f"{base_kwargs['save_dir']}/tier1_alpha_sweep_partial.json", all_results)
    return all_results


# ---------------------------------------------------------------------
# TIER 1b: weak:medium:strong proportion sweep (mirrors base paper Table 3)
# ---------------------------------------------------------------------
def run_tier1_proportion_sweep(base_kwargs: dict,
                                proportions=((4, 3, 3), (8, 1, 1), (1, 8, 1), (1, 1, 8)),
                                methods=("heterofl", "adaptivefl", "rafl"), seeds=(42,)):
    all_results = {}
    for (w, m, s) in proportions:
        tag = f"{w}_{m}_{s}"
        print(f"\n{'#' * 70}\nTIER1 proportion sweep: weak:medium:strong = {tag} ({len(seeds)} seed(s))\n{'#' * 70}")
        prop_save_dir = f"{base_kwargs['save_dir']}/tier1_proportion_{tag}"
        kwargs = {**base_kwargs, "save_dir": prop_save_dir}
        method_seed_summary, method_seed_history = _run_methods_across_seeds(
            kwargs, methods, seeds, overrides={"weak_ratio": w, "medium_ratio": m, "strong_ratio": s},
            resume=False, tag=f"TIER1 proportion={tag}")
        method_agg = _report_sweep_point(prop_save_dir, method_seed_summary, method_seed_history)
        all_results[f"proportion_{tag}"] = method_agg
        _save_json(f"{base_kwargs['save_dir']}/tier1_proportion_sweep_partial.json", all_results)
    return all_results


# ---------------------------------------------------------------------
# TIER 1c: sync vs. hybrid aggregation ablation for RAFL (answers RQ3)
# ---------------------------------------------------------------------
def run_tier1_aggregation_ablation(base_kwargs: dict, modes=("sync", "async", "hybrid"), seeds=(42,)):
    abl_save_dir = f"{base_kwargs['save_dir']}/tier1_aggregation_ablation"
    variants = {mode: {"aggregation_mode": mode} for mode in modes}
    return _run_component_ablation(base_kwargs, abl_save_dir, variants, seeds,
                                    tag_prefix="TIER1 aggregation ablation")


# ---------------------------------------------------------------------
# TIER 1d: leave-one-out ablation over RAFL's three additive layers
# (capability gate / fairness+energy reward / hybrid aggregation) -- answers
# "which of RAFL's three contributions actually drives the improvement."
# See Corrected_Methodology.md Section 4.
# ---------------------------------------------------------------------
RAFL_ABLATION_VARIANTS = {
    "full":               dict(use_capability_gate=True,  use_fairness_energy_reward=True,  aggregation_mode="sync"),
    "no_capability_gate": dict(use_capability_gate=False, use_fairness_energy_reward=True,  aggregation_mode="sync"),
    "no_fairness_energy": dict(use_capability_gate=True,  use_fairness_energy_reward=False, aggregation_mode="sync"),
}


def run_tier1_component_ablation(base_kwargs: dict, variants=None, seeds=(42,)):
    abl_save_dir = f"{base_kwargs['save_dir']}/tier1_component_ablation"
    variants = variants or RAFL_ABLATION_VARIANTS
    return _run_component_ablation(base_kwargs, abl_save_dir, variants, seeds,
                                    tag_prefix="TIER1 component ablation")


def _run_component_ablation(base_kwargs, abl_save_dir, variants: dict, seeds, tag_prefix):
    """Shared machinery for both ablation drivers above: run RAFL once per
    (variant, seed), aggregate across seeds, and write the same table/CSV/plot
    artifacts as a regular sweep point, plus ablation-specific bar charts."""
    variant_seed_summary = {v: {} for v in variants}
    variant_seed_history = {v: {} for v in variants}
    for variant_name, overrides in variants.items():
        for seed in seeds:
            print(f"\n{'#' * 70}\n{tag_prefix}: variant={variant_name} seed={seed}\n{'#' * 70}")
            kwargs = {**base_kwargs, **overrides, "baseline_names": ("rafl",), "seed": seed,
                      "save_dir": f"{abl_save_dir}/{variant_name}/seed_{seed}"}
            cfg, trainset, testset, test_loader, client_indices, num_classes = build_round(kwargs)
            _, history, elapsed = RUNNERS["rafl"](
                cfg, trainset, testset, test_loader, client_indices, num_classes,
                method_name=f"rafl_{variant_name}", resume=False)
            variant_seed_summary[variant_name][seed] = summarize_history(history, elapsed)
            variant_seed_history[variant_name][seed] = history
            save_history_csv(history, f"{abl_save_dir}/{variant_name}/seed_{seed}/history.csv")

    variant_agg = _report_sweep_point(abl_save_dir, variant_seed_summary, variant_seed_history)
    plot_ablation_bars(variant_agg, f"{abl_save_dir}/plots")
    _save_json(f"{abl_save_dir}/ablation_summary.json", variant_agg)
    return variant_agg


def benchmark_seconds_per_round(save_dir, data_root, n_rounds=10, num_clients=100, participation_rate=0.10):
    """Run n_rounds real RAFL rounds at full scale and report seconds/round plus an
    estimated GPU-hour cost for a full Tier-2 run (800 rounds x 4 methods). Run this
    on whatever GPU you're actually using BEFORE committing to a long run -- the
    numbers in Corrected_Methodology.md Section 6 are planning estimates, not
    measurements for your specific hardware."""
    bench_kwargs = dict(
        dataset="cifar10", data_root=data_root, save_dir=f"{save_dir}/_bench",
        num_clients=num_clients, participation_rate=participation_rate, local_epochs=5, batch_size=64,
        num_rounds=n_rounds, eval_every=n_rounds, checkpoint_every=n_rounds + 1,
        baseline_names=("rafl",),
    )
    cfg, trainset, testset, test_loader, client_indices, num_classes = build_round(bench_kwargs)
    _, history, elapsed = RUNNERS["rafl"](cfg, trainset, testset, test_loader, client_indices, num_classes,
                                           method_name="_bench_rafl", resume=False)
    per_round = elapsed / n_rounds
    print(f"RAFL: {elapsed:.1f}s for {n_rounds} rounds -> {per_round:.2f}s/round at "
          f"{num_clients} clients, {participation_rate:.0%} participation.")
    print(f"Estimated cost of a full Tier-2 run (800 rounds, 4 methods): "
          f"~{per_round * 800 * 4 / 3600:.1f} GPU-hours (x n_seeds if you run Tier 2 with multiple seeds).")
    return per_round


def tier1_default_config(save_dir="/kaggle/working/RAFL_Results", data_root="/kaggle/working/data"):
    """Fast-sweep scale: compact backbone, 50 clients, 20% participation,
    200-300 rounds per config — designed to fit comfortably inside one
    Kaggle session per sweep entry. Benchmark 10 rounds first and adjust
    num_rounds if your GPU is slower/faster than assumed."""
    return dict(
        dataset="cifar10", data_root=data_root, save_dir=save_dir,
        num_clients=50, participation_rate=0.20, local_epochs=3, batch_size=64,
        num_rounds=250, eval_every=10, checkpoint_every=250,  # tier1 runs don't need mid-run checkpoints
    )


# ---------------------------------------------------------------------
# TIER 2: paper-scale validation, checkpointed across sessions
# ---------------------------------------------------------------------
def tier2_default_config(save_dir="/kaggle/working/RAFL_Results/tier2", data_root="/kaggle/working/data"):
    """Paper-scale: 100 clients, 10% participation (matches AdaptiveFL's
    protocol), alpha=0.3, proportion 4:3:3 (base paper's hardest reported
    non-IID case). num_rounds is a *target ceiling*: with checkpointing,
    re-running this same cell in later Kaggle sessions resumes from the
    last saved checkpoint instead of restarting. IMPORTANT: /kaggle/working
    is only persisted within one session unless you commit the kernel and
    carry `RAFL_Results/checkpoints/` forward as a Kaggle Dataset input to
    the next session — see the README markdown cell for the exact steps."""
    return dict(
        dataset="cifar10", data_root=data_root, save_dir=save_dir,
        num_clients=100, participation_rate=0.10, local_epochs=5, batch_size=64,
        alpha=0.3, weak_ratio=4, medium_ratio=3, strong_ratio=3,
        num_rounds=800, eval_every=20, checkpoint_every=25,
        baseline_names=("fedavg", "heterofl", "adaptivefl", "rafl"),  # scalefl/decoupled optional, see README
    )


def run_tier2(cfg_kwargs=None, resume=True, seeds=(42,)):
    cfg_kwargs = cfg_kwargs or tier2_default_config()
    methods = list(cfg_kwargs.get("baseline_names", tier2_default_config()["baseline_names"]))
    root_save_dir = cfg_kwargs["save_dir"]
    print(f"Tier 2 run: {cfg_kwargs.get('num_clients')} clients, "
          f"{cfg_kwargs.get('num_rounds')} round ceiling, "
          f"checkpoint every {cfg_kwargs.get('checkpoint_every')} rounds, "
          f"resume={resume}, seeds={list(seeds)}")

    method_seed_summary, method_seed_history = _run_methods_across_seeds(
        cfg_kwargs, methods, seeds, resume=resume, tag="TIER2")

    method_agg = _report_sweep_point(root_save_dir, method_seed_summary, method_seed_history)

    summary_rows = comparison_table_rows(method_agg)
    summary_df = pd.DataFrame(summary_rows)
    sort_col = "best_global_acc_mean" if "best_global_acc_mean" in summary_df.columns else None
    if sort_col:
        summary_df = summary_df.sort_values(sort_col, ascending=False)
    _save_json(f"{root_save_dir}/tier2_summary.json", summary_rows)
    summary_df.to_csv(f"{root_save_dir}/tier2_summary.csv", index=False)

    # outputs kept in the old {method: {"history":..., "summary":...}} shape for
    # backward compatibility, using the first seed's run as the representative one.
    outputs = {m: {"history": method_seed_history[m][seeds[0]], "summary": method_seed_summary[m][seeds[0]]}
               for m in methods}
    return outputs, summary_df
