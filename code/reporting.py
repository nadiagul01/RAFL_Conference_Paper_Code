"""
Publication-grade reporting for RAFL results.

- TeeLogger: duplicates stdout into a timestamped .log file without touching any of
  the existing print() call sites throughout rafl_unified.py / rafl_runners.py.
- save_history_csv / save_long_format_csv: per-round metric export. The long-format
  CSV (one row per method/seed/round) is the single reusable artifact everything
  else -- every plot, every table -- can be regenerated from, which is what a
  reproducibility/supplementary-material reviewer will actually want.
- aggregate_summaries / comparison_table_rows / save_comparison_table: turn a set of
  per-seed summarize_history() outputs into mean +/- std tables (CSV, Markdown, and
  LaTeX), degrading gracefully to a bare value when only one seed was run.

See Corrected_Methodology.md and RAFL_README.md for what these numbers mean.
"""
import csv
import datetime
import statistics
import sys
from pathlib import Path


class TeeLogger:
    """Context manager: while active, everything written to sys.stdout is also
    appended to a timestamped log file on disk. Use around a whole run so a crash,
    a dropped SSH session, or a closed terminal doesn't lose the run's history --
    only the terminal scrollback is ephemeral, the .log file isn't.
    """

    def __init__(self, log_path):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8")
        self._stdout = None
        self._line_start = True

    def write(self, text):
        self._stdout.write(text)
        for line in text.splitlines(keepends=True):
            if self._line_start and line.strip():
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._file.write(f"[{ts}] {line}")
            else:
                self._file.write(line)
            self._line_start = line.endswith("\n")
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        self._file.close()


_HISTORY_FIELDS = [
    "round", "global_acc", "global_loss", "avg_submodel_acc",
    "tier_acc_weak", "tier_acc_medium", "tier_acc_strong",
    "comm_waste", "comm_cost_cumulative", "energy_cost_cumulative",
    "selection_fairness", "perf_fairness", "gated_out_frac",
]


def _history_rows(history: dict):
    rounds = history.get("round", [])
    rows = []
    for i, r in enumerate(rounds):
        tier_acc = history["tier_acc"][i] if i < len(history.get("tier_acc", [])) else {}
        rows.append({
            "round": r,
            "global_acc": history["global_acc"][i],
            "global_loss": history["global_loss"][i],
            "avg_submodel_acc": history["avg_submodel_acc"][i],
            "tier_acc_weak": tier_acc.get("weak"),
            "tier_acc_medium": tier_acc.get("medium"),
            "tier_acc_strong": tier_acc.get("strong"),
            "comm_waste": history["comm_waste"][i],
            "comm_cost_cumulative": history["comm_cost_cumulative"][i],
            "energy_cost_cumulative": history["energy_cost_cumulative"][i],
            "selection_fairness": history["selection_fairness"][i],
            "perf_fairness": history["perf_fairness"][i],
            "gated_out_frac": history["gated_out_frac"][i] if i < len(history.get("gated_out_frac", [])) else None,
        })
    return rows


def save_history_csv(history: dict, path):
    """Per-round metric dump for a single (method, seed) run."""
    rows = _history_rows(history)
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_long_format_csv(method_to_seed_histories: dict, path):
    """method_to_seed_histories: {method: {seed: history_dict}}. Tidy long-format
    CSV -- one row per (method, seed, round) -- the master reproducibility artifact:
    every plot and every summary table in this module can be regenerated from it."""
    rows = []
    for method, seed_hists in method_to_seed_histories.items():
        for seed, history in seed_hists.items():
            for row in _history_rows(history):
                rows.append({"method": method, "seed": seed, **row})
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "seed"] + _HISTORY_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summaries(seed_to_summary: dict):
    """seed_to_summary: {seed: summary_dict} (one summarize_history() output per
    seed). Returns {metric: {"mean", "std", "n", "values"}} for every numeric key.
    With a single seed, std is 0.0 and that's reported honestly as n=1 -- callers
    should render it as a bare value, not a fake +/- 0.0."""
    if not seed_to_summary:
        return {}
    summaries = list(seed_to_summary.values())
    keys = summaries[0].keys()
    agg = {}
    for k in keys:
        values = [s[k] for s in summaries if s.get(k) is not None]
        if not values or not isinstance(values[0], (int, float)):
            agg[k] = {"mean": values[0] if values else None, "std": None, "n": len(values), "values": values}
            continue
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        agg[k] = {"mean": mean, "std": std, "n": len(values), "values": values}
    return agg


DEFAULT_TABLE_METRICS = [
    "best_global_acc", "final_global_acc", "best_avg_submodel_acc", "final_avg_submodel_acc",
    "avg_comm_waste", "final_comm_cost", "final_energy_cost",
    "final_selection_fairness", "final_perf_fairness",
]


def comparison_table_rows(method_to_agg: dict, metrics=None):
    """method_to_agg: {method_name: aggregate_summaries() output}. Flat rows with
    '<metric>_mean' / '<metric>_std' columns, ready for csv.DictWriter or pandas."""
    metrics = metrics or DEFAULT_TABLE_METRICS
    rows = []
    for method, agg in method_to_agg.items():
        n_seeds = next(iter(agg.values()))["n"] if agg else 0
        row = {"method": method, "n_seeds": n_seeds}
        for m in metrics:
            if m in agg:
                row[f"{m}_mean"] = agg[m]["mean"]
                row[f"{m}_std"] = agg[m]["std"]
        rows.append(row)
    return rows


def _fmt(row, base_metric):
    mean = row.get(f"{base_metric}_mean")
    std = row.get(f"{base_metric}_std")
    if mean is None:
        return "-"
    if isinstance(mean, (int, float)):
        if std:
            return f"{mean:.4f} +/- {std:.4f}"
        return f"{mean:.4f}"
    return str(mean)


def save_comparison_table(rows, out_path_csv, out_path_md=None, out_path_tex=None):
    """Writes a mean+-std comparison table as CSV (always), and optionally as a
    Markdown table and a LaTeX tabular -- both directly pastable into a manuscript."""
    if not rows:
        return
    Path(out_path_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_bases = sorted({fn[:-5] for fn in fieldnames if fn.endswith("_mean")})

    if out_path_md:
        Path(out_path_md).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_md, "w", encoding="utf-8") as f:
            f.write("| method | n_seeds | " + " | ".join(metric_bases) + " |\n")
            f.write("|" + "---|" * (2 + len(metric_bases)) + "\n")
            for row in rows:
                f.write(f"| {row['method']} | {row['n_seeds']} | " +
                        " | ".join(_fmt(row, m) for m in metric_bases) + " |\n")

    if out_path_tex:
        Path(out_path_tex).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_tex, "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{l" + "c" * len(metric_bases) + "}\n\\toprule\n")
            f.write("Method & " + " & ".join(m.replace("_", " ") for m in metric_bases) + " \\\\\n\\midrule\n")
            for row in rows:
                f.write(f"{row['method']} & " +
                        " & ".join(_fmt(row, m) for m in metric_bases) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
