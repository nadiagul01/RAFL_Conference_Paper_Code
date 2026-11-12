# RAFL: Resource-Aware Federated Learning with Fairness- and Energy-Aware Client Selection in Resource-Constrained Networks

Code and results for the HITE 2026 conference paper (Paper ID 167).

Authors: Nadia Gul, Muhammad Mateen Yaqoob, Ayesha Kalsoom, Kashif Nasr

## Repository structure

```
PAPER_RUN.py            # entry point used for the paper runs (RAFL / AdaptiveFL, seeds, alpha, ablation variants)
code/
  rafl_unified.py       # config, CIFAR-10 loading, Dirichlet split, LiteNet, device simulation, core FL logic
  rafl_runners.py       # method runners: RAFL, AdaptiveFL, FedAvg, HeteroFL, ScaleFL, Decoupled
  rafl_drivers.py       # multi-seed / sweep / ablation drivers
  reporting.py          # summary tables (CSV / Markdown / LaTeX)
  plots.py              # figures
  run_published.py      # published AdaptiveFL reference values
results/                # history CSVs, summary tables and plots of the reported runs
requirements.txt        # package versions of the environment used
MD5SUMS.txt             # checksums of the code files
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically into `./data` on the first run.

## Experimental protocol (as in the paper)

CIFAR-10, 100 clients, 10% participation per round, Dirichlet non-IID split (alpha = 0.3 and 0.6),
5 local epochs, batch size 50, SGD (lr 0.01, momentum 0.5, weight decay 5e-4), 1000 rounds,
device tiers weak:medium:strong = 4:3:3, LiteNet backbone (618,474 parameters), seeds 42, 43, 44.

## Running

`PAPER_RUN.py` is configured with environment variables:

| Variable | Values |
|---|---|
| `METHOD` | `rafl`, `adaptivefl` |
| `SEED` | `42`, `43`, `44` |
| `ALPHA` | `0.3`, `0.6` |
| `VARIANT` | `none`, `async`, `hybrid`, `noCG`, `noFE`, `noE`, `noF` (RAFL ablations) |
| `MODE` | `smoke` (30 rounds), `full` (1000 rounds), `report` (tables from saved CSVs) |

Examples:

```bash
# quick check
MODE=smoke METHOD=rafl SEED=42 ALPHA=0.3 python PAPER_RUN.py

# full runs
MODE=full METHOD=rafl       SEED=42 ALPHA=0.3 python PAPER_RUN.py
MODE=full METHOD=adaptivefl SEED=42 ALPHA=0.3 python PAPER_RUN.py
MODE=full METHOD=rafl       SEED=42 ALPHA=0.3 VARIANT=async python PAPER_RUN.py

# tables (mean ± std over the seeds found)
MODE=report python PAPER_RUN.py
```

Results are written to `RAFL_Results/` and `PAPER_REPORT/`.

## Notes

Battery, network and timing are simulated; energy is reported in relative units, not hardware measurements.
