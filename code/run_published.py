"""
Run RAFL's complete methodology at the PUBLISHED AdaptiveFL protocol (paper Sec.4),
with your approved LiteNet. Crash-safe: checkpoints every 25 rounds and resumes.

Run RAFL FIRST (fast, ~7h):        edit METHODS below to ("rafl",)  then run this file.
Run the full comparison table:     leave METHODS as all six.
"""
from rafl_drivers import run_tier2

# ---- PUBLISHED base-paper protocol (identical to the paper) + LiteNet ----------
METHODS = ("fedavg", "heterofl", "scalefl", "decoupled", "adaptivefl", "rafl")

PUBLISHED = dict(
    dataset="cifar10", data_root="/kaggle/working/data",
    save_dir="/kaggle/working/RAFL_Results/published",
    num_clients=100, participation_rate=0.1, local_epochs=5, batch_size=50,
    lr=0.01, momentum=0.5, weight_decay=5e-4,
    alpha=0.1, weak_ratio=4, medium_ratio=3, strong_ratio=3,
    num_rounds=1000, eval_every=10,
    checkpoint_every=25,             # crash-safety: at most ~25 rounds (~10 min) can be lost
    aggregation_mode="sync",         # headline mode (no update discarded)
    simulate_failures=False,         # benign, same as the baselines -> fair comparison
    baseline_names=METHODS,
)

if __name__ == "__main__":
    # resume=True -> on any crash/shutdown, re-running this file reloads the last
    # checkpoint and continues; methods that already finished are skipped instantly.
    outputs, summary = run_tier2(PUBLISHED, resume=True, seeds=(42,))
    print(summary.to_string(index=False))
    print("\nDone. Table + per-round CSVs + plots under:", PUBLISHED["save_dir"])
