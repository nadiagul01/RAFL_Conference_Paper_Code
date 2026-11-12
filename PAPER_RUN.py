#!/usr/bin/env python3
"""
PAPER_RUN.py - extra runs for the CONFERENCE paper, with the ORIGINAL (submitted) RAFL code in ./code
(the September code; it reproduces the paper's F_select 0.6210, waste 0.1546, comm 4.8586e9, energy 4583.81 at seed 42).

Same protocol as the paper: CIFAR-10, Dirichlet alpha=0.3, 100 clients, 10%/round, 5 local epochs, batch 50,
SGD lr 0.01 momentum 0.5, weight decay 5e-4, 1000 rounds, tiers 4:3:3, LiteNet (BatchNorm), synchronous aggregation.

  METHOD=rafl|adaptivefl|fedavg|heterofl|scalefl|decoupled  SEED=42|43|44  ALPHA=0.3|0.6  VARIANT=none|async|hybrid|noCG|noFE|noE|noF  MODE=smoke|full|report
  EVAL_EVERY=1 (default, as the submitted run: every round evaluated)   MAX_HOURS=7.5 (stop cleanly before Colab ends)
  NUM_WORKERS=0 (default; data loading only, no effect on the method)
  smoke : 30 rounds in a separate *_smoke folder, prints PASS/FAIL and the time per round
  full  : 1000 rounds, checkpoint every 10 rounds, resumes automatically after a disconnect
  report: tables (mean +- std over the seeds found) and the paper's figures, from the CSV files only

Results: <WORK>/RAFL_Results/paper_c10_a03[_<variant>]/seed_<S>/<method>_history.csv  (+ checkpoints/)
Communication waste is also recorded at EVERY round (the paper's 0.1546 is the mean over all 1000 rounds;
the history CSV only has the 101 logged rounds) in <folder>/seed_<S>/<method>_waste_all_rounds.json.
"""
import os, sys, json, time, glob, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK", HERE)
sys.path.insert(0, os.path.join(HERE, "code"))
os.chdir(HERE)
import numpy as np
import pandas as pd

MODE = os.environ.get("MODE", "smoke").lower()
METHOD = os.environ.get("METHOD", "rafl").lower()
SEED = int(os.environ.get("SEED", "42"))
VARIANT = os.environ.get("VARIANT", "none")
ALPHA = float(os.environ.get("ALPHA", "0.3"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "1"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "7.5"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
METHODS = ("rafl", "adaptivefl", "fedavg", "heterofl", "scalefl", "decoupled")
assert ALPHA in (0.3, 0.6), "ALPHA must be 0.3 or 0.6"
VARS = {"none": {}, "async": {"aggregation_mode": "async"}, "hybrid": {"aggregation_mode": "hybrid"},
        "noCG": {"use_capability_gate": False}, "noFE": {"use_fairness_energy_reward": False},
        "noE": {"energy_weight": 0.0}, "noF": {"fairness_gamma": 0.0}}
assert MODE in ("smoke", "full", "report"), "MODE must be smoke, full or report"
assert METHOD in METHODS, f"METHOD must be one of {METHODS}"
assert VARIANT in VARS, f"VARIANT must be one of {list(VARS)}"
assert VARIANT == "none" or METHOD == "rafl", "variants are RAFL ablations"
RES = os.path.join(WORK, "RAFL_Results")
OUT = os.path.join(WORK, "PAPER_REPORT")
FOLDER = ("paper_c10_a03" if ALPHA == 0.3 else "paper_c10_a06") + ("" if VARIANT == "none" else f"_{VARIANT}")
PAPER_V1 = {"selection_fairness": 0.6210, "energy": 4583.81, "comm": 4.8586e9}   # submitted run, seed 42
FINGERPRINT = {42: 4583.81, 43: 4450.4, 44: 4460.9}   # RAFL full energy at round 1000 (replay of this code)


# ============================================================================ report
def hist(folder, method, seed):
    f = os.path.join(RES, folder, f"seed_{seed}", f"{method}_history.csv")
    try:
        h = pd.read_csv(f)
        return h if len(h) and int(h["round"].max()) >= 1000 else None
    except Exception:
        return None


def waste_all(folder, method, seed):
    try:
        w = json.load(open(os.path.join(RES, folder, f"seed_{seed}", f"{method}_waste_all_rounds.json")))
        return w["sum"] / w["n"] if w["n"] >= 1000 else np.nan
    except Exception:
        h = hist(folder, method, seed)          # a run evaluated every round (like the submitted one) has it in the CSV
        return float(h["comm_waste"].mean()) if h is not None and len(h) >= 1000 else np.nan


def clf(folder, method, seed):
    """macro precision / recall / F1 of the final-round model (round 1000) on the CIFAR-10 test set"""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, precision_score, recall_score
    import rafl_unified as RU
    ck = os.path.join(RES, folder, f"seed_{seed}", "checkpoints", f"{method}_ckpt.pt")
    if not os.path.exists(ck):
        return {}
    st = torch.load(ck, map_location="cpu", weights_only=False)
    if int(st.get("round_idx", 0)) < 1000:
        return {}
    if "test" not in clf.__dict__:
        clf.test = RU.load_dataset(RU.Config(dataset="cifar10", data_root=os.path.join(WORK, "data")))[1]
    m = RU.LiteNetAdaptive(RU.LITE_CHANNELS, num_classes=10)
    m.load_state_dict(st["model_state"] if "model_state" in st else st["model_state_L"]); m.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in DataLoader(clf.test, batch_size=256):
            yp.extend(m(x).argmax(1).numpy()); yt.extend(np.asarray(y))
    yt, yp = np.array(yt), np.array(yp)
    return {"F1": f1_score(yt, yp, average="macro"), "Precision": precision_score(yt, yp, average="macro", zero_division=0),
            "Recall": recall_score(yt, yp, average="macro"), "final_model_acc": float((yt == yp).mean())}


def row(folder, method, seed):
    h = hist(folder, method, seed)
    if h is None:
        return None
    g = h["global_acc"]
    r = {"seed": seed, "best_global": g.max(), "best_round": int(h["round"][g.idxmax()]), "final_global": g.iloc[-1],
         "best_submodel": h["avg_submodel_acc"].max(), "F_select": h["selection_fairness"].iloc[-1],
         "F_perf": h["perf_fairness"].iloc[-1], "waste_all_rounds": waste_all(folder, method, seed),
         "comm_e9": h["comm_cost_cumulative"].iloc[-1] / 1e9, "comm_GB": 4 * h["comm_cost_cumulative"].iloc[-1] / 1e9,
         "energy": h["energy_cost_cumulative"].iloc[-1]}
    if method == "rafl" and folder == "paper_c10_a03":
        r["same_code_as_paper"] = abs(r["energy"] - FINGERPRINT.get(seed, -1)) < 0.5
    try:
        r.update(clf(folder, method, seed))
    except Exception as e:
        print(f"[report] F1 not computed for {folder}/{method}/seed {seed}: {e}")
    return r


def mean_std(df):
    num = df.drop(columns=[c for c in ("seed", "same_code_as_paper") if c in df.columns]).astype(float)
    if len(df) == 1:
        return num.iloc[0].map(lambda v: f"{v:.4f}")
    return num.mean().map(lambda v: f"{v:.4f}") + " ± " + num.std(ddof=1).map(lambda v: f"{v:.4f}")


def report():
    os.makedirs(OUT, exist_ok=True)
    rows = {}
    for folder in ["paper_c10_a03", "paper_c10_a06"] + [f"paper_c10_a03_{v}" for v in VARS if v != "none"]:
        for m in METHODS:
            got = [r for r in (row(folder, m, s) for s in (42, 43, 44)) if r is not None]
            if got:
                rows[(folder, m)] = pd.DataFrame(got)
    if not rows:
        print("[report] no finished 1000-round run yet"); return
    lines = []
    for (folder, m), df in rows.items():
        lines.append(f"\n=== {folder} / {m}  seeds {list(df['seed'])}")
        lines.append(df.round(4).to_string(index=False))
        lines.append("mean ± std (sample, n=%d):" % len(df))
        lines.append(mean_std(df).to_string())
        df.to_csv(os.path.join(OUT, f"{folder}_{m}.csv"), index=False)
    if ("paper_c10_a03", "rafl") in rows:
        d = rows[("paper_c10_a03", "rafl")]
        bad = d[~d["same_code_as_paper"]]["seed"].tolist() if "same_code_as_paper" in d else []
        lines.append("\nfingerprint check (energy at round 1000 must equal the original code): "
                     + ("OK for all seeds" if not bad else f"NOT the paper code for seeds {bad} - do not use them"))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "paper_report.txt"), "w").write(txt + "\n")
    figures(rows)
    print(f"\n[report] written to {OUT}")


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = [s for s in (42, 43, 44) if hist("paper_c10_a03", "rafl", s) is not None]
    if not seeds:
        return
    H = [hist("paper_c10_a03", "rafl", s) for s in seeds]
    common = sorted(set.intersection(*[set(h["round"].astype(int)) for h in H]))   # runs may log every round or every 10
    H = [h[h["round"].astype(int).isin(common)].sort_values("round").reset_index(drop=True) for h in H]
    r = H[0]["round"].values

    def band(ax, col, label, scale=100.0):
        y = np.vstack([h[col].values for h in H]) * scale
        ax.plot(r, y.mean(0), label=label)
        if len(H) > 1:
            ax.fill_between(r, y.mean(0) - y.std(0, ddof=1), y.mean(0) + y.std(0, ddof=1), alpha=0.25)

    # Fig. 3: global accuracy over rounds (+ AdaptiveFL re-run if present, + published AdaptiveFL line)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    band(ax, "global_acc", f"RAFL (mean of {len(H)} seeds)")
    A = [hist("paper_c10_a03", "adaptivefl", s) for s in (42, 43, 44)]
    A = [a for a in A if a is not None]
    if A:
        ya = np.vstack([a["global_acc"].values for a in A]) * 100
        ax.plot(A[0]["round"], ya.mean(0), label=f"AdaptiveFL re-run, LiteNet ({len(A)} seed{'s' if len(A) > 1 else ''})")
    ax.axhline(78.99, ls="--", c="k", lw=1, label="AdaptiveFL published (VGG16) 78.99%")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Global accuracy (%)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_global_accuracy.png"), dpi=300); plt.close(fig)
    # Fig. 4: per-tier accuracy
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for t in ("weak", "medium", "strong"):
        band(ax, f"tier_acc_{t}", t)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Tier accuracy (%)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_tier_accuracy.png"), dpi=300); plt.close(fig)
    # Fig. 5: selection fairness and performance fairness
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    band(axs[0], "selection_fairness", "F_select", 1.0); axs[0].set_ylabel("$F_{select}$")
    band(axs[1], "perf_fairness", "F_perf", 1.0); axs[1].set_ylabel("$F_{perf}$ (variance)")
    for a in axs:
        a.set_xlabel("Communication round"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_fairness.png"), dpi=300); plt.close(fig)


if MODE == "report":
    report()
    sys.exit(0)

# ============================================================================ training (original code, unchanged)
import torch

_orig_save = torch.save


def _atomic_save(obj, f, *a, **k):
    if isinstance(f, (str, os.PathLike)):
        tmp = str(f) + ".tmp"
        _orig_save(obj, tmp, *a, **k)
        os.replace(tmp, f)
    else:
        _orig_save(obj, f, *a, **k)


torch.save = _atomic_save
import rafl_unified as RU
import rafl_runners as RR
from rafl_drivers import run_tier2
from run_published import PUBLISHED

ROUNDS = 30 if MODE == "smoke" else 1000
ROOT = os.path.join(RES, FOLDER + ("_smoke" if MODE == "smoke" else ""))
SAVE = os.path.join(ROOT, f"seed_{SEED}")
WFILE = os.path.join(SAVE, f"{METHOD}_waste_all_rounds.json")
W = {"sum": 0.0, "n": 0}
if os.path.exists(os.path.join(SAVE, "checkpoints", f"{METHOD}_ckpt.pt")) and os.path.exists(WFILE):
    W.update(json.load(open(WFILE)))     # resume: continue the per-round waste from the checkpointed round
_orig_comm = RR.communication_cost_params


def communication_cost_params(sent, returned, pc):
    s = sum(pc[m] for m in sent)
    W["sum"] += (1.0 - sum(pc[m] for m in returned) / s) if s else 0.0
    W["n"] += 1
    return _orig_comm(sent, returned, pc)


RR.communication_cost_params = communication_cost_params
_orig_ckpt = RR.save_checkpoint


T_START = time.time()


def save_checkpoint(*a, **k):
    _orig_ckpt(*a, **k)
    os.makedirs(SAVE, exist_ok=True)
    json.dump(W, open(WFILE + ".tmp", "w")); os.replace(WFILE + ".tmp", WFILE)
    r = a[2] if len(a) > 2 else k.get("round_idx", 0)
    if MODE == "full" and r < ROUNDS and (time.time() - T_START) / 3600 > MAX_HOURS:
        print(f"[pause] saved at round {r} after {MAX_HOURS} h - run the SAME cell again to continue", flush=True)
        os._exit(0)


RR.save_checkpoint = save_checkpoint

hf = os.path.join(SAVE, f"{METHOD}_history.csv")
try:
    done = int(pd.read_csv(hf)["round"].max()) >= ROUNDS
except Exception:
    done = False
if done:
    print(f"[skip] {FOLDER} {METHOD} seed {SEED} already finished")
else:
    P = dict(PUBLISHED)
    P.update(baseline_names=(METHOD,), dataset="cifar10", alpha=ALPHA, num_rounds=ROUNDS, checkpoint_every=10,
             eval_every=EVAL_EVERY, num_workers=NUM_WORKERS, data_root=os.path.join(WORK, "data"), save_dir=ROOT)
    P.update(VARS[VARIANT])
    print("#" * 80 + f"\nPAPER_RUN {METHOD} seed {SEED} alpha {ALPHA} variant {VARIANT} rounds {ROUNDS} eval_every {EVAL_EVERY} workers {NUM_WORKERS} | GPU: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU - too slow)'}\n-> {SAVE}\n" + "#" * 80, flush=True)
    t0 = time.time()
    try:
        run_tier2(P, resume=True, seeds=(SEED,))
    except Exception:
        print(f"[ERROR]\n{traceback.format_exc()}"); sys.exit(1)
    dt = time.time() - t0
    print(f"[OK] {FOLDER} {METHOD} seed {SEED}  {dt / 3600:.2f} h", flush=True)
    if MODE == "smoke":
        print(f"[time] {dt / ROUNDS:.1f} s per round  ->  about {dt / ROUNDS * 1000 / 3600:.1f} h for 1000 rounds")

if MODE == "smoke":
    d = pd.read_csv(hf)
    last = float(d["global_acc"].iloc[-1])
    print(d[["round", "global_acc", "avg_submodel_acc", "selection_fairness"]].round(4).to_string(index=False))
    print(f"round-30 acc = {100 * last:.2f}%  ->  {'PASS - learning' if last > 0.2 else 'FAIL - not learning'}")
else:
    h = pd.read_csv(hf)
    print(f"best global {h.global_acc.max():.4f} | best sub {h.avg_submodel_acc.max():.4f} | F_select "
          f"{h.selection_fairness.iloc[-1]:.4f} | waste(all rounds) {W['sum'] / max(W['n'], 1):.4f} | comm "
          f"{h.comm_cost_cumulative.iloc[-1]:.4e} | energy {h.energy_cost_cumulative.iloc[-1]:.2f}")
    if METHOD == "rafl" and VARIANT == "none" and ALPHA == 0.3 and SEED in FINGERPRINT:
        ok = abs(h.energy_cost_cumulative.iloc[-1] - FINGERPRINT[SEED]) < 0.5
        print(f"[check] same code as the paper (energy {FINGERPRINT[SEED]}): {'YES' if ok else 'NO - stop and tell me'}")
