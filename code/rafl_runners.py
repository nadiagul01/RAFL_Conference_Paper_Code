"""
RAFL — method runners (FedAvg / HeteroFL / ScaleFL / Decoupled / AdaptiveFL / RAFL)
+ checkpoint/resume wiring + Tier 1 sweep / Tier 2 long-run drivers.

Depends on rafl_unified.py being imported first (or merged into the same
notebook — see build_notebook.py for how this gets assembled into cells).
"""

import copy
import json
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch

from rafl_unified import (
    Config, device, set_seed, LiteNetAdaptive, LITE_CHANNELS, MODEL_ORDER, TIER_MODELS,
    build_model_specs, build_model_channels, build_model_pool, copy_prefix_weights,
    assert_monotonic_model_sizes,
    create_client_profiles, RLTables, SelectionTracker,
    sample_unique_clients, select_clients_adaptivefl, select_clients_rafl, adaptive_local_prune,
    resolve_actual_model, profile_step, selected_client_fails,
    train_local, train_local_with_distillation, aggregate_weighted,
    evaluate_model, evaluate_submodel_average, communication_waste, communication_cost_params,
    simulate_train_time, energy_cost_round,
    save_checkpoint, load_checkpoint, restore_from_checkpoint, checkpoint_path,
    capture_rng_state, restore_rng_state,
    init_history, maybe_log_round,
)


def make_client_loader(client_id, cfg, trainset, client_indices):
    from torch.utils.data import DataLoader, Subset
    return DataLoader(Subset(trainset, client_indices[client_id]), batch_size=cfg.batch_size,
                       shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)


def summarize_history(history, elapsed):
    return {
        "best_global_acc": max(history["global_acc"]) if history["global_acc"] else None,
        "final_global_acc": history["global_acc"][-1] if history["global_acc"] else None,
        "best_avg_submodel_acc": max(history["avg_submodel_acc"]) if history["avg_submodel_acc"] else None,
        "final_avg_submodel_acc": history["avg_submodel_acc"][-1] if history["avg_submodel_acc"] else None,
        "avg_comm_waste": float(np.mean(history["comm_waste"])) if history["comm_waste"] else None,
        "final_comm_cost": history["comm_cost_cumulative"][-1] if history["comm_cost_cumulative"] else None,
        "final_energy_cost": history["energy_cost_cumulative"][-1] if history["energy_cost_cumulative"] else None,
        "final_selection_fairness": history["selection_fairness"][-1] if history["selection_fairness"] else None,
        "final_perf_fairness": history["perf_fairness"][-1] if history["perf_fairness"] else None,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------
# FedAvg (All-Large) — unchanged baseline
# ---------------------------------------------------------------------
def run_fedavg(cfg: Config, trainset, testset, test_loader, client_indices, num_classes,
                method_name="fedavg", resume=True):
    set_seed(cfg.seed)
    specs = build_model_specs(cfg)
    global_model = LiteNetAdaptive(LITE_CHANNELS, num_classes=num_classes).to(device)
    profiles = create_client_profiles(cfg)
    tracker = SelectionTracker(cfg.num_clients)
    history = init_history("FedAvg")
    start_round, cum_comm, cum_energy = 1, 0.0, 0.0

    ckpt = load_checkpoint(cfg, method_name) if resume else None
    if ckpt is not None:
        start_round, history = restore_from_checkpoint(ckpt, global_model, None, profiles, tracker)
        start_round += 1
        cum_comm = ckpt["extra"].get("cum_comm", 0.0)
        cum_energy = ckpt["extra"].get("cum_energy", 0.0)
        print(f"[{method_name}] resumed from round {start_round}")

    start_time = time.time()
    full_params = LiteNetAdaptive(LITE_CHANNELS, num_classes=num_classes).num_parameters()
    param_counts = {"L1": full_params}
    for round_idx in range(start_round, cfg.num_rounds + 1):
        selected_clients = sample_unique_clients(cfg, cfg.num_clients)
        tracker.record(selected_clients)
        client_updates = []
        sent_models = ["L1"] * len(selected_clients)
        returned_models = ["L1"] * len(selected_clients)
        for client_id in selected_clients:
            local_model = LiteNetAdaptive(LITE_CHANNELS, num_classes=num_classes).to(device)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_state, _ = train_local(local_model, make_client_loader(client_id, cfg, trainset, client_indices), cfg)
            client_updates.append((local_state, len(client_indices[client_id])))
        global_model = aggregate_weighted(global_model, client_updates)
        sim_times = {cid: simulate_train_time(profiles[cid], "L1", param_counts, cfg) for cid in selected_clients}
        cum_energy += energy_cost_round(list(zip(selected_clients, sent_models)), profiles, param_counts, cfg, sim_times)
        cum_comm += full_params * 2 * len(selected_clients)
        maybe_log_round(history, round_idx, cfg, global_model, sent_models, returned_models,
                         specs, num_classes, test_loader, tracker, cum_comm, cum_energy)
        if round_idx % cfg.checkpoint_every == 0 or round_idx == cfg.num_rounds:
            save_checkpoint(cfg, method_name, round_idx, global_model, None, profiles, tracker, history,
                             extra={"cum_comm": cum_comm, "cum_energy": cum_energy})
        torch.cuda.empty_cache()
    return global_model, history, time.time() - start_time


# ---------------------------------------------------------------------
# HeteroFL — unchanged baseline (static tier -> model-size mapping)
# ---------------------------------------------------------------------
def run_heterofl(cfg: Config, trainset, testset, test_loader, client_indices, num_classes,
                  method_name="heterofl", resume=True):
    set_seed(cfg.seed)
    specs = build_model_specs(cfg)
    global_model = LiteNetAdaptive(LITE_CHANNELS, num_classes=num_classes).to(device)
    profiles = create_client_profiles(cfg)
    tracker = SelectionTracker(cfg.num_clients)
    history = init_history("HeteroFL")
    start_round, cum_comm, cum_energy = 1, 0.0, 0.0

    ckpt = load_checkpoint(cfg, method_name) if resume else None
    if ckpt is not None:
        start_round, history = restore_from_checkpoint(ckpt, global_model, None, profiles, tracker)
        start_round += 1
        cum_comm = ckpt["extra"].get("cum_comm", 0.0)
        cum_energy = ckpt["extra"].get("cum_energy", 0.0)
        print(f"[{method_name}] resumed from round {start_round}")

    start_time = time.time()
    profile_to_model = {"weak": "S1", "medium": "M1", "strong": "L1"}
    for round_idx in range(start_round, cfg.num_rounds + 1):
        model_pool = build_model_pool(global_model, num_classes, specs)
        param_counts = {n: model_pool[n].num_parameters() for n in MODEL_ORDER}
        selected_clients = sample_unique_clients(cfg, cfg.num_clients)
        tracker.record(selected_clients)
        client_updates, sent_models, returned_models = [], [], []
        for client_id in selected_clients:
            target_model = profile_to_model[profiles[client_id].device_type]
            local_model = LiteNetAdaptive(build_model_channels(target_model, specs), num_classes=num_classes).to(device)
            local_model = copy_prefix_weights(model_pool[target_model], local_model)
            local_state, _ = train_local(local_model, make_client_loader(client_id, cfg, trainset, client_indices), cfg)
            client_updates.append((local_state, len(client_indices[client_id])))
            sent_models.append(target_model)
            returned_models.append(target_model)
        global_model = aggregate_weighted(global_model, client_updates)
        sim_times = {cid: simulate_train_time(profiles[cid], m, param_counts, cfg)
                     for cid, m in zip(selected_clients, returned_models)}
        cum_energy += energy_cost_round(list(zip(selected_clients, returned_models)), profiles, param_counts, cfg, sim_times)
        cum_comm += communication_cost_params(sent_models, returned_models, param_counts)
        maybe_log_round(history, round_idx, cfg, global_model, sent_models, returned_models,
                         specs, num_classes, test_loader, tracker, cum_comm, cum_energy)
        if round_idx % cfg.checkpoint_every == 0 or round_idx == cfg.num_rounds:
            save_checkpoint(cfg, method_name, round_idx, global_model, None, profiles, tracker, history,
                             extra={"cum_comm": cum_comm, "cum_energy": cum_energy})
        torch.cuda.empty_cache()
    return global_model, history, time.time() - start_time


# ---------------------------------------------------------------------
# ScaleFL — FIXED to include self-distillation (Corrected_Methodology §5)
# ---------------------------------------------------------------------
def run_scalefl(cfg: Config, trainset, testset, test_loader, client_indices, num_classes,
                 method_name="scalefl", resume=True):
    set_seed(cfg.seed)
    specs = build_model_specs(cfg)
    global_model = LiteNetAdaptive(LITE_CHANNELS, num_classes=num_classes).to(device)
    profiles = create_client_profiles(cfg)
    tracker = SelectionTracker(cfg.num_clients)
    history = init_history("ScaleFL")
    start_round, cum_comm, cum_energy = 1, 0.0, 0.0

    ckpt = load_checkpoint(cfg, method_name) if resume else None
    if ckpt is not None:
        start_round, history = restore_from_checkpoint(ckpt, global_model, None, profiles, tracker)
        start_round += 1
        cum_comm = ckpt["extra"].get("cum_comm", 0.0)
        cum_energy = ckpt["extra"].get("cum_energy", 0.0)
        print(f"[{method_name}] resumed from round {start_round}")

    start_time = time.time()
    for round_idx in range(start_round, cfg.num_rounds + 1):
        model_pool = build_model_pool(global_model, num_classes, specs)
        param_counts = {n: model_pool[n].num_parameters() for n in MODEL_ORDER}
        selected_clients = sample_unique_clients(cfg, cfg.num_clients)
        tracker.record(selected_clients)
        client_updates, sent_models, returned_models = [], [], []
        for client_id in selected_clients:
            profile = profiles[client_id]
            affordable = [m for m in MODEL_ORDER if profile.can_train(m)]
            sent_model_name = str(np.random.choice(affordable))
            local_model = LiteNetAdaptive(build_model_channels(sent_model_name, specs), num_classes=num_classes).to(device)
            local_model = copy_prefix_weights(model_pool[sent_model_name], local_model)

            sent_idx = MODEL_ORDER.index(sent_model_name)
            larger = [m for m in affordable if MODEL_ORDER.index(m) > sent_idx]
            if cfg.scalefl_distillation and larger:
                teacher_name = larger[-1]
                teacher_model = model_pool[teacher_name]
                local_state, _ = train_local_with_distillation(
                    local_model, teacher_model, make_client_loader(client_id, cfg, trainset, client_indices), cfg)
            else:
                local_state, _ = train_local(local_model, make_client_loader(client_id, cfg, trainset, client_indices), cfg)

            client_updates.append((local_state, len(client_indices[client_id])))
            sent_models.append(sent_model_name)
            returned_models.append(sent_model_name)
        global_model = aggregate_weighted(global_model, client_updates)
        sim_times = {cid: simulate_train_time(profiles[cid], m, param_counts, cfg)
                     for cid, m in zip(selected_clients, returned_models)}
        cum_energy += energy_cost_round(list(zip(selected_clients, returned_models)), profiles, param_counts, cfg, sim_times)
        cum_comm += communication_cost_params(sent_models, returned_models, param_counts)
        maybe_log_round(history, round_idx, cfg, global_model, sent_models, returned_models,
                         specs, num_classes, test_loader, tracker, cum_comm, cum_energy)
        if round_idx % cfg.checkpoint_every == 0 or round_idx == cfg.num_rounds:
            save_checkpoint(cfg, method_name, round_idx, global_model, None, profiles, tracker, history,
                             extra={"cum_comm": cum_comm, "cum_energy": cum_energy})
        torch.cuda.empty_cache()
    return global_model, history, time.time() - start_time


# ---------------------------------------------------------------------
# Decoupled — unchanged baseline (3 separate global models, no cross-tier aggregation)
# ---------------------------------------------------------------------
def run_decoupled(cfg: Config, trainset, testset, test_loader, client_indices, num_classes,
                   method_name="decoupled", resume=True):
    set_seed(cfg.seed)
    specs = build_model_specs(cfg)
    global_model_L = LiteNetAdaptive(build_model_channels("L1", specs), num_classes=num_classes).to(device)
    global_model_M = LiteNetAdaptive(build_model_channels("M1", specs), num_classes=num_classes).to(device)
    global_model_S = LiteNetAdaptive(build_model_channels("S1", specs), num_classes=num_classes).to(device)
    param_counts = {"L1": global_model_L.num_parameters(), "M1": global_model_M.num_parameters(),
                    "S1": global_model_S.num_parameters()}
    profiles = create_client_profiles(cfg)
    tracker = SelectionTracker(cfg.num_clients)
    history = init_history("Decoupled")
    start_round, cum_comm, cum_energy = 1, 0.0, 0.0

    # Decoupled keeps three separate global models, so it can't reuse the
    # single-model save_checkpoint/restore_from_checkpoint helpers -- it gets
    # its own bespoke (but still RNG- and device-state-preserving) checkpoint.
    ckpt_path = checkpoint_path(cfg, method_name)
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        global_model_L.load_state_dict(ckpt["model_state_L"])
        global_model_M.load_state_dict(ckpt["model_state_M"])
        global_model_S.load_state_dict(ckpt["model_state_S"])
        for cid, ds in ckpt.get("device_states", {}).items():
            cid = int(cid)
            if cid in profiles:
                profiles[cid].device_state.cpu_score = ds["cpu_score"]
                profiles[cid].device_state.battery_pct = ds["battery_pct"]
                profiles[cid].device_state.network_stability = ds["network_stability"]
        tracker.counts = np.array(ckpt["tracker_counts"], dtype=np.float64)
        history = ckpt["history"]
        start_round = ckpt["round_idx"] + 1
        cum_comm = ckpt["extra"].get("cum_comm", 0.0)
        cum_energy = ckpt["extra"].get("cum_energy", 0.0)
        if ckpt.get("rng_state") is not None:
            restore_rng_state(ckpt["rng_state"])
        print(f"[{method_name}] resumed from round {start_round}")

    start_time = time.time()
    for round_idx in range(start_round, cfg.num_rounds + 1):
        selected_clients = sample_unique_clients(cfg, cfg.num_clients)
        tracker.record(selected_clients)
        updates_L, updates_M, updates_S = [], [], []
        sent_models, returned_models = [], []
        for client_id in selected_clients:
            dtype = profiles[client_id].device_type
            if dtype == "strong":
                target_model, base_global = "L1", global_model_L
            elif dtype == "medium":
                target_model, base_global = "M1", global_model_M
            else:
                target_model, base_global = "S1", global_model_S
            local_model = LiteNetAdaptive(build_model_channels(target_model, specs), num_classes=num_classes).to(device)
            local_model.load_state_dict(copy.deepcopy(base_global.state_dict()))
            local_state, _ = train_local(local_model, make_client_loader(client_id, cfg, trainset, client_indices), cfg)
            if dtype == "strong":
                updates_L.append((local_state, len(client_indices[client_id])))
            elif dtype == "medium":
                updates_M.append((local_state, len(client_indices[client_id])))
            else:
                updates_S.append((local_state, len(client_indices[client_id])))
            sent_models.append(target_model)
            returned_models.append(target_model)
        if updates_L:
            global_model_L = aggregate_weighted(global_model_L, updates_L)
        if updates_M:
            global_model_M = aggregate_weighted(global_model_M, updates_M)
        if updates_S:
            global_model_S = aggregate_weighted(global_model_S, updates_S)

        sim_times = {cid: simulate_train_time(profiles[cid], m, param_counts, cfg)
                     for cid, m in zip(selected_clients, sent_models)}
        cum_energy += energy_cost_round(list(zip(selected_clients, sent_models)), profiles, param_counts, cfg, sim_times)
        cum_comm += communication_cost_params(sent_models, returned_models, param_counts)

        if round_idx % cfg.eval_every == 0 or round_idx == 1 or round_idx == cfg.num_rounds:
            global_acc, global_loss = evaluate_model(global_model_L, test_loader)
            acc_L = global_acc
            acc_M, _ = evaluate_model(global_model_M, test_loader)
            acc_S, _ = evaluate_model(global_model_S, test_loader)
            tier_acc = {"weak": acc_S, "medium": acc_M, "strong": acc_L}
            avg_sub_acc = float(np.mean([acc_L, acc_M, acc_S]))
            perf_fair = float(np.var([acc_S, acc_M, acc_L]))
            history["round"].append(round_idx)
            history["global_acc"].append(global_acc)
            history["global_loss"].append(global_loss)
            history["avg_submodel_acc"].append(avg_sub_acc)
            history["tier_acc"].append(tier_acc)
            history["comm_waste"].append(0.0)
            history["comm_cost_cumulative"].append(cum_comm)
            history["energy_cost_cumulative"].append(cum_energy)
            history["selection_fairness"].append(tracker.fairness_index())
            history["perf_fairness"].append(perf_fair)
            history["gated_out_frac"].append(0.0)
            print(f"{history['name']:12s} | round {round_idx:04d}/{cfg.num_rounds} | "
                  f"global_acc={global_acc:.4f} | avg_sub_acc={avg_sub_acc:.4f} | "
                  f"F_select={tracker.fairness_index():.3f} | F_perf={perf_fair:.5f}")
        if round_idx % cfg.checkpoint_every == 0 or round_idx == cfg.num_rounds:
            torch.save({
                "round_idx": round_idx,
                "model_state_L": global_model_L.state_dict(),
                "model_state_M": global_model_M.state_dict(),
                "model_state_S": global_model_S.state_dict(),
                "device_states": {cid: {"cpu_score": p.device_state.cpu_score,
                                         "battery_pct": p.device_state.battery_pct,
                                         "network_stability": p.device_state.network_stability}
                                   for cid, p in profiles.items()},
                "tracker_counts": tracker.counts.tolist(),
                "history": history,
                "rng_state": capture_rng_state(),
                "extra": {"cum_comm": cum_comm, "cum_energy": cum_energy},
            }, ckpt_path)
        torch.cuda.empty_cache()
    return global_model_L, history, time.time() - start_time


# ---------------------------------------------------------------------
# AdaptiveFL — faithful reproduction of the base paper (RL selection, sync agg)
# ---------------------------------------------------------------------
def run_adaptivefl(cfg: Config, trainset, testset, test_loader, client_indices, num_classes,
                    method_name="adaptivefl", resume=True):
    set_seed(cfg.seed)
    specs = build_model_specs(cfg)
    global_model = LiteNetAdaptive(LITE_CHANNELS, num_classes=num_classes).to(device)
    profiles = create_client_profiles(cfg)
    rl = RLTables(cfg.num_clients, MODEL_ORDER, beta=cfg.curiosity_beta)
    tracker = SelectionTracker(cfg.num_clients)
    history = init_history("AdaptiveFL")
    start_round, cum_comm, cum_energy = 1, 0.0, 0.0

    ckpt = load_checkpoint(cfg, method_name) if resume else None
    if ckpt is not None:
        start_round, history = restore_from_checkpoint(ckpt, global_model, rl, profiles, tracker)
        start_round += 1
        cum_comm = ckpt["extra"].get("cum_comm", 0.0)
        cum_energy = ckpt["extra"].get("cum_energy", 0.0)
        print(f"[{method_name}] resumed from round {start_round}")

    start_time = time.time()
    for round_idx in range(start_round, cfg.num_rounds + 1):
        if cfg.simulate_failures:
            for p in profiles.values():
                profile_step(p.device_state, cfg)  # robustness experiment: devices drift for AdaptiveFL too
        model_pool = build_model_pool(global_model, num_classes, specs)
        param_counts = {n: model_pool[n].num_parameters() for n in MODEL_ORDER}
        dispatched_models = list(np.random.choice(MODEL_ORDER, size=cfg.num_selected, replace=True))
        selections = select_clients_adaptivefl(dispatched_models, profiles, rl)
        for cid, _ in selections:
            profiles[cid].times_selected += 1
        tracker.record([cid for cid, _ in selections])

        client_updates, sent_models, returned_models = [], [], []
        for client_id, sent_model_name in selections:
            profile = profiles[client_id]
            actual_model_name = resolve_actual_model(profile, sent_model_name)  # for aligned comm bookkeeping
            sent_models.append(sent_model_name)
            returned_models.append(actual_model_name)
            if selected_client_fails(profile.device_state, cfg):
                continue  # robustness experiment only: AdaptiveFL has no gate, so it eats this dropout
            _, local_model = adaptive_local_prune(profile, sent_model_name, model_pool, num_classes, specs)
            local_state, _ = train_local(local_model, make_client_loader(client_id, cfg, trainset, client_indices), cfg)
            client_updates.append((local_state, len(client_indices[client_id])))
            rl.update_after_round_trip(client_id, sent_model_name, actual_model_name)
        if client_updates:
            global_model = aggregate_weighted(global_model, client_updates)
        sim_times = {cid: simulate_train_time(profiles[cid], m, param_counts, cfg)
                     for (cid, _), m in zip(selections, returned_models)}
        cum_energy += energy_cost_round(list(zip([cid for cid, _ in selections], returned_models)),
                                         profiles, param_counts, cfg, sim_times)
        cum_comm += communication_cost_params(sent_models, returned_models, param_counts)
        maybe_log_round(history, round_idx, cfg, global_model, sent_models, returned_models,
                         specs, num_classes, test_loader, tracker, cum_comm, cum_energy)
        if round_idx % cfg.checkpoint_every == 0 or round_idx == cfg.num_rounds:
            save_checkpoint(cfg, method_name, round_idx, global_model, rl, profiles, tracker, history,
                             extra={"cum_comm": cum_comm, "cum_energy": cum_energy})
        torch.cuda.empty_cache()
    return global_model, history, time.time() - start_time


# ---------------------------------------------------------------------
# RAFL — the thesis contribution: capability gate + fairness/energy reward
# + sync-or-hybrid staleness-weighted aggregation (Corrected_Methodology §4)
# ---------------------------------------------------------------------
def run_rafl(cfg: Config, trainset, testset, test_loader, client_indices, num_classes,
             method_name="rafl", resume=True):
    set_seed(cfg.seed)
    specs = build_model_specs(cfg)
    global_model = LiteNetAdaptive(LITE_CHANNELS, num_classes=num_classes).to(device)
    profiles = create_client_profiles(cfg)
    rl = RLTables(cfg.num_clients, MODEL_ORDER, beta=cfg.curiosity_beta)
    tracker = SelectionTracker(cfg.num_clients)
    variant_tag = ("" if cfg.use_capability_gate else "-noCG") + ("" if cfg.use_fairness_energy_reward else "-noFE")
    history = init_history(f"RAFL-{cfg.aggregation_mode}{variant_tag}")
    start_round, cum_comm, cum_energy = 1, 0.0, 0.0
    straggler_buffer = []  # list of (local_state, weight, staleness) waiting to be folded in

    ckpt = load_checkpoint(cfg, method_name) if resume else None
    if ckpt is not None:
        start_round, history = restore_from_checkpoint(ckpt, global_model, rl, profiles, tracker)
        start_round += 1
        cum_comm = ckpt["extra"].get("cum_comm", 0.0)
        cum_energy = ckpt["extra"].get("cum_energy", 0.0)
        print(f"[{method_name}] resumed from round {start_round}")

    start_time = time.time()
    for round_idx in range(start_round, cfg.num_rounds + 1):
        model_pool = build_model_pool(global_model, num_classes, specs)
        param_counts = {n: model_pool[n].num_parameters() for n in MODEL_ORDER}
        dispatched_models = list(np.random.choice(MODEL_ORDER, size=cfg.num_selected, replace=True))
        selections, gated_out_frac = select_clients_rafl(dispatched_models, profiles, rl, param_counts, cfg)
        for cid, _ in selections:
            profiles[cid].times_selected += 1
        tracker.record([cid for cid, _ in selections])

        # Resolve each client's actual (capacity-capped) model BEFORE simulating
        # time/energy, so the straggler deadline and the energy-cost metric
        # reflect what's really trained -- not the dispatched size, which a
        # weak/medium client is often capped down from (adaptive_local_prune
        # below applies the same capping; resolve_actual_model is the shared,
        # pure-function version of that decision so both stay in sync).
        actual_models = {cid: resolve_actual_model(profiles[cid], m) for cid, m in selections}
        sim_times = {cid: simulate_train_time(profiles[cid], actual_models[cid], param_counts, cfg)
                     for cid, _ in selections}
        deadline = (np.quantile(list(sim_times.values()), cfg.round_deadline_quantile)
                    if sim_times and cfg.aggregation_mode == "hybrid" else float("inf"))
        t_min = min(sim_times.values()) if sim_times else 1.0  # fastest client, for async staleness

        client_updates_weighted, sent_models, returned_models = [], [], []
        this_round_late = []
        for client_id, sent_model_name in selections:
            profile = profiles[client_id]
            # append comm bookkeeping every iteration (keeps sent/returned aligned in
            # both benign and dropout modes); actual_models[cid] == the model
            # adaptive_local_prune resolves below, so benign behavior is unchanged.
            sent_models.append(sent_model_name)
            returned_models.append(actual_models[client_id])
            if selected_client_fails(profile.device_state, cfg):
                continue  # robustness experiment only: client dropped out (rare for RAFL)
            _, local_model = adaptive_local_prune(profile, sent_model_name, model_pool, num_classes, specs)
            local_state, _ = train_local(local_model, make_client_loader(client_id, cfg, trainset, client_indices), cfg)
            n_samples = len(client_indices[client_id])
            rl.update_after_round_trip(client_id, sent_model_name, actual_models[client_id])

            on_time = sim_times[client_id] <= deadline
            if cfg.aggregation_mode == "async":
                # ASYNCHRONOUS: no round barrier; every update is applied the round it
                # arrives, weighted down by staleness (slower clients are more stale),
                # FedAsync-style. Distinct from sync (waits for all, no time weighting)
                # and hybrid (defers stragglers to the next round).
                staleness = (sim_times[client_id] / t_min) - 1.0
                w = float(n_samples) / (1.0 + cfg.staleness_lambda * staleness)
                client_updates_weighted.append((local_state, w))
            elif cfg.aggregation_mode == "sync" or on_time:
                client_updates_weighted.append((local_state, float(n_samples)))
            else:
                this_round_late.append((local_state, float(n_samples), 0))

        # fold in stragglers buffered from previous round(s), gentle staleness penalty
        for local_state, n_samples, staleness in straggler_buffer:
            staleness_penalty = 1.0 / (1.0 + cfg.staleness_lambda * staleness)
            client_updates_weighted.append((local_state, n_samples * staleness_penalty))
        straggler_buffer = [(s, n, stale + 1) for (s, n, stale) in this_round_late]

        energy_this_round = energy_cost_round(list(actual_models.items()), profiles, param_counts, cfg, sim_times)
        cum_energy += energy_this_round
        cum_comm += communication_cost_params(sent_models, returned_models, param_counts)

        if client_updates_weighted:
            global_model = aggregate_weighted(global_model, client_updates_weighted)
        maybe_log_round(history, round_idx, cfg, global_model, sent_models, returned_models,
                         specs, num_classes, test_loader, tracker, cum_comm, cum_energy, gated_out_frac)
        if round_idx % cfg.checkpoint_every == 0 or round_idx == cfg.num_rounds:
            save_checkpoint(cfg, method_name, round_idx, global_model, rl, profiles, tracker, history,
                             extra={"cum_comm": cum_comm, "cum_energy": cum_energy,
                                    "straggler_buffer_size": len(straggler_buffer)})
        torch.cuda.empty_cache()
    return global_model, history, time.time() - start_time


RUNNERS = {
    "fedavg": run_fedavg,
    "heterofl": run_heterofl,
    "scalefl": run_scalefl,
    "decoupled": run_decoupled,
    "adaptivefl": run_adaptivefl,
    "rafl": run_rafl,
}


def run_all_methods(cfg: Config, trainset, testset, test_loader, client_indices, num_classes, resume=True):
    assert_monotonic_model_sizes(build_model_specs(cfg), num_classes=num_classes)
    outputs = {}
    for method_name in cfg.baseline_names:
        print(f"\n{'=' * 70}\nRunning {method_name}\n{'=' * 70}")
        model, history, elapsed = RUNNERS[method_name](
            cfg, trainset, testset, test_loader, client_indices, num_classes,
            method_name=method_name, resume=resume)
        outputs[method_name] = {"model": model, "history": history, "summary": summarize_history(history, elapsed)}
    return outputs
