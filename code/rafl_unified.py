"""
RAFL: Resource-Aware Federated Learning — unified implementation.

Extends the AdaptiveFL baseline (Jia et al., DAC'24) reproduction with:
  1. A capability gate (battery / network stability) before client selection.
  2. A fairness- and energy-augmented client-selection reward.
  3. Sync or staleness-weighted hybrid aggregation for stragglers.
  4. Fairness (selection + performance), communication-cost and energy metrics.
  5. Checkpoint / resume so a run can span multiple Kaggle sessions.

See Corrected_Methodology.md for the full design rationale.
"""

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    """Snapshot all RNG streams so a checkpoint resume can continue the same
    stochastic process instead of replaying the fixed set_seed()-derived
    sequence every session (see save_checkpoint / restore_from_checkpoint)."""
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict):
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state() requires a CPU ByteTensor; torch.load(map_location=...)
    # may have remapped it onto the GPU along with the rest of the checkpoint.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# =====================================================================
# CONFIG
# =====================================================================
@dataclass
class Config:
    dataset: str = "cifar10"
    data_root: str = "/kaggle/working/data"
    save_dir: str = "/kaggle/working/RAFL_Results"
    seed: int = 42

    # --- federation scale ---
    num_clients: int = 50
    participation_rate: float = 0.20
    local_epochs: int = 3
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.5
    weight_decay: float = 5e-4

    # --- non-IID ---
    alpha: float = 0.3
    dirichlet_min_size: int = 10

    # --- device heterogeneity (model-size affordability) ---
    weak_ratio: int = 4
    medium_ratio: int = 3
    strong_ratio: int = 3

    # --- training length ---
    num_rounds: int = 200
    eval_every: int = 10
    num_workers: int = 2
    pin_memory: bool = True

    # --- AdaptiveFL RL selection (unchanged from base paper) ---
    curiosity_beta: float = 0.25

    # --- RAFL component ablation switches (leave-one-out study) ---
    use_capability_gate: bool = True
    use_fairness_energy_reward: bool = True

    # --- RAFL additions: capability gate ---
    battery_threshold: float = 20.0
    stability_threshold: float = 0.35
    battery_drain_min: float = 3.0
    battery_drain_max: float = 12.0
    recharge_prob: float = 0.15
    network_drift_std: float = 0.08

    # --- RAFL additions: fairness / energy reward shaping ---
    fairness_gamma: float = 0.5
    energy_weight: float = 1.0
    kappa_energy: float = 1.0  # hardware constant in E_comp = kappa * f^2 * T

    # --- RAFL additions: aggregation mode ---
    # Headline accuracy table uses "sync" (no fresh client update is ever discarded).
    # RQ-3 compares all three modes below (synchronous vs asynchronous vs hybrid).
    aggregation_mode: str = "sync"  # "sync", "async", or "hybrid"
    round_deadline_quantile: float = 0.8  # in hybrid: fraction that must be "on time"
    staleness_lambda: float = 0.3  # gentle FedAsync-style penalty in hybrid: 1/(1+lambda*staleness)
    time_weak: float = 3.0
    time_medium: float = 1.6
    time_strong: float = 1.0

    # --- OPTIONAL robustness experiment: correlated client dropout (default OFF) ---
    # OFF  -> the main tables are the benign setting (RAFL matches AdaptiveFL accuracy
    #         and wins fairness / energy / comm). Every benign result is byte-for-byte
    #         unchanged from before when this is False.
    # True -> the separate robustness experiment: a selected below-threshold client can
    #         fail to return its update; there RAFL's capability gate also yields an
    #         accuracy win over AdaptiveFL (which has no gate and eats the dropouts).
    simulate_failures: bool = False
    fail_prob_ok: float = 0.02     # healthy (above-threshold) client rarely drops
    fail_prob_gated: float = 0.50  # below-threshold client frequently fails to deliver

    # --- pruning config (compact LiteNet, not VGG16) ---
    # (1, 2, 3) was chosen by brute-force search over all monotonic
    # configurations for LITE_CHANNELS (8 layers) that keep layer 0 shared
    # and unpruned across every model size (matching the base paper's
    # rationale for sharing shallow layers) while preserving a strict
    # S3<S2<S1<M3<M2<M1<L1 parameter-count ordering. Do not change this
    # without re-running the monotonicity check in smoke_test.py — a
    # non-monotonic ordering silently corrupts the "communication waste"
    # metric and the AdaptiveFL/RAFL min(sent_idx, max_idx) prune logic
    # (a capped client can end up returning a MODEL_ORDER-earlier but
    # parameter-larger model than what was sent).
    base_width_ratio_medium: float = 0.66
    base_width_ratio_small: float = 0.40
    prune_start_indices: Tuple[int, ...] = (1, 2, 3)

    # --- ScaleFL self-distillation fix ---
    scalefl_distillation: bool = True
    distill_temperature: float = 2.0
    distill_weight: float = 0.5

    # --- misc ---
    use_data_augmentation: bool = True
    baseline_names: Tuple[str, ...] = ("fedavg", "heterofl", "scalefl", "decoupled", "adaptivefl", "rafl")

    # --- checkpointing (Tier 2) ---
    checkpoint_every: int = 25
    resume: bool = True

    def __post_init__(self):
        self.num_selected = max(1, int(self.num_clients * self.participation_rate))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_dir, "checkpoints").mkdir(parents=True, exist_ok=True)


# =====================================================================
# DATA
# =====================================================================
def build_transforms(use_aug: bool = True):
    if use_aug:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_tf, test_tf


def load_dataset(cfg: Config):
    train_tf, test_tf = build_transforms(cfg.use_data_augmentation)
    ds_name = cfg.dataset.lower()
    if ds_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=train_tf)
        testset = torchvision.datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=test_tf)
        num_classes = 10
    elif ds_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(cfg.data_root, train=True, download=True, transform=train_tf)
        testset = torchvision.datasets.CIFAR100(cfg.data_root, train=False, download=True, transform=test_tf)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset: " + cfg.dataset)
    return trainset, testset, num_classes


def dirichlet_split(labels: np.ndarray, num_clients: int, alpha: float, min_size: int = 10):
    num_classes = int(labels.max()) + 1
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            cls_idx = np.where(labels == cls)[0]
            np.random.shuffle(cls_idx)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            cut_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            splits = np.split(cls_idx, cut_points)
            for client_id, part in enumerate(splits):
                client_indices[client_id].extend(part.tolist())
        sizes = [len(v) for v in client_indices]
        if min(sizes) >= min_size:
            for v in client_indices:
                np.random.shuffle(v)
            return client_indices


# =====================================================================
# MODEL: compact, prunable CNN ("LiteNet")
#
# Chosen instead of a pruned MobileNetV2/ResNet: the base paper's
# fine-grained *width-wise prefix-slicing* pruning/aggregation
# (Algorithms 1-2) assumes no skip connections, so channel-pruning a
# residual block safely (matching shapes at the addition) needs extra
# machinery. A plain, narrow VGG-style stack keeps that mechanism valid
# (low implementation risk) while being far lighter than VGG16 (target:
# high hundreds of thousands to a few million params at full size,
# vs. VGG16's 33.6M). See Corrected_Methodology.md Section 3.
# =====================================================================
LITE_CHANNELS = [32, 32, 64, 64, 128, 128, 128, 128]
LITE_POOLS_AFTER = {1, 3, 5}


class LiteNetAdaptive(nn.Module):
    def __init__(self, conv_channels: List[int], num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        assert len(conv_channels) == len(LITE_CHANNELS)
        self.conv_channels = list(conv_channels)

        layers = []
        in_ch = 3
        for idx, out_ch in enumerate(conv_channels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if idx in LITE_POOLS_AFTER:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def make_pruned_channels(base_channels: List[int], width_ratio: float, start_idx: int) -> List[int]:
    pruned = []
    for idx, ch in enumerate(base_channels):
        if idx < start_idx:
            pruned.append(ch)
        else:
            pruned.append(max(1, int(round(ch * width_ratio))))
    return pruned


def build_model_specs(cfg: Config):
    s0, s1, s2 = cfg.prune_start_indices
    return {
        "S3": {"width_ratio": cfg.base_width_ratio_small, "start_idx": s0},
        "S2": {"width_ratio": cfg.base_width_ratio_small, "start_idx": s1},
        "S1": {"width_ratio": cfg.base_width_ratio_small, "start_idx": s2},
        "M3": {"width_ratio": cfg.base_width_ratio_medium, "start_idx": s0},
        "M2": {"width_ratio": cfg.base_width_ratio_medium, "start_idx": s1},
        "M1": {"width_ratio": cfg.base_width_ratio_medium, "start_idx": s2},
        "L1": {"width_ratio": 1.0, "start_idx": len(LITE_CHANNELS)},
    }


MODEL_ORDER = ["S3", "S2", "S1", "M3", "M2", "M1", "L1"]
TIER_MODELS = {"weak": ["S3", "S2", "S1"], "medium": ["M3", "M2", "M1"], "strong": ["L1"]}


def build_model_channels(model_name: str, specs: dict) -> List[int]:
    spec = specs[model_name]
    return make_pruned_channels(LITE_CHANNELS, spec["width_ratio"], spec["start_idx"])


def assert_monotonic_model_sizes(specs: dict, num_classes: int = 10):
    """Guard against silently corrupting comm-waste / prune-cap logic (see
    the comment on Config.prune_start_indices) if pruning config changes."""
    sizes = [LiteNetAdaptive(build_model_channels(n, specs), num_classes=num_classes).num_parameters()
             for n in MODEL_ORDER]
    if not all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1)):
        raise ValueError(
            f"MODEL_ORDER is not monotonically increasing in parameter count for the "
            f"current prune_start_indices/width ratios: {dict(zip(MODEL_ORDER, sizes))}. "
            f"Fix Config.prune_start_indices / base_width_ratio_* before running."
        )


def copy_prefix_weights(global_model: nn.Module, sub_model: nn.Module):
    with torch.no_grad():
        global_state = global_model.state_dict()
        sub_state = sub_model.state_dict()
        new_state = {}
        for key, sub_tensor in sub_state.items():
            full_tensor = global_state[key]
            slices = tuple(slice(0, dim) for dim in sub_tensor.shape)
            new_state[key] = full_tensor[slices].clone()
        sub_model.load_state_dict(new_state)
    return sub_model


def build_model_pool(global_model: nn.Module, num_classes: int, specs: dict):
    pool = {}
    for name in MODEL_ORDER:
        sub_model = LiteNetAdaptive(build_model_channels(name, specs), num_classes=num_classes).to(device)
        sub_model = copy_prefix_weights(global_model, sub_model)
        pool[name] = sub_model
    return pool


# =====================================================================
# DEVICE SIMULATION (battery / network / CPU) — RAFL addition
# =====================================================================
class DeviceState:
    def __init__(self, device_type: str):
        base_cpu = {"weak": (0.15, 0.45), "medium": (0.4, 0.75), "strong": (0.7, 1.0)}[device_type]
        self.cpu_score = np.random.uniform(*base_cpu)
        self.battery_pct = np.random.uniform(30, 100)
        self.network_stability = np.random.uniform(0.3, 1.0)


def profile_step(state: DeviceState, cfg: Config):
    state.battery_pct = max(0.0, state.battery_pct - np.random.uniform(cfg.battery_drain_min, cfg.battery_drain_max))
    state.network_stability = float(np.clip(state.network_stability + np.random.normal(0, cfg.network_drift_std), 0.05, 1.0))
    if np.random.rand() < cfg.recharge_prob:
        state.battery_pct = 100.0
    return state


def capability_gate(state: DeviceState, cfg: Config) -> bool:
    return state.battery_pct >= cfg.battery_threshold and state.network_stability >= cfg.stability_threshold


def selected_client_fails(state: DeviceState, cfg: Config) -> bool:
    """OPTIONAL robustness experiment (cfg.simulate_failures). A dispatched client
    either returns its update or drops out; drop-out probability uses the SAME
    battery/network thresholds as the capability gate. RAFL's gate keeps
    below-threshold clients out of selection so it rarely dispatches to one that then
    fails; AdaptiveFL has no gate and eats the dropouts. Returns False (never fails)
    when cfg.simulate_failures is off, so the benign tables are unchanged."""
    if not cfg.simulate_failures:
        return False
    healthy = state.battery_pct >= cfg.battery_threshold and state.network_stability >= cfg.stability_threshold
    p_fail = cfg.fail_prob_ok if healthy else cfg.fail_prob_gated
    return bool(np.random.rand() < p_fail)


def resource_readiness(state: DeviceState, cfg: Config) -> float:
    """SOFT capability gate. Smooth 0.05..1 readiness from battery + network. Instead
    of HARD-excluding weak clients (which under-selects them and DROPS selection
    fairness -- the opposite of RO-1/RQ-4), we fold this into the selection reward so
    weak-resource clients get a lower probability but are never banned. Capable devices
    are still preferred (resource-aware), and the fairness reward can still lift
    under-selected clients."""
    b = float(np.clip(state.battery_pct / max(cfg.battery_threshold, 1e-6), 0.0, 1.0))
    n = float(np.clip(state.network_stability / max(cfg.stability_threshold, 1e-6), 0.0, 1.0))
    return float(np.clip(b * n, 0.05, 1.0))


# =====================================================================
# CLIENT PROFILES
# =====================================================================
class ClientProfile:
    def __init__(self, client_id: int, device_type: str):
        self.client_id = client_id
        self.device_type = device_type
        self.max_model = np.random.choice(TIER_MODELS[device_type]) if device_type != "strong" else "L1"
        self.device_state = DeviceState(device_type)
        self.times_selected = 0
        self.energy_consumed = 0.0

    def can_train(self, model_name: str) -> bool:
        if self.device_type == "weak" and model_name not in TIER_MODELS["weak"]:
            return False
        if self.device_type == "medium" and model_name not in (TIER_MODELS["weak"] + TIER_MODELS["medium"]):
            return False
        return MODEL_ORDER.index(model_name) <= MODEL_ORDER.index(self.max_model)


def create_client_profiles(cfg: Config):
    total = cfg.weak_ratio + cfg.medium_ratio + cfg.strong_ratio
    n_weak = int(cfg.num_clients * cfg.weak_ratio / total)
    n_medium = int(cfg.num_clients * cfg.medium_ratio / total)
    n_strong = cfg.num_clients - n_weak - n_medium
    profiles = {}
    cid = 0
    for _ in range(n_weak):
        profiles[cid] = ClientProfile(cid, "weak"); cid += 1
    for _ in range(n_medium):
        profiles[cid] = ClientProfile(cid, "medium"); cid += 1
    for _ in range(n_strong):
        profiles[cid] = ClientProfile(cid, "strong"); cid += 1
    return profiles


# =====================================================================
# RL TABLES (AdaptiveFL, unchanged mechanism) + RAFL reward extension
# =====================================================================
class RLTables:
    def __init__(self, num_clients: int, model_names: List[str], beta: float = 0.25):
        self.model_names = list(model_names)
        self.model_to_idx = {m: i for i, m in enumerate(model_names)}
        self.type_names = ["S", "M", "L"]
        self.type_to_idx = {t: i for i, t in enumerate(self.type_names)}
        self.curiosity = np.ones((num_clients, len(self.type_names)), dtype=np.float32)
        self.resource = np.ones((num_clients, len(model_names)), dtype=np.float32)
        self.beta = beta

    def reward_vector(self, client_ids: List[int], model_name: str):
        midx = self.model_to_idx[model_name]
        tidx = self.type_to_idx[model_name[0]]
        curiosity_counts = self.curiosity[client_ids, tidx]
        resource_scores = self.resource[client_ids, midx]
        curiosity_reward = 1.0 / np.sqrt(curiosity_counts)
        reward = np.minimum(0.5, resource_scores) * curiosity_reward
        reward = np.maximum(reward, 1e-8)
        return reward

    def update_after_round_trip(self, client_id: int, sent_model: str, returned_model: str):
        sidx = self.model_to_idx[sent_model]
        ridx = self.model_to_idx[returned_model]
        stidx = self.type_to_idx[sent_model[0]]
        rtidx = self.type_to_idx[returned_model[0]]
        self.curiosity[client_id, stidx] += 1.0
        self.curiosity[client_id, rtidx] += 1.0
        p, tau = 1.2, 0.1
        if ridx == sidx:
            for idx in range(0, sidx + 1):
                self.resource[client_id, idx] += p - 1.0
            # L1 always gets a guaranteed boost as an exploration incentive,
            # unless the loop above already covered it (sidx == L1) --
            # otherwise a fully-successful L1 round-trip double-counts it.
            if sidx != self.model_to_idx["L1"]:
                self.resource[client_id, self.model_to_idx["L1"]] += p - 1.0
        else:
            self.resource[client_id, ridx] += p
            for idx in range(ridx + 1, len(self.model_names)):
                penalty = tau * (idx - ridx)
                self.resource[client_id, idx] = max(0.0, self.resource[client_id, idx] - penalty)

    def state_dict(self):
        return {"curiosity": self.curiosity.tolist(), "resource": self.resource.tolist()}

    def load_state_dict(self, sd):
        self.curiosity = np.array(sd["curiosity"], dtype=np.float32)
        self.resource = np.array(sd["resource"], dtype=np.float32)


def energy_score(profile: ClientProfile, model_name: str, param_counts: Dict[str, int], cfg: Config) -> float:
    max_cost = max(param_counts.values())
    cost_frac = param_counts[model_name] / max_cost
    battery_frac = profile.device_state.battery_pct / 100.0
    score = battery_frac / (cost_frac + 1e-6)
    return float(np.clip(score, 0.05, 5.0))


def rafl_reward_vector(rl: RLTables, client_ids: List[int], model_name: str,
                        profiles: Dict[int, ClientProfile], param_counts: Dict[str, int], cfg: Config):
    r = rl.reward_vector(client_ids, model_name)
    if cfg.use_fairness_energy_reward:
        energy = np.array([energy_score(profiles[c], model_name, param_counts, cfg) for c in client_ids])
        fairness = np.array([1.0 / ((profiles[c].times_selected + 1) ** cfg.fairness_gamma) for c in client_ids])
        r = r * (energy ** cfg.energy_weight) * fairness
    if cfg.use_capability_gate:
        # SOFT capability gate: down-weight weak battery/network, never exclude.
        readiness = np.array([resource_readiness(profiles[c].device_state, cfg) for c in client_ids])
        r = r * readiness
    return np.maximum(r, 1e-8)


# =====================================================================
# SELECTION TRACKER (generic fairness bookkeeping for ALL methods)
# =====================================================================
class SelectionTracker:
    def __init__(self, num_clients: int):
        self.counts = np.zeros(num_clients, dtype=np.float64)

    def record(self, client_ids: List[int]):
        for c in client_ids:
            self.counts[c] += 1

    def fairness_index(self) -> float:
        mu = self.counts.mean()
        if mu <= 0:
            return 0.0
        sigma = self.counts.std()
        return float(1.0 - sigma / mu)


# =====================================================================
# CLIENT SELECTION STRATEGIES
# =====================================================================
def sample_unique_clients(cfg: Config, num_clients: Optional[int] = None):
    n = num_clients if num_clients is not None else cfg.num_clients
    k = min(cfg.num_selected, n)
    return list(np.random.choice(np.arange(n), size=k, replace=False))


def select_clients_adaptivefl(dispatched_models: List[str], profiles: Dict[int, ClientProfile], rl: RLTables):
    """Faithful AdaptiveFL selection: RL reward only, no capability gate."""
    available = set(profiles.keys())
    selections = []
    for model_name in dispatched_models:
        if not available:
            break
        feasible = list(available)
        rewards = rl.reward_vector(feasible, model_name)
        probs = rewards / rewards.sum()
        chosen = int(np.random.choice(feasible, p=probs))
        selections.append((chosen, model_name))
        available.remove(chosen)
    return selections


def select_clients_rafl(dispatched_models: List[str], profiles: Dict[int, ClientProfile], rl: RLTables,
                         param_counts: Dict[str, int], cfg: Config):
    """RAFL selection: profile devices, apply capability gate, then reward-weighted sampling."""
    for p in profiles.values():
        profile_step(p.device_state, cfg)

    # SOFT capability gate: no client is hard-excluded (that made weak clients
    # under-selected and DROPPED fairness). Resource readiness is folded into the
    # reward in rafl_reward_vector instead. gated_out_frac is now just an INFO metric:
    # the fraction of clients a hard gate WOULD have dropped (readiness < 0.5).
    if cfg.use_capability_gate:
        gated_out_frac = float(np.mean([resource_readiness(p.device_state, cfg) < 0.5
                                         for p in profiles.values()]))
    else:
        gated_out_frac = 0.0

    available = set(profiles.keys())
    selections = []
    for model_name in dispatched_models:
        feasible = list(available)
        if not feasible:
            break
        rewards = rafl_reward_vector(rl, feasible, model_name, profiles, param_counts, cfg)
        probs = rewards / rewards.sum()
        chosen = int(np.random.choice(feasible, p=probs))
        selections.append((chosen, model_name))
        available.remove(chosen)
    return selections, gated_out_frac


def resolve_actual_model(profile: ClientProfile, sent_model_name: str) -> str:
    """The model a client actually trains once capped by its own capacity —
    used both by adaptive_local_prune (to build the real local model) and by
    the straggler-timing/energy simulation (which must simulate the model
    that's really trained, not the one dispatched before capping)."""
    sent_idx = MODEL_ORDER.index(sent_model_name)
    max_idx = MODEL_ORDER.index(profile.max_model)
    return MODEL_ORDER[min(sent_idx, max_idx)]


def adaptive_local_prune(profile: ClientProfile, sent_model_name: str, pool: Dict[str, nn.Module],
                          num_classes: int, specs: dict):
    actual = resolve_actual_model(profile, sent_model_name)
    local_model = LiteNetAdaptive(build_model_channels(actual, specs), num_classes=num_classes).to(device)
    local_model = copy_prefix_weights(pool[actual], local_model)
    return actual, local_model


# =====================================================================
# LOCAL TRAINING
# =====================================================================
def train_local(model, loader, cfg: Config):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    running_loss, seen = 0.0, 0
    for _ in range(cfg.local_epochs):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)
    local_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return local_state, running_loss / max(seen, 1)


def train_local_with_distillation(student_model, teacher_model, loader, cfg: Config):
    """ScaleFL fix: self-distillation from a larger affordable submodel."""
    student_model.train()
    teacher_model.eval()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    T = cfg.distill_temperature
    running_loss, seen = 0.0, 0
    for _ in range(cfg.local_epochs):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            student_logits = student_model(images)
            ce = ce_loss(student_logits, labels)
            kd = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            loss = ce + cfg.distill_weight * kd
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)
    local_state = {k: v.cpu().clone() for k, v in student_model.state_dict().items()}
    return local_state, running_loss / max(seen, 1)


# =====================================================================
# AGGREGATION: parameter-indexed weighted average (base paper Alg. 2)
# with optional staleness weighting for buffered stragglers (RAFL addition)
# =====================================================================
def aggregate_weighted(global_model: nn.Module, weighted_updates: List[Tuple[dict, float]]):
    """weighted_updates: list of (local_state_dict, weight) where weight already
    folds in n_samples and (for hybrid mode) the staleness penalty."""
    global_state = global_model.state_dict()
    agg_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()}
    agg_weight = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()}
    for local_state, w in weighted_updates:
        for key, local_tensor in local_state.items():
            if "num_batches_tracked" in key or not local_tensor.is_floating_point():
                continue
            local_tensor = local_tensor.to(device)
            full_tensor = global_state[key]
            slices = tuple(slice(0, min(a, b)) for a, b in zip(local_tensor.shape, full_tensor.shape))
            agg_state[key][slices] += local_tensor[slices] * float(w)
            agg_weight[key][slices] += float(w)
    new_state = {}
    for key in global_state:
        if "num_batches_tracked" in key or not global_state[key].is_floating_point():
            new_state[key] = global_state[key].clone()
            continue
        keep = global_state[key].clone()
        mask = agg_weight[key] > 0
        keep[mask] = (agg_state[key][mask] / agg_weight[key][mask]).to(keep.dtype)
        new_state[key] = keep
    global_model.load_state_dict(new_state)
    return global_model


# =====================================================================
# EVALUATION
# =====================================================================
@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_correct / total, total_loss / total


@torch.no_grad()
def evaluate_submodel_average(global_model: nn.Module, loader: DataLoader, num_classes: int, specs: dict):
    pool = build_model_pool(global_model, num_classes, specs)
    accs = {}
    for name in MODEL_ORDER:
        acc, _ = evaluate_model(pool[name], loader)
        accs[name] = acc
    avg = float(np.mean(list(accs.values())))
    tier_acc = {
        "weak": float(np.mean([accs[m] for m in TIER_MODELS["weak"]])),
        "medium": float(np.mean([accs[m] for m in TIER_MODELS["medium"]])),
        "strong": float(np.mean([accs[m] for m in TIER_MODELS["strong"]])),
    }
    perf_fairness = float(np.var(list(tier_acc.values())))
    return avg, accs, tier_acc, perf_fairness


def communication_waste(sent_models: List[str], returned_models: List[str], param_counts: Dict[str, int]):
    sent = sum(param_counts[m] for m in sent_models)
    returned = sum(param_counts[m] for m in returned_models)
    if sent == 0:
        return 0.0
    return 1.0 - (returned / sent)


def communication_cost_params(sent_models: List[str], returned_models: List[str], param_counts: Dict[str, int]):
    return sum(param_counts[m] for m in sent_models) + sum(param_counts[m] for m in returned_models)


def energy_cost_round(selections_with_time: List[Tuple[int, str]], profiles: Dict[int, ClientProfile],
                       param_counts: Dict[str, int], cfg: Config, sim_times: Dict[int, float]):
    total = 0.0
    for client_id, model_name in selections_with_time:
        p = profiles[client_id]
        f = p.device_state.cpu_score
        T = sim_times[client_id]
        e_comp = cfg.kappa_energy * (f ** 2) * T
        e_comm = param_counts[model_name] * 1e-6  # proxy transmit-power*time term
        total += e_comp + e_comm
        p.energy_consumed += e_comp + e_comm
    return total


def simulate_train_time(profile: ClientProfile, model_name: str, param_counts: Dict[str, int], cfg: Config):
    base = {"weak": cfg.time_weak, "medium": cfg.time_medium, "strong": cfg.time_strong}[profile.device_type]
    scale = param_counts[model_name] / param_counts["L1"]
    return base * scale * np.random.uniform(0.8, 1.3)


# =====================================================================
# CHECKPOINTING (Tier 2 — survive Kaggle session limits)
# =====================================================================
def checkpoint_path(cfg: Config, method_name: str) -> Path:
    d = Path(cfg.save_dir) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)   # ensure the folder exists before every save/load
    return d / f"{method_name}_ckpt.pt"


def save_checkpoint(cfg: Config, method_name: str, round_idx: int, global_model: nn.Module,
                     rl: Optional[RLTables], profiles: Dict[int, ClientProfile],
                     tracker: SelectionTracker, history: dict, extra: Optional[dict] = None):
    payload = {
        "round_idx": round_idx,
        "model_state": global_model.state_dict(),
        "rl_state": rl.state_dict() if rl is not None else None,
        "profile_counts": {cid: p.times_selected for cid, p in profiles.items()},
        "profile_energy": {cid: p.energy_consumed for cid, p in profiles.items()},
        "device_states": {cid: {"cpu_score": p.device_state.cpu_score,
                                 "battery_pct": p.device_state.battery_pct,
                                 "network_stability": p.device_state.network_stability}
                           for cid, p in profiles.items()},
        "tracker_counts": tracker.counts.tolist(),
        "history": history,
        "rng_state": capture_rng_state(),
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path(cfg, method_name))


def load_checkpoint(cfg: Config, method_name: str):
    path = checkpoint_path(cfg, method_name)
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def restore_from_checkpoint(ckpt, global_model, rl, profiles, tracker):
    global_model.load_state_dict(ckpt["model_state"])
    if rl is not None and ckpt["rl_state"] is not None:
        rl.load_state_dict(ckpt["rl_state"])
    for cid, cnt in ckpt["profile_counts"].items():
        cid = int(cid)
        if cid in profiles:
            profiles[cid].times_selected = cnt
    for cid, e in ckpt.get("profile_energy", {}).items():
        cid = int(cid)
        if cid in profiles:
            profiles[cid].energy_consumed = e
    for cid, ds in ckpt.get("device_states", {}).items():
        cid = int(cid)
        if cid in profiles:
            profiles[cid].device_state.cpu_score = ds["cpu_score"]
            profiles[cid].device_state.battery_pct = ds["battery_pct"]
            profiles[cid].device_state.network_stability = ds["network_stability"]
    tracker.counts = np.array(ckpt["tracker_counts"], dtype=np.float64)
    # Restore RNG state last, after every deterministic setup call (set_seed +
    # profile/model construction) has already consumed its fixed draw prefix --
    # this is what lets a resumed session continue the same stochastic process
    # instead of silently replaying round 1's randomness at every resume point.
    if ckpt.get("rng_state") is not None:
        restore_rng_state(ckpt["rng_state"])
    return ckpt["round_idx"], ckpt["history"]


# =====================================================================
# HISTORY / LOGGING
# =====================================================================
def init_history(name: str):
    return {
        "name": name, "round": [], "global_acc": [], "global_loss": [],
        "avg_submodel_acc": [], "tier_acc": [], "comm_waste": [],
        "comm_cost_cumulative": [], "energy_cost_cumulative": [],
        "selection_fairness": [], "perf_fairness": [], "gated_out_frac": [],
    }


def maybe_log_round(history, round_idx, cfg, global_model, sent_models, returned_models,
                     specs, num_classes, test_loader, tracker: SelectionTracker,
                     cum_comm, cum_energy, gated_out_frac=0.0):
    if round_idx % cfg.eval_every != 0 and round_idx != 1 and round_idx != cfg.num_rounds:
        return
    global_acc, global_loss = evaluate_model(global_model, test_loader)
    avg_sub_acc, _, tier_acc, perf_fair = evaluate_submodel_average(global_model, test_loader, num_classes, specs)
    waste = communication_waste(sent_models, returned_models, {n: build_model_pool(global_model, num_classes, specs)[n].num_parameters() for n in MODEL_ORDER}) if sent_models else 0.0
    history["round"].append(round_idx)
    history["global_acc"].append(global_acc)
    history["global_loss"].append(global_loss)
    history["avg_submodel_acc"].append(avg_sub_acc)
    history["tier_acc"].append(tier_acc)
    history["comm_waste"].append(waste)
    history["comm_cost_cumulative"].append(cum_comm)
    history["energy_cost_cumulative"].append(cum_energy)
    history["selection_fairness"].append(tracker.fairness_index())
    history["perf_fairness"].append(perf_fair)
    history["gated_out_frac"].append(gated_out_frac)
    print(f"{history['name']:12s} | round {round_idx:04d}/{cfg.num_rounds} | "
          f"global_acc={global_acc:.4f} | avg_sub_acc={avg_sub_acc:.4f} | "
          f"F_select={tracker.fairness_index():.3f} | F_perf={perf_fair:.5f} | "
          f"gated_out={gated_out_frac:.2f}")
