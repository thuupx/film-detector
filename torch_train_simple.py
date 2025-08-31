#!/usr/bin/env python3
"""
Patched training script for FilmSimulation-focused multi-task model.

Highlights of fixes/improvements:
- Safe NUM_WORKERS handling
- Effective number class weights (Cui et al.)
- Sampler that prioritizes FilmSimulation but also considers WB/ISO
- Correct imbalance ratio calculation from class counts
- Proper Top-5 computation
- Mixup/CutMix that supports multi-head (soft labels for mixed samples)
- Replace heavy GradNorm with a simple DWA-like adaptive weighting (stable)
- Per-head loss policy:
    * FilmSimulation: Weighted CrossEntropy + label smoothing (eps=0.1)
    * Very imbalanced heads: FocalLoss with effective weights
    * Medium imbalance: Weighted CrossEntropy (label smoothing small)
    * Balanced: standard CrossEntropy
- Per-task metrics logging; early stopping on FilmSimulation macro-F1
- Freeze backbone for initial epochs, then unfreeze
"""

import json
import os
import pickle
import random
import math
from collections import Counter, deque
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

# -------------------------
# Configuration
# -------------------------
DATA_FILE = "recipes.json"
IMG_SIZE = 224
MODEL_PATH = "model/model.pt"
TOOLS_PATH = "model/tools.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = max(4, (os.cpu_count() or 1) - 1)
BATCH_SIZE = 16
EPOCHS = 60
FREEZE_BACKBONE_EPOCHS = 3  # freeze backbone initially
PATIENCE = 10  # early stopping for FilmSimulation macro-F1
SWA_START = None  # optional: set epoch to start SWA, None to disable
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CATEGORICAL_OUTPUTS = [
    "FilmSimulation",
    "DynamicRange",
    "GrainEffect",
    "ColorChromeEffect",
    "ColorChromeEffectBlue",
    "WhiteBalance",
    "ISO",
    "ExposureCompensation",
]

NUMERICAL_OUTPUTS = [
    "Highlight",
    "Shadow",
    "Color",
    "NoiseReduction",
    "Sharpening",
    "Clarity",
    "WBShiftRed",
    "WBShiftBlue",
]

NUM_RANGES = {
    "Highlight": (-2, 4),
    "Shadow": (-2, 4),
    "Color": (-4, 4),
    "NoiseReduction": (-4, 4),
    "Sharpening": (-4, 4),
    "Clarity": (-5, 5),
    "WBShiftRed": (-9, 9),
    "WBShiftBlue": (-9, 9),
}

os.makedirs("model", exist_ok=True)
torch.backends.cudnn.benchmark = True

# -------------------------
# Utilities: Effective Number weights
# -------------------------
def effective_num_weights(counts: np.ndarray, beta: float = 0.9999) -> torch.Tensor:
    """
    Compute class weights using Effective Number of Samples (Cui et al.)
    counts: array of class counts (length = n_classes)
    returns: torch tensor of weights (device=DEVICE)
    """
    counts = np.array(counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    # avoid divide by zero
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# -------------------------
# Dataset & Sampler
# -------------------------
class FilmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cat_cols = CATEGORICAL_OUTPUTS
        self.num_cols = NUMERICAL_OUTPUTS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["PhotoPath"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # fallback to gray image to avoid crash
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        # categorical targets: already encoded into ints in load_data
        cat_targets = torch.tensor([int(row[c]) for c in self.cat_cols], dtype=torch.long)
        # numerical targets: assume normalized beforehand
        num_targets = torch.tensor([float(row[c]) for c in self.num_cols], dtype=torch.float32)

        return img, cat_targets, num_targets


def make_balanced_sampler(df: pd.DataFrame, priority_cols: List[str] = ["FilmSimulation"], priorities: List[float] = None) -> WeightedRandomSampler:
    """
    Create a weighted sampler that prioritizes FilmSimulation but also considers other columns (like WB, ISO).
    priority_cols: ordered list of columns to consider in weighting (first = most important)
    priorities: relative weights for those columns (sum should be <=1). remainder ignored.
    """
    if priorities is None:
        priorities = [0.7] + [0.2, 0.1]  # default: 70% film, 20% WB, 10% ISO (if present)

    n = len(df)
    # compute per-column inverse freq maps
    col_counts = {col: Counter(df[col].values) for col in priority_cols if col in df.columns}
    # precompute normalized inverse-frequency per value
    col_invfreq = {}
    for i, col in enumerate(priority_cols):
        if col not in col_counts:
            continue
        counts = col_counts[col]
        inv = {k: 1.0 / max(1, v) for k, v in counts.items()}
        # normalize so avg ~1 for stability
        vals = np.array(list(inv.values()), dtype=float)
        meanv = vals.mean() if len(vals) > 0 else 1.0
        inv_norm = {k: v / meanv for k, v in inv.items()}
        col_invfreq[col] = inv_norm

    weights = []
    for _, row in df.iterrows():
        w = 0.0
        for i, col in enumerate(priority_cols):
            if col not in col_invfreq:
                continue
            p = priorities[i] if i < len(priorities) else 0.0
            w += p * col_invfreq[col].get(row[col], 1.0)
        # avoid zero
        weights.append(max(w, 1e-6))

    weights = np.array(weights, dtype=np.float32)
    # scale to have mean 1
    weights = weights / weights.mean()
    sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
    return sampler


# -------------------------
# Losses
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        # weight argument will be passed to cross_entropy inside forward
        self.register_buffer("weight_buf", weight if weight is not None else torch.tensor([]))  # keep on-device
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: (B, C), targets: (B,)
        ce = F.cross_entropy(logits, targets, weight=(self.weight_buf if len(self.weight_buf) > 0 else None), reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def mixed_loss(criterion, outputs: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    return lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)


# -------------------------
# Mixup / CutMix helper
# -------------------------
class MixupCutmix:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob

    def _rand_beta(self, alpha):
        if alpha <= 0:
            return 1.0
        return np.random.beta(alpha, alpha)

    def __call__(self, x, y_cat, y_num):
        """
        x: tensor (B,C,H,W)
        y_cat: tensor (B, n_cat_heads)
        y_num: tensor (B, n_num)
        returns mixed_x, y_a_cat, y_b_cat, y_a_num, y_b_num, lam
        """
        if random.random() < 0.5:
            # apply mixing
            if random.random() < self.cutmix_prob:
                # CutMix
                lam = self._rand_beta(self.cutmix_alpha)
                B, C, H, W = x.shape
                index = torch.randperm(B, device=x.device)
                cut_rat = math.sqrt(1.0 - lam)
                cut_w = int(W * cut_rat)
                cut_h = int(H * cut_rat)
                cx = random.randint(0, W)
                cy = random.randint(0, H)
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)
                x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                y_a_cat, y_b_cat = y_cat, y_cat[index]
                y_a_num, y_b_num = y_num, y_num[index]
                return x, y_a_cat, y_b_cat, y_a_num, y_b_num, lam
            else:
                # Mixup
                lam = self._rand_beta(self.mixup_alpha)
                B = x.size(0)
                index = torch.randperm(B, device=x.device)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                y_a_cat, y_b_cat = y_cat, y_cat[index]
                y_a_num, y_b_num = y_num, y_num[index]
                return mixed_x, y_a_cat, y_b_cat, y_a_num, y_b_num, lam
        # no mix
        return x, y_cat, None, y_num, None, 1.0


# -------------------------
# Model
# -------------------------
class FilmNet(nn.Module):
    def __init__(self, class_counts: Dict[str, int], num_numeric: int):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feat_dim = backbone.classifier[1].in_features

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # categorical heads
        self.categorical_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),
                    nn.Linear(128, num_classes),
                )
                for name, num_classes in class_counts.items()
            }
        )

        self.numerical_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_numeric),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        x = self.head(x)
        cat_out = {name: head(x) for name, head in self.categorical_heads.items()}
        num_out = self.numerical_head(x)
        return cat_out, num_out


# -------------------------
# Transforms (per-task policy)
# -------------------------
def get_transforms(is_train: bool = True):
    if is_train:
        # We will use a moderate color jitter (not too strong) because FilmSimulation and WB are color-sensitive.
        return T.Compose(
            [
                T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                # moderate color jitter
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
                T.RandomRotation(8),
                T.RandomApply([T.GaussianBlur(3)], p=0.25),
                T.ToTensor(),
                T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(IMG_SIZE),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


# -------------------------
# Metrics helper for FilmSimulation
# -------------------------
class FilmMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.extend(preds.detach().cpu().numpy().tolist())
        self.targets.extend(targets.detach().cpu().numpy().tolist())

    def compute(self):
        preds = np.array(self.preds, dtype=int)
        targets = np.array(self.targets, dtype=int)
        if len(preds) == 0:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "balanced_accuracy": 0.0,
                "per_class_recall": np.zeros(self.num_classes),
                "confusion_matrix": np.zeros((self.num_classes, self.num_classes), dtype=int),
            }
        acc = (preds == targets).mean()
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(targets, preds)
        per_rec = recall_score(targets, preds, average=None, zero_division=0, labels=np.arange(self.num_classes))
        cm = confusion_matrix(targets, preds, labels=np.arange(self.num_classes))
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "per_class_recall": per_rec,
            "confusion_matrix": cm,
        }


# -------------------------
# Data loading
# -------------------------
def load_data() -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, torch.Tensor], Dict[str, int]]:
    print("Loading dataset...")
    with open(DATA_FILE) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Validate images
    valid_idx = []
    for i, p in enumerate(tqdm(df["PhotoPath"].values, desc="validating images")):
        try:
            Image.open(p).verify()
            valid_idx.append(i)
        except Exception:
            continue
    df = df.iloc[valid_idx].reset_index(drop=True)
    print(f"Valid samples: {len(df)}")

    encoders = {}
    class_weights = {}
    class_counts = {}

    # encode categorical labels to integers and compute class counts
    for col in CATEGORICAL_OUTPUTS:
        df[col] = df[col].fillna("Unknown").astype(str)
        le = pd.factorize(df[col])[0]  # faster than sklearn here
        # get unique label strings
        uniques = pd.unique(df[col])
        encoders[col] = {label: i for i, label in enumerate(uniques)}
        # map strings to integers consistently:
        mapping = {label: idx for idx, label in enumerate(uniques)}
        df[col] = df[col].map(mapping).astype(int)
        # counts
        counts = np.bincount(df[col].values, minlength=len(uniques))
        class_counts[col] = len(uniques)
        # effective number weights
        class_weights[col] = effective_num_weights(counts, beta=0.9999)
        print(f"{col}: classes={len(uniques)}, counts min/max={counts.min()}/{counts.max()}")

    # numeric columns: coerce and normalize into [0,1] per provided NUM_RANGES
    for col in NUMERICAL_OUTPUTS:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
        mn, mx = NUM_RANGES[col]
        df[col] = df[col].clip(mn, mx)
        df[col] = (df[col] - mn) / (mx - mn)

    return df, encoders, class_weights, class_counts


# -------------------------
# Training / Validation loops
# -------------------------
def train_one_epoch(model, loader, optimizer, cat_losses, num_loss, mixup_fn, film_metrics, task_names, task_weights):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, cat_targets, num_targets in loader:
        images = images.to(DEVICE)
        cat_targets = cat_targets.to(DEVICE)
        num_targets = num_targets.to(DEVICE)

        mixed_images, y_a_cat, y_b_cat, y_a_num, y_b_num, lam = mixup_fn(images, cat_targets, num_targets)
        mixed_images = mixed_images.to(DEVICE)

        optimizer.zero_grad()
        cat_outputs, num_outputs = model(mixed_images)

        # per-task loss list for DWA
        individual_losses = []
        total_cat_loss = 0.0

        for i, name in enumerate(task_names):
            crit = cat_losses[name]
            logits = cat_outputs[name]
            if y_b_cat is not None:
                loss_i = mixed_loss(crit, logits, y_a_cat[:, i].to(DEVICE), y_b_cat[:, i].to(DEVICE), lam)
            else:
                loss_i = crit(logits, cat_targets[:, i])
            # priority scaling: FilmSimulation gets extra weight via task_weights
            loss_i = loss_i * float(task_weights[i])
            individual_losses.append(loss_i.detach().cpu().item())
            total_cat_loss = total_cat_loss + loss_i

            # update film metrics only for real (non-mixed) samples, fallback: use y_a (approx)
            if name == "FilmSimulation" and y_b_cat is None:
                preds = torch.argmax(logits, dim=1)
                film_metrics.update(preds, cat_targets[:, i])

        # numerical loss
        if y_b_num is not None:
            num_l = mixed_loss(num_loss, num_outputs, y_a_num.to(DEVICE), y_b_num.to(DEVICE), lam)
        else:
            num_l = num_loss(num_outputs, num_targets)
        num_l = num_l * float(task_weights[-1])
        total_loss = total_cat_loss + num_l

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += total_loss.item()
        n_batches += 1

    avg_loss = running_loss / max(1, n_batches)
    return avg_loss, individual_losses  # return last batch losses as sample for DWA tracking


def validate(model, loader, cat_losses, num_loss, film_metrics, task_names):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_samples = 0

    # auxiliary stats
    aux_correct = {name: 0 for name in task_names if name != "FilmSimulation"}
    aux_total = {name: 0 for name in task_names if name != "FilmSimulation"}

    film_metrics.reset()
    film_top5_correct = 0

    with torch.no_grad():
        for images, cat_targets, num_targets in loader:
            images = images.to(DEVICE)
            cat_targets = cat_targets.to(DEVICE)
            num_targets = num_targets.to(DEVICE)
            batch_size = images.size(0)

            cat_outputs, num_outputs = model(images)

            # compute losses
            batch_cat_loss = 0.0
            for i, name in enumerate(task_names):
                logits = cat_outputs[name]
                crit = cat_losses[name]
                loss_i = crit(logits, cat_targets[:, i])
                batch_cat_loss = batch_cat_loss + loss_i

                # aux accuracy
                if name != "FilmSimulation":
                    preds = torch.argmax(logits, dim=1)
                    aux_correct[name] += (preds == cat_targets[:, i]).sum().item()
                    aux_total[name] += preds.size(0)

            num_l = num_loss(num_outputs, num_targets)
            batch_loss = batch_cat_loss + num_l
            total_loss += batch_loss.item()
            n_batches += 1
            total_samples += batch_size

            # film metrics and top5
            film_logits = cat_outputs["FilmSimulation"]
            film_targets = cat_targets[:, 0]
            _, film_preds = torch.max(film_logits, dim=1)
            film_metrics.update(film_preds, film_targets)

            # top-5 counting
            k = min(5, film_logits.size(1))
            topk = torch.topk(film_logits, k=k, dim=1).indices  # (B,k)
            # check if true label present in topk
            film_top5_correct += (topk == film_targets.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = total_loss / max(1, n_batches)
    film_results = film_metrics.compute()
    film_results["top5"] = film_top5_correct / max(1, total_samples)

    aux_accuracies = {name: (aux_correct[name] / max(1, aux_total[name])) for name in aux_correct}

    return avg_loss, film_results, aux_accuracies


# -------------------------
# DWA (simple adaptive weighting)
# -------------------------
class DWA:
    def __init__(self, num_tasks: int, T: float = 2.0):
        # track previous two epochs losses per task
        self.loss_hist = [deque(maxlen=2) for _ in range(num_tasks)]
        self.T = T
        self.num_tasks = num_tasks

    def update(self, task_losses: List[float]):
        # task_losses: list of per-task scalar losses from current epoch
        # returns weights (torch tensor) for tasks normalized to sum=num_tasks
        assert len(task_losses) == self.num_tasks
        for i, l in enumerate(task_losses):
            self.loss_hist[i].append(l)
        # if not enough history, return uniform weights
        for h in self.loss_hist:
            if len(h) < 2:
                return torch.ones(self.num_tasks, dtype=torch.float32, device=DEVICE)

        r = np.array([ (self.loss_hist[i][-1] / (self.loss_hist[i][-2] + 1e-12)) for i in range(self.num_tasks)])
        exp_r = np.exp(r / self.T)
        w = (self.num_tasks * exp_r / exp_r.sum()).astype(np.float32)
        return torch.tensor(w, device=DEVICE)


# -------------------------
# Main
# -------------------------
def main():
    print(f"Device: {DEVICE}, num_workers: {NUM_WORKERS}")

    df, encoders, class_weights, class_counts = load_data()

    # prepare CV split on FilmSimulation
    film_labels = df["FilmSimulation"].values
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    best_model_state = None
    best_film_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, film_labels)):
        print("\n" + "="*50)
        print(f"FOLD {fold+1}/3")
        print("="*50)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # datasets and sampler
        train_ds = FilmDataset(train_df, transform=get_transforms(is_train=True))
        val_ds = FilmDataset(val_df, transform=get_transforms(is_train=False))

        sampler = make_balanced_sampler(train_df, priority_cols=["FilmSimulation", "WhiteBalance", "ISO"], priorities=[0.7, 0.2, 0.1])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # model
        model = FilmNet(class_counts, num_numeric=len(NUMERICAL_OUTPUTS)).to(DEVICE)
        # freeze backbone for a few epochs
        for p in model.features.parameters():
            p.requires_grad = False

        # param groups: backbone (lower lr after unfreeze) and heads
        backbone_params = list(model.features.parameters())
        head_params = list(model.head.parameters()) + list(model.categorical_heads.parameters()) + list(model.numerical_head.parameters())

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": 1e-5},
                {"params": head_params, "lr": 5e-4},
            ],
            weight_decay=1e-3,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

        # build per-head loss functions using counts from df
        cat_losses = {}
        task_names = list(CATEGORICAL_OUTPUTS)
        # compute counts for each head for imbalance ratio
        counts_map = {}
        for name in CATEGORICAL_OUTPUTS:
            counts = np.bincount(train_df[name].values, minlength=class_counts[name])
            counts_map[name] = counts
        for name in CATEGORICAL_OUTPUTS:
            counts = counts_map[name]
            imbalance_ratio = float(counts.max()) / max(1, counts.min())

            if name == "FilmSimulation":
                # Weighted CrossEntropy + label smoothing 0.1
                w = class_weights[name]
                cat_losses[name] = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
                print(f"{name}: Weighted CE + LS (imbalance {imbalance_ratio:.1f})")
            elif imbalance_ratio > 100:
                # very imbalanced: focal
                w = class_weights[name]
                cat_losses[name] = FocalLoss(weight=w, gamma=2.0, alpha=1.0)
                print(f"{name}: FocalLoss (imbalance {imbalance_ratio:.1f})")
            elif imbalance_ratio > 10:
                w = class_weights[name]
                cat_losses[name] = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
                print(f"{name}: Weighted CE (imbalance {imbalance_ratio:.1f})")
            else:
                cat_losses[name] = nn.CrossEntropyLoss()
                print(f"{name}: Standard CE (imbalance {imbalance_ratio:.1f})")

        num_loss = nn.SmoothL1Loss()

        mixup = MixupCutmix(mixup_alpha=0.2, cutmix_alpha=1.0, cutmix_prob=0.5)
        film_metrics = FilmMetrics(num_classes=class_counts["FilmSimulation"])

        # DWA for dynamic task weighting (num_tasks = n_cat + 1 numerical)
        num_tasks = len(task_names) + 1
        dwa = DWA(num_tasks=num_tasks, T=2.0)
        # initialize uniform task weights
        task_weights = torch.ones(num_tasks, dtype=torch.float32, device=DEVICE)  # last pos for numerical

        # training loop
        best_fold_f1 = 0.0
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

            # unfreeze backbone after initial freeze epochs
            if epoch == FREEZE_BACKBONE_EPOCHS:
                for p in model.features.parameters():
                    p.requires_grad = True
                print("Unfroze backbone parameters; continuing fine-tune.")

            train_loss, last_task_losses = train_one_epoch(
                model, train_loader, optimizer, cat_losses, num_loss, mixup, film_metrics, task_names, task_weights
            )

            val_loss, film_results, aux_accuracies = validate(model, val_loader, cat_losses, num_loss, film_metrics, task_names)

            # Film primary metrics
            film_f1 = film_results["macro_f1"]
            film_top1 = film_results["accuracy"]
            film_top5 = film_results.get("top5", 0.0)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"FilmSimulation - Top1: {film_top1:.3f} Top5: {film_top5:.3f} MacroF1: {film_f1:.3f} BalancedAcc: {film_results['balanced_accuracy']:.3f}")

            # auxiliary metrics
            print("Auxiliary Task Accuracies:")
            for n, a in aux_accuracies.items():
                print(f"  - {n}: {a:.3f}")

            # update scheduler and DWA
            scheduler.step(val_loss)
            # create a quick per-task loss vector from last_task_losses if available
            # last_task_losses is per-batch list - we'll aggregate mean as proxy
            if isinstance(last_task_losses, list) and len(last_task_losses) >= num_tasks:
                # If train_one_epoch returned individual_losses as last batch, use them
                per_task_losses = np.array(last_task_losses[:num_tasks], dtype=float).tolist()
            else:
                per_task_losses = [1.0] * num_tasks
            task_weights = dwa.update(per_task_losses)

            # early stopping & checkpoint
            if film_f1 > best_fold_f1:
                best_fold_f1 = film_f1
                epochs_no_improve = 0
                # save best model state for this fold
                best_model_state = model.state_dict().copy()
                print(f"New best fold FilmSimulation Macro F1: {best_fold_f1:.4f}")
                # also print/save confusion matrix shape
                cm = film_results["confusion_matrix"]
                print(f"FilmSimulation confusion matrix shape: {cm.shape}")
            else:
                epochs_no_improve += 1
                print(f"No improvement epochs: {epochs_no_improve}/{PATIENCE}")
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping triggered on fold {fold+1} at epoch {epoch+1}")
                    break

        # end fold
        # keep best across folds
        if best_fold_f1 > best_film_f1:
            best_film_f1 = best_fold_f1
            best_model_state = best_model_state  # already set

    # save best model and tools
    if best_model_state is not None:
        final_model = FilmNet(class_counts, num_numeric=len(NUMERICAL_OUTPUTS))
        final_model.load_state_dict(best_model_state)
        torch.save({"state_dict": best_model_state, "class_counts": class_counts, "num_numeric": len(NUMERICAL_OUTPUTS)}, MODEL_PATH)
        tools = {"encoders": encoders, "class_counts": class_counts, "num_numeric": len(NUMERICAL_OUTPUTS), "num_ranges": NUM_RANGES}
        with open(TOOLS_PATH, "wb") as f:
            pickle.dump(tools, f)
        print(f"\nSaved model to {MODEL_PATH} and tools to {TOOLS_PATH}")

    print(f"\nDone. Best FilmSimulation Macro F1 across folds: {best_film_f1:.4f}")


if __name__ == "__main__":
    main()
