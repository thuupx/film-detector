# torch_train.py
#
# PyTorch training script equivalent to TensorFlow `main.py` with
# - MobileNetV2 backbone
# - Multi-output heads (8 categorical + 1 numerical vector of 8)
# - Focal loss for imbalanced categorical heads
# - Stratified K-Fold training (by FilmSimulation)
# - Data augmentation
# - Checkpoints and best model selection

import json
import os
import pickle
from typing import Dict, List, Tuple

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

# --- Configuration ---
JSON_DATA_FILE = "recipes.json"
IMG_WIDTH, IMG_HEIGHT = 224, 224
SAVED_MODEL_PATH = "model/model.pt"
TOOLS_PATH = "model/tools.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Performance/batching config (override via env vars if needed)
BATCH_TRAIN = int(os.getenv("BATCH_TRAIN", "32"))
BATCH_VAL = int(os.getenv("BATCH_VAL", "64"))
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1) - 1)  # leave 1 core free by default
NUM_WORKERS = int(os.getenv("NUM_WORKERS", str(DEFAULT_WORKERS)))
PIN_MEMORY = bool(torch.cuda.is_available())

# Let cuDNN pick optimal algorithms for current input sizes
torch.backends.cudnn.benchmark = True

CATEGORICAL_OUTPUT_NAMES = [
    "FilmSimulation",
    "DynamicRange",
    "GrainEffect",
    "ColorChromeEffect",
    "ColorChromeEffectBlue",
    "WhiteBalance",
    "ISO",
    "ExposureCompensation",
]
NUMERICAL_OUTPUT_NAMES = [
    "Highlight",
    "Shadow",
    "Color",
    "NoiseReduction",
    "Sharpening",
    "Clarity",
    "WBShiftRed",
    "WBShiftBlue",
]

# Default set (will be overridden dynamically per fold)
SEVERELY_IMBALANCED = set(
    ["FilmSimulation", "DynamicRange", "WhiteBalance", "ISO", "ExposureCompensation"]
)

FIXED_NUM_RANGES = {
    "Highlight": (-2, 4),
    "Shadow": (-2, 4),
    "Color": (-4, 4),
    "NoiseReduction": (-4, 4),
    "Sharpening": (-4, 4),
    "Clarity": (-5, 5),
    "WBShiftRed": (-9, 9),
    "WBShiftBlue": (-9, 9),
}


# --- Utils ---
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), target: (B,) class indices
        ce_loss = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )  # (B,)
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class RecipesDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, encoders: Dict[str, LabelEncoder], transforms: T.Compose
    ):
        self.df = df.reset_index(drop=True)
        self.encoders = encoders
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["PhotoPath"]
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        # categorical targets as indices
        cat_targets = {}
        for name in CATEGORICAL_OUTPUT_NAMES:
            label = str(row[name]) if pd.notna(row[name]) else "Unknown"
            target_idx = self.encoders[name].transform([label])[0]
            cat_targets[name] = target_idx
        # numerical scaled [0,1]
        num_values = []
        for name in NUMERICAL_OUTPUT_NAMES:
            v = row[name]
            if pd.isna(v):
                # will be filled by median earlier, but guard anyway
                v = 0
            min_v, max_v = FIXED_NUM_RANGES[name]
            scaled = (float(v) - min_v) / (max_v - min_v)
            num_values.append(scaled)
        numerical = torch.tensor(num_values, dtype=torch.float32)
        return image, cat_targets, numerical


class MultiHeadMobileNet(nn.Module):
    def __init__(self, class_counts: Dict[str, int], num_numeric: int):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.classifier[1].in_features
        # Shared head
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        # Categorical heads
        self.cat_heads = nn.ModuleDict(
            {
                f"{name}": nn.Linear(128, num_classes)
                for name, num_classes in class_counts.items()
            }
        )
        # Numerical head
        self.num_head = nn.Linear(128, num_numeric)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        feats = self.fc1(x)
        cat_logits = {name: head(feats) for name, head in self.cat_heads.items()}
        num_out = self.num_head(feats)
        return cat_logits, num_out


def create_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.Resize((IMG_HEIGHT, IMG_WIDTH)),
                T.RandomHorizontalFlip(),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02
                        )
                    ],
                    p=0.7,
                ),
                T.RandAugment(num_ops=2, magnitude=7),
                T.RandomAffine(
                    degrees=20, translate=(0.15, 0.15), shear=12, scale=(0.85, 1.15)
                ),
                T.ToTensor(),
                T.RandomErasing(
                    p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((IMG_HEIGHT, IMG_WIDTH)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def load_and_prepare_dataframe(
    json_path: str,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], Dict[str, Dict[int, float]]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Filter rows with missing or unreadable images
    valid_idx = []
    for i, p in enumerate(tqdm(df["PhotoPath"], desc="Verifying images")):
        try:
            Image.open(p).convert("RGB").close()
            valid_idx.append(i)
        except Exception:
            pass
    df = df.iloc[valid_idx].reset_index(drop=True)

    # Encoders for categorical
    encoders: Dict[str, LabelEncoder] = {}
    class_weights: Dict[str, Dict[int, float]] = {}
    for name in CATEGORICAL_OUTPUT_NAMES:
        df[name] = df[name].fillna("Unknown").astype(str)
        le = LabelEncoder()
        encoded = le.fit_transform(df[name])
        encoders[name] = le
        classes = np.unique(encoded)
        weights = compute_class_weight("balanced", classes=classes, y=encoded)
        class_weights[name] = {int(c): float(w) for c, w in zip(classes, weights)}

    # Numerical fill with median
    for name in NUMERICAL_OUTPUT_NAMES:
        df[name] = pd.to_numeric(df[name], errors="coerce")
        med = df[name].median()
        df[name] = df[name].fillna(med)

    return df, encoders, class_weights


def detect_severely_imbalanced(
    df: pd.DataFrame, encoders: Dict[str, LabelEncoder], threshold: float = 5.0
) -> set:
    severe = set()
    for name in CATEGORICAL_OUTPUT_NAMES:
        labels = encoders[name].transform(df[name].astype(str).tolist())
        unique, counts = np.unique(labels, return_counts=True)
        if counts.min(initial=1) == 0:
            continue
        ratio = float(counts.max() / max(1, counts.min()))
        if ratio >= threshold:
            severe.add(name)
    return severe


def build_weighted_sampler(
    df: pd.DataFrame, encoders: Dict[str, LabelEncoder], head: str = "FilmSimulation"
) -> WeightedRandomSampler:
    labels = encoders[head].transform(df[head].astype(str).tolist())
    unique, counts = np.unique(labels, return_counts=True)
    freq = {int(u): int(c) for u, c in zip(unique, counts)}
    weights = [1.0 / max(1, freq[int(lbl)]) for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_model(class_counts: Dict[str, int], num_numeric: int) -> nn.Module:
    model = MultiHeadMobileNet(class_counts, num_numeric)
    return model


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_counts: Dict[str, int],
    severely_imbalanced: set,
    class_weights: Dict[str, Dict[int, float]],
    epochs: int = 50,
    lr: float = 1e-4,
    freeze_epochs: int = 5,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    model.to(DEVICE)
    # Initially freeze backbone
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Cosine schedule with warmup
    warmup_epochs = max(1, epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # cosine from warmup to epochs
        progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # Losses with class weights
    ce_losses: Dict[str, nn.Module] = {}
    for name, num_classes in class_counts.items():
        # Build weight tensor aligned by class index
        weights = class_weights.get(name, {})
        w_t = torch.ones(num_classes, device=DEVICE)
        for idx, w in weights.items():
            if 0 <= int(idx) < num_classes:
                w_t[int(idx)] = float(w)
        if name in severely_imbalanced:
            ce_losses[name] = FocalLoss(weight=w_t)
        else:
            ce_losses[name] = nn.CrossEntropyLoss(weight=w_t)
    huber = nn.SmoothL1Loss()

    history = {"train_loss": [], "val_loss": []}

    def run_epoch(loader: DataLoader, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        n = 0
        # metrics
        correct: Dict[str, int] = {k: 0 for k in class_counts.keys()}
        total: Dict[str, int] = {k: 0 for k in class_counts.keys()}
        mae_sum = 0.0
        mae_count = 0
        with torch.set_grad_enabled(train_mode):
            for images, cat_targets, num_targets in loader:
                images = images.to(device=DEVICE, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                num_targets = num_targets.to(device=DEVICE, non_blocking=True)
                # Move already-collated CPU tensors to device without re-wrapping to avoid warnings
                cat_targets_t = {
                    k: v.detach().to(device=DEVICE, non_blocking=True)
                    for k, v in cat_targets.items()
                }

                if train_mode:
                    optimizer.zero_grad()
                with torch.amp.autocast(
                    enabled=torch.cuda.is_available(), device_type="cuda"
                ):
                    cat_logits, num_out = model(images)
                    loss = 0.0
                    for name, logits in cat_logits.items():
                        loss += ce_losses[name](logits, cat_targets_t[name])
                    loss += huber(num_out, num_targets)

                if train_mode:
                    scaler.scale(loss).backward()
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                bs = images.size(0)
                total_loss += loss.item() * bs
                n += bs
                # metrics update
                with torch.no_grad():
                    for name, logits in cat_logits.items():
                        pred = torch.argmax(logits, dim=1)
                        correct[name] += int((pred == cat_targets_t[name]).sum().item())
                        total[name] += bs
                    mae_sum += F.l1_loss(num_out, num_targets, reduction="sum").item()
                    mae_count += num_targets.numel()
        avg_loss = total_loss / max(n, 1)
        # compute metrics
        metrics = {
            "acc": {k: (correct[k] / max(1, total[k])) for k in correct},
            "mae": (mae_sum / max(1, mae_count)),
        }
        return avg_loss, metrics

    best_val = float("inf")
    best_state = None
    patience, patience_counter = 12, 0
    for epoch in range(epochs):
        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        train_loss, train_metrics = run_epoch(train_loader, True)
        val_loss, val_metrics = run_epoch(val_loader, False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        # step scheduler per epoch
        scheduler.step()
        # Logging concise metrics
        head_accs = " ".join(
            [f"{k[:6]}:{val_metrics['acc'][k] * 100:.1f}%" for k in class_counts.keys()]
        )
        print(
            f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_metrics['mae']:.4f} - {head_accs}"
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_with_cv(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    class_weights: Dict[str, Dict[int, float]],
):
    # Prepare stratify labels from FilmSimulation
    film_labels = encoders["FilmSimulation"].transform(
        df["FilmSimulation"].astype(str).tolist()
    )

    # Class counts
    class_counts = {
        name: len(encoders[name].classes_) for name in CATEGORICAL_OUTPUT_NAMES
    }
    num_numeric = len(NUMERICAL_OUTPUT_NAMES)

    # K-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    transforms_train = create_transforms(train=True)
    transforms_val = create_transforms(train=False)

    fold_histories = []
    fold_models = []

    indices = np.arange(len(df))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(indices, film_labels)):
        print(f"\n=== FOLD {fold + 1}/3 ===")
        train_subset = df.iloc[tr_idx].reset_index(drop=True)
        val_subset = df.iloc[va_idx].reset_index(drop=True)

        train_ds = RecipesDataset(train_subset, encoders, transforms_train)
        val_ds = RecipesDataset(val_subset, encoders, transforms_val)
        # Build sampler on FilmSimulation to balance batches
        sampler = build_weighted_sampler(train_subset, encoders, head="FilmSimulation")
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_TRAIN,
            shuffle=False,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_VAL,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )

        model = build_model(class_counts, num_numeric)
        # Detect severely imbalanced heads on the training subset
        severely_imbalanced = detect_severely_imbalanced(
            train_subset, encoders, threshold=10.0
        )
        model, history = train_one_fold(
            model,
            train_loader,
            val_loader,
            class_counts,
            severely_imbalanced,
            class_weights,
            epochs=150,
            lr=1e-4,
            freeze_epochs=5,
        )

        fold_histories.append(history)
        fold_models.append(model)

    # Select best model by min val loss
    best_fold = int(np.argmin([min(h["val_loss"]) for h in fold_histories]))
    best_model = fold_models[best_fold]
    return best_model, fold_histories[best_fold], class_counts, num_numeric


def main():
    if not os.path.exists(JSON_DATA_FILE):
        print(f"Error: Data file not found at '{JSON_DATA_FILE}'")
        return

    print("--- Loading and preprocessing data ---")
    df, encoders, class_weights = load_and_prepare_dataframe(JSON_DATA_FILE)

    print("--- Starting cross-validation training ---")
    best_model, history, class_counts, num_numeric = train_with_cv(
        df, encoders, class_weights
    )

    print("\n=== FINAL RESULTS ===")
    best_epoch = int(np.argmin(history["val_loss"]))
    best_val_loss = float(history["val_loss"][best_epoch])
    best_train_loss = float(history["train_loss"][best_epoch])
    overfit_ratio = best_val_loss / max(best_train_loss, 1e-8)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training loss at best epoch: {best_train_loss:.4f}")
    print(f"Overfitting ratio: {overfit_ratio:.2f}")

    print(f"Saving best model to {SAVED_MODEL_PATH}")
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "class_counts": class_counts,
            "num_numeric": num_numeric,
        },
        SAVED_MODEL_PATH,
    )

    tools = {
        **encoders,
        **{f"{k}_class_weights": v for k, v in class_weights.items()},
        **{name: FIXED_NUM_RANGES[name] for name in NUMERICAL_OUTPUT_NAMES},
        "class_counts": class_counts,
        "num_numeric": num_numeric,
    }
    with open(TOOLS_PATH, "wb") as f:
        pickle.dump(tools, f)
    print(f"Tools saved to {TOOLS_PATH}")


if __name__ == "__main__":
    main()
