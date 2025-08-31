# torch_predict.py
#
# PyTorch prediction script equivalent to TensorFlow `predict.py`.
# Loads the trained model from torch_train.py and decodes predictions.

import os
import pickle
from typing import Dict
import argparse

from PIL import Image
import numpy as np
import torch

from torch_train import (
    CATEGORICAL_OUTPUTS,
    FilmNet,
    NUMERICAL_OUTPUTS,
    get_transforms,
)

MODEL_PATH = "model/model.pt"
TOOLS_PATH = "model/tools.pkl"
DEFAULT_IMAGE = "test_image.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_gpu():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"--- Found {n} CUDA GPU(s) ---")
        for i in range(n):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("--- No GPU found. Using CPU ---")


def get_eval_transform():
    # Use the same validation/eval transforms as training script
    return get_transforms(is_train=False)


def preprocess_image(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    tensor = get_eval_transform()(img)
    return tensor.unsqueeze(0)  # add batch dim


def decode_predictions(
    cat_logits: Dict[str, torch.Tensor], num_out: torch.Tensor, tools: Dict
):
    results = {}

    # tools structure from torch_train.py:
    # { 'encoders': {task: {label_str -> idx}}, 'class_counts': {...}, 'num_numeric': int, 'num_ranges': {...} }
    encoders = tools.get("encoders", {})
    num_ranges = tools.get("num_ranges", {})

    # Decode categorical: argmax then map idx -> original label via inverse of encoder dict
    for name in CATEGORICAL_OUTPUTS:
        logits = cat_logits[name].detach().cpu().numpy()
        idx = int(np.argmax(logits, axis=1)[0])
        enc_map = encoders.get(name, {})  # label_str -> idx
        inv_map = {v: k for k, v in enc_map.items()}
        label = inv_map.get(idx, str(idx))
        results[name] = label

    # Decode numerical: inverse of normalization using provided ranges
    numerical = num_out.detach().cpu().numpy()[0]
    for i, name in enumerate(NUMERICAL_OUTPUTS):
        mn, mx = num_ranges.get(name, (0.0, 1.0))
        val = float(numerical[i]) * (mx - mn) + mn
        results[name] = int(round(val))

    return results


def main():
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Predict film settings for an image")
    parser.add_argument("--image", "-i", type=str, default=DEFAULT_IMAGE, help="Path to image file")
    args = parser.parse_args()
    image_path = args.image

    check_gpu()

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOOLS_PATH):
        print(
            "\nError: Model or tools file not found. Run torch_train.py first."
        )
        return
    if not os.path.exists(image_path):
        print(
            f"\nError: Test image not found at '{image_path}'. Pass with --image."
        )
        return

    print("\n--- Loading model and tools ---")
    # Load tools first (for fallback metadata)
    with open(TOOLS_PATH, "rb") as f:
        tools = pickle.load(f)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    # Support both formats: {'state_dict': ... , metadata...} or raw state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Derive metadata if not present
    if isinstance(ckpt, dict) and "class_counts" in ckpt:
        class_counts = ckpt["class_counts"]
    else:
        class_counts = tools.get(
            "class_counts", {name: 1 for name in CATEGORICAL_OUTPUTS}
        )

    if isinstance(ckpt, dict) and "num_numeric" in ckpt:
        num_numeric = ckpt["num_numeric"]
    else:
        num_numeric = tools.get("num_numeric", len(NUMERICAL_OUTPUTS))

    model = FilmNet(class_counts, num_numeric)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"--- Predicting for image: {image_path} ---")
    x = preprocess_image(image_path).to(DEVICE)
    with torch.no_grad():
        cat_logits, num_out = model(x)
        # Wrap cat logits into tensors with batch dim preserved
        cat_logits = {k: v for k, v in cat_logits.items()}
        results = decode_predictions(cat_logits, num_out, tools)

    print("\n--- Predicted Settings ---")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
