# torch_predict.py
#
# PyTorch prediction script equivalent to TensorFlow `predict.py`.
# Loads the trained model from torch_train.py and decodes predictions.

import os
import pickle
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from torch_train import MultiHeadMobileNet, CATEGORICAL_OUTPUT_NAMES, NUMERICAL_OUTPUT_NAMES, IMG_HEIGHT, IMG_WIDTH

MODEL_PATH = 'fujifilm_recipe_model_improved.pt'
TOOLS_PATH = 'fujifilm_recipe_tools_improved.pkl'
IMAGE_TO_TEST = 'test_image.jpg'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_gpu():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"--- Found {n} CUDA GPU(s) ---")
        for i in range(n):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("--- No GPU found. Using CPU ---")


def get_eval_transform():
    return T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_image(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    tensor = get_eval_transform()(img)
    return tensor.unsqueeze(0)  # add batch dim


def decode_predictions(cat_logits: Dict[str, torch.Tensor], num_out: torch.Tensor, tools: Dict):
    results = {}

    # Decode categorical: argmax then inverse_transform via LabelEncoder
    for name in CATEGORICAL_OUTPUT_NAMES:
        logits = cat_logits[name].detach().cpu().numpy()
        idx = int(np.argmax(logits, axis=1)[0])
        encoder = tools[name]
        label = encoder.inverse_transform([idx])[0]
        results[name] = label

    # Decode numerical: inverse of fixed scaling
    numerical = num_out.detach().cpu().numpy()[0]
    for i, name in enumerate(NUMERICAL_OUTPUT_NAMES):
        min_v, max_v = tools[name]
        val = float(numerical[i]) * (max_v - min_v) + min_v
        results[name] = int(round(val))

    return results


def main():
    check_gpu()

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOOLS_PATH):
        print("\nError: Model or tools file not found. Run torch_train.py first.")
        return
    if not os.path.exists(IMAGE_TO_TEST):
        print(f"\nError: Test image not found at '{IMAGE_TO_TEST}'. Update IMAGE_TO_TEST.")
        return

    print("\n--- Loading model and tools ---")
    # Load tools first (for fallback metadata)
    with open(TOOLS_PATH, 'rb') as f:
        tools = pickle.load(f)

    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    # Support both formats: {'state_dict': ... , metadata...} or raw state_dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Derive metadata if not present
    if isinstance(ckpt, dict) and 'class_counts' in ckpt:
        class_counts = ckpt['class_counts']
    else:
        class_counts = {name: len(tools[name].classes_) for name in CATEGORICAL_OUTPUT_NAMES}

    if isinstance(ckpt, dict) and 'num_numeric' in ckpt:
        num_numeric = ckpt['num_numeric']
    else:
        num_numeric = len(NUMERICAL_OUTPUT_NAMES)

    model = MultiHeadMobileNet(class_counts, num_numeric)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"--- Predicting for image: {IMAGE_TO_TEST} ---")
    x = preprocess_image(IMAGE_TO_TEST).to(DEVICE)
    with torch.no_grad():
        cat_logits, num_out = model(x)
        # Wrap cat logits into tensors with batch dim preserved
        cat_logits = {k: v for k, v in cat_logits.items()}
        results = decode_predictions(cat_logits, num_out, tools)

    print("\n--- Predicted Settings ---")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("--------------------------\n")


if __name__ == '__main__':
    main()
