#!/usr/bin/env python3
"""
Convert PyTorch FilmNet model to ONNX format for web deployment.
Exports model with static input shape and saves metadata for decoding.
"""

import os
import json
import pickle
import torch
import torch.nn as nn
from torch_train import FilmNet, IMG_SIZE, CATEGORICAL_OUTPUTS, NUMERICAL_OUTPUTS

MODEL_PATH = "model/model.pt"
TOOLS_PATH = "model/tools.pkl"
ONNX_PATH = "model/filmnet.onnx"
METADATA_PATH = "model/metadata.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_to_onnx():
    """Export PyTorch model to ONNX format with metadata."""
    
    print("Loading PyTorch model...")
    
    # Load tools and model
    with open(TOOLS_PATH, "rb") as f:
        tools = pickle.load(f)
    
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        class_counts = ckpt.get("class_counts", tools.get("class_counts", {}))
        num_numeric = ckpt.get("num_numeric", tools.get("num_numeric", len(NUMERICAL_OUTPUTS)))
    else:
        state_dict = ckpt
        class_counts = tools.get("class_counts", {})
        num_numeric = tools.get("num_numeric", len(NUMERICAL_OUTPUTS))
    
    # Create model
    base_model = FilmNet(class_counts, num_numeric)
    base_model.load_state_dict(state_dict)
    base_model.eval()

    # Wrapper ensures deterministic two-output signature for ONNX
    class ExportWrapper(nn.Module):
        def __init__(self, model: FilmNet, cat_order):
            super().__init__()
            self.model = model
            self.cat_order = list(cat_order)

        def forward(self, x):
            cat_dict, num = self.model(x)
            # Concatenate categorical logits in a fixed order
            cats = [cat_dict[name] for name in self.cat_order]
            cat_concat = torch.cat(cats, dim=1)
            return cat_concat, num

    model = ExportWrapper(base_model, CATEGORICAL_OUTPUTS)
    
    # Create dummy input - batch size 1, 3 channels, IMG_SIZE x IMG_SIZE
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    print(f"Exporting to ONNX: {ONNX_PATH}")
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        input_names=['image'],
        output_names=['categorical_outputs', 'numerical_outputs'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'categorical_outputs': {0: 'batch_size'},
            'numerical_outputs': {0: 'batch_size'}
        }
    )
    
    # Prepare metadata for web app
    encoders = tools.get("encoders", {})
    num_ranges = tools.get("num_ranges", {})
    
    # Create inverse encoders for decoding
    inverse_encoders = {}
    for task, enc_map in encoders.items():
        inverse_encoders[task] = {v: k for k, v in enc_map.items()}
    
    metadata = {
        "model_info": {
            "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
            "image_size": IMG_SIZE,
            "categorical_outputs": CATEGORICAL_OUTPUTS,
            "numerical_outputs": NUMERICAL_OUTPUTS
        },
        "encoders": {
            "forward": encoders,
            "inverse": inverse_encoders
        },
        "class_counts": class_counts,
        "num_ranges": num_ranges,
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    
    # Save metadata as JSON
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model exported to: {ONNX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Categorical tasks: {len(CATEGORICAL_OUTPUTS)}")
    print(f"Numerical tasks: {len(NUMERICAL_OUTPUTS)}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOOLS_PATH):
        print("Error: Model or tools file not found. Run torch_train.py first.")
        exit(1)
    
    export_to_onnx()
    print("ONNX export completed successfully!")
