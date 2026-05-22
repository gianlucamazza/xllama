#!/usr/bin/env python3
"""Merge model.onnx.data into model.onnx (make the model self-contained).

ORT Runtime 1.24.4 calls weakly_canonical() in ValidateExternalDataPath()
when loading a model that has external data files. On Xbox AppContainer,
MSVC STL's weakly_canonical walks path segments from root and fails at
Q:/Users/UserMgr0 (inaccessible intermediate segment). Embedding all
weights in model.onnx prevents that code path from being reached.

Usage: python scripts/merge_onnx_external_data.py <model_dir>
"""

import sys
import os

import onnx
from onnx.external_data_helper import load_external_data_for_model


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_dir>", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    onnx_path = os.path.join(model_dir, "model.onnx")
    data_path = os.path.join(model_dir, "model.onnx.data")

    if not os.path.exists(onnx_path):
        print(f"ERROR: {onnx_path} not found", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"No external data file found at {data_path}, nothing to merge.")
        return

    print(f"Loading {onnx_path} (external data from {model_dir})...")
    model = onnx.load(onnx_path, load_external_data=False)
    load_external_data_for_model(model, model_dir)

    print("Saving self-contained model.onnx...")
    onnx.save_model(model, onnx_path, save_as_external_data=False)

    os.remove(data_path)
    size_mb = os.path.getsize(onnx_path) // 1024 // 1024
    print(f"Done. model.onnx embedded size: {size_mb} MB")


if __name__ == "__main__":
    main()
