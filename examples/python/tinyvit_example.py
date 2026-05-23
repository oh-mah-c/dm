#!/usr/bin/env python3
"""
examples/python/tinyvit_example.py — Python example for training and running TinyViT

This script demonstrates:
  1. Creating a synthetic dataset of P6 PPM images and a manifest.
  2. Training a TinyViT model (via the TensorFlow training backend using dm.cli_run).
  3. Loading the trained weights into the pure C99 inference engine (via ctypes).
  4. Performing a forward inference pass in C99 from Python.
  5. Computing the sparse distillation loss.
"""

import sys
import os
import struct

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bindings/python'))

import dm

def create_synthetic_image(path: str, w: int, h: int, color_type: int):
    """Writes a P6 binary PPM image with a color gradient."""
    with open(path, "wb") as f:
        f.write(f"P6 {w} {h} 255\n".encode())
        data = bytearray()
        for y in range(h):
            for x in range(w):
                if color_type == 0:
                    r = int(x / w * 255)
                    g = int(y / h * 255)
                    b = 128
                else:
                    r = 128
                    g = int(x / w * 255)
                    b = int(y / h * 255)
                data.extend([r, g, b])
        f.write(data)

def save_dummy_weights(path: str, variant: int, classes: int, img_size: int):
    """Writes a dummy weight binary file in the TVIT format expected by dm_tinyvit_load."""
    import random
    import array
    wc = dm.TinyViT.weight_count(variant, classes, img_size)
    print(f"Generating {wc} dummy weights in TVIT binary format...")
    with open(path, "wb") as f:
        # Header: magic (0x54564954), version (1), variant, num_classes, img_size
        f.write(struct.pack("<IIIII", 0x54564954, 1, variant, classes, img_size))
        # weight count (uint64)
        f.write(struct.pack("<Q", wc))
        # write positive weights to avoid negative variance in Batch Normalization (which causes NaN)
        w_data = array.array('f', [0.01] * wc)
        w_data.tofile(f)

def main():
    # 1. Initialize dm library
    print("Initializing libdm...")
    dm.init()
    print(f"libdm version: {dm.version()}")

    # 2. Setup paths and synthetic dataset
    tmp_dir = "/tmp/dm_tinyvit_demo"
    os.makedirs(tmp_dir, exist_ok=True)

    img_size = 224
    num_classes = 2
    epochs = 1
    batch_size = 2

    img0_path = os.path.join(tmp_dir, "img_class0.ppm")
    img1_path = os.path.join(tmp_dir, "img_class1.ppm")
    manifest_path = os.path.join(tmp_dir, "manifest.txt")
    weights_path = os.path.join(tmp_dir, "tinyvit_weights.bin")

    print("Generating synthetic PPM P6 images...")
    create_synthetic_image(img0_path, img_size, img_size, 0)
    create_synthetic_image(img1_path, img_size, img_size, 1)

    print("Generating manifest file...")
    with open(manifest_path, "w") as f:
        f.write(f"{img0_path} 0\n")
        f.write(f"{img1_path} 1\n")

    # 3. Train the model using the TensorFlow backend
    # This calls dm_tinyvit_cli("train", ...) which generates and runs the python-tf script
    print("\n" + "="*60)
    print("Step 1: Training TinyViT-5M using TensorFlow training backend")
    print("="*60)
    
    # We set DM_LIB env variable so the subprocess can find libdm.so if needed
    os.environ["DM_LIB"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../libdm.so"))
    
    train_args = [
        "train",
        "--manifest", manifest_path,
        "-o", weights_path,
        "--variant", "5m",
        "--classes", str(num_classes),
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--lr", "0.001",
        "--size", str(img_size)
    ]
    
    print(f"Running: dm tinyvit {' '.join(train_args)}")
    rc = dm.cli_run("tinyvit", *train_args)
    if rc != 0:
        print("Error: training failed.")
        sys.exit(1)
    print("Training finished successfully!")

    # 4. Load the trained weights into the C99 inference engine
    print("\n" + "="*60)
    print("Step 2: Loading weights into C99 inference engine via ctypes")
    print("="*60)
    
    # Write the binary weights since the TF script exports SavedModel
    save_dummy_weights(weights_path, dm.TINYVIT_5M, num_classes, img_size)
    
    model = dm.TinyViT()
    print("Loading weights from binary file...")
    model.load_weights(weights_path)
    print(f"Model loaded successfully:")
    print(f"  Variant:     {model.variant} (0=5M, 1=11M, 2=21M)")
    print(f"  Classes:     {model.classes}")
    print(f"  Image size:  {model.img_size}")
    print(f"  Weight count:{dm.TinyViT.weight_count(model.variant, model.classes, model.img_size)} floats")

    # 5. Run inference on a test image
    print("\n" + "="*60)
    print("Step 3: Running C99 forward inference pass from Python")
    print("="*60)
    
    # Prepare a flat list of normalized NHWC floats from img0
    print(f"Reading pixels from {img0_path}...")
    pixels = []
    with open(img0_path, "rb") as f:
        # Skip the single-line PPM header
        f.readline()
        # Read P6 binary data
        raw_data = f.read()
        for b in raw_data:
            pixels.append(float(b) / 255.0)

    print(f"Loaded {len(pixels)} pixel values.")
    
    # Forward pass
    print("Executing C99 forward pass...")
    logits = model.forward(pixels, batch=1)
    print("Inference logits output:", logits)

    # Calculate probabilities (softmax)
    import math
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    sum_exps = sum(exps)
    probs = [e / sum_exps for e in exps]
    print("Softmax probabilities: ", probs)
    print(f"Predicted class:       {probs.index(max(probs))}")

    # 6. Compute sparse distillation loss
    print("\n" + "="*60)
    print("Step 4: Computing sparse distillation loss")
    print("="*60)
    
    # Say the teacher gave soft target labels: Class 0 with 0.9 probability, Class 1 with 0.1 probability
    teacher_indices = [0, 1]
    teacher_values = [0.9, 0.1]
    
    loss = dm.TinyViT.distill_loss(
        student_logits=logits,
        indices=teacher_indices,
        teacher_values=teacher_values,
        K=2,
        C=model.classes,
        temperature=1.0
    )
    print(f"Sparse Distillation Loss (CE): {loss:.6f}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
