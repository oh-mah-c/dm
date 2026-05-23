#!/usr/bin/env python3
"""
examples/python/bindings_example.py — Verification script for MobileNetTiny and BERT Python bindings.
"""

import sys
import os
import struct

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bindings/python'))

import dm

def verify_mobilenet_tiny():
    print("=== Verifying MobileNetTiny Bindings ===")
    
    # 1. Initialize MobileNetTiny object
    image_size = 224
    classes = 5
    seed = 42
    model = dm.MobileNetTiny(image_size=image_size, classes=classes, seed=seed)
    print(f"Initialized MobileNetTiny: image_size={model.image_size}, classes={model.classes}, seed={model.seed}")
    
    # 2. Run forward pass with randomly initialized weights (head_w=None, head_b=None)
    dummy_input = [0.5] * (3 * image_size * image_size)
    print("Running forward pass with default head weights...")
    logits = model.forward(dummy_input)
    print(f"Logits: {logits}")
    assert len(logits) == classes, f"Expected {classes} logits, got {len(logits)}"
    print("Forward pass successful!")
    
    # 3. Create dummy head weights to test saving and loading
    checkpoint_path = "test_mobilenet_tiny_head.bin"
    feature_dim = 960
    
    print(f"Writing dummy head checkpoint to {checkpoint_path}...")
    magic_bytes = b"DMMNH1\x00\x00"
    w_data = [0.01] * (classes * feature_dim)
    b_data = [0.0] * classes
    
    with open(checkpoint_path, "wb") as f:
        f.write(magic_bytes)
        f.write(struct.pack("<iiii", classes, feature_dim, image_size, seed))
        f.write(struct.pack(f"<{len(w_data)}f", *w_data))
        f.write(struct.pack(f"<{len(b_data)}f", *b_data))
        
    print(f"Loading head weights from {checkpoint_path}...")
    model.load_head(checkpoint_path)
    print(f"Loaded successfully: classes={model.classes}, feature_dim={model._feature_dim}, image_size={model.image_size}")
    assert model.classes == classes
    assert model._feature_dim == feature_dim
    assert model.image_size == image_size
    
    # 4. Run forward pass using the loaded weights
    print("Running forward pass with loaded weights...")
    logits2 = model.forward(dummy_input)
    print(f"Logits (loaded weights): {logits2}")
    
    # 5. Save the weights back using bindings
    saved_checkpoint_path = "test_mobilenet_tiny_head_saved.bin"
    print(f"Saving head weights back to {saved_checkpoint_path}...")
    model.save_head(saved_checkpoint_path)
    
    # Clean up files
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(saved_checkpoint_path):
        os.remove(saved_checkpoint_path)
        
    print("MobileNetTiny bindings verification completed successfully!\n")


def verify_bert():
    print("=== Verifying BERT Bindings ===")
    
    # 1. Initialize BERT parameters
    variant = dm.BERT_BASE
    
    # Use a small configuration to avoid creating massive files
    small_vocab = 100
    small_seq = 16
    small_weight_count = dm.BERT.weight_count(variant, small_vocab, small_seq)
    print(f"Small BERT config weight count (vocab={small_vocab}, seq={small_seq}): {small_weight_count}")
    
    # 2. Create a dummy BERT weights checkpoint file
    checkpoint_path = "test_bert_weights.bin"
    magic = 0x42455254
    ver = 1
    print(f"Writing dummy BERT checkpoint to {checkpoint_path}...")
    dummy_weight_val = 0.01
    
    with open(checkpoint_path, "wb") as f:
        # Header: magic (uint32), ver (uint32), variant (uint32), vocab (uint32), max_len (uint32), wc (uint64)
        f.write(struct.pack("<IIIIIQ", magic, ver, variant, small_vocab, small_seq, small_weight_count))
        # Write dummy weights
        chunk_size = 10000
        written = 0
        while written < small_weight_count:
            to_write = min(chunk_size, small_weight_count - written)
            f.write(struct.pack(f"<{to_write}f", *([dummy_weight_val] * to_write)))
            written += to_write
            
    print(f"Loading BERT weights from {checkpoint_path}...")
    model = dm.BERT(variant=variant, vocab_size=small_vocab, max_seq_len=small_seq)
    model.load_weights(checkpoint_path)
    print(f"Loaded successfully: variant={model.variant}, vocab_size={model.vocab_size}, max_seq_len={model.max_seq_len}")
    assert model.variant == variant
    assert model.vocab_size == small_vocab
    assert model.max_seq_len == small_seq
    
    # 3. Run forward pass
    token_ids = [1, 2, 3, 4]
    segment_ids = [0, 0, 0, 0]
    print(f"Running BERT forward pass with input token_ids={token_ids}...")
    hidden_out, cls_out = model.forward(token_ids, segment_ids)
    
    hidden_dim = 768
    print(f"Success! Output shapes: hidden_out={len(hidden_out)} (expected {len(token_ids) * hidden_dim}), cls_out={len(cls_out)} (expected {hidden_dim})")
    assert len(hidden_out) == len(token_ids) * hidden_dim
    assert len(cls_out) == hidden_dim
    
    # Test with attention mask
    attention_mask = [1, 1, 1, 0]
    print(f"Running BERT forward pass with attention_mask={attention_mask}...")
    hidden_out_m, cls_out_m = model.forward(token_ids, segment_ids, attention_mask=attention_mask)
    assert len(hidden_out_m) == len(token_ids) * hidden_dim
    assert len(cls_out_m) == hidden_dim
    print("BERT forward masked pass successful!")
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    print("BERT bindings verification completed successfully!\n")


if __name__ == "__main__":
    verify_mobilenet_tiny()
    verify_bert()
    print("All bindings tests passed!")
