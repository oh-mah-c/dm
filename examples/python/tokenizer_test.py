#!/usr/bin/env python3
import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bindings/python'))

import dm

def test_tokenizer():
    print("=== Testing Tokenizer Bindings ===")
    corpus_path = "test_corpus.txt"
    model_path = "test_bpe_model"

    # Write a simple corpus
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("This is a simple tokenizer test corpus for BPE and unigram.\n")
        f.write("We want to test if Python bindings can train and run tokenizer models.\n")

    try:
        # Create BPE tokenizer
        print("Creating BPE tokenizer...")
        tokenizer = dm.Tokenizer("bpe")

        # Train BPE tokenizer
        print("Training BPE tokenizer...")
        tokenizer.train(corpus_path, vocab_size=100, output_path=model_path)
        print(f"Training complete. Vocab size: {tokenizer.vocab_size}")

        # Encode text
        test_text = "The quick brown fox"
        print(f"Encoding text: '{test_text}'")
        ids = tokenizer.encode(test_text)
        print(f"Encoded IDs: {ids}")

        # Decode IDs
        decoded_text = tokenizer.decode(ids)
        print(f"Decoded text: '{decoded_text}'")

        # Check token text
        for token_id in ids[:5]:
            print(f"Token ID {token_id} text: '{tokenizer.token_text(token_id)}'")

    finally:
        # Clean up
        if os.path.exists(corpus_path):
            os.remove(corpus_path)
        if os.path.exists(model_path):
            os.remove(model_path)
        # BPE training might generate other files like model_path + '.voc' etc. Let's see
        for f in os.listdir('.'):
            if f.startswith(model_path):
                os.remove(f)

if __name__ == "__main__":
    test_tokenizer()
