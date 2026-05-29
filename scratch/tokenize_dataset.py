import sys
import struct

def build_vocab(vocab_file):
    vocab = {}
    # Reserve 0 for <unk>
    vocab['<unk>'] = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if parts:
                # Add 1 because 0 is <unk>
                vocab[parts[0]] = i + 1
    return vocab

def tokenize(input_bpe_file, output_bin_file, vocab):
    tokens_ids = []
    unk_id = 0
    with open(input_bpe_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            for w in words:
                if w.endswith('@@'):
                    token = w[:-2]
                else:
                    token = w + '</w>'
                
                tid = vocab.get(token, unk_id)
                tokens_ids.append(tid)
                
    with open(output_bin_file, 'wb') as f:
        for tid in tokens_ids:
            f.write(struct.pack('<i', tid))  # write as 32-bit little-endian int
            
    print(f"Tokenized {len(tokens_ids)} tokens. Saved to {output_bin_file}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python tokenize_dataset.py <vocab_file> <input_bpe_file> <output_bin_file>")
        sys.exit(1)
    
    vocab = build_vocab(sys.argv[1])
    print(f"Loaded {len(vocab)} tokens into vocabulary.")
    tokenize(sys.argv[2], sys.argv[3], vocab)
