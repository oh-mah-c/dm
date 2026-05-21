"""
VOLT: Vocabulary Learning via Optimal Transport for Neural Machine Translation
ACL 2021 - Xu, Zhou, Gan, Zheng, Li (ByteDance AI Lab)
https://aclanthology.org/2021.acl-long.571

Implements Algorithm 1 (VOLT) exactly as described in the paper:

1. Generate BPE token candidates L ranked by frequency
2. For each size S[t] in incremental sequence S:
   a. T = L[:S[t]]  (top S[t] candidates)
   b. Build kernel K[|T|x|C|]: K[i,j] = 1/len(T[i]) if char_j in T[i] else 0
   c. Sinkhorn: u = P(T)/(K@v), v = P(C)/(K^T@u), P_opt = diag(u)@K@diag(v)
   d. vocab = {T[i]: row_sum(P_opt[i]) >= threshold * P(T[i])}
   e. H_v = -1/l_v * sum_{i in vocab} P(i)*log P(i)
3. Select vocab with maximum MUV = -(H(t)-H(t-1))/(S[t]-S[t-1])

Usage:
    python volt.py -i corpus.txt --max-merges 10000 -o vocab.json [--stats]
    python volt.py -i corpus.txt --bpe-model bpe_codes.txt -o vocab.json [--stats]
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# BPE Training  (Sennrich 2016, word-level with </w> end-of-word marker)
# ---------------------------------------------------------------------------

def read_word_freq(paths, max_lines=0):
    counts = Counter()
    sources = paths if paths else [None]
    for p in sources:
        fh = open(p, encoding="utf-8", errors="replace") if p else sys.stdin
        for n, line in enumerate(fh):
            if max_lines and n >= max_lines:
                break
            for word in line.strip().split():
                counts[word] += 1
        if p:
            fh.close()
    return counts


def _get_pairs(sym_vocab):
    """Count bigram frequencies in symbol vocabulary."""
    pairs = Counter()
    for symbols, freq in sym_vocab.items():
        syms = symbols.split()
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i + 1])] += freq
    return pairs


_MERGE_CACHE = {}


def _merge_vocab(best, sym_vocab):
    """Apply one BPE merge to symbol vocabulary."""
    out = {}
    bigram = re.escape(" ".join(best))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    merged = "".join(best)
    for word, freq in sym_vocab.items():
        out[pattern.sub(merged, word)] = freq
    return out


def learn_bpe(word_freq, max_merges):
    """
    Train BPE, return (merge_rules, final_sym_vocab).
    merge_rules: list of (left, right) string tuples
    final_sym_vocab: {symbol_string: count}
    """
    sym_vocab = {}
    for word, freq in word_freq.items():
        sym = " ".join(list(word)) + " </w>"
        sym_vocab[sym] = freq

    merges = []
    for _ in range(max_merges):
        pairs = _get_pairs(sym_vocab)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        if pairs[best] < 2:
            break
        sym_vocab = _merge_vocab(best, sym_vocab)
        merges.append(best)

    return merges, sym_vocab


def _encode_word(word, merge_rules):
    """Encode one word into BPE tokens using merge rules (greedy left-to-right)."""
    symbols = list(word) + ["</w>"]
    for left, right in merge_rules:
        i = 0
        new_syms = []
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                new_syms.append(left + right)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        symbols = new_syms
    return symbols


def compute_token_freqs(word_freq, merge_rules):
    """Tokenize corpus with BPE and count token frequencies."""
    token_counts = Counter()
    for word, freq in word_freq.items():
        tokens = _encode_word(word, merge_rules)
        for t in tokens:
            token_counts[t] += freq
    return token_counts


# ---------------------------------------------------------------------------
# Character frequency counting
# ---------------------------------------------------------------------------

def compute_char_freqs(word_freq):
    """
    Count Unicode character frequencies from word corpus.
    Each word character (not </w>) is counted word_freq times.
    """
    char_counts = Counter()
    for word, freq in word_freq.items():
        for ch in word:
            char_counts[ch] += freq
    return char_counts


# ---------------------------------------------------------------------------
# VOLT core: kernel, Sinkhorn, vocab extraction
# ---------------------------------------------------------------------------

def build_kernel(tokens, chars, char_to_idx):
    """
    Build kernel matrix K of shape (|T|, |C|).
    K[i,j] = 1/len(token_i)  if char_j appears in token_i (excluding </w>)
            = 0               otherwise.
    len(token_i) = number of Unicode codepoints in token_i excluding </w>.
    """
    n_tok = len(tokens)
    n_char = len(chars)
    K = np.zeros((n_tok, n_char), dtype=np.float64)

    for i, tok in enumerate(tokens):
        clean = tok.replace("</w>", "")
        if not clean:
            continue
        tok_len = len(clean)  # Unicode codepoint count
        for ch in set(clean):
            if ch in char_to_idx:
                K[i, char_to_idx[ch]] = 1.0 / tok_len

    return K


def sinkhorn(p_token, p_char, K, max_iter=1000, tol=1e-9):
    """
    Balanced Sinkhorn iterations (Algorithm 1, VOLT paper):
      u = P(T) / (K @ v)
      v = P(C) / (K^T @ u)
      P_opt = u[:,None] * K * v[None,:]

    Parameters
    ----------
    p_token : ndarray (|T|,)  token probability distribution (sums to 1)
    p_char  : ndarray (|C|,)  char probability distribution (sums to 1)
    K       : ndarray (|T|,|C|)  kernel matrix
    Returns P_opt (|T|,|C|)
    """
    EPS = 1e-300
    u = np.ones(len(p_token), dtype=np.float64)
    v = np.ones(len(p_char), dtype=np.float64)

    for _ in range(max_iter):
        u_prev = u.copy()
        v_prev = v.copy()

        Kv = K @ v                     # (|T|,)
        u = p_token / np.maximum(Kv, EPS)

        KTu = K.T @ u                  # (|C|,)
        v = p_char / np.maximum(KTu, EPS)

        if np.max(np.abs(u - u_prev)) < tol and np.max(np.abs(v - v_prev)) < tol:
            break

    return u[:, None] * K * v[None, :]   # (|T|, |C|)


def extract_vocab(P_opt, tokens, p_token, threshold=1e-3):
    """
    Extract vocabulary from optimal transport matrix.
    Keep token i iff row_sum(P_opt[i]) >= threshold * p_token[i].
    Paper: "We remove tokens with distributed chars less than 0.001 token frequencies."
    """
    row_sums = P_opt.sum(axis=1)
    return [tokens[i] for i in range(len(tokens))
            if row_sums[i] >= threshold * p_token[i]]


def compute_entropy(vocab, token_freqs):
    """
    Corpus entropy with vocabulary v (Eq. 2 from paper):
      H_v = -1/l_v * sum_{i in v} P(i) * log P(i)
    where P(i) = Token(i) / sum_{j in v} Token(j)
    and   l_v = mean len of tokens in v (Unicode chars, excl. </w>)
    """
    if not vocab:
        return 0.0

    total_freq = sum(token_freqs.get(t, 0) for t in vocab)
    if total_freq == 0:
        return 0.0

    total_len = sum(len(t.replace("</w>", "")) for t in vocab)
    l_v = total_len / len(vocab)
    if l_v <= 0:
        return 0.0

    H = 0.0
    for t in vocab:
        freq = token_freqs.get(t, 0)
        if freq > 0:
            p = freq / total_freq
            H -= p * math.log(p)

    return H / l_v


# ---------------------------------------------------------------------------
# VOLT Algorithm 1
# ---------------------------------------------------------------------------

def volt(tokens_ranked, token_freqs, char_freqs, S_sequence,
         threshold=1e-3, sinkhorn_iters=1000, sinkhorn_tol=1e-9,
         verbose=True):
    """
    Algorithm 1 from the VOLT paper.

    Parameters
    ----------
    tokens_ranked : list[str]
        All token candidates, sorted by frequency descending (BPE output).
    token_freqs   : Counter  {token: count}
    char_freqs    : Counter  {char: count}
    S_sequence    : list[int]
        Incremental size sequence, e.g. [1000, 2000, ..., 10000].
    threshold     : float
        Minimum row-sum fraction to keep a token (default 1e-3).
    Returns list of (size, entropy, vocab) tuples.
    """
    chars = sorted(char_freqs.keys())
    char_to_idx = {c: i for i, c in enumerate(chars)}
    total_chars = sum(char_freqs.values()) or 1
    p_char = np.array([char_freqs.get(c, 0) / total_chars for c in chars],
                      dtype=np.float64)

    total_tokens = sum(token_freqs.values()) or 1
    results = []

    for size in S_sequence:
        T = tokens_ranked[:size]
        if not T:
            continue

        # Unnormalized token probabilities, normalized over current T
        raw = np.array([token_freqs.get(t, 0) for t in T], dtype=np.float64)
        p_sum = raw.sum()
        p_token = (raw / p_sum) if p_sum > 0 else np.full(len(T), 1.0 / len(T))

        K = build_kernel(T, chars, char_to_idx)
        if K.sum() == 0:
            continue

        P_opt = sinkhorn(p_token, p_char, K,
                         max_iter=sinkhorn_iters, tol=sinkhorn_tol)

        vocab = extract_vocab(P_opt, T, p_token, threshold)
        if not vocab:
            continue

        H = compute_entropy(vocab, token_freqs)
        results.append((size, H, vocab))

        if verbose:
            print(f"  t={size:6d}: |vocab|={len(vocab):5d}  H={H:.6f}",
                  file=sys.stderr)

    return results


def select_optimal_vocab(results):
    """
    Eq. 3 from paper: maximize MUV = -(H(t) - H(t-1)) / (S[t] - S[t-1]).
    Returns (best_vocab, best_muv, best_size).
    """
    if not results:
        return [], float("-inf"), 0
    if len(results) == 1:
        return results[0][2], float("-inf"), results[0][0]

    best_muv = float("-inf")
    best_vocab = results[-1][2]
    best_size = results[-1][0]

    for i in range(1, len(results)):
        s_prev, H_prev, _ = results[i - 1]
        s_curr, H_curr, vocab = results[i]
        step = s_curr - s_prev
        if step == 0:
            continue
        muv = -(H_curr - H_prev) / step
        if muv > best_muv:
            best_muv = muv
            best_vocab = vocab
            best_size = s_curr

    return best_vocab, best_muv, best_size


# ---------------------------------------------------------------------------
# VOLT Encoder: greedy BPE-style encoding with the found vocabulary
# ---------------------------------------------------------------------------

def encode_text(text, vocab_set, unk="<unk>"):
    """
    Greedy BPE-style encoding with VOLT vocabulary.
    1. Split into chars + </w> at word boundaries.
    2. Iteratively merge consecutive tokens if merged form is in vocab.
    3. OOV chars replaced with <unk>.
    """
    result = []
    for word in text.split():
        # Character-level initial split
        syms = list(word) + ["</w>"]
        changed = True
        while changed:
            changed = False
            new_syms = []
            i = 0
            while i < len(syms):
                if i + 1 < len(syms):
                    merged = syms[i] + syms[i + 1]
                    if merged in vocab_set:
                        new_syms.append(merged)
                        i += 2
                        changed = True
                        continue
                new_syms.append(syms[i])
                i += 1
            syms = new_syms
        # Replace OOV tokens
        for s in syms:
            result.append(s if s in vocab_set else unk)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_S_sequence(max_candidates, bilingual=True):
    """Generate default size sequence S based on corpus size."""
    if bilingual:
        # Paper uses 1K..10K for bilingual
        step = max(1, min(1000, max_candidates // 20))
        upper = min(max_candidates, 10000)
        upper = max(upper, step * 2)
        return list(range(step, upper + 1, step))
    else:
        # Paper uses 40K..160K for multilingual
        step = 10000
        return list(range(40000, 161000, step))


def main():
    ap = argparse.ArgumentParser(
        description="VOLT: Vocabulary Learning via Optimal Transport (ACL 2021)")
    ap.add_argument("-i", "--input", nargs="+", metavar="FILE",
                    help="Training corpus file(s)")
    ap.add_argument("-o", "--output", required=True, metavar="FILE",
                    help="Output vocabulary JSON file")
    ap.add_argument("--max-merges", type=int, default=10000, metavar="N",
                    help="Number of BPE merge operations for candidate generation (default: 10000)")
    ap.add_argument("--bpe-model", metavar="FILE",
                    help="Use pre-trained BPE model file instead of training BPE")
    ap.add_argument("--S-min", type=int, default=0, metavar="N",
                    help="Minimum size in S sequence (0=auto)")
    ap.add_argument("--S-max", type=int, default=0, metavar="N",
                    help="Maximum size in S sequence (0=auto)")
    ap.add_argument("--S-step", type=int, default=0, metavar="N",
                    help="Step size in S sequence (0=auto)")
    ap.add_argument("--threshold", type=float, default=1e-3, metavar="F",
                    help="Transport threshold for vocab extraction (default: 0.001)")
    ap.add_argument("--sinkhorn-iters", type=int, default=1000, metavar="N",
                    help="Maximum Sinkhorn iterations (default: 1000)")
    ap.add_argument("--sinkhorn-tol", type=float, default=1e-9, metavar="F",
                    help="Sinkhorn convergence tolerance (default: 1e-9)")
    ap.add_argument("--max-lines", type=int, default=0, metavar="N",
                    help="Max lines per input file (0=all)")
    ap.add_argument("--multilingual", action="store_true",
                    help="Use multilingual S sequence (40K..160K instead of 1K..10K)")
    ap.add_argument("--encode", metavar="FILE",
                    help="After finding vocab, encode this file and print tokens")
    ap.add_argument("--stats", action="store_true",
                    help="Print JSON stats to stderr")
    args = ap.parse_args()

    if not args.input:
        ap.print_help()
        sys.exit(2)

    # ---- Load corpus ----
    print("Loading corpus...", file=sys.stderr)
    word_freq = read_word_freq(args.input, args.max_lines)
    print(f"  {len(word_freq)} word types, {sum(word_freq.values())} tokens",
          file=sys.stderr)

    # ---- BPE training or loading ----
    if args.bpe_model:
        print(f"Loading BPE model from {args.bpe_model}...", file=sys.stderr)
        merge_rules = []
        with open(args.bpe_model, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    merge_rules.append((parts[0], parts[1]))
    else:
        print(f"Training BPE with {args.max_merges} merges...", file=sys.stderr)
        merge_rules, _ = learn_bpe(word_freq, args.max_merges)
        print(f"  Learned {len(merge_rules)} merge rules", file=sys.stderr)

    # ---- Compute token and char frequencies ----
    print("Computing token frequencies...", file=sys.stderr)
    token_freqs = compute_token_freqs(word_freq, merge_rules)
    print(f"  {len(token_freqs)} distinct tokens", file=sys.stderr)

    char_freqs = compute_char_freqs(word_freq)
    print(f"  {len(char_freqs)} distinct chars", file=sys.stderr)

    # ---- Build token candidate list ranked by frequency ----
    tokens_ranked = [tok for tok, _ in token_freqs.most_common()]

    # ---- Build S sequence ----
    max_cands = len(tokens_ranked)
    if args.S_step > 0 and args.S_min > 0 and args.S_max > 0:
        S = list(range(args.S_min, args.S_max + 1, args.S_step))
    elif args.S_step > 0 and args.S_max > 0:
        step = args.S_step
        S = list(range(step, args.S_max + 1, step))
    else:
        S = make_S_sequence(max_cands, bilingual=not args.multilingual)

    # Clamp S to available candidates
    S = [s for s in S if s <= max_cands]
    if not S:
        S = [max_cands]
    print(f"Size sequence S: {S[0]}..{S[-1]} ({len(S)} steps)", file=sys.stderr)

    # ---- Run VOLT ----
    print("Running VOLT (Sinkhorn OT)...", file=sys.stderr)
    results = volt(
        tokens_ranked, token_freqs, char_freqs, S,
        threshold=args.threshold,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_tol=args.sinkhorn_tol,
        verbose=True,
    )

    if not results:
        print("ERROR: No valid vocabularies found.", file=sys.stderr)
        sys.exit(1)

    # ---- Select optimal vocabulary ----
    best_vocab, best_muv, best_size = select_optimal_vocab(results)

    # ---- Write output ----
    # Clean token representations (strip </w> for output)
    vocab_clean = [t.replace("</w>", "") for t in best_vocab]
    # Remove empty strings from </w>-only tokens
    vocab_clean = [t for t in vocab_clean if t]

    # Compute final entropy and stats
    final_H = compute_entropy(best_vocab, token_freqs)
    vocab_set = set(best_vocab)

    out = {
        "version": "dm-volt-acl2021",
        "algorithm": "vocabulary-learning-optimal-transport",
        "vocab_size": len(best_vocab),
        "muv": best_muv,
        "optimal_S": best_size,
        "entropy": final_H,
        "vocab": vocab_clean,
        "vocab_with_markers": list(best_vocab),
        "all_sizes": [{"size": r[0], "vocab_size": len(r[2]), "entropy": r[1]}
                      for r in results],
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"\nOptimal vocabulary: {len(best_vocab)} tokens at S={best_size}, "
          f"MUV={best_muv:.6f}, H={final_H:.6f}", file=sys.stderr)
    print(f"Written to {args.output}", file=sys.stderr)

    # ---- Optional encoding ----
    if args.encode:
        print(f"\nEncoding {args.encode}...", file=sys.stderr)
        with open(args.encode, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                tokens = encode_text(line.strip(), vocab_set)
                print(" ".join(tokens))

    if args.stats:
        print(json.dumps({
            "vocab_size": len(best_vocab),
            "optimal_S": best_size,
            "muv": best_muv,
            "entropy": final_H,
            "num_chars": len(char_freqs),
            "num_token_candidates": len(tokens_ranked),
        }), file=sys.stderr)


if __name__ == "__main__":
    main()
