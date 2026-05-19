# High-Information Entropy Pattern Mining for Embedded Text Streams

## Research landscape and analytical gap

Classical pattern mining is built on support. Apriori exploited the anti-monotonicity of support to prune the search space level by level, FP-Growth replaced candidate generation with a compressed FP-tree traversal, and PrefixSpan did the same for sequences through prefix projection. Those ideas remain foundational because support obeys a clean downward-closure law: every super-pattern can only be supported by a subset of the transactions supporting its subpatterns. The difficulty for text is that natural language is heavily Zipfian, so support is systematically inflated by boilerplate, stopwords, discourse markers, and workflow tokens such as “please”, “system”, and “user”. In other words, the mathematically convenient measure is often semantically misaligned with the target notion of informativeness. citeturn3search0turn6search0turn33search0turn6search18turn17search5

Information retrieval solved the single-term version of this problem decades ago by downweighting ubiquitous terms through inverse document frequency. In the Stanford IR text, the idf of a term is explicitly defined as \( \log \frac{N}{df_t} \), so rare terms receive larger weights than common ones. Church and Gale linked idf to deviations from a Poisson model, while keyword extraction work by Tomokiyo and Hurst argued that useful phrases require both phraseness and informativeness, and Yang et al. later showed that entropy differences can discriminate relevant words from randomly distributed ones. These strands all point in the same direction: in text, frequency alone is too crude, and information content matters. citeturn5search2turn20search0turn19search13turn20search1

High-utility itemset mining addressed a different version of the same structural problem. Because raw utility is not anti-monotone, Two-Phase introduced transaction-weighted utilization as an upper bound, HUI-Miner introduced utility-lists, FHM tightened pruning with estimated utility co-occurrence, and EFIM pushed the state of the art forward using tighter upper bounds and transaction merging. Closed variants such as CHUI-Miner and EFIM-Closed then reduced redundancy while preserving exactness. The lesson from HUIM is important here: if the target score is not support, exact mining is still possible, but only if one engineers safe upper bounds and data structures around the objective. citeturn27search0turn16search0turn16search1turn16search2turn16search8turn16search10turn16search14turn27search17

Information-theoretic pattern mining does exist, but its focus is different from the embedded text-stream setting. Tatti’s maximum-entropy significance of itemsets asks whether an itemset is more surprising than expected under a background model. De Bie’s information-theoretic framework formalized exploratory data mining as information exchange relative to background knowledge, and maximum-entropy models were later used to quantify subjective interestingness. MDL-based approaches such as Krimp compress databases by selecting a compact descriptive set of patterns, and Galbrun’s MDL survey shows how broad that line of work has become. These are conceptually adjacent to HIEPM, but they do not solve the problem of exact, high-throughput mining of rare-but-informative token patterns directly from integerized text streams in a zero-dependency C99 engine. citeturn36search16turn22search4turn15search0turn15search2turn21search2turn21search4turn22search17

Recent NLP work strengthens the case for an information-theoretic mining objective. In 2025, surprisal slope was studied as a feature for multi-word expression detection, and in 2026 surprisal was again used as a measure of word-level information content in research on information status and uniform information density. In software engineering, a 2025 study analyzed token and structural entropy to identify unusual source-code changes. These works show that token-level information content remains an active and productive lens in 2025–2026, yet they stop at scoring or supervised detection; they do not provide an exact exhaustive miner for high-information token combinations. citeturn32search1turn32search3turn32search6turn18search0

This is the central gap. If one defines the score of a textual pattern directly from joint surprisal or support-weighted surprisal, the objective is not anti-monotone: extending a pattern decreases support but can sharply increase information content, so some supersets improve and others collapse. Classical support-based SPM can therefore prune away semantically valuable low-support patterns too early, while standard HUIM bounds do not apply out of the box when the score is tied to corpus-derived information weights, window compactness, and token-order constraints. The situation is exactly the kind of non-anti-monotone search space that prior work on constrained sequence mining and utility mining identified as the main source of combinatorial blow-up. citeturn17search0turn28search0turn16search0turn16search3turn27search17

The proposal below therefore treats HIEPM as an entropy-derived utility mining problem for text. The key design move is to separate two things that are often conflated: the *information-theoretic meaning* of a pattern and the *search-time objective* used to mine it exactly. The former can be expressed with self-information and surprisal; the latter must be decomposable enough to admit tight upper bounds.

## Formal problem formulation

Strictly speaking, entropy and self-information are not the same quantity. Shannon entropy is an expectation over a distribution, whereas the information content of a realized event is its self-information, or surprisal, \( -\log p \). Modern NLP continues to use surprisal as the operational measure of information content for individual words and contexts. Accordingly, HIEPM is best understood as mining *entropy-derived high self-information patterns*; the shorter name “High-Information Entropy Pattern Mining” is retained as a label for the paradigm. citeturn0search0turn32search3turn32search6turn32search5

Let the tokenizer emit an integer token stream
\[
Z = \langle z_1, z_2, \dots, z_N\rangle,\qquad z_j \in I,
\]
where \(I\) is the token universe produced by the flat hash tokenizer in `dm.exe`. From this stream, define a windowization operator \( \mathcal W_{L,\Delta} \) with window length \(L\) and stride \( \Delta \). It produces \(n\) window-bounded transactions:
\[
D = \{T_1,\dots,T_n\}.
\]
In itemset mode, each transaction is the deduplicated set of tokens in the window:
\[
T_q = \mathrm{uniq}\big(\langle z_{a_q},\dots,z_{b_q}\rangle\big)\subseteq I.
\]
In sequence mode, each transaction is the ordered subsequence itself:
\[
S_q = \langle z_{a_q},\dots,z_{b_q}\rangle.
\]
The notation \(X \preceq T\) will mean either \(X\subseteq T\) for itemsets or “\(X\) occurs as a subsequence of \(T\)” for sequences.

Define window support and document probability as
\[
\sigma(X)=|\Gamma(X)|,\qquad \Gamma(X)=\{T\in D: X\preceq T\},
\]
and for single tokens,
\[
df(x)=\sigma(\{x\}),\qquad
p(x)=\frac{df(x)+\alpha}{n+\alpha |I|},
\]
with smoothing parameter \( \alpha>0 \). The self-information weight of token \(x\) is then
\[
w(x)=I(x)=-\log_2 p(x).
\]
This is the direct text-mining analogue of idf-style distinctiveness weighting, but expressed in Shannon units. citeturn5search2turn20search0

An attractive but computationally difficult “ideal” pattern score would be the joint surprisal
\[
J(X)= -\log_2 \Pr(X)
\approx -\log_2 \frac{\sigma(X)+\alpha}{n+\alpha},
\]
possibly aggregated across supporting windows as \( \sigma(X)\,J(X) \). The problem is that this score is neither downward-closed nor upward-closed. A superset can gain enough information content to dominate its parent despite lower support, but another superset can also lose too much support and fall below it. That destroys the direct pruning logic of Apriori, GSP, and support-driven prefix growth. This is the same structural difficulty that earlier work on non-anti-monotone constraints and utility mining had to circumvent with upper bounds rather than exact monotone objectives. citeturn28search0turn17search0turn16search0turn27search0

For exact mining, HIEPM therefore uses a decomposable entropy-derived utility surrogate. I propose the per-transaction semantic information utility
\[
u(X,T)
=
\mathbf 1[X\preceq T]\;
\kappa(X,T)\;
\sum_{x\in X} w(x),
\]
where \( \kappa(X,T)\in(0,1] \) is a compactness or cohesion factor. In itemset mode, set \( \kappa(X,T)=1 \). In sequence mode, let
\[
\mathrm{span}_T(X)
=
\min_{\pi_1<\cdots<\pi_m,\; T[\pi_j]=x_j}
(\pi_m-\pi_1+1),
\]
for \(X=\langle x_1,\dots,x_m\rangle\), and define
\[
\kappa(X,T)=
\exp\!\big(-\gamma(\mathrm{span}_T(X)-|X|)\big),
\qquad \gamma\ge 0.
\]
This rewards patterns that are both informative and locally compact inside the window.

The global HIEPM utility is then
\[
U(X)=\sum_{T\in D}u(X,T).
\]
In itemset mode, this collapses to
\[
U(X)=\sigma(X)\sum_{x\in X}w(x),
\]
which is exactly a support-weighted information-density score. In sequence mode, the score remains transaction-decomposable but is discounted by span compactness.

The target output of HIEPM is
\[
\mathcal H_{\theta,\sigma_0}
=
\left\{
X\neq\emptyset:
U(X)\ge \theta,\;
\sigma(X)\ge \sigma_0
\right\},
\]
where \( \theta \) is a strict information-utility threshold and \( \sigma_0 \) is a small persistence floor, typically 2 or 3 in noisy corpora. The role of \( \sigma_0 \) is not to recover support mining; it is to suppress one-off artifacts, typos, and pathological singletons. That safeguard is important because purely information-driven association measures are known to over-reward extremely rare events, much as PMI is biased toward infrequent co-occurrences. citeturn25search17turn25search7

For interpretability, each output pattern can also be annotated post hoc with its empirical joint surprisal \(J(X)\), but the thresholding and exact search operate on \(U(X)\). This is the crucial modeling compromise: HIEPM remains faithful to Shannon-style information weighting while using an objective that can still be mined exactly.

## HIEP-Miner architecture

HIEP-Miner is designed for the kind of low-level engine described in the prompt: a memory-mapped reader, an integer tokenizer, and a flat-array mining core. On Linux, `mmap()` maps file contents directly into the process address space, the file descriptor can be closed after the mapping is established, and the region remains valid until `munmap()`; for large sequential scans, `POSIX_FADV_SEQUENTIAL` increases readahead, `MADV_SEQUENTIAL` expresses page-order locality, and `POSIX_FADV_DONTNEED` or `MADV_DONTNEED` can be used to reduce cache pollution in streaming experiments. This is a natural fit for an embedded text miner that wants to avoid redundant copies and per-string heap allocation. citeturn29view0turn29view2turn31view0turn31view3turn30view0

The physical layout should be flat and pointer-minimal. A practical HIEP-Miner core can be built from four contiguous arenas:

| Array | Suggested type | Meaning |
|---|---:|---|
| `item_meta[m]` | struct-of-arrays | `df`, `w`, `TIUB1`, posting offsets |
| `txn_off[n+1]` + `txn_items[nnz]` | `uint32_t` | compact transaction store |
| `occ_tid[]`, `occ_pos[]` | `uint32_t` | vertical postings for sequence mode |
| `ul_tid[]`, `ul_last[]`, `ul_eu[]`, `ul_ru[]` | SoA arena | utility-list workspace |

Here \(m=|I|\), \(n=|D|\), and \(nnz=\sum_q |T_q|\). The singleton vertical postings are built once. Sequence mode stores positions; itemset mode may omit them. The utility-list arena is recycled depth-first with a monotonic allocator, so recursion only moves a frontier pointer forward and rewinds it on return. That yields an exact search without per-node `malloc()` overhead, while keeping the utility-list spirit of HUI-Miner and the bound-tightening philosophy of FHM and EFIM. citeturn16search12turn16search1turn16search2

The first pruning device is a coarse transaction-level upper bound, analogous in spirit to TWU but adapted to information weights. Define the transaction information utility
\[
ITU(T)=\sum_{x\in T} w(x),
\]
and for any prefix pattern \(X\),
\[
TIUB(X)=\sum_{T\in \Gamma(X)} ITU(T).
\]
Then, for every extension \(Y\succeq X\),
\[
U(Y)\le TIUB(X).
\]
The proof is immediate: if \(Y\) occurs in \(T\), then \(X\) also occurs in \(T\), and because \( \kappa(Y,T)\le 1\), the exact utility contributed by \(Y\) inside \(T\) cannot exceed the sum of all token information weights present in \(T\). Summing across the support set of \(X\) yields the bound. This bound is loose, but it is extremely cheap and useful for eliminating hopeless singletons and many two-item prefixes early. It is the HIEPM analogue of TWU-style first-pass pruning. citeturn27search0turn27search8turn27search17

The second pruning device is the core theorem.

**Information-Weighted Remaining Utility Theorem.**  
Fix a global item order \( \prec \), preferably ascending \(TIUB(\{x\})\) with ties broken by descending \(w(x)\). For each supporting transaction \(T\) of a prefix \(X\), let \( \mathrm{tail}_T(X) \) be the set of extension-eligible items after the last item of \(X\) in the search order, or after the last matched position in sequence mode. Define
\[
\mathrm{pwt}(X)=\sum_{x\in X} w(x),
\qquad
\mathrm{ru}(X,T)=\sum_{y\in \mathrm{tail}_T(X)} w(y).
\]
Then
\[
IWRU(X)
=
\sum_{T\in \Gamma(X)}
\left(
\mathrm{pwt}(X)+\mathrm{ru}(X,T)
\right)
\]
is a safe upper bound for every extension \(Y\succeq X\):
\[
U(Y)\le IWRU(X).
\]

The proof is transaction-local. In any supporting transaction \(T\), an extension \(Y\) consists of the current prefix items plus some subset of the extension-eligible tail. Therefore,
\[
u(Y,T)
\le
\sum_{y\in Y} w(y)
=
\mathrm{pwt}(X)+\sum_{z\in Y\setminus X} w(z)
\le
\mathrm{pwt}(X)+\mathrm{ru}(X,T).
\]
Summing over \( \Gamma(Y)\subseteq \Gamma(X) \) gives the bound.

This immediately yields a monotone pruning envelope:
\[
Y\succeq X \implies IWRU(Y)\le IWRU(X),
\]
because both the supporting transaction set and the admissible extension tails only shrink along any depth-first branch. This is the exact property needed to control a non-anti-monotone information objective without sacrificing completeness. It plays the role that remaining utility, EUCP, local utility, and subtree utility play in HUIM, but with Shannon-derived weights and optional sequence compactness. citeturn16search0turn16search1turn16search2turn27search8

The utility-list entry for a pattern \(X\) in sequence mode is
\[
e = \langle tid,\ last,\ eu,\ ru\rangle,
\]
where `tid` is the transaction id, `last` is the last matched position, \(eu=u(X,T)\) is the exact utility contribution in that transaction, and \(ru=\mathrm{ru}(X,T)\) is the remaining information upper bound from that point onward. In itemset mode, `last` can be replaced by an offset into the sorted transaction slice or omitted entirely if transactions are intersected as ordered sets. A candidate extension is produced by a two-finger merge on `tid`, with an additional position check in sequence mode. Because both singleton postings and derived utility lists are sorted and contiguous, the join cost is linear in the list sizes seen by the branch.

The search itself is straightforward depth-first branch-and-bound:

```text
HIEP-MINER(D, theta, sigma0):
    build singleton postings and item weights
    compute TIUB for singletons
    discard items with TIUB < theta or df < sigma0
    order remaining items by ascending TIUB, tie by descending w
    build singleton utility lists
    for each singleton x in order:
        if exactU({x}) >= theta: emit {x}
        if IWRU({x}) >= theta:
            DFS(prefix={x}, ULx, suffix_items_after_x)

DFS(prefix X, UL(X), Ext):
    for each item a in Ext:
        UL(Y) = JOIN(UL(X), UL(a))
        if support(Y) < sigma0:
            continue
        if exactU(Y) >= theta:
            emit Y
        if IWRU(Y) >= theta:
            DFS(Y, UL(Y), items_after_a)
```

The correctness argument follows the two inequalities above. **Soundness** is immediate because a pattern is reported only when its exact \(U(X)\) is computed and exceeds \( \theta \). **Completeness** follows because a branch is pruned only when \(IWRU(X)<\theta\); by the theorem, no extension of \(X\) can then reach the threshold. Everything else is explored. This is the exact same proof shape that made one-phase utility-list miners viable, but the bound has been rebuilt so it remains valid under entropy-derived weighting rather than business-profit utility. citeturn16search0turn16search2

The runtime is output-sensitive in the standard utility-list sense:
\[
O\!\left(
\sum_{v\in \mathcal T_{\mathrm{visited}}}
\mathrm{join\_cost}(v)
\right),
\]
where each join cost is linear in the support-list slices processed by that node. The space cost is
\[
O(nnz + \sum_{x\in I'} df(x)),
\]
for retained items \(I'\), plus a recursion-local arena that is reused branch-wise. The worst case remains exponential because exact pattern mining is worst-case exponential, but the dominant practical lever is no longer support monotonicity; it is the tightness of \(IWRU\) under a good item order.

## Experimental validation blueprint

The evaluation must separate two questions. One is **systems efficiency**: can HIEP-Miner scan and search faster, with lower RSS, than generic baselines on integerized transactions? The other is **semantic value**: does it suppress high-frequency noise while recovering rare informative phrases better than support-only miners? Real pattern-mining datasets from SPMF are useful for the first question because they are standard and allow direct comparison across sparse and dense regimes; SPMF explicitly lists Retail, Accidents, and Chess as common evaluation datasets, and prior HUIM studies use the same families to contrast sparse and dense behavior. For the second question, a synthetic Zipfian corpus with planted high-information phrases is essential because it provides known signal and known junk. citeturn23search0turn23search1turn23search12turn23search2

The baselines should be chosen by objective class, not by convenience. Apriori and FP-Growth are the canonical exact support-based itemset miners; PrefixSpan is the canonical support-based sequence miner. HUI-Miner and EFIM are the natural exact utility baselines when each token is assigned utility \(w(x)\) and quantity 1. That last comparison is especially important: if HIEP-Miner cannot beat generic HUIM on the same transformed utility semantics, then the proposed bound and data layout are not doing enough work. citeturn6search0turn33search0turn6search18turn16search0turn16search2

The core metrics should be computed as follows:
\[
\mathrm{Throughput}_{MB/s}
=
\frac{\text{bytes processed}/2^{20}}{\Delta t},
\qquad
\mathrm{Throughput}_{tok/s}
=
\frac{N}{\Delta t}.
\]
For memory, use `ru_maxrss` from `getrusage(RUSAGE_SELF, ...)` as the primary system metric and optionally record `/proc/self/status` `VmHWM` as a cross-check. For time, use `clock_gettime(CLOCK_MONOTONIC, ...)`, since `CLOCK_MONOTONIC` is not affected by discontinuous wall-clock jumps. For file-mapped scans, issue `posix_fadvise(..., POSIX_FADV_SEQUENTIAL)` before mining and `madvise(..., MADV_SEQUENTIAL)` on the mapped region so the benchmark reflects the intended streaming regime. citeturn11view2turn11view3turn10view0turn10view1turn31view0turn31view3turn30view0

For semantic quality on the synthetic corpus, define a planted junk vocabulary \( \mathcal J \) and a planted signal set \( \mathcal S \) of rare multiword patterns. Then measure:
\[
\mathrm{NFE}
=
1-
\frac{
|\{X\in \mathcal O : X\subseteq \mathcal J\}|
}{
|\mathcal O|
},
\]
where \( \mathcal O \) is the output pattern set, and
\[
\mathrm{SignalRecall}
=
\frac{|\mathcal O\cap \mathcal S|}{|\mathcal S|}.
\]
A useful density metric is
\[
\mathrm{IDO}
=
\frac{1}{|\mathcal O|}
\sum_{X\in\mathcal O}
\frac{U(X)}{|X|},
\]
which reports how many information-utility units are emitted per output token on average. If HIEPM is doing what it should, NFE and IDO should increase relative to support-only miners, while SignalRecall should remain high under reasonable thresholds.

The synthetic generator should reflect three empirical facts: token frequencies in natural language are approximately Zipfian, informative phrases require co-occurrence structure rather than isolated rare words, and modern text streams are often multilingual and noisy. The script below uses only the Python standard library for portability.

```python
#!/usr/bin/env python3
import argparse
import bisect
import math
import os
import random
import string

STOP = [
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with",
    "please", "system", "user", "assistant", "is", "are", "was", "were",
    "that", "this", "it", "be", "as", "by", "from", "at", "if", "not"
]

COMMON = [
    "model", "data", "mining", "token", "window", "pattern", "query",
    "search", "memory", "stream", "index", "graph", "document", "topic",
    "vector", "context", "cluster", "feature", "signal", "noise", "entropy",
    "utility", "support", "prefix", "suffix", "kernel", "thread", "batch"
]

RARE_TECH = [
    "zero_copy", "subword_lattice", "homomorphic", "differential", "spectrogram",
    "transformerless", "knowledge_router", "quantization", "adjacency_sketch",
    "microkernel", "orthogonalization", "hypergraph", "surprisal", "retrieval_fusion",
    "ondevice", "token_compactor", "systolic", "delta_encoding"
]

RARE_MULTI = [
    "điện_toán", "ngôn_ngữ", "học_máy", "dữ_liệu", "truy_vấn", "niño", "mañana",
    "señal", "canción", "jalapeño", "über", "straße", "façade", "crème",
    "北京", "上海", "数据", "模型", "語義", "東京", "한국어", "데이터"
]

SIGNALS = [
    ["zero_copy", "token_compactor", "surprisal"],
    ["knowledge_router", "retrieval_fusion", "context"],
    ["quantization", "hypergraph", "entropy"],
    ["dữ_liệu", "học_máy", "ngôn_ngữ"],
    ["北京", "数据", "模型"],
    ["niño", "mañana", "señal"]
]

PUNCT = [",", ".", ";", ":", "!", "?", "...", "—", "(", ")", "[", "]", "\""]

def build_vocab():
    vocab = STOP + COMMON + RARE_TECH + RARE_MULTI
    # Ordered from most common to least common; Zipf weights depend on rank.
    return vocab

def build_cdf(vocab, s=1.07):
    weights = [1.0 / ((i + 1) ** s) for i in range(len(vocab))]
    z = sum(weights)
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / z
        cdf.append(acc)
    return cdf

def sample_zipf(vocab, cdf):
    r = random.random()
    idx = bisect.bisect_left(cdf, r)
    return vocab[idx]

def noisy_token(tok):
    if random.random() < 0.08:
        tok = tok.upper()
    elif random.random() < 0.05:
        tok = tok.capitalize()
    if random.random() < 0.12:
        tok = random.choice(PUNCT) + tok
    if random.random() < 0.12:
        tok = tok + random.choice(PUNCT)
    return tok

def make_line(vocab, cdf, avg_len, signal_prob):
    length = max(5, int(random.expovariate(1.0 / avg_len)))
    out = []
    if random.random() < signal_prob:
        sig = random.choice(SIGNALS)
        insert_at = random.randint(0, max(0, length - len(sig)))
        for i in range(length):
            if insert_at <= i < insert_at + len(sig):
                out.append(sig[i - insert_at])
            else:
                out.append(sample_zipf(vocab, cdf))
    else:
        out = [sample_zipf(vocab, cdf) for _ in range(length)]

    # Add punctuation noise and occasional repeated filler bursts.
    if random.random() < 0.15:
        burst = random.choice(["please", "system", "user", "assistant"])
        for _ in range(random.randint(1, 4)):
            out.insert(random.randint(0, len(out)), burst)

    out = [noisy_token(t) for t in out]
    return " ".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="zipf_hiepm_corpus.txt")
    ap.add_argument("--target-mb", type=int, default=512)
    ap.add_argument("--avg-len", type=float, default=24.0)
    ap.add_argument("--signal-prob", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    vocab = build_vocab()
    cdf = build_cdf(vocab)

    target_bytes = args.target_mb * 1024 * 1024
    written = 0
    line_count = 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        while written < target_bytes:
            line = make_line(vocab, cdf, args.avg_len, args.signal_prob)
            f.write(line + "\n")
            written += len((line + "\n").encode("utf-8"))
            line_count += 1

    print(f"Wrote {args.out}")
    print(f"Bytes: {written}")
    print(f"Lines: {line_count}")
    print("Injected signal patterns:", len(SIGNALS))

if __name__ == "__main__":
    main()
```

This generator deliberately creates a head-heavy stopword region, a middle technical vocabulary, a long-tail rare multilingual tail, and planted high-information patterns. That makes it suitable both for mining evaluation and for stressing the tokenizer’s UTF-8 path. The shape is motivated by the Zipfian structure of word frequencies and by recent interest in surprisal-based phrase indicators. citeturn3search0turn32search1

A minimal benchmark harness in C should follow this order: open file, `posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)`, `mmap` the file, `madvise(addr, len, MADV_SEQUENTIAL)`, `clock_gettime(CLOCK_MONOTONIC, &t0)`, run tokenization plus HIEP-Miner, `clock_gettime(CLOCK_MONOTONIC, &t1)`, `getrusage(RUSAGE_SELF, &ru)`, then parse `/proc/self/status` for `VmHWM` as a secondary memory report, and finally `munmap` the region. For cache-cold runs, either drop caches with privileged system support or iterate over separate files larger than memory; for cache-warm runs, rerun the same mapped file. Report medians over at least 11 repetitions, plus geometric means across threshold sweeps. citeturn29view0turn29view2turn11view2turn10view0turn10view1

A fair protocol should include four ablations: removing compactness \( \kappa \), removing the coarse \(TIUB\) stage, removing \(IWRU\), and replacing information weights \(w(x)\) with uniform weights. The expected pattern is clear. Without \( \kappa \), sequence mode will admit more dispersed, less phrase-like patterns. Without \(TIUB\), startup candidate pressure increases. Without \(IWRU\), the search degenerates toward a much broader utility-pattern enumerator. With uniform weights, the miner collapses back toward support-driven co-occurrence mining and should emit more junk-heavy outputs. Those ablations are what will make the “better than other tokenizers/miners” claim experimentally defensible rather than rhetorical.

## Implications for dm.exe and open questions

If implemented as above, HIEPM gives `dm.exe` a mathematically coherent way to move beyond support while still keeping exact search and systems-level efficiency. Conceptually, it sits between IR-style distinctiveness weighting and HUIM-style exact utility mining: from IR it borrows the idea that rare terms carry more information, and from HUIM it borrows the insight that non-support objectives are mineable if one engineers the right upper bounds. That makes it far more appropriate for text logs, chats, command histories, and technical corpora than plain Apriori, FP-Growth, or PrefixSpan used naively on token ids. citeturn5search2turn19search13turn16search0turn16search2

Two limitations deserve to be stated openly. First, information weights are corpus-relative. If the engine runs in a truly online regime where the corpus distribution drifts continuously, then \(w(x)\) changes over time; exact incremental maintenance under changing weights is harder than static snapshot mining and should be treated as a future problem. Second, any information-theoretic score can overvalue extreme rarity unless it is coupled to recurrence, compactness, or semantic constraints. The use of \( \sigma_0 \), compactness \( \kappa \), and possibly part-of-speech or vocabulary-class filters is therefore not an implementation nuisance; it is a theoretical necessity to keep “informative” from collapsing into “accidental”. Recent surprisal work in NLP and longstanding warnings about rare-event bias in association measures both point in that direction. citeturn32search1turn32search12turn25search17turn25search7

The most promising next extension is a two-layer score: keep HIEP-Miner’s additive \(U(X)\) for exact search, but post-rank exact outputs by a richer semantic criterion such as contextual surprisal under a compact language model, or by divergence from a background model in the spirit of subjective interestingness and maximum-entropy data exploration. That would preserve the pruning guarantees of the current design while moving the notion of “information” from corpus rarity toward contextual semantic specificity. In that sense, HIEPM is best viewed not as the endpoint, but as the first exact embedded miner that makes Shannon-style informativeness operational inside a high-performance text-stream engine. citeturn22search4turn15search0turn18search22