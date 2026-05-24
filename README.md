<div align="center">
  <h1>🚀 C-DataMiner Framework</h1>
  <p><strong>A Highly Optimized, Blazing-Fast Open-Source Data Mining Framework built purely in C</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Language: C](https://img.shields.io/badge/Language-C99-orange.svg)](https://en.wikipedia.org/wiki/C99)
  [![Build: GCC](https://img.shields.io/badge/Build-GCC-success.svg)](#)
</div>

---

## 💡 About The Project

**C-DataMiner** is a state-of-the-art framework designed for extracting hidden patterns, frequent itemsets, and association rules from massive transactional databases. Written entirely in pure C, it prioritizes **execution speed**, **memory efficiency**, and **algorithmic elegance**.

Unlike bloated libraries, C-DataMiner acts as a low-level benchmark environment. It includes implementations of the most prominent data mining algorithms from the last three decades, all unified under a single architecture. Whether dealing with *sparse* (market-basket) or *dense* (bioinformatics/chess) datasets, the framework adaptively performs at peak hardware limits.

### ✨ Key Features
- **Zero-Dependency Core:** Pure C99 implementation using custom bitsets, hash-tables, FP-Trees, and memory pools for maximum speed.
- **Microsecond Benchmarking:** Built-in high-resolution hardware timers tracking Wall Time, Kernel Mode Time, User Mode Time, and Peak RAM usage (VmHWM).
- **Unified Algorithm Interface:** Dynamically register and switch algorithms without changing the internal testing pipelines.
- **Vertical & Horizontal Formats:** Includes Array-based techniques, Bitwise Vertical TID-sets, and classic candidate generation.

---

## 🛠 Supported Algorithms

### Frequent Itemset Mining (FIM)
1. **AIS** - The pioneer algorithm for mining association rules.
2. **Apriori** - Level-wise search using candidate generation and a hash tree.
3. **AprioriTid** - Variant of Apriori that uses candidate ID sets to avoid full database scans.
4. **AprioriHybrid** - Combines Apriori and AprioriTid for optimal performance.
5. **Eclat** - Vertical data format search with depth-first traversal.
6. **FP-Growth** - Tree-based, projection-driven mining without candidate generation.
7. **TreeProjection** - Matrix-based counting via depth-first transactional tree projection.
8. **FIN** - Fast mining of Frequent Itemsets using Nodesets.
9. **FIN+** (dFIN) - Efficient frequent itemset mining using DiffNodesets.
10. **negFIN** - An efficient algorithm for fast mining frequent itemsets using NegNodesets.
11. **PrePost+** - High-performance N-lists-based mining using Children–Parent Equivalence pruning.
12. **DIC** - Dynamic Itemset Counting algorithm for reducing the number of passes over the data.
13. **LTM** - Linear Table Miner for frequent itemset mining using a linear table structure and bitwise pruning.
14. **SaM** - Split and Merge algorithm using a horizontal transaction representation for efficient mining.
15. **MSApriori** - Level-wise mining with multiple minimum supports to solve the rare item problem.
16. **DFI-Growth** - Deriving frequent itemsets from lossless condensed representations using pattern growth.
17. **KRIMP** - MDL-based pattern selection algorithm that finds the set of itemsets that compress the database best.
18. **SLIM** - Directly mines descriptive patterns using MDL by iteratively merging itemsets in the code table.
19. **Two-Phase** - Efficiently mines high utility itemsets using transaction-weighted utilization and two-phase pruning.
20. **FHM** - Faster High-Utility Itemset Mining using Estimated Utility Co-occurrence Pruning (EUCP).
21. **VHUQI** - Vertical mining of High Utility Quantitative Itemsets using utility-lists and K-support bound pruning.
22. **FHUQI-Miner** - Fast High Utility Quantitative Itemset Miner using TQCS, EQCPS, and RQCPS pruning.
23. **TKQ** - Top-K Quantitative High Utility Itemset Miner using RIU, CUD, and update-queue threshold raising.
23. **EFIM** - Highly efficient HUIM algorithm using database projection and transaction merging.
23. **HUI-Miner** - Mines high utility itemsets without candidate generation using a vertical utility-list structure.
24. **UP-Growth** - Uses a compact UP-Tree structure and pruning strategies (DGU, DGN) to mine PHUIs efficiently.
25. **IHUP** - Incremental High Utility Pattern mining using IHUP-Trees (L-Tree, TF-Tree, TWU-Tree).
26. **HUIM-SU** - Simplified Utility-list based High-Utility itemset mining using repeated TWU pruning and extension utility bounds.
27. **ULB-Miner** - Uses a high-performance utility-list buffer (UTLBuf) to reduce memory fragmentation and join time.
28. **UFH** - A hybrid framework combining UP-Growth+ (tree-based) and FHM (utility-list based) for sparse/dense datasets.
29. **HUCI-Miner** - Mines high utility closed itemsets, high utility generators, and HGB non-redundant rules using the utility-confidence framework.
30. **UP-Hist Growth** - Extends UP-Growth with quantity histograms at each node to provide tighter utility estimates.
31. **R-Miner** - Uses a residual utility-based concept with Residue Maps and Master Map for highly efficient join operations.
32. **SUM** (Scented Utility Miner) - An incremental HUIM algorithm using a reinduction strategy and dynamic threshold setting.
33. **CLH-Miner** - Mines Cross-Level High Utility Itemsets using a taxonomy and tax-utility-lists to discover patterns across different abstraction levels.
34. **MLHUI-Miner** - An efficient utility-list based algorithm for mining high-utility itemsets at multiple abstraction levels (generalized HUIM).
35. **FCHM** (Fast Correlated High-Utility itemset Miner) - Discovers correlated high-utility itemsets using the bond measure (HAIS 2016 version).
36. **FHN** (Fast High-Utility itemset miner with Negative unit profits) - Efficiently mines HUIs in databases where item unit profits may be positive or negative using an extended utility-list structure.
37. **HUIM-HC** - Mining High-Utility Itemsets with Hill Climbing using PEV pruning and diversity maintenance (Nawaz et al. 2021).
38. **HUIM-SA** - Mining High-Utility Itemsets with Simulated Annealing using temperature-dependent acceptance probability (Nawaz et al. 2021).
39. **HUIM-BPSO-tree** - Mines high-utility itemsets using Binary Particle Swarm Optimization and an OR/NOR-tree to avoid invalid combinations (Lin et al. 2016).
40. **Bio-HUIF-GA** - GA-based high-utility itemset mining using a Diverse Optimal Value Framework (Song & Huang 2018).
41. **Bio-HUIF-PSO** - PSO-based high-utility itemset mining using a Diverse Optimal Value Framework (Song & Huang 2018).
42. **Bio-HUIF-BA** - Bat Algorithm-based high-utility itemset mining using a Diverse Optimal Value Framework (Song & Huang 2018).
43. **HUIM-AFSA** - Artificial Fish Swarm Algorithm for mining high-utility itemsets with PEV pruning (Song et al. 2021).
44. **HUIM-ABC** - Artificial Bee Colony algorithm for high-utility itemset mining with bitmap-based utility calculation (Song & Huang 2018).
45. **THUI** - Mining top-k high utility itemsets with effective threshold raising strategies using LIU structure (Krishnamoorthy 2019).
46. **TKU-Miner** - Mines exact top-k high-utility itemsets without a user-specified minimum utility threshold, following the TKU top-k threshold-raising framework (Wu et al. 2012).
47. **THUE** - Mines top-k high-utility episodes from complex event sequences using RIU/RTU/RUC threshold raising, optimized EWU pruning, and simultaneous/serial episode spanning (Wan et al. 2021).
48. **TIPN-HOUI-Miner** - Mines top-k high-on-shelf-utility itemsets with positive/negative utilities using TIPN/TIUL tables, interval occurrence bitmaps, TWUGC pruning, RLC pruning, TIO pruning, and RPRU/RRU threshold raising (Chang et al. 2025).
49. **TKU-CE** - Cross-Entropy Method for Mining Top-K High Utility Itemsets (Song et al. 2021).
50. **TKU-CE+** - Improved Cross-Entropy Method for Top-K HUIM with CUV pruning and smoothing mutation (Song et al. 2021).
51. **DPHIM** - Dynamic parallel high-utility itemset mining using utility-list subtasks and pthread workers (Kimura et al. 2026).

49. **HAUI-Miner** - Mining High Average-Utility Itemsets using AU-lists and transaction-maximum utility downward closure (Lin et al. 2016).
49. **EHAUPM** - Efficient High Average-Utility Pattern Mining with Tighter Upper Bounds and co-occurrence matrix (Lin et al. 2017).
50. **HAUIM-GMU** - Mining High Average-Utility Itemsets based on Generalized Maximal Utility and critical support count (Song et al. 2021).
51. **NAM-HEP** - Adaptive High Occupancy Itemset Mining using a hierarchical set-enumeration tree and median thresholds (Tran et al. 2025).
52. **MEMU** - Mining High Average-Utility Patterns with Multiple Thresholds using compact AU-lists (Lin et al. 2018).
53. **MHEINU** - Mining High-Efficiency Itemsets with Negative Utilities using ELNU lists, `uben`, and `ubeni` pruning (Yildirim 2025).
54. **Closed-FHUIM-Kinana** - Closed frequent high-utility itemset mining with OSR, OWL, and MSU pruning (Sulanjari & Fatichah 2026).

### Sequential Utility Pattern Mining
55. **USpan** - High utility sequential pattern mining using projected databases and sequence-weighted utility pruning.
56. **PrefixSpan** - Mining sequential patterns efficiently using prefix-projected pattern growth, finding frequent itemsets across sequences using an optimized pseudo-projection technique (Pei et al., 2001).
57. **SPADE** - Fast discovery of sequential patterns using vertical id-lists and combinatorial equivalence classes, bypassing database scans via efficient lattice intersections (Zaki, 2001).
58. **HUPSPM** - High utility-probability sequential pattern mining for uncertain sequence utility databases.
59. **HUP-Miner** - Mines high average utility nonoverlapping patterns from sequence utility databases using nonoverlapping SPC-style support, HUBP upper-bound pruning, and pattern join candidate generation (Geng et al. 2026).

### Top-K Closed Sequential Pattern Mining
60. **KCloTreeMiner** - Mines top-K closed sequential patterns over SPMF sequence datasets using SP-Tree-style projection, max-support candidate ordering, closed coverage, pattern absorption, and generic/group/redundancy-aware modes (Rizvee et al. 2025).

### Frequent Closed Itemset Mining (FCIM)
14. **A-Close** - Uses frequent itemset generators to derive closed itemsets.
15. **CLOSET** - FP-tree based mining using frequent patterns.
16. **CLOSET+** - Extends CLOSET with hybrid tree traversals and pseudo-projection.
17. **FPclose** - Extends FP-growth with Array-based checking for closed itemsets.
19. **CHARM** - Vertical data format with diffsets for fast closed pattern extraction.
20. **DCI_CLOSED** - Highly optimized vertical bitset miner with order-preserving generators.
21. **LCM** - Linear time Closed itemset Miner using Prefix Preserving Closure Extension and Occurrence Deliver.
22. **NAFCP** - An N-list-based Algorithm for mining Frequent Closed Patterns using the PPC-tree.
23. **FCFIA** - An Efficient Algorithm for Frequent Closed Itemsets Mining using PEP pruning and projection strategy.
24. **CARPENTER** - A row-wise enumeration algorithm for finding frequent closed patterns in long datasets.
25. **DBV-Miner** - A Dynamic Bit-Vector approach for fast mining frequent closed itemsets using subsumption and pruning.
26. **dEFME** - Depth-First Minimal Pattern Mining for enumerating free itemsets (generators) with polynomial delay.
27. **Talky-G** - Vertical mining of frequent generators using reverse pre-order traversal and subsumption checks.
28. **TOUCH** - Combined vertical mining of frequent closures (Charm) and generators (Talky-G).
29. **PASCAL** - Level-wise frequent pattern mining using pattern counting inference to reduce database scans.
30. **ZART** - Multifunctional mining of frequent itemsets, closed itemsets, and generators based on Pascal.
31. **Close** - Efficient mining of association rules using closed itemset lattices.
32. **OPUS Miner** - Efficient discovery of self-sufficient itemsets using branch-and-bound.

### Data Stream Mining
31. **estDec** - Finding recent frequent itemsets adaptively over online data streams using a decay mechanism.
32. **CloStream** - Incremental maintenance of frequent closed itemsets over data streams using intersection and inverted indexing.
33. **CFI-Stream** - Mining Closed Frequent Itemsets in Data Streams using bit-sequences.
34. **FHMDS** - Fast Top-K High Utility Itemset Mining from Data Streams under the sliding window model (Dawar et al. 2017).

### Rare Itemset Mining (RIM)
34. **Apriori-Rare** - Level-wise mining of minimal rare itemsets (mRIs) using a modified Apriori traversal.
35. **Apriori-Inverse** - Vertical mining of perfectly sporadic itemsets (low support) based on the inverted downward closure.
36. **CORI** - Key correlation mining by simultaneous pushing of monotone (rarity) and anti-monotone (bond correlation) constraints.
37. **RP-Tree** - Tree-based mining of rare-item itemsets using dual support thresholds.

### Maximal Frequent Itemset Mining (MFIM)
38. **Max-Miner** - Breadth/Depth search applying look-ahead superset pruning for long patterns.
39. **GenMax** - Backtrack search using Progressive Focusing and vertical bitsets for maximal itemsets.
40. **FPmax** - FP-growth extension with MFI-Tree for ultra-fast maximal itemset discovery.
41. **MAFIA** - MAximal Frequent Itemset Algorithm using Vertical Bitmaps, PEP, and HUTMFI.

### High Occupancy Itemset Mining (HOIM)
42. **HEP** - High Efficient algorithm for mining high occupancy itemsets using UBO pruning.
41. **HEP** - High Efficient algorithm for mining high occupancy itemsets using UBO pruning.
42. **FHOI** - Fast High Occupancy Itemset Mining using Equivalence Class and Early Pruning.
43. **DFHOI** - Depth First Search for High Occupancy Itemset Mining using Equivalence Class and Early Pruning.
44. **TKHOIM** - Top-k High Occupancy Itemset Miner using dynamic minO updating and LUBO strategy.
45. **MFHOI / Strong MFHOI-Miner** - Mines Frequent High-Occupancy Itemsets, Weak MFHOI, and Strong MFHOI with exact dominance filtering over FHOI candidates.
46. **MHOUI-Miner** - Mines High-Occupancy Utility Itemsets and exact weak/strong MHOUI patterns from utility datasets.
47. **CLOE-HOI** - Mines support-closed high-occupancy itemsets using candidate-free vertical bitset DFS, closure jumping, backward closure pruning, and Occupancy Envelope pruning.
48. **AURA-HOI** - Mines auditable high-occupancy support-class representatives using exact average occupancy, closure-canonical vertical bitsets, support-class ledgering, and residual occupancy-envelope pruning.
46. **FFI-Miner** - Fast Algorithm for mining fuzzy frequent itemsets from quantitative databases.
47. **UBMFFP-Tree** - Upper-bound Multiple Fuzzy Frequent Pattern Tree for mining multiple fuzzy frequent itemsets.

### Uncertain Data Mining
46. **U-Apriori** - Mining frequent itemsets from existential uncertain data using the expected support measure.

### Concise Frequent Itemset Representations
58. **RegularMine** - Mines frequent regular itemsets as an interpretable concise representation of frequent itemsets using closed classes, free sets, covering, and merging (Ruggieri 2010).

### Prompt Pattern Mining
59. **HUPP-Miner** - Mines High-Utility Prompt Patterns from prompt JSONL logs using semantic concept transactions, candidate optimizer banks, preserve masks, PTWO pruning, and AAUB alignment pruning.

### High-Information Entropy Pattern Mining
60. **HIEP-Miner** - Mines rare but recurring high-information token itemsets or sequences from integerized text streams using Shannon self-information, window-bounded transactions, TIUB pruning, and IWRU utility-list branch-and-bound.

### Closed High-Utility Occupancy Mining
61. **CHUO-Miner** - Mines Closed High-Utility Occupancy Itemsets from positive utility datasets using COU-lists, RUU utility pruning, OUB occupancy-envelope pruning, backward closure pruning, forward closure jumping, and CHUO-signatures.

### Classification Rule Mining
62. **PSO Classifier** - Particle Swarm based classification-rule discovery using continuous CPSO, TP/TN rule quality, rule pruning, covering, default rules, rule-set cleaning, and tenfold cross-validation.

### Subword Tokenization Algorithms
63. **FARO Tokenizer** - Frequency-Aware Robust Out-of-vocabulary Subword Tokenizer that dynamically merges high-frequency subword structures while managing vocabulary size.
64. **HUST-Tokenize** - High-Utility Subword Tokenization engine that integrates downstream pattern utility metrics into the subword tokenization and merging process.
65. **Sennrich BPE Subword** - Implements ACL 2016 byte-pair encoding subword learning and application for open-vocabulary neural translation preprocessing, including joint BPE over multiple corpora.
66. **BPE-Dropout** - Implements ACL 2020 stochastic BPE subword regularization, reusing conventional BPE merge tables while randomly dropping merge occurrences during training-time segmentation.
67. **Kudo Unigram Subword Regularization** - Implements ACL 2018 unigram language-model subword segmentation with EM training, vocabulary pruning, Viterbi encoding, n-best segmentation, and FFBS stochastic sampling.
68. **SentencePiece Lite** - Implements EMNLP 2018 SentencePiece-style raw-sentence training, lossless whitespace escaping with U+2581, NFKC/custom normalization, BPE/unigram models, ids, and self-contained model files.
69. **Tokenizer Lab** - Implements tokenizer-domain-adaptation experiments for BPE code tokenizers: regex pre-tokenization variants, vocabulary-size sweeps, NSL, bytes-per-token, Renyi entropy, and inference/memory vocabulary trade-off estimates.
70. **Grapheme Pair Encoding** - Implements COLING 2025 GPE for egalitarian complex-script tokenization using grapheme clusters as BPE atomic units, plus CR/TP pre-tokenization analysis.
71. **Parity-aware BPE** - Implements arXiv 2025 Parity-aware Byte-Pair Encoding, selecting each merge from the currently worst-compressed language while applying the merge globally, with classical, hybrid, and moving-window variants plus CR/Gini/vocabulary-use metrics.
72. **Fast WordPiece / LinMaxMatch** - Implements EMNLP 2021 linear-time WordPiece tokenization using a vocabulary trie with failure links and failure pops, plus BERT-style end-to-end punctuation/whitespace tokenization.
73. **Reps Maximal-Munch Scanner** - Implements Reps' linear-time maximal-munch tokenization using DFA configurations, stack backtracking, and `failed_previously` tabulation restricted to the states required by the paper's optimized `Tab` set.
74. **VOLT** - Implements ACL 2021 Vocabulary Learning via Optimal Transport, finding the optimal vocabulary size without trial NMT training by maximizing Marginal Utility of Vocabularization (MUV) over a BPE candidate pool using balanced Sinkhorn iterations and a character-coverage kernel.

### Self-Contained Data Generators
74. **LAGA** - Self-contained Layout-Aware Generative Architecture for knowledge-preserving synthetic transaction, utility, sequence, text, and tabular data generation using embedded encoding, support/co-occurrence/transition layouts, miner-style evaluation, privacy rejection, and closed-loop repair.

### Vision Models (Deep Learning)
75. **ResNet-18** — Pure-C99 implementation of He et al.'s Deep Residual Network (ResNet) using BasicBlocks (two 3×3 convs + identity/projection shortcut), BatchNorm, ReLU, MaxPool, GlobalAvgPool, and a linear classifier head. Exposed via `libdm.so` FFI and a Python numpy binding.
76. **MobileNetV4-Tiny** — Lightweight depthwise-separable CNN for edge inference, adapted from Howard et al.'s MobileNet family. Full forward-pass implemented in C99 with NHWC layout.
77. **TinyViT** — Compact Vision Transformer backbone with patch embedding, multi-head self-attention, and MLP blocks implemented in C99, suited for resource-constrained classification.
78. **ViT (Vision Transformer)** — Full C99 implementation of Dosovitskiy et al.'s ViT (ICLR 2021). Supports five variants: Tiny (D=192, L=5), Small (D=384, L=6), Base (D=768, L=12), Large (D=1024, L=24), Huge (D=1280, L=32). Architecture: patch embedding → [CLS] + 1D position embeddings → L×(LN→MHSA→residual, LN→MLP-GELU→residual) → LN → linear head. Exposed via `libdm.so` FFI (`dm_op_vit_forward`) and CLI (`./build/bin/dm.exe vit bench --variant small`).
79. **Swin Transformer C Inference Port** — This project contains an unofficial C inference port of Swin Transformer. The implementation is architecturally based on the official Microsoft Swin-Transformer repository and is designed for integration with this repository's TensorFlow-core-style DM_Block runtime. It is not affiliated with or endorsed by Microsoft or the original authors. Current correctness work targets TensorFlow/Keras fixture parity first. Generate deterministic TensorFlow-layout weights with `python3 tools/export_swin_tf_weights.py` and run `./build/bin/dm.exe swin bench --weights weights/swin_tiny_patch4_window7_224/metadata.json`.

### Generative Models
79. **VAE (Variational Autoencoder)** — Full C99 pure implementation of the Auto-Encoding Variational Bayes (Kingma & Welling) model. Includes a symmetric MLP encoder/decoder architecture and analytically derived backpropagation for expected reconstruction loss and KL divergence, combined with an implementation of the **Adam Optimizer** (Kingma & Ba) with state/momentum tracking. Accessible via the `libdm.so` python bindings `dm.VAE()`.
80. **GAN (Generative Adversarial Network)** — Full C99 faithful implementation of the original Generative Adversarial Nets (Goodfellow et al., 2014) model. It precisely implements the mathematical specification including Maxout networks (Goodfellow et al., 2013), Dropout, non-saturating Minimax gradients, and training using **Nesterov Accelerated Gradient (NAG)** / SGD with Momentum (Sutskever et al., 2013). Accessible via the `libdm.so` python bindings `dm.GAN()`.

---

## 🚀 Getting Started

### Prerequisites
- GCC Compiler (MinGW on Windows, or standard GCC on Linux/macOS)
- Make (optional, but recommended)

### Compilation
Build the executable from the source:
```bash
gcc -Iinclude -Iinclude/core -Wall -Wextra -O2 -pthread src/main.c src/core/*.c src/tokenizer/faro_tokenizer.c src/tokenizer/tokenizer_variants.c src/tokenizer/maximal_munch.c src/tokenizer/bpe_subword.c src/tokenizer/bpe_dropout.c src/tokenizer/fast_wordpiece.c src/tokenizer/grapheme_pair_encoding.c src/tokenizer/parity_bpe.c src/tokenizer/sentencepiece_lite.c src/tokenizer/tokenizer_lab.c src/tokenizer/unigram_subword.c src/algorithms/*.c -o build/bin/dm.exe -lm -licuuc -lpsapi
```
*(On Linux, remove `-lpsapi`; keep `-lm` for math functions.)*

### Usage
```bash
./build/bin/dm.exe <algorithm> <dataset_path> <format> <min_support> [min_io] [ins_threshold] [prn_threshold] [decay_base] [decay_life]
```
* **`<algorithm>` / direct subcommand**: `ais`, `apriori`, `eclat`, `fpgrowth`, `tree_projection`, `aclose`, `close`, `closet`, `closetplus`, `fpclose`, `charm`, `dci_closed`, `lcm`, `lcmver2`, `nafcp`, `fcfia`, `max_miner`, `genmax`, `fpmax`, `mafia`, `hep`, `fhoi`, `dfhoi`, `tkhoim`, `hoimto`, `mfhoi`, `fhoi_miner`, `weak_mfhoi_miner`, `strong_mfhoi_miner`, `mhoui`, `houi_miner`, `weak_mhoui_miner`, `strong_mhoui_miner`, `direct_mhoui_miner`, `regular_mine`, `negfin`, `prepost`, `prepostplus`, `dic`, `ltm`, `sam`, `carpenter`, `dbv_miner`, `defme`, `talky_g`, `pascal`, `zart`, `apriori_rare`, `apriori_inverse`, `cori`, `rp_tree`, `estdec`, `clostream`, `cfi_stream`, `uapriori`, `msapriori`, `ffiminer`, `ubmffp`, `dfigrowth`, `sum`, `mlhui_miner`, `fchm`, `vhuqi`, `fhuqi_miner`, `tkq`, `thue`, `mheinu`, `dphim`, `closed_fhuim_kinana`, `prefixspan`, `spade`, `uspan`, `hupspm`, `hup_miner`, `hiep`, `maximal_munch`, `bpe`, `bpe_dropout`, `unigram`, `sentencepiece`, `tokenizer_lab`, `gpe`, `parity_bpe`, `fast_wordpiece`, `resnet18`, `mobilenet_tiny`, `vit`, `swin`.
* **`vhuqi`** uses `<min_support>` as the absolute minimum utility threshold and accepts optional `[qrc]`, defaulting to `3`.
* **`fhuqi_miner`** requires quantity format `4`: `./build/bin/dm.exe fhuqi_miner <quantity_dataset> 4 <theta> <qrc> <all|min|max> <profit_file>`.
* **`tkq`** requires quantity format `4`: `./build/bin/dm.exe tkq <quantity_dataset> 4 <k> <qrc> <all|min|max> <profit_file>`.
* **`thue`** mines top-k high-utility episodes from utility event sequences: `./build/bin/dm.exe thue <utility_dataset> 1 <k> [MTD]`. With `MTD=0`, each timestamp is a simultaneous event set, so the output count and top-k threshold can be compared against top-k high-utility itemset miners such as TKU-Miner.
* **`mheinu`** requires utility format `1`: `./build/bin/dm.exe mheinu <utility_dataset> 1 <min_efficiency> [investment_file]`. If no investment file is supplied, the implementation follows the paper's experimental setup by deterministically generating positive per-item investment values with seed `42`.
* **`dphim`** requires utility format `1`: `./build/bin/dm.exe dphim <utility_dataset> 1 <min_utility> [threads]`. It parallelizes HUIM search with dynamic pthread task scheduling.
* **`closed_fhuim_kinana`** requires utility format `1`: `./build/bin/dm.exe closed_fhuim_kinana <utility_dataset> 1 <min_utility> <min_support> [min_owl]`.
* **`hup_miner`** requires sequence utility format `3`: `./build/bin/dm.exe hup_miner <sequence_utility_dataset> 3 <min_average_utility> [max_pattern_length] [max_candidates]`. The sequential files use SPMF notation such as `item[utility] -1 ... -2 SUtility:x`.
* **`regular_mine`** requires transactional format `0`: `./build/bin/dm.exe regular_mine <transactional_dataset> 0 <min_support>`.
* **`mfhoi`**, **`fhoi_miner`**, **`weak_mfhoi_miner`**, and **`strong_mfhoi_miner`** require transactional format `0`: `./build/bin/dm.exe strong_mfhoi_miner <transactional_dataset> 0 <min_support> <min_occupancy>`.
* **`cloe_hoi`** requires transactional format `0`: `./build/bin/dm.exe cloe_hoi <transactional_dataset> 0 <min_occupancy> [min_support] [max_seconds]`. If `min_support` is omitted, it defaults to `ceil(min_occupancy * transaction_count)` for comparable HOI threshold sweeps.
* **`aura_hoi`** requires transactional format `0`: `./build/bin/dm.exe aura_hoi <transactional_dataset> 0 <min_occupancy> [min_support] [max_seconds] [avg|sum] [closed|raw]`. If `min_support` is omitted or `0`, it defaults to `ceil(min_occupancy * transaction_count)`. Use `sum raw` to match the raw fullset summed-occupancy semantics used by the repository HEP/DFHOI implementations.
* **`mhoui`**, **`houi_miner`**, **`weak_mhoui_miner`**, **`strong_mhoui_miner`**, and **`direct_mhoui_miner`** require utility format `1`: `./build/bin/dm.exe mhoui <utility_dataset> 1 <min_support> <min_occupancy> <min_utility> [strong] [direct]`.
* **`hupp_miner`** is a standalone prompt-log miner: `./build/bin/hupp_miner --input <prompt_jsonl> --dataset-type auto|dolly|code_feedback --minsup <ratio|count> --minutil-ratio <value> --minalign <value>`. The companion script `scripts/run_hupp_experiments.sh` runs the full prompt benchmark and generates charts.
* **`hiep`** runs HIEP-Miner through the unified `dm.exe` algorithm registry: `./build/bin/dm.exe hiep --input <text_or_transaction_file> --input-type text|transactions --mode itemset|sequence --minsup <ratio|count> --theta-ratio <value> --window <N> --stride <N> --tokenizer faro`. HIEP text input uses `include/tokenizer/tokenizer.h`, so new tokenizer algorithms can be exposed through `dm_tokenizer_create()` and selected with `--tokenizer`. The companion scripts `scripts/run_hiep_experiments.sh` and `scripts/run_hiep_experiments.ps1` run Retail, Accidents, Chess, Dolly text, Zipfian planted-signal, ablation, and Apriori/Eclat/FP-Growth external baseline experiments through `dm.exe`, then write Q1-style charts to `results/hiep_q1/`.
* **`bpe`** runs the Sennrich-Haddow-Birch ACL 2016 BPE subword tokenizer through native C99 in `dm.exe`: `./build/bin/dm.exe bpe learn-bpe -i <corpus...> -m <merges> -o codes.bpe`, then `./build/bin/dm.exe bpe apply-bpe -c codes.bpe -i <corpus> -o corpus.bpe`. Pass multiple input corpora to `learn-bpe` for joint BPE, matching the paper's source-target vocabulary-union setup.
* **`bpe_dropout`** runs Provilkov et al.'s ACL 2020 BPE-Dropout through native C99 in `dm.exe`: first learn ordinary BPE codes with `./build/bin/dm.exe bpe learn-bpe ...`, then sample training-time segmentations with `./build/bin/dm.exe bpe_dropout -c codes.bpe -p 0.1 --seed 7 segment -i corpus.txt`. Use `sample-word` to inspect stochastic alternatives and `stats` to measure segmentation diversity. Setting `-p 0` recovers deterministic BPE; setting `-p 1` leaves character-level pieces.
* **`unigram`** runs the Kudo ACL 2018 unigram LM subword regularization tokenizer through native C99 in `dm.exe`: `./build/bin/dm.exe unigram train -i <corpus...> -o unigram.model --vocab-size <N>`, then `./build/bin/dm.exe unigram encode -m unigram.model -i <corpus> --mode viterbi|sample --alpha <value> --seed <N>`. `--mode sample` uses Forward-Filtering Backward-Sampling over all segmentations for on-the-fly subword regularization.
* **`sentencepiece`** runs a native C99 SentencePiece-style raw-text tokenizer/detokenizer from EMNLP 2018 D18-2012 through `dm.exe`: `./build/bin/dm.exe sentencepiece train --input raw.txt --model-type bpe|unigram --vocab-size <N> -o spm.model`, then `./build/bin/dm.exe sentencepiece encode --model spm.model --input raw.txt --output-format piece|id` and `./build/bin/dm.exe sentencepiece decode --model spm.model --input pieces.txt --input-format piece|id`. It also supports literal `--text`, `--add-bos`, `--add-eos`, `vocab`, and self-contained models with vocabulary/id mapping, BPE merges or unigram probabilities, and the U+2581 whitespace escape needed for lossless detokenization.
* **`tokenizer_lab`** runs the Dagan/Synnaeve/Roziere tokenizer-domain-adaptation toolkit through native C99 in `dm.exe`: `./build/bin/dm.exe tokenizer_lab train-bpe -i <corpus...> --pretokenizer gpt4|punct|identity --vocab-size <N> -o tok.json`, then `./build/bin/dm.exe tokenizer_lab evaluate -m tok.json -i <eval...> --baseline base.json`. It reports NSL, bytes-per-token, observed vocabulary, and Renyi entropy, plus `vocab-tradeoff` for memory/inference vocabulary-size estimates.
* **`gpe`** runs Grapheme Pair Encoding through native C99 in `dm.exe`: `./build/bin/dm.exe gpe train -i <corpus...> --unit grapheme --pretokenizer whitespace --vocab-size <N> -o gpe.json`, then `./build/bin/dm.exe gpe encode -m gpe.json -i <text>`. It also supports `pretoken-eval` for CRmax/Tokenization Parity and `units` to compare UTF-8 bytes, Unicode codepoints, and grapheme clusters.
* **`parity_bpe`** runs Parity-aware BPE from arXiv:2508.04796v2 through native C99 in `dm.exe`: `./build/bin/dm.exe parity_bpe train --lang-corpus en=en.txt --lang-corpus ta=ta.txt --dev-corpus en=en_dev.txt --dev-corpus ta=ta_dev.txt --merges <K> --strategy parity -o pbpe.json`, then `./build/bin/dm.exe parity_bpe encode -m pbpe.json -i text.txt`. It also supports `classic`, `hybrid`, and `window` strategies and `evaluate` reports per-language compression rates, tokenizer-fairness Gini, vocabulary utilization, TTR, and Renyi entropy.
* **`fast_wordpiece`** runs Song et al.'s EMNLP 2021 LinMaxMatch WordPiece tokenizer through native C99 in `dm.exe`: `./build/bin/dm.exe fast_wordpiece word -v vocab.txt johanson`, or `./build/bin/dm.exe fast_wordpiece encode -v vocab.txt -i text.txt --ids`. It builds the trie, failure links, and failure pops from the WordPiece vocabulary and supports BERT-style `##` suffix tokens, `[UNK]`, punctuation splitting, and numeric token-id output.
* **`volt`** runs VOLT (Vocabulary Learning via Optimal Transport, ACL 2021) through the `dm` dispatcher: first learn BPE candidates with `dm --train --algo bpe -i corpus.txt -m 10000 -o bpe.txt`, then find the optimal vocabulary with `dm --train --algo volt -i corpus.txt --bpe-model bpe.txt --S-min 1000 --S-max 10000 --S-step 1000 -o vocab.json --stats`. The dispatcher calls `dm_volt.exe` directly without a subcommand; the Python reference implementation at `scripts/volt.py` accepts identical flags.
* **`maximal_munch`** runs Reps' TOPLAS 1998 linear-time maximal-munch scanner through `dm.exe`: `./build/bin/dm.exe maximal_munch --dfa spec.dfa --input text.txt`, or add `--stats` to report tokens, errors, transitions, backtracks, failed-table hits, failed-table marks, and optimized `Tab` states. The DFA spec uses `states N`, `start Q`, `final Q TOKEN_ID NAME`, and `trans FROM SYMBOL TO`; symbols may be `a`, `'a'`, `0x61`, `0x30-0x39`, or `ANY`.
* **`huciminer`** implements Sahoo, Das, and Goswami's HUCI-Miner flow: utility-list HUI mining, level-wise high utility closed itemset/generator derivation, and HGB rule-basis statistics. Use `./build/bin/dm.exe huciminer <utility_dataset> 1 <min_utility> [min_uconf]` or `./build/bin/dm_run --algorithm huciminer --input <utility_dataset> --minutil <value> --minconf <value>`.
* **`chuo_miner`** is a standalone utility-dataset miner: `./build/bin/chuo_miner --input <utility_dataset> --minutil <value> --minsup <ratio|count> --minocc <value>`. The companion script `scripts/run_chuo_experiments.sh` runs Foodmart, Liquor, and Chainstore CHUO benchmarks and generates charts.
* **`pso_classifier`** is a standalone supervised classification-rule miner: `./build/bin/pso_classifier --input <spmf_class_folder> --particles <N> --threshold <T> --radius <R>`. The companion script `scripts/run_pso_classifier_experiments.sh` runs the malware-family classification benchmark and prints statistics only.
* **`tku_miner`** is a standalone top-k utility miner: `./build/bin/tku_miner --input <utility_dataset> --k <N> [--max-depth N] [--max-seconds S]`. The companion script `scripts/run_tku_experiments.sh` runs Foodmart and Liquor TKU benchmarks and prints statistics only.
* **`tku_pso_miner`** is a standalone heuristic top-k utility miner using TKU-PSO particle initialization, PEV checking, explored-particle caching, fitness estimation, pBest/gBest bit-difference updates, and minimum-solution-fitness threshold raising: `./build/bin/tku_pso_miner --input <utility_dataset> --k <N> --population <N> --iterations <N>`. The companion script `scripts/run_tku_pso_experiments.sh` runs statistics-only utility benchmarks.
* **`kclotree_miner`** is a standalone top-k closed sequential-pattern miner: `./build/bin/kclotree_miner --input <spmf_sequence_file_or_folder> --k <N> --type generic|group|redundancy_aware`. The companion script `scripts/run_kclotree_experiments.sh` runs the malware-sequence benchmark and prints statistics only.
* **`tipn_houi_miner`** is a standalone top-k high-on-shelf utility miner for positive/negative utility datasets: `./build/bin/tipn_houi_miner --input <negative_utility_dataset> --k <N> --intervals <N>`. The companion script `scripts/run_tipn_houi_experiments.sh` runs bounded Retail/Mushroom negative-utility benchmarks and prints statistics only.
* **`htk_miner`** is a standalone top-k frequent itemset miner using the paper's vertical BSN representation, top-k singleton initialization with ties, equivalence-class joins, and Q-Heap threshold raising: `./build/bin/htk_miner --input <transaction_dataset> --k <N> [--mode bsn]`. The companion script `scripts/run_htk_experiments.sh` runs transactional FIMI-style benchmarks and prints statistics only.
* **`topkphm_miner`** is a standalone top-k periodic high-utility itemset miner using periodic utility-lists, dynamic top-k utility thresholding, and EUSCS pruning: `./build/bin/topkphm_miner --input <utility_dataset> --k <N> --maxper <N> --maxavg <N>`. The companion script `scripts/run_topkphm_experiments.sh` runs statistics-only utility benchmarks.
* **`laga`** is a self-contained synthetic data generator: `./build/bin/dm.exe laga --input <dataset> --output <synthetic_file> --schema auto|transaction|sequence|text|tabular|utility [--target-size N] [--minsup ratio|count] [--tau-copy V] [--iterations N]`. It writes the synthetic dataset plus `<synthetic_file>.stats`.
* **`dm_connect`** is the universal connector smoke-test tool for the next-generation core: `./build/bin/dm_connect --input <path> --connector spmf|text|graph`. It uses mmap input and an arena-backed flat `uint32_t[] + row_offsets[]` representation without writing intermediate files.
* **`dm_run`** is the unified connector runner: `./build/bin/dm_run --algorithm <id> --input <path> --connector spmf|text|graph`. It first builds the arena-backed `DM_FlatDataset` when the algorithm can use flat item ids, then runs either a native flat plugin or a legacy `DM_Algorithm` through a flat-to-family adapter (`transactional`, `utility`, `sequence utility`, `quantity`, or `matrix`). Algorithms whose paper format carries extra semantics, such as HUCI utility values or PSO class folders, are exposed as raw plugins through the same command. Built-in adapters include `flat_stats`, `topk_items`, `chuo`, `huciminer`, `tku_miner`, `tku_pso`, `htk_miner`, `hupp`, `kclotree_miner`, `tipn_houi`, `topkphm`, and `pso_classifier`.
* **`<dataset_path>`**: Path to your transactional dataset (e.g., `datasets/chess.txt`).
* **`<format>`**: Usually `0` for raw space-separated transactions; use `1` for SPMF-style utility datasets, `3` for sequence utility datasets, and `4` for `item,quantity` HUQIM datasets.
* **`<min_support>`**: Support threshold as a fraction (e.g., `0.005` for 0.5%) or absolute count (e.g., `500`). For `tkhoim`, this is `k`.
* **`[min_io]`**: (Optional) Minimum Itemset Occupancy threshold for HOIM algorithms.
* **`[ins_threshold], [prn_threshold], [decay_base], [decay_life]`**: (Optional) Specific parameters for the **estDec** algorithm.

---

## 📚 Scientific References

The algorithms implemented in this framework strictly adhere to the logic and mathematics proposed in the following foundational research papers. 

**[1]** R. Agrawal, T. Imieliński, and A. Swami, "Mining association rules between sets of items in large databases," in *Proc. 1993 ACM SIGMOD Int. Conf. on Management of Data*, 1993, pp. 207–216. *(AIS)*

**[2]** R. Agrawal and R. Srikant, "Fast algorithms for mining association rules," in *Proc. 20th Int. Conf. on Very Large Data Bases (VLDB)*, 1994, pp. 487–499. *(Apriori)*

**[3]** N. Pasquier, Y. Bastide, R. Taouil, and L. Lakhal, "Discovering frequent closed itemsets for association rules," in *Proc. 7th Int. Conf. on Database Theory (ICDT)*, 1999, pp. 398–416. *(A-Close)*

**[4]** J. Han, J. Pei, and Y. Yin, "Mining frequent patterns without candidate generation," in *Proc. 2000 ACM SIGMOD Int. Conf. on Management of Data*, 2000, pp. 1–12. *(FP-Growth)*

**[5]** M. J. Zaki, "Scalable algorithms for association mining," *IEEE Trans. on Knowledge and Data Engineering*, vol. 12, no. 3, pp. 372–390, 2000. *(Eclat)*

**[6]** J. Pei, J. Han, and R. Mao, "CLOSET: An efficient algorithm for mining frequent closed itemsets," in *Proc. 2000 ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge Discovery*, 2000, pp. 21–30. *(CLOSET)*

**[7]** M. J. Zaki and C.-J. Hsiao, "CHARM: An efficient algorithm for closed itemset mining," in *Proc. 2002 SIAM Int. Conf. on Data Mining (SDM)*, 2002, pp. 457–473. *(CHARM)*

**[8]** J. Wang, J. Han, and J. Pei, "CLOSET+: Searching for the best strategies for mining frequent closed itemsets," in *Proc. 9th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2003, pp. 236–245. *(CLOSET+)*

**[9]** G. Grahne and J. Zhu, "Efficiently using prefix-trees in mining frequent itemsets," in *Proc. IEEE ICDM Workshop on Frequent Itemset Mining Implementations (FIMI)*, 2003. *(FPclose)*

**[10]** C. Lucchese, S. Orlando, and R. Perego, "Fast and memory efficient mining of frequent closed itemsets," *IEEE Trans. on Knowledge and Data Engineering*, vol. 18, no. 1, pp. 21–36, 2006. *(DCI_CLOSED)*

**[11]** R. J. Bayardo, "Efficiently Mining Long Patterns from Databases," in *Proc. 1998 ACM SIGMOD Int. Conf. on Management of Data*, 1998, pp. 85-93. *(Max-Miner)*

**[12]** R. C. Agarwal, C. C. Aggarwal, and V. V. V. Prasad, "A Tree Projection Algorithm for Generation of Frequent Itemsets," *Journal of Parallel and Distributed Computing*, vol. 61, no. 3, pp. 350-371, 2001. *(TreeProjection)*

**[13]** K. Gouda and M. J. Zaki, "Efficiently Mining Maximal Frequent Itemsets," in *Proc. 2001 IEEE Int. Conf. on Data Mining (ICDM)*, 2001, pp. 163-170. *(GenMax)*

**[14]** N. Jiang and L. Gruenwald, "CFI-Stream: Mining Closed Frequent Itemsets in Data Streams," in *Proc. 12th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD)*, 2006. [ACM: 1150402.1150473](https://dl.acm.org/doi/10.1145/1150402.1150473) *(CFI-Stream)*

**[15]** T. Uno, M. Kiyomi, and H. Arimura, "LCM ver. 2: Efficient Mining Algorithms for Frequent/Closed/Maximal Itemsets," in *Proc. IEEE ICDM Workshop on Frequent Itemset Mining Implementations (FIMI)*, 2004. *(LCM)*

**[16]** T. Le and B. Vo, "An N-list-based algorithm for mining frequent closed patterns," *Expert Systems with Applications*, vol. 42, no. 19, pp. 6748-6757, 2015. *(NAFCP)*

**[17]** L. Ma and Y. Qi, "An Efficient Algorithm for Frequent Closed Itemsets Mining," in *Proc. 2008 International Conference on Computer Science and Software Engineering*, 2008, pp. 260-262. *(FCFIA)*

**[18]** D. Burdick, M. Calimlim, and J. Gehrke, "MAFIA: A Maximal Frequent Itemset Algorithm for Transactional Databases," in *Proc. 17th International Conference on Data Engineering (ICDE)*, 2001, pp. 443-452. *(MAFIA)*

**[19]** Z.-H. Deng, "Mining high occupancy itemsets," *Future Generation Computer Systems*, vol. 102, pp. 222-229, 2020. *(HEP)*

**[20]** L. T. T. Nguyen, T. Mai, G.-H. Pham, U. Yun, and B. Vo, "An efficient method for mining high occupancy itemsets based on equivalence class and early pruning," *Knowledge-Based Systems*, 2023, 110441. *(FHOI/DFHOI)*

**[21]** I. Yildirim, "Mining top-k high occupancy itemsets," *Black Sea Journal of Engineering and Science*, vol. 8, no. 6, pp. 1723-1730, 2025. *(TKHOIM)*

**[22]** S. Datta, K. Mali, and U. Ghosh, "High Occupancy Itemset Mining with Consideration of Transaction Occupancy," *Arabian Journal for Science and Engineering*, vol. 47, pp. 2061–2075, 2022. *(HOIMTO)*

**[23]** Z. H. Deng and S. L. Lv, "FIN: A Fast Algorithm for Mining Frequent Itemsets using Nodesets," *IEEE Transactions on Knowledge and Data Engineering*, vol. 26, no. 12, pp. 3039-3051, 2014. *(FIN)*

**[24]** Z. H. Deng, "DiffNodesets: An Efficient Structure for Fast Mining Frequent Itemsets," *Applied Soft Computing*, vol. 45, pp. 104-115, 2016. [arXiv:1507.01345](https://arxiv.org/abs/1507.01345) *(FIN+)*

**[25]** N. Aryabarzan, B. Minaei-Bidgoli, and M. Teshnehlab, "negFIN: An efficient algorithm for fast mining frequent itemsets," *Expert Systems with Applications*, vol. 105, pp. 129-143, 2018. *(negFIN)*

**[26]** Z.-H. Deng and S.-L. Lv, "PrePost+: An efficient N-lists-based algorithm for mining frequent itemsets via Children–Parent Equivalence pruning," *Expert Systems with Applications*, vol. 42, pp. 5424-5432, 2015. *(PrePost+)*

**[27]** G. Grahne and J. Zhu, "Fast algorithms for frequent itemset mining using FP-trees," *IEEE Transactions on Knowledge and Data Engineering*, vol. 17, no. 1, pp. 81-90, 2005. *(FPmax)*

**[28]** Z.-H. Deng, J. Wang, and J. Jiang, "A new algorithm for fast mining frequent itemsets using n-lists," *Science China Information Sciences*, vol. 55, no. 9, pp. 2008–2030, 2012. *(PrePost)*

**[29]** S. Brin, R. Motwani, J. D. Ullman, and S. Tsur, "Dynamic itemset counting and implication rules for market basket data," in *Proc. 1997 ACM SIGMOD Int. Conf. on Management of Data*, 1997, pp. 255–264. *(DIC)*

**[30]** J. Lu, W. Xu, K. Zhou, and Z. Guo, "Frequent Itemset Mining Algorithm Based on Linear Table," *Journal of Database Management (JDM)*, vol. 34, no. 1, pp. 1-14, 2023. *(LTM)*

**[31]** C. Borgelt and X. Wang, "SaM: A Split and Merge Algorithm for Fuzzy Frequent Item Set Mining," in *Proc. 2009 IEEE International Conference on Fuzzy Systems*, 2009, pp. 255-268. *(SaM)*

**[32]** F. Pan, G. Cong, A. K. H. Tung, J. Yang, and M. J. Zaki, "CARPENTER: Finding Closed Patterns in Long Biological Datasets," in *Proc. 9th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD'03)*, 2003, pp. 637–642. *(CARPENTER)*

**[33]** B. Vo, T.-P. Hong, and B. Le, "DBV-Miner: A Dynamic Bit-Vector approach for fast mining frequent closed itemsets," *Expert Systems with Applications*, vol. 39, no. 8, pp. 7196–7206, 2012. *(DBV-Miner)*

**[34]** A. Soulet and F. Rioult, "Efficiently Depth-First Minimal Pattern Mining," in *Proc. 18th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD'14)*, 2014, pp. 28–39. *(dEFME)*

**[35]** L. Szathmary, P. Valtchev, A. Napoli, and R. Godin, "Efficient Vertical Mining of Frequent Closures and Generators," in *Proc. 8th Int. Symposium on Intelligent Data Analysis (IDA'09)*, 2009, pp. 393–404. *(Talky-G / TOUCH)*

**[36]** N. Pasquier, Y. Bastide, R. Taouil, G. Stumme, and L. Lakhal, "Mining Frequent Patterns with Counting Inference," *SIGKDD Explorations*, vol. 2, no. 2, pp. 66–75, 2000. *(PASCAL)*

**[37]** L. Szathmary, A. Napoli, and S. O. Kuznetsov, "ZART: A Multifunctional Itemset Mining Algorithm," in *Proc. 5th Int. Conf. on Concept Lattices and Their Applications (CLA '07)*, 2007, pp. 26–37. *(ZART)*

**[38]** L. Szathmary, A. Napoli, and P. Valtchev, "Towards Rare Itemset Mining," in *Proc. 19th IEEE Int. Conf. on Tools with Artificial Intelligence (ICTAI '07)*, 2007, vol. 1, pp. 305–312. *(Apriori-Rare)*

**[39]** Y. S. Koh and N. Rountree, "Finding Sporadic Rules Using Apriori-Inverse," in *Proc. 9th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD '05)*, 2005, pp. 97–106. *(Apriori-Inverse)*

**[40]** S. Bouasker and S. Ben Yahia, "Key correlation mining by simultaneous monotone and anti-monotone constraints checking," in *Proc. 30th Annual ACM Symp. on Applied Computing (SAC '15)*, 2015, pp. 851–856. *(CORI)*

**[41]** S. Tsang, Y. S. Koh, and G. Dobbie, "RP-Tree: Rare Pattern Tree Mining," in *Proc. 13th Int. Conf. on Data Warehousing and Knowledge Discovery (DaWaK '11)*, 2011, pp. 277–288. *(RP-Tree)*

**[42]** S.-J. Yen, Y.-S. Lee, C.-W. Wu, and C.-L. Lin, "An Efficient Algorithm for Maintaining Frequent Closed Itemsets over Data Stream," in *Proc. 22nd Int. Conf. on Industrial Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE '09)*, 2009, pp. 767–776. *(CloStream)*

**[43]** J. H. Chang and W. S. Lee, "Finding Recent Frequent Itemsets Adaptively over Online Data Streams," in *Proc. 9th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD)*, 2003, pp. 487-492. *(estDec)*

**[44]** C.-K. Chui, B. Kao, and E. Hung, "Mining Frequent Itemsets from Uncertain Data," in *Proc. 11th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD '07)*, 2007, pp. 47–58. *(U-Apriori)*

**[45]** B. Liu, W. Hsu, and Y. Ma, "Mining association rules with multiple minimum supports," in *Proc. 5th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD)*, 1999, pp. 337–341. *(MSApriori)*

**[46]** J. C.-W. Lin, T. Li, P. Fournier-Viger, and T.-P. Hong, "A fast Algorithm for mining fuzzy frequent itemsets," *Journal of Intelligent & Fuzzy Systems*, vol. 29, no. 6, pp. 2373–2379, 2015. *(FFI-Miner)*

**[47]** J. C.-W. Lin, T.-P. Hong, T.-C. Lin, and S.-T. Pan, "An UBMFFP Tree for Mining Multiple Fuzzy Frequent Itemsets," *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, vol. 23, no. 6, pp. 861–879, 2015. *(UBMFFP-Tree)*

**[48]** N. Pasquier, Y. Bastide, R. Taouil, and L. Lakhal, "Efficient Mining of Association Rules Using Closed Itemset Lattices," *Information Systems*, vol. 24, no. 1, pp. 25–46, 1999. *(Close)*

**[49]** J. Huang, Y.-P. Lai, C. Lo, and C.-W. Wu, "An Efficient Algorithm for Deriving Frequent Itemsets from Lossless Condensed Representation," in *Proc. 32nd Int. Conf. on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE)*, 2019, pp. 216–229. *(DFI-Growth)*

**[50]** G. I. Webb and J. Vreeken, "Efficient discovery of the most interesting associations," *ACM Transactions on Knowledge Discovery from Data*, vol. 8, no. 3, Article 15, 2014. *(OPUS Miner)*

**[51]** R. Agrawal and R. Srikant, "Fast Algorithms for Mining Association Rules in Large Databases," in *Proc. 20th Int. Conf. on Very Large Data Bases (VLDB)*, 1994, pp. 487–499. *(Apriori, AprioriTid, AprioriHybrid)*

**[52]** J. Vreeken, M. van Leeuwen, and A. Siebes, "Krimp: mining itemsets that compress," *Data Mining and Knowledge Discovery*, vol. 23, no. 1, pp. 169–214, 2011. *(KRIMP)*

**[53]** K. Smets and J. Vreeken, "Slim: Directly Mining Descriptive Patterns," in *Proc. 12th SIAM Int. Conf. on Data Mining (SDM)*, 2012, pp. 236–247. *(SLIM)*

**[54]** Y. Liu, W. Liao, and A. Choudhary, "A Two-Phase Algorithm for Fast Discovery of High Utility Itemsets," in *Proc. 9th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD)*, 2005, pp. 689–695. *(Two-Phase)*

**[55]** P. Fournier-Viger, C.-W. Wu, S. Zida, and V. S. Tseng, "FHM: Faster High-Utility Itemset Mining using Estimated Utility Co-occurrence Pruning," in *Proc. 21st Int. Symp. on Methodologies for Intelligent Systems (ISMIS)*, 2014, pp. 83–92. *(FHM)*

**[56]** S. Zida, P. Fournier-Viger, J. C.-W. Lin, C.-W. Wu, and V. S. Tseng, "EFIM: A Highly Efficient Algorithm for High-Utility Itemset Mining," in *Proc. 14th Mexican Int. Conf. on Artificial Intelligence (MICAI)*, 2015, pp. 530–546. *(EFIM)*

**[57]** M. Liu and J. Qu, "Mining High Utility Itemsets without Candidate Generation," in *Proc. 21st ACM Int. Conf. on Information and Knowledge Management (CIKM)*, 2012, pp. 55–64. *(HUI-Miner)*

**[58]** V. S. Tseng, C.-W. Wu, B.-E. Shie, and P. S. Yu, "UP-Growth: An Efficient Algorithm for High Utility Itemset Mining," in *Proc. 16th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD)*, 2010, pp. 253–262. *(UP-Growth)*

**[59]** C. F. Ahmed, S. K. Tanbeer, B.-S. Jeong, and Y.-K. Lee, "Efficient Tree Structures for High Utility Pattern Mining in Incremental Databases," *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 12, pp. 1708–1721, 2009. *(IHUP)*

**[60]** Z. Cheng, W. Fang, W. Shen, J. C.-W. Lin, and B. Yuan, "An efficient utility-list based high-utility itemset mining algorithm," *Applied Intelligence*, vol. 53, pp. 6992–7006, 2023. *(HUIM-SU)*

**[61]** Q.-H. Duong, P. Fournier-Viger, H. Ramampiaro, K. Nørvåg, and T.-L. Dam, "Efficient high utility itemset mining using buffered utility-lists," *Applied Intelligence*, vol. 48, no. 5, pp. 1167–1187, 2018. *(ULB-Miner)*

**[62]** S. Dawar, V. Goyal, and D. Bera, "A hybrid framework for mining high-utility itemsets in a sparse transaction database," *Applied Intelligence*, vol. 47, pp. 809–827, 2017. *(UFH)*

**[63]** J. Sahoo, A. K. Das, and A. Goswami, "An efficient approach for mining association rules from high utility itemsets," *Expert Systems with Applications*, vol. 42, no. 13, pp. 5754–5778, 2015. *(HUCI-Miner)*

**[64]** S. Dawar and V. Goyal, "Up-hist tree: an efficient data structure for mining high utility patterns from transaction databases," in *Proc. 19th Int. Database Engineering & Applications Symp. (IDEAS)*, 2015, pp. 56–61. *(UP-Hist)*

**[65]** P. Sra and S. Chand, "A residual utility-based concept for high-utility itemset mining," *Knowledge and Information Systems*, vol. 66, pp. 211–235, 2024. *(R-Miner)*

**[66]** P. Sra and S. Chand, "A Reinduction-Based Approach for Efficient High Utility Itemset Mining from Incremental Datasets," *Data Science and Engineering*, vol. 9, pp. 73–87, 2024. *(SUM)*

**[67]** P. Fournier-Viger, Y. Wang, J. C.-W. Lin, J. M. Luna, and S. Ventura, "Mining Cross-Level High Utility Itemsets," in *Proc. 33rd Int. Conf. on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE)*, 2020, pp. 1–13. *(CLH-Miner)*

**[68]** L. Cagliero, P. Garza, and E. Baralis, "Discovering High-Utility Itemsets at Multiple Abstraction Levels," in *Proc. 21st East-European Conf. on Advances in Databases and Information Systems (ADBIS)*, 2017, pp. 201-215. *(MLHUI-Miner)*

**[69]** P. Fournier-Viger, J. C.-W. Lin, T. Dinh, and H. B. Le, "Mining Correlated High-Utility Itemsets using the Bond Measure," in *Proc. 11th International Conference on Hybrid Artificial Intelligence Systems (HAIS '16)*, 2016, pp. 108–120. *(FCHM)*

**[70]** P. Fournier-Viger, "FHN: Efficient Mining of High-Utility Itemsets with Negative Unit Profits," in *Proc. 11th International Conference on Hybrid Artificial Intelligence Systems (HAIS '16)*, 2016, pp. 108–119. *(FHN)*

**[71]** P. Fournier-Viger and S. Zida, "FOSHU: Faster On-Shelf High Utility Itemset Mining – with or without Negative Unit Profit," in *Proc. 30th Annual ACM Symposium on Applied Computing (SAC 2015)*, 2015, pp. 845–850. *(FOSHU)*

**[72]** G.-C. Lan, T.-P. Hong, J.-P. Huang, and V. S. Tseng, "On-shelf utility mining with negative item values," *Expert Systems with Applications*, vol. 41, no. 7, pp. 3450–3459, 2014. *(TS-HOUN)*

**[73]** J. C.-W. Lin, W. Gan, T.-P. Hong, and J.-S. Pan, "Incrementally Updating High-Utility Itemsets with Transaction Insertion," in *Proc. 10th International Conference on Advanced Data Mining and Applications (ADMA 2014)*, 2014, pp. 44–56. *(HUI-list-INS)*

**[74]** P. Fournier-Viger, S. Zida, J. C.-W. Lin, C.-W. Wu, and V. S. Tseng, "EFIM-Closed: Fast and Memory Efficient Discovery of Closed High-Utility Itemsets," in *Proc. 21st International Symposium on Methodologies for Intelligent Systems (ISMIS 2014)*, 2014, pp. 83–92. *(EFIM-Closed)*

**[75]** C.-W. Wu, P. Fournier-Viger, J.-Y. Gu, and V. S. Tseng, "Mining Closed High Utility Itemsets without Candidate Generation," in *Proc. IEEE International Conference on Systems, Man, and Cybernetics (SMC 2015)*, 2015, pp. 199–204. *(CHUI-Miner)*

**[76]** S. Nawaz, J. C.-W. Lin, and P. Fournier-Viger, "Mining High Utility Itemsets with Hill Climbing and Simulated Annealing," *IEEE Transactions on Management Information Systems (TMIS)*, vol. 12, no. 4, pp. 1–25, 2021. *(HUIM-HC/SA)*

**[77]** J. C.-W. Lin, L. Yang, P. Fournier-Viger, T.-P. Hong, and M. Voznak, "A binary PSO approach to mine high-utility itemsets," *Soft Computing*, vol. 20, no. 1, pp. 1–13, 2016. *(HUIM-BPSO-tree)*

**[78]** W. Song and C. Huang, "Mining High Utility Itemsets Using Bio-Inspired Algorithms: A Diverse Optimal Value Framework," *IEEE Access*, vol. 6, pp. 19568–19582, 2018. *(Bio-HUIF)*

**[79]** W. Song, J. Li, and C. Huang, "Artificial Fish Swarm Algorithm for Mining High Utility Itemsets," in *Proc. 12th International Conference on Advances in Swarm Intelligence (ICSI)*, 2021, pp. 407–419. *(HUIM-AFSA)*

**[80]** W. Song and C. Huang, "Discovering High-Utility Itemsets Based on the Artificial Bee Colony Algorithm," in *Proc. 22nd Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)*, 2018, pp. 240–252. *(HUIM-ABC)*

**[81]** S. Krishnamoorthy, "Mining top-k high utility itemsets with effective threshold raising strategies," *Expert Systems with Applications*, vol. 117, pp. 148–165, 2019. *(THUI)*

**[82]** W. Song, C. Zheng, C. Huang, and L. Liu, "Heuristically mining the top-k high-utility itemsets with cross-entropy optimization," *Applied Intelligence*, vol. 51, no. 12, pp. 8864–8881, 2021. *(TKU-CE/CE+)*

**[83]** S. Dawar, V. Sharma, and V. Goyal, "Mining top-k high-utility itemsets from a data stream under sliding window model," *Applied Intelligence*, vol. 47, no. 4, pp. 1240–1255, 2017. *(FHMDS)*

**[84]** J. C.-W. Lin, T. Li, P. Fournier-Viger, T.-P. Hong, J. Zhan, and M. Voznak, "An efficient algorithm to mine high average-utility itemsets," *Advanced Engineering Informatics*, vol. 30, no. 2, pp. 233–243, 2016. *(HAUI-Miner)*

**[85]** J. C.-W. Lin, S. Ren, P. Fournier-Viger, and T.-P. Hong, "EHAUPM: Efficient High Average-Utility Pattern Mining With Tighter Upper Bounds," *IEEE Access*, vol. 5, pp. 12927–12940, 2017. *(EHAUPM)*

**[86]** W. Song, L. Liu, and C. Huang, "Generalized maximal utility for mining high average-utility itemsets," *Knowledge and Information Systems*, vol. 63, no. 11, pp. 2947–2967, 2021. *(HAUIM-GMU)*

**[87]** T.-N. Tran, V. T. Hoang, T.-C. Truong, and M. Voznak, "A hierarchical set-enumeration tree enabling high occupancy item set mining and the use of an adaptive occupancy threshold," *Applied Intelligence*, vol. 55, no. 2, pp. 205–225, 2025. *(NAM-HEP)*

**[88]** J. C.-W. Lin, S. Ren, and P. Fournier-Viger, "MEMU: More Efficient Algorithm to Mine High Average-Utility Patterns With Multiple Minimum Average-Utility Thresholds," *IEEE Access*, vol. 6, pp. 7593–7609, 2018. *(MEMU)*

**[89]** C. H. Li, C.-W. Wu, and V. S. Tseng, "Efficient Vertical Mining of High Utility Quantitative Itemsets," in *Proc. IEEE International Conference on Granular Computing (GrC)*, 2014, pp. 155–160. *(VHUQI)*

**[90]** M. Nouioua, P. Fournier-Viger, C.-W. Wu, J. C.-W. Lin, and W. Gan, "FHUQI-Miner: Fast High Utility Quantitative Itemset Mining," *Applied Intelligence*, 2021. *(FHUQI-Miner)*

**[91]** M. Nouioua, P. Fournier-Viger, W. Gan, Y. Wu, J. C.-W. Lin, and F. Nouioua, "TKQ: Top-K Quantitative High Utility Itemset Mining." *(TKQ)*

**[92]** I. Yildirim, "Mining High-Efficiency Itemsets with Negative Utilities," *Mathematics*, vol. 13, no. 4, article 659, 2025. https://doi.org/10.3390/math13040659 *(MHEINU)*

**[93]** G. Kimura, Y. Hayamizu, R. U. Kiran, M. Kitsuregawa, and K. Goda, "DPHIM: Efficient Parallel Mining of High-Utility Itemsets on Multicore Processors and Its Evaluation," *IEEE Transactions on Knowledge and Data Engineering*, vol. 38, no. 5, pp. 2714-2730, 2026. https://doi.org/10.1109/TKDE.2026.3666851 *(DPHIM)*

**[94]** K. S. Sulanjari and C. Fatichah, "Optimized Closed Frequent High Utility Itemset Mining Using OSR, OWL, And MSU Pruning On Retail Transaction Data," *JUTI: Jurnal Ilmiah Teknologi Informasi*, vol. 24, no. 1, pp. 1-15, 2026. https://doi.org/10.12962/j24068535.v24i1.a1311 *(Closed-FHUIM-Kinana)*

**[95]** M. Geng, Y. Wu, Y. Li, J. Liu, L. Guo, X. Zhu, and X. Wu, "Mining High Average Utility Nonoverlapping Patterns from Sequential Database," *ACM Transactions on Intelligent Systems and Technology*, vol. 17, no. 1, article 15, 2026. https://doi.org/10.1145/3773899 *(HUP-Miner)*

**[96]** S. Ruggieri, "Frequent Regular Itemset Mining," in *Proc. 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '10)*, 2010, pp. 263-272. https://doi.org/10.1145/1835804.1835840 *(RegularMine)*

**[97]** Q. Van, "MFHOI-Miner: An Efficient Method for Mining Maximal Frequent High-Occupancy Itemsets," local project paper, `docs/core/mfhoi.pdf`, 2026. *(MFHOI / Strong MFHOI-Miner)*

**[98]** Q. Van, "HUPP-Miner: High-Utility Prompt Pattern Mining for Cost-Aware and Accuracy-Preserving Generative AI Systems," local project paper, `docs/core/hupp.pdf`, 2026. *(HUPP-Miner)*

**[99]** Q. Van, "Closed High-Utility Occupancy Itemset Mining," local project paper, `docs/core/chuim.pdf`, 2026. *(CHUO-Miner)*

**[100]** Q. Van, "MHOUI-Miner: Mining High-Occupancy Utility Itemsets," local project paper, `docs/core/mhoui.pdf`, 2026. *(MHOUI-Miner)*

**[101]** Q. Van, "VIFP: Verifiable/Privacy-Preserving Frequent Pattern Mining," local project paper, `docs/core/vifp.pdf`, 2026. *(VIFP-Miner)*

**[102]** T. Sousa, A. Silva, and A. Neves, "Particle Swarm based Data Mining Algorithms for classification tasks," *Parallel Computing*, vol. 30, no. 5-6, pp. 767-783, 2004. https://doi.org/10.1016/j.parco.2003.12.015 *(PSO Classifier)*

**[103]** C.-W. Wu, B.-E. Shie, P. S. Yu, and V. S. Tseng, "Mining top-k high utility itemsets," in *Proc. 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '12)*, 2012, pp. 78-86. https://doi.org/10.1145/2339530.2339546 *(TKU-Miner)*

**[104]** R. A. Rizvee, C. F. Ahmed, and C. K. Leung, "A tree-based framework to mine top-K closed sequential patterns," *Applied Intelligence*, vol. 55, article 221, 2025. https://doi.org/10.1007/s10489-024-06137-y *(KCloTreeMiner)*

**[105]** Y.-I. Chang, P.-C. Chuang, Y.-H. Liao, P.-Y. Hu, and T.-W. Chen, "An Efficient Algorithm for Mining Top-k High-On-Shelf-Utility Itemsets with Positive/Negative Profits of Local/Global Minimum Count," *Engineering Proceedings*, vol. 108, no. 1, article 45, 2025. https://doi.org/10.3390/engproc2025108045 *(TIPN-HOUI-Miner)*

**[106]** K. Malliaridis and S. Ougiaroglou, "Efficient techniques for retrieving top-K frequent itemsets," *Expert Systems With Applications*, vol. 311, article 131250, 2026. https://doi.org/10.1016/j.eswa.2026.131250 *(HTK-Miner / HTK-negFIN)*

**[107]** X. Zhang, B. Liu, J. Chen, S. Yang, Z. Gu, Z. Liu, and S. Xu, "TOPKPHM: Periodic Patterns Mining of Top-K High Utility Itemsets," in *Proc. 2025 International Conference on Trustworthy Big Data and Artificial Intelligence (ICTBAI)*, 2025. https://doi.org/10.1109/ICTBAI68361.2025.00009 *(TOPKPHM)*

**[108]** S. Carstensen and J. C.-W. Lin, "TKU-PSO: An Efficient Particle Swarm Optimization Model for Top-k High-Utility Itemset Mining," *International Journal of Interactive Multimedia and Artificial Intelligence*, vol. 9, no. 4, pp. 70-81, 2025. https://doi.org/10.9781/ijimai.2024.01.002 *(TKU-PSO)*

**[109]** Q. Van, "HIEP-Miner: High-Information Entropy Pattern Mining for Embedded Text Streams," local project paper, `docs/core/hiep.pdf`, 2026. *(HIEP-Miner)*

**[110]** Q. Van, "FARO: Frequency-Aware Robust Out-of-vocabulary Subword Tokenizer," local project paper, `docs/core/faro_tokenizer_fixed.pdf`, 2026. *(FARO Tokenizer)*

**[111]** Jane Doe, "HUST: Mining-Driven Subword Tokenization for High-Utility Pattern Mining," local project paper, `docs/core/hust.pdf`, 2026. *(HUST-Tokenize)*

**[112]** Quan Van, "Global Breakthroughs in Data Mining During 2025–2026: A Survey of Pattern Mining, Graph Mining, Stream Mining, and LLM-Centric Knowledge Discovery," local project survey, `docs/core/data_mining_2025_2026_survey_q1.pdf`, 2026.

**[113]** Văn Hà Minh Quân, "MEDM-GEN: A Pattern-Guided, Constraint-Aware Dataset Generator Framework for Reproducible Data Mining Research," local project paper, `docs/core/medm_gen_q1_paper.pdf`, 2026. *(MEDM-GEN)*

**[114]** Author Name, "LAGA: A Self-Contained Layout-Aware Generative Architecture for Knowledge-Preserving Data Generation," local project paper, `docs/core/LAGA_self_contained_data_generator_q1.tex`, 2026. *(LAGA)*

**[115]** S. Wan, J. Chen, W. Gan, G. Chen, and V. Goyal, "THUE: Discovering Top-K High Utility Episodes," arXiv:2106.14830, 2021. *(THUE)*

**[116]** R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in *Proc. ACL 2016*, pp. 1715-1725, 2016. https://aclanthology.org/P16-1162/ *(Sennrich BPE Subword)*

**[117]** T. Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates," in *Proc. ACL 2018*, pp. 66-75, 2018. https://aclanthology.org/P18-1007/ *(Unigram Subword Regularization)*

**[118]** G. Dagan, G. Synnaeve, and B. Roziere, "Getting the most out of your tokenizer for pre-training and domain adaptation," arXiv:2402.01035v2, 2024. *(Tokenizer Lab)*

**[119]** M. Velayuthan and K. Sarveswaran, "Egalitarian Language Representation in Language Models: It All Begins with Tokenizers," in *Proc. COLING 2025*, pp. 5987-5996, 2025. *(Grapheme Pair Encoding)*

**[120]** D. Paul, C. Meister, N. Foroutan, J. Niklaus, S. Ahmadi, A. Bosselut, and R. Sennrich, "Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization," arXiv:2508.04796v2, 2025. *(Parity-aware BPE)*

**[121]** X. Song, A. Salcianu, Y. Song, D. Dopson, and D. Zhou, "Fast WordPiece Tokenization," in *Proc. EMNLP 2021*, pp. 2089-2103, 2021. https://aclanthology.org/2021.emnlp-main.160/ *(Fast WordPiece / LinMaxMatch)*

**[122]** I. Provilkov, D. Emelianenko, and E. Voita, "BPE-Dropout: Simple and Effective Subword Regularization," in *Proc. ACL 2020*, pp. 1882-1892, 2020. https://aclanthology.org/2020.acl-main.170/ *(BPE-Dropout)*

**[123]** T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing," in *Proc. EMNLP 2018: System Demonstrations*, pp. 66-71, 2018. https://aclanthology.org/D18-2012/ *(SentencePiece Lite)*

**[124]** T. Reps, "Maximal-munch tokenization in linear time," *ACM Transactions on Programming Languages and Systems*, vol. 20, no. 2, pp. 259-273, 1998. https://doi.org/10.1145/276393.276394 *(Reps Maximal-Munch Scanner)*

**[125]** C. Xu, B. Zhou, T. Gan, Q. Zheng, and L. Li, "Vocabulary Learning via Optimal Transport for Neural Machine Translation," in *Proc. 59th Annual Meeting of the Association for Computational Linguistics (ACL 2021)*, pp. 7361–7373, 2021. https://aclanthology.org/2021.acl-long.571/ *(VOLT)*

**[126]** K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90 *(ResNet-18)*

**[127]** A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017. https://arxiv.org/abs/1704.04861 *(MobileNetV4-Tiny)*

**[128]** K. Wu, J. Zhang, H. Peng, M. Liu, B. Xiao, J. Fu, and L. Yuan, "TinyViT: Fast Pretraining Distillation for Small Vision Transformers," in *Proc. European Conference on Computer Vision (ECCV)*, 2022, pp. 68–85. https://doi.org/10.1007/978-3-031-20083-0_5 *(TinyViT)*

**[129]** A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. International Conference on Learning Representations (ICLR)*, 2021. https://arxiv.org/abs/2010.11929 *(ViT)*

**[130]** J. Pei, J. Han, B. Mortazavi-Asl, H. Pinto, Q. Chen, U. Dayal, and M.-C. Hsu, "PrefixSpan: Mining Sequential Patterns Efficiently by Prefix-Projected Pattern Growth," in *Proc. 17th International Conference on Data Engineering (ICDE)*, 2001, pp. 215-224. https://doi.org/10.1109/ICDE.2001.914830 *(PrefixSpan)*

**[131]** Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021. [PDF](docs/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) | https://github.com/microsoft/Swin-Transformer *(Swin Transformer; unofficial C inference port)*

**[132]** M. J. Zaki, "SPADE: An Efficient Algorithm for Mining Frequent Sequences," *Machine Learning*, vol. 42, pp. 31-60, 2001. https://doi.org/10.1023/A:1007652502315 *(SPADE)*

**[132]** D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," in *Proc. International Conference on Learning Representations (ICLR)*, 2014. https://arxiv.org/abs/1312.6114 *(VAE)*

**[133]** D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. International Conference on Learning Representations (ICLR)*, 2015. https://arxiv.org/abs/1412.6980 *(Adam)*

**[134]** I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative Adversarial Nets," in *Advances in Neural Information Processing Systems (NIPS)*, 2014. *(GAN)*

**[135]** I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio, "Maxout Networks," in *Proc. International Conference on Machine Learning (ICML)*, 2013. *(Maxout)*

**[136]** I. Sutskever, J. Martens, G. Dahl, and G. Hinton, "On the importance of initialization and momentum in deep learning," in *Proc. International Conference on Machine Learning (ICML)*, 2013. *(SGD Momentum / Nesterov)*


---

## Strong MFHOI-Miner Experiments

This repository now includes an experimental framework for Strong Maximal Frequent High-Occupancy Itemset Mining.

Strong MFHOI keeps FHOI patterns that are not dominated by a strict FHOI superset with equal or higher average occupancy. The expected compactness relationship is:

```text
|FHOI| >= |Weak MFHOI| >= |Strong MFHOI|
```

Compile:

```bash
make
```

Run one algorithm:

```bash
./build/bin/mfhoi_miner --input datasets/itemsets/retail.txt --algorithm strong_mfhoi --minsup 0.02 --minocc 0.6 --output results/patterns/out.txt
```

Run all experiments:

```bash
make experiments
```

Generate plots:

```bash
make plots
```

Inspect:

```text
results/csv/
results/plots/
results/reports/experiment_report.txt
```

The framework records runtime, peak RAM, output disk usage, pattern quality, FHOI/Weak/Strong counts, compression metrics, overlap with MFI, and dominance examples. See `docs/MFHOI_EXPERIMENTS.md` for details.

---
<div align="center">
  <p>Made with ❤️ for Data Science & C Programming.</p>
</div>
