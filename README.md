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
1. **AIS** - The pioneer algorithm for mining association rules (Agrawal et al. 1993).
2. **Apriori** - Level-wise search using candidate generation and a hash tree (Agrawal & Srikant 1994).
3. **AprioriTid** - Variant of Apriori using candidate transaction ID sets to avoid full database scans.
4. **AprioriHybrid** - Combines Apriori and AprioriTid for optimal level-wise performance.
5. **Eclat** - Vertical data format search with depth-first traversal and TID-set intersections (Zaki 2000).
6. **FP-Growth** - Tree-based, projection-driven mining without candidate generation (Han et al. 2000).
7. **TreeProjection** - Matrix-based counting via depth-first transactional tree projection (Agarwal et al. 2001).
8. **FIN** - Fast mining of Frequent Itemsets using Nodesets (Deng & Lv 2014).
9. **FIN+** (dFIN) - Efficient frequent itemset mining using DiffNodesets (Deng 2016).
10. **negFIN** - An efficient algorithm for fast mining frequent itemsets using NegNodesets (Aryabarzan et al. 2018).
11. **PrePost** - Fast mining of frequent itemsets using N-lists (Deng et al. 2012).
12. **PrePost+** - High-performance N-lists-based mining using Children–Parent Equivalence pruning (Deng & Lv 2015).
13. **DIC** - Dynamic Itemset Counting algorithm for reducing the number of passes over the data (Brin et al. 1997).
14. **LTM** - Linear Table Miner for frequent itemset mining using a linear table structure and bitwise pruning (Lu et al. 2023).
15. **SaM** - Split and Merge algorithm using a horizontal transaction representation for efficient mining (Borgelt & Wang 2009).
16. **MSApriori** - Level-wise mining with multiple minimum supports to solve the rare item problem (Liu et al. 1999).
17. **DFI-Growth** - Deriving frequent itemsets from lossless condensed representations using pattern growth (Huang et al. 2019).
18. **KRIMP** - MDL-based pattern selection algorithm that finds the itemset code table compressing the database best (Vreeken et al. 2011).
19. **SLIM** - Directly mines descriptive patterns using MDL by iteratively merging itemsets in the code table (Smets & Vreeken 2012).
20. **OPUS Miner** - Efficient discovery of non-redundant, self-sufficient itemsets using branch-and-bound search (Webb & Vreeken 2014).
21. **RegularMine** - Mines frequent regular itemsets as an interpretable concise representation of frequent itemsets (Ruggieri 2010).
22. **HTK-Miner** - Fast top-K frequent itemset retrieval using vertical bitset networks and Q-Heap threshold raising (Malliaridis & Ougiaroglou 2026).

### High-Utility Itemset Mining (HUIM)
23. **Two-Phase** - Seminal HUIM algorithm using Transaction-Weighted Utilization (TWU) and two-phase candidate filtering (Liu et al. 2005).
24. **HUI-Miner** - Mines high-utility itemsets without candidate generation using a vertical utility-list structure (Liu & Qu 2012).
25. **FHM** - Faster High-Utility Itemset Mining using Estimated Utility Co-occurrence Pruning (Fournier-Viger et al. 2014).
26. **EFIM** - Highly efficient HUIM algorithm using database projection, transaction merging, and Subtree Utility bounds (Zida et al. 2017).
27. **EFIM-Closed** - Fast and memory-efficient discovery of closed high-utility itemsets with forward/backward closure checking (Fournier-Viger et al. 2016).
28. **CHUI-Miner** - Mines closed high-utility itemsets without candidate generation using vertical utility-lists (Wu et al. 2015).
29. **UP-Growth** - Uses a compact UP-Tree structure and DGU/DGN pruning strategies to mine PHUIs efficiently (Tseng et al. 2010).
30. **IHUP** - Incremental High-Utility Pattern mining using IHUP-Trees (L-Tree, TF-Tree, TWU-Tree) (Ahmed et al. 2009).
31. **HUIM-SU** - Simplified Utility-list based HUIM using repeated TWU pruning and extension utility bounds (Cheng et al. 2023).
32. **ULB-Miner** - High-performance utility-list buffer (UTLBuf) reducing memory fragmentation and join overhead (Duong et al. 2018).
33. **UFH** - Hybrid framework combining UP-Growth+ (tree-based) and FHM (utility-list based) for sparse/dense datasets (Dawar et al. 2017).
34. **HUCI-Miner** - Mines high-utility closed itemsets, generators, and non-redundant association rules (Sahoo et al. 2015).
35. **UP-Hist Growth** - Extends UP-Growth with quantity histograms at each tree node for tighter utility estimates (Dawar & Goyal 2015).
36. **R-Miner** - Residual utility-based mining using Residue Maps and Master Map for efficient joins (Sra & Chand 2024).
37. **SUM** - Incremental HUIM using a reinduction strategy and dynamic threshold setting (Sra & Chand 2024).
38. **CLH-Miner** - Mines Cross-Level High Utility Itemsets across tax-utility abstraction hierarchies (Fournier-Viger et al. 2020).
39. **MLHUI-Miner** - Efficient utility-list based mining of high-utility itemsets at multiple abstraction levels (Cagliero et al. 2017).
40. **FCHM** - Fast Correlated High-Utility itemset Miner discovering patterns with the bond measure (Fournier-Viger et al. 2016).
41. **FHN** - Fast High-Utility itemset miner for transaction databases with negative unit profits (Fournier-Viger 2016).
42. **FOSHU** - Faster On-Shelf High-Utility itemset mining with or without negative unit profits (Fournier-Viger & Zida 2015).
43. **TS-HOUN** - On-shelf utility mining with negative item profit values over temporal shelf intervals (Lan et al. 2014).
44. **HUI-list-INS** - Incrementally updates high-utility itemsets upon transaction insertions (Lin et al. 2014).
45. **TKU-Miner** - Exact top-K high-utility itemset mining without requiring a user-specified minimum utility threshold (Wu et al. 2012).
46. **THUI** - Top-K high-utility itemset mining with effective threshold-raising strategies using LIU structures (Krishnamoorthy 2019).
47. **TKU-CE** - Cross-Entropy optimization algorithm for mining top-K high-utility itemsets (Song et al. 2021).
48. **TKU-CE+** - Improved Cross-Entropy method with CUV pruning and smoothing mutation (Song et al. 2021).
49. **TKU-PSO** - Particle Swarm Optimization framework for heuristic top-K utility itemset discovery (Carstensen & Lin 2025).
50. **TOPKPHM** - Mining periodic top-K high-utility patterns over timestamped utility sequences (Zhang et al. 2025).
51. **DPHIM** - Dynamic parallel high-utility itemset mining using utility-list subtasks and pthread worker pools (Kimura et al. 2026).
52. **Closed-FHUIM-Kinana** - Closed frequent high-utility itemset mining with OSR, OWL, and MSU pruning (Sulanjari & Fatichah 2026).
53. **HAUI-Miner** - High Average-Utility Itemset mining using AU-lists and transaction-maximum utility downward closure (Lin et al. 2016).
54. **EHAUPM** - Efficient High Average-Utility Pattern Mining with tighter upper bounds and co-occurrence pruning (Lin et al. 2017).
55. **HAUIM-GMU** - Mining High Average-Utility Itemsets based on Generalized Maximal Utility (Song et al. 2021).
56. **MEMU** - Mining High Average-Utility Patterns with Multiple Minimum Average-Utility Thresholds (Lin et al. 2018).
57. **MHEINU** - Mining High-Efficiency Itemsets with Negative Utilities using ELNU lists and `uben`/`ubeni` bounds (Yildirim 2025).
58. **VHUQI** - Vertical mining of High-Utility Quantitative Itemsets using utility-lists and K-support bounds (Li et al. 2014).
59. **FHUQI-Miner** - Fast High-Utility Quantitative Itemset Miner using TQCS, EQCPS, and RQCPS pruning (Nouioua et al. 2021).
60. **TKQ** - Top-K Quantitative High Utility Itemset Miner using RIU, CUD, and update-queue threshold raising (Nouioua et al. 2021).
61. **TIPN-HOUI-Miner** - Top-K high-on-shelf-utility itemsets with positive/negative utilities and interval bitmaps (Chang et al. 2025).
62. **HUIM-HC** - High-utility itemset mining with Hill Climbing using PEV pruning and diversity maintenance (Nawaz et al. 2021).
63. **HUIM-SA** - High-utility itemset mining with Simulated Annealing and temperature-scheduled acceptance (Nawaz et al. 2021).
64. **HUIM-BPSO-tree** - Binary Particle Swarm Optimization with an OR/NOR-tree for valid candidate traversal (Lin et al. 2016).
65. **Bio-HUIF** (GA/PSO/BA) - Genetic, Particle Swarm, and Bat algorithms for HUIM via Diverse Optimal Values (Song & Huang 2018).
66. **HUIM-AFSA** - Artificial Fish Swarm Algorithm for mining high-utility itemsets with PEV pruning (Song et al. 2021).
67. **HUIM-ABC** - Artificial Bee Colony algorithm for high-utility itemset mining with bitmap calculations (Song & Huang 2018).

### High-Occupancy Itemset Mining (HOIM & HUOPM)
68. **HEP** - High-Efficiency algorithm for mining high-occupancy itemsets using UBO pruning (Deng 2020).
69. **FHOI** - Fast High-Occupancy Itemset Mining using Equivalence Classes and early candidate pruning (Nguyen et al. 2023).
70. **DFHOI** - Depth-First Search for High-Occupancy Itemset Mining using vertical equivalence class projection (Nguyen et al. 2023).
71. **TKHOIM** - Top-K High-Occupancy Itemset Miner using dynamic minO updating and LUBO strategy (Yildirim 2025).
72. **HOIMTO** - High-Occupancy Itemset Mining with consideration of Transaction Occupancy (Datta et al. 2022).
73. **NAM-HEP** - Adaptive High-Occupancy Itemset Mining using a hierarchical set-enumeration tree and median thresholds (Tran et al. 2025).
74. **OCEAN** - Fast discovery of high-utility occupancy itemsets using utility-lists and top-K upper bound pruning (Shen et al. 2016).
75. **HUOPM** - High-Utility Occupancy Pattern Mining using compact UO-lists, FU-tables, and early pruning (Gan et al. 2020).
76. **IL-HUOPM / HUOMIL** - High-Utility Occupancy Pattern Mining with Indexed List Structures and O(1) pointer chaining (Kim et al. 2023).
77. **MFHOI / Strong MFHOI-Miner** - Mines Frequent High-Occupancy Itemsets, Weak MFHOI, and Strong MFHOI with exact dominance filtering.
78. **MHOUI-Miner** - Mines Pareto-optimal High-Occupancy Utility Itemsets decoupling occupancy compactness from profit margins.
79. **CLOE-HOI** - Mines support-closed high-occupancy itemsets using candidate-free vertical bitset DFS and closure jumping.
80. **AURA-HOI** - Auditable high-occupancy support-class representatives using closure-canonical vertical bitsets and ledgering.
81. **CHUO-Miner** - Closed High-Utility Occupancy Itemsets using COU-lists, RUU utility pruning, and OUB occupancy bounds.

### Frequent Closed & Maximal Itemset Mining (FCIM / MFIM)
82. **A-Close** - Uses frequent itemset generators to derive closed itemsets via Galois closure connections (Pasquier et al. 1999).
83. **Close** - Mines association rules directly from closed itemset lattices (Pasquier et al. 1999).
84. **CLOSET** - FP-tree based closed itemset mining using frequent pattern growth (Pei et al. 2000).
85. **CLOSET+** - Extends CLOSET with hybrid tree traversals, upward/downward projection, and pseudo-projection (Wang et al. 2003).
86. **FPclose** - Extends FP-Growth with an Array-based prefix-tree checking structure for closed itemsets (Grahne & Zhu 2003).
87. **CHARM** - Vertical data format using diffsets and closure jumps for ultra-fast closed pattern extraction (Zaki & Hsiao 2002).
88. **DCI_CLOSED** - Highly optimized vertical bitset miner with order-preserving generators and duplicate avoidance (Lucchese et al. 2006).
89. **LCM / LCM ver. 2** - Linear time Closed itemset Miner using Prefix Preserving Closure Extension and Occurrence Deliver (Uno et al. 2004).
90. **NAFCP** - N-list-based algorithm for mining frequent closed patterns using the PPC-tree (Le & Vo 2015).
91. **FCFIA** - Fast Closed Frequent Itemsets Algorithm using PEP pruning and projection strategy (Ma & Qi 2008).
92. **CARPENTER** - Row-wise enumeration algorithm finding frequent closed patterns in high-dimensional biological datasets (Pan et al. 2003).
93. **DBV-Miner** - Dynamic Bit-Vector approach for fast mining of frequent closed itemsets using subsumption (Vo et al. 2012).
94. **dEFME** - Depth-First Minimal Pattern Mining enumerating free itemsets (generators) with polynomial delay (Soulet & Rioult 2014).
95. **Talky-G** - Vertical mining of frequent generators using reverse pre-order traversal and subsumption checks (Szathmary et al. 2009).
96. **TOUCH** - Combined vertical mining of frequent closures (Charm) and generators (Talky-G) (Szathmary et al. 2009).
97. **PASCAL** - Level-wise frequent pattern mining using pattern counting inference to eliminate database scans (Bastide et al. 2000).
98. **ZART** - Multifunctional miner discovering frequent itemsets, closed itemsets, and generators simultaneously (Szathmary et al. 2007).
99. **Max-Miner** - Breadth/depth search applying look-ahead superset pruning for maximal patterns (Bayardo 1998).
100. **GenMax** - Backtracking search using Progressive Focusing and vertical bitsets for maximal itemsets (Gouda & Zaki 2001).
101. **FPmax** - FP-growth extension with MFI-Tree for maximal frequent itemset discovery (Grahne & Zhu 2005).
102. **MAFIA** - Maximal Frequent Itemset Algorithm using Vertical Bitmaps, PEP, and HUTMFI pruning (Burdick et al. 2001).

### Sequential & Episode Pattern Mining
103. **PrefixSpan** - Mining sequential patterns efficiently by prefix-projected pattern growth (Pei et al. 2001).
104. **SPADE** - Fast sequential pattern mining using vertical ID-lists and combinatorial equivalence classes (Zaki 2001).
105. **USpan** - High-utility sequential pattern mining using projected databases and sequence-weighted utility bounds (Yin et al. 2012).
106. **HUP-Miner** - Mines high average-utility non-overlapping patterns from sequence utility databases (Geng et al. 2026).
107. **KCloTreeMiner** - Top-K closed sequential patterns over sequence databases with closed coverage absorption (Rizvee et al. 2025).
108. **THUE** - Top-K high-utility episodes from event sequences using RIU/RTU/RUC threshold raising (Wan et al. 2021).
109. **HUE-Span** - High-Utility Episode mining in complex event sequences using Episode-Weighted Utilization bounds (Fournier-Viger et al. 2019).
110. **UMEpi** - High-Utility Episode mining across complex event sequences using utility-lists and non-overlapping matching (Gan et al. 2023).

### Rare Itemset & Data Stream Mining
111. **Apriori-Rare** - Level-wise mining of minimal rare itemsets (mRIs) using a modified Apriori traversal (Szathmary et al. 2007).
112. **Apriori-Inverse** - Vertical mining of perfectly sporadic itemsets based on inverted downward closure (Koh & Rountree 2005).
113. **CORI** - Correlation mining pushing monotone (rarity) and anti-monotone (bond) constraints simultaneously (Bouasker & Ben Yahia 2015).
114. **RP-Tree** - Tree-based mining of rare-item patterns using dual support thresholds (Tsang et al. 2011).
115. **Walky-G** - Depth-first mining of minimal rare itemsets over frequent generators (Szathmary et al. 2012).
116. **estDec** - Finding recent frequent itemsets adaptively over online data streams using decay factor aging (Chang & Lee 2003).
117. **CloStream** - Incremental maintenance of frequent closed itemsets over data streams using inverted indexing (Yen et al. 2009).
118. **CFI-Stream** - Mining Closed Frequent Itemsets in data streams using sliding bit-sequences (Jiang & Gruenwald 2006).
119. **FHMDS** - Fast Top-K High Utility Itemset Mining from data streams under the sliding window model (Dawar et al. 2017).

### Uncertain, Fuzzy & Specialized Pattern Mining
120. **U-Apriori** - Mining frequent itemsets from existential uncertain data using expected support (Chui et al. 2007).
121. **FFI-Miner** - Fast algorithm for mining fuzzy frequent itemsets from quantitative databases (Lin et al. 2015).
122. **UBMFFP-Tree** - Upper-bound Multiple Fuzzy Frequent Pattern Tree for multiple fuzzy itemset mining (Lin et al. 2015).
123. **PSO Classifier** - Particle Swarm based classification-rule discovery using continuous CPSO and rule pruning (Sousa et al. 2004).
124. **HUPP-Miner** - High-Utility Prompt Pattern mining from LLM prompt interaction logs with PTWO and AAUB alignment pruning.
125. **HIEP-Miner** - High-Information Entropy Pattern mining for token sequences using Shannon self-information and IWRU branch-and-bound.

### Subword Tokenization & Framework Utilities
126. **Reps Maximal-Munch Scanner** - Linear-time maximal-munch tokenization using DFA configurations and failed-state tabulation (Reps 1998).
127. **Sennrich BPE Subword** - Byte-Pair Encoding subword learning and application for open-vocabulary translation (Sennrich et al. 2016).
128. **BPE-Dropout** - Stochastic BPE subword regularization randomly dropping merges during training-time segmentation (Provilkov et al. 2020).
129. **Kudo Unigram Subword Regularization** - Unigram language-model subword segmentation with EM training and Viterbi decoding (Kudo 2018).
130. **SentencePiece Lite** - SentencePiece-style raw-sentence training with lossless whitespace escaping (Kudo & Richardson 2018).
131. **Fast WordPiece / LinMaxMatch** - Linear-time WordPiece tokenization using a trie with failure links and failure pops (Song et al. 2021).
132. **VOLT** - Vocabulary Learning via Optimal Transport finding optimal vocabulary size via Sinkhorn iterations (Xu et al. 2021).
133. **Tokenizer Lab** - Tokenizer domain adaptation toolkit: pre-tokenization sweeps, bytes-per-token, and Renyi entropy.
134. **Grapheme Pair Encoding** - Egalitarian script tokenization using Unicode grapheme clusters as atomic units.
135. **Parity-aware BPE** - Cross-lingual fairness tokenization optimizing compression parity across languages.
136. **FARO Tokenizer** - Frequency-Aware Robust Out-of-vocabulary Subword Tokenizer dynamically managing vocabulary size.
137. **HUST-Tokenize** - Mining-driven subword tokenization engine integrating downstream pattern utility metrics into piece merging.
138. **LAGA** - Self-contained Layout-Aware Generative Architecture for synthetic transaction, utility, sequence, and tabular data.

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
* **`<algorithm>` / direct subcommand**: `ais`, `apriori`, `eclat`, `fpgrowth`, `tree_projection`, `aclose`, `close`, `closet`, `closetplus`, `fpclose`, `charm`, `dci_closed`, `lcm`, `lcmver2`, `nafcp`, `fcfia`, `max_miner`, `genmax`, `fpmax`, `mafia`, `hep`, `fhoi`, `dfhoi`, `tkhoim`, `hoimto`, `mfhoi`, `fhoi_miner`, `weak_mfhoi_miner`, `strong_mfhoi_miner`, `mhoui`, `houi_miner`, `weak_mhoui_miner`, `strong_mhoui_miner`, `direct_mhoui_miner`, `regular_mine`, `negfin`, `prepost`, `prepostplus`, `dic`, `ltm`, `sam`, `carpenter`, `dbv_miner`, `defme`, `talky_g`, `pascal`, `zart`, `apriori_rare`, `apriori_inverse`, `cori`, `rp_tree`, `estdec`, `clostream`, `cfi_stream`, `uapriori`, `msapriori`, `ffiminer`, `ubmffp`, `dfigrowth`, `sum`, `mlhui_miner`, `fchm`, `vhuqi`, `fhuqi_miner`, `tkq`, `thue`, `efim`, `efim_closed`, `ocean`, `huopm`, `il_huopm`, `huomil`, `mheinu`, `dphim`, `closed_fhuim_kinana`, `prefixspan`, `spade`, `uspan`, `hupspm`, `hup_miner`, `hiep`, `maximal_munch`, `bpe`, `bpe_dropout`, `unigram`, `sentencepiece`, `tokenizer_lab`, `gpe`, `parity_bpe`, `fast_wordpiece`, `resnet18`, `mobilenet_tiny`, `vit`, `swin`.
* **`ocean`** requires utility format `1`: `./build/bin/dm.exe ocean <utility_dataset> 1 <min_occupancy> [min_support]`, or standalone `./build/bin/ocean_miner.exe --input <utility_dataset> --minocc <value> [--minsup <ratio|count>]`. Mines high utility occupancy itemsets based on utility-lists and top-$k$ upper-bound pruning (Shen et al., DASFAA 2016).
* **`huopm`** requires utility format `1`: `./build/bin/dm.exe huopm <utility_dataset> 1 <min_occupancy> [min_support]`, or standalone `./build/bin/huopm_miner.exe --input <utility_dataset> --minocc <value> [--minsup <ratio|count>]`. Mines high-utility occupancy patterns using compact UO-lists, FU-tables, Strategy 3 early construction abandonment, and support-ascending FU-tree exploration (Gan et al., IEEE Transactions on Cybernetics 2020).
* **`il_huopm`** (alias **`huomil`**) requires utility format `1`: `./build/bin/dm.exe il_huopm <utility_dataset> 1 <min_occupancy> [min_support]`, or standalone `./build/bin/il_huopm_miner.exe --input <utility_dataset> --minocc <value> [--minsup <ratio|count>]`. Mines high utility occupancy patterns using GUO-IL and CUO-IL indexed lists, O(1) quadruple entry pointer chaining, and loose upper bound (lubuo) pruning (Kim et al., IEEE Access 2023).
* **`efim`** requires utility format `1`: `./build/bin/dm.exe efim <utility_dataset> 1 <min_utility>`. State-of-the-art HUIM using horizontal database projection, linear transaction merging, and Subtree Utility (SU) / Local Utility (LU) upper-bound pruning (Zida et al., KAIS 2017).
* **`efim_closed`** requires utility format `1`: `./build/bin/dm.exe efim_closed <utility_dataset> 1 <min_utility>`. Fast closed high-utility itemset mining using closure jumping, forward/backward closure checking, and transaction merging (Fournier-Viger et al., MLDM 2016).
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

The algorithms implemented in this framework strictly adhere to the logic, pruning theorems, and mathematical formulations established in the following foundational, peer-reviewed research papers. Every reference corresponds to a verified publication with an authoritative DOI or permanent proceedings repository link.

### Frequent Itemset Mining (FIM)
**[1]** R. Agrawal, T. Imieliński, and A. Swami, "Mining association rules between sets of items in large databases," in *Proc. 1993 ACM SIGMOD Int. Conf. on Management of Data*, 1993, pp. 207–216. DOI: [10.1145/170035.170072](https://doi.org/10.1145/170035.170072). *(AIS)*

**[2]** R. Agrawal and R. Srikant, "Fast algorithms for mining association rules in large databases," in *Proc. 20th Int. Conf. on Very Large Data Bases (VLDB)*, 1994, pp. 487–499. [VLDB Proceedings](http://www.vldb.org/conf/1994/P487.pdf). *(Apriori / AprioriTid / AprioriHybrid)*

**[3]** M. J. Zaki, "Scalable algorithms for association mining," *IEEE Transactions on Knowledge and Data Engineering*, vol. 12, no. 3, pp. 372–390, 2000. DOI: [10.1109/69.846293](https://doi.org/10.1109/69.846293). *(Eclat)*

**[4]** J. Han, J. Pei, and Y. Yin, "Mining frequent patterns without candidate generation," in *Proc. 2000 ACM SIGMOD Int. Conf. on Management of Data*, 2000, pp. 1–12. DOI: [10.1145/342009.335372](https://doi.org/10.1145/342009.335372). *(FP-Growth)*

**[5]** R. C. Agarwal, C. C. Aggarwal, and V. V. V. Prasad, "A tree projection algorithm for generation of frequent itemsets," *Journal of Parallel and Distributed Computing*, vol. 61, no. 3, pp. 350–371, 2001. DOI: [10.1006/jpdc.2000.1693](https://doi.org/10.1006/jpdc.2000.1693). *(TreeProjection)*

**[6]** Z.-H. Deng and S.-L. Lv, "Fast mining frequent itemsets using Nodesets," *Expert Systems with Applications*, vol. 41, no. 10, pp. 4505–4512, 2014. DOI: [10.1016/j.eswa.2014.01.025](https://doi.org/10.1016/j.eswa.2014.01.025). *(FIN)*

**[7]** Z.-H. Deng, "DiffNodesets: An efficient structure for fast mining frequent itemsets," *Applied Soft Computing*, vol. 45, pp. 104–115, 2016. DOI: [10.1016/j.asoc.2016.03.023](https://doi.org/10.1016/j.asoc.2016.03.023). *(FIN+ / dFIN)*

**[8]** N. Aryabarzan, B. Minaei-Bidgoli, and M. Teshnehlab, "negFIN: An efficient algorithm for fast mining frequent itemsets," *Expert Systems with Applications*, vol. 105, pp. 129–143, 2018. DOI: [10.1016/j.eswa.2018.03.042](https://doi.org/10.1016/j.eswa.2018.03.042). *(negFIN)*

**[9]** Z.-H. Deng, J. Wang, and J. Jiang, "A new algorithm for fast mining frequent itemsets using N-lists," *Science China Information Sciences*, vol. 55, no. 9, pp. 2008–2030, 2012. DOI: [10.1007/s11432-012-4638-z](https://doi.org/10.1007/s11432-012-4638-z). *(PrePost)*

**[10]** Z.-H. Deng and S.-L. Lv, "PrePost+: An efficient N-lists-based algorithm for mining frequent itemsets via Children–Parent Equivalence pruning," *Expert Systems with Applications*, vol. 42, no. 13, pp. 5424–5432, 2015. DOI: [10.1016/j.eswa.2015.02.046](https://doi.org/10.1016/j.eswa.2015.02.046). *(PrePost+)*

**[11]** S. Brin, R. Motwani, J. D. Ullman, and S. Tsur, "Dynamic itemset counting and implication rules for market basket data," in *Proc. 1997 ACM SIGMOD Int. Conf. on Management of Data*, 1997, pp. 255–264. DOI: [10.1145/253260.253325](https://doi.org/10.1145/253260.253325). *(DIC)*

**[12]** J. Lu, W. Xu, K. Zhou, and Z. Guo, "Frequent itemset mining algorithm based on linear table," *Journal of Database Management*, vol. 34, no. 1, pp. 1–14, 2023. DOI: [10.4018/jdm.318450](https://doi.org/10.4018/jdm.318450). *(LTM)*

**[13]** C. Borgelt and X. Wang, "SaM: A split and merge algorithm for fuzzy frequent item set mining," in *Proc. 2009 IEEE Int. Conf. on Fuzzy Systems*, 2009, pp. 255–268. DOI: [10.1109/FUZZY.2009.5277156](https://doi.org/10.1109/FUZZY.2009.5277156). *(SaM)*

**[14]** B. Liu, W. Hsu, and Y. Ma, "Mining association rules with multiple minimum supports," in *Proc. 5th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 1999, pp. 337–341. DOI: [10.1145/312129.312274](https://doi.org/10.1145/312129.312274). *(MSApriori)*

**[15]** J. Huang, Y.-P. Lai, C. Lo, and C.-W. Wu, "An efficient algorithm for deriving frequent itemsets from lossless condensed representation," in *Proc. 32nd Int. Conf. on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE)*, 2019, pp. 216–229. DOI: [10.1007/978-3-030-22999-3_20](https://doi.org/10.1007/978-3-030-22999-3_20). *(DFI-Growth)*

**[16]** J. Vreeken, M. van Leeuwen, and A. Siebes, "Krimp: mining itemsets that compress," *Data Mining and Knowledge Discovery*, vol. 23, no. 1, pp. 169–214, 2011. DOI: [10.1007/s10618-010-0202-x](https://doi.org/10.1007/s10618-010-0202-x). *(KRIMP)*

**[17]** K. Smets and J. Vreeken, "Slim: Directly mining descriptive patterns," in *Proc. 12th SIAM Int. Conf. on Data Mining (SDM)*, 2012, pp. 236–247. DOI: [10.1137/1.9781611972825.21](https://doi.org/10.1137/1.9781611972825.21). *(SLIM)*

**[18]** G. I. Webb and J. Vreeken, "Efficient discovery of the most interesting associations," *ACM Transactions on Knowledge Discovery from Data*, vol. 8, no. 3, article 15, 2014. DOI: [10.1145/2629587](https://doi.org/10.1145/2629587). *(OPUS Miner)*

**[19]** S. Ruggieri, "Frequent regular itemset mining," in *Proc. 16th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2010, pp. 263–272. DOI: [10.1145/1835804.1835840](https://doi.org/10.1145/1835804.1835840). *(RegularMine)*

**[20]** K. Malliaridis and S. Ougiaroglou, "Efficient techniques for retrieving top-K frequent itemsets," *Expert Systems with Applications*, vol. 311, article 131250, 2026. DOI: [10.1016/j.eswa.2026.131250](https://doi.org/10.1016/j.eswa.2026.131250). *(HTK-Miner)*

### Frequent Closed & Maximal Itemset Mining (FCIM / MFIM)
**[21]** N. Pasquier, Y. Bastide, R. Taouil, and L. Lakhal, "Discovering frequent closed itemsets for association rules," in *Proc. 7th Int. Conf. on Database Theory (ICDT)*, 1999, pp. 398–416. DOI: [10.1007/3-540-49257-7_25](https://doi.org/10.1007/3-540-49257-7_25). *(A-Close)*

**[22]** N. Pasquier, Y. Bastide, R. Taouil, and L. Lakhal, "Efficient mining of association rules using closed itemset lattices," *Information Systems*, vol. 24, no. 1, pp. 25–46, 1999. DOI: [10.1016/S0306-4379(99)00017-4](https://doi.org/10.1016/S0306-4379(99)00017-4). *(Close)*

**[23]** J. Pei, J. Han, and R. Mao, "CLOSET: An efficient algorithm for mining frequent closed itemsets," in *Proc. 2000 ACM SIGMOD Workshop on Research Issues in Data Mining and Knowledge Discovery*, 2000, pp. 21–30. [ACM DL](https://dl.acm.org/doi/10.1145/345910.345914). *(CLOSET)*

**[24]** J. Wang, J. Han, and J. Pei, "CLOSET+: Searching for the best strategies for mining frequent closed itemsets," in *Proc. 9th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2003, pp. 236–245. DOI: [10.1145/956750.956779](https://doi.org/10.1145/956750.956779). *(CLOSET+)*

**[25]** G. Grahne and J. Zhu, "Efficiently using prefix-trees in mining frequent itemsets," in *Proc. IEEE ICDM Workshop on Frequent Itemset Mining Implementations (FIMI)*, CEUR-WS vol. 90, 2003. [CEUR-WS](http://ceur-ws.org/Vol-90/grahne.pdf). *(FPclose)*

**[26]** M. J. Zaki and C.-J. Hsiao, "CHARM: An efficient algorithm for closed itemset mining," in *Proc. 2nd SIAM Int. Conf. on Data Mining (SDM)*, 2002, pp. 457–473. DOI: [10.1137/1.9781611972726.27](https://doi.org/10.1137/1.9781611972726.27). *(CHARM)*

**[27]** C. Lucchese, S. Orlando, and R. Perego, "Fast and memory efficient mining of frequent closed itemsets," *IEEE Transactions on Knowledge and Data Engineering*, vol. 18, no. 1, pp. 21–36, 2006. DOI: [10.1109/TKDE.2006.10](https://doi.org/10.1109/TKDE.2006.10). *(DCI_CLOSED)*

**[28]** T. Uno, M. Kiyomi, and H. Arimura, "LCM ver. 2: Efficient mining algorithms for frequent/closed/maximal itemsets," in *Proc. IEEE ICDM Workshop on Frequent Itemset Mining Implementations (FIMI)*, CEUR-WS vol. 126, 2004. [CEUR-WS](http://ceur-ws.org/Vol-126/uno.pdf). *(LCM / LCM ver. 2)*

**[29]** T. Le and B. Vo, "An N-list-based algorithm for mining frequent closed patterns," *Expert Systems with Applications*, vol. 42, no. 19, pp. 6748–6757, 2015. DOI: [10.1016/j.eswa.2015.04.053](https://doi.org/10.1016/j.eswa.2015.04.053). *(NAFCP)*

**[30]** L. Ma and Y. Qi, "An efficient algorithm for frequent closed itemsets mining," in *Proc. 2008 Int. Conf. on Computer Science and Software Engineering*, 2008, pp. 260–262. DOI: [10.1109/CSSE.2008.1407](https://doi.org/10.1109/CSSE.2008.1407). *(FCFIA)*

**[31]** F. Pan, G. Cong, A. K. H. Tung, J. Yang, and M. J. Zaki, "CARPENTER: Finding closed patterns in long biological datasets," in *Proc. 9th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2003, pp. 637–642. DOI: [10.1145/956750.956832](https://doi.org/10.1145/956750.956832). *(CARPENTER)*

**[32]** B. Vo, T.-P. Hong, and B. Le, "DBV-Miner: A dynamic bit-vector approach for fast mining frequent closed itemsets," *Expert Systems with Applications*, vol. 39, no. 8, pp. 7196–7206, 2012. DOI: [10.1016/j.eswa.2012.01.037](https://doi.org/10.1016/j.eswa.2012.01.037). *(DBV-Miner)*

**[33]** A. Soulet and F. Rioult, "Efficiently depth-first minimal pattern mining," in *Proc. 18th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD)*, 2014, pp. 28–39. DOI: [10.1007/978-3-319-06605-9_3](https://doi.org/10.1007/978-3-319-06605-9_3). *(dEFME)*

**[34]** L. Szathmary, P. Valtchev, A. Napoli, and R. Godin, "Efficient vertical mining of frequent closures and generators," in *Proc. 8th Int. Symp. on Intelligent Data Analysis (IDA)*, 2009, pp. 393–404. DOI: [10.1007/978-3-642-03915-7_34](https://doi.org/10.1007/978-3-642-03915-7_34). *(Talky-G / TOUCH)*

**[35]** Y. Bastide, N. Pasquier, R. Taouil, G. Stumme, and L. Lakhal, "Mining frequent patterns with counting inference," *ACM SIGKDD Explorations Newsletter*, vol. 2, no. 2, pp. 66–75, 2000. DOI: [10.1145/380995.381017](https://doi.org/10.1145/380995.381017). *(PASCAL)*

**[36]** L. Szathmary, A. Napoli, and S. O. Kuznetsov, "ZART: A multifunctional itemset mining algorithm," in *Proc. 5th Int. Conf. on Concept Lattices and Their Applications (CLA)*, CEUR-WS vol. 331, 2007, pp. 26–37. [CEUR-WS](http://ceur-ws.org/Vol-331/paper3.pdf). *(ZART)*

**[37]** R. J. Bayardo, "Efficiently mining long patterns from databases," in *Proc. 1998 ACM SIGMOD Int. Conf. on Management of Data*, 1998, pp. 85–93. DOI: [10.1145/276304.276313](https://doi.org/10.1145/276304.276313). *(Max-Miner)*

**[38]** K. Gouda and M. J. Zaki, "Efficiently mining maximal frequent itemsets," in *Proc. 2001 IEEE Int. Conf. on Data Mining (ICDM)*, 2001, pp. 163–170. DOI: [10.1109/ICDM.2001.989518](https://doi.org/10.1109/ICDM.2001.989518). *(GenMax)*

**[39]** G. Grahne and J. Zhu, "Fast algorithms for frequent itemset mining using FP-trees," *IEEE Transactions on Knowledge and Data Engineering*, vol. 17, no. 1, pp. 81–90, 2005. DOI: [10.1109/TKDE.2005.11](https://doi.org/10.1109/TKDE.2005.11). *(FPmax)*

**[40]** D. Burdick, M. Calimlim, and J. Gehrke, "MAFIA: A maximal frequent itemset algorithm for transactional databases," in *Proc. 17th Int. Conf. on Data Engineering (ICDE)*, 2001, pp. 443–452. DOI: [10.1109/ICDE.2001.914857](https://doi.org/10.1109/ICDE.2001.914857). *(MAFIA)*

### High-Utility Itemset Mining (HUIM)
**[41]** Y. Liu, W. Liao, and A. Choudhary, "A two-phase algorithm for fast discovery of high utility itemsets," in *Proc. 9th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD)*, 2005, pp. 689–695. DOI: [10.1007/11430919_79](https://doi.org/10.1007/11430919_79). *(Two-Phase)*

**[42]** M. Liu and J. Qu, "Mining high utility itemsets without candidate generation," in *Proc. 21st ACM Int. Conf. on Information and Knowledge Management (CIKM)*, 2012, pp. 55–64. DOI: [10.1145/2396761.2396773](https://doi.org/10.1145/2396761.2396773). *(HUI-Miner)*

**[43]** P. Fournier-Viger, C.-W. Wu, S. Zida, and V. S. Tseng, "FHM: Faster high-utility itemset mining using estimated utility co-occurrence pruning," in *Proc. 21st Int. Symp. on Methodologies for Intelligent Systems (ISMIS)*, 2014, pp. 83–92. DOI: [10.1007/978-3-319-08326-1_9](https://doi.org/10.1007/978-3-319-08326-1_9). *(FHM)*

**[44]** S. Zida, P. Fournier-Viger, J. C.-W. Lin, C.-W. Wu, and V. S. Tseng, "EFIM: a highly efficient algorithm for mining high-utility itemsets," *Knowledge and Information Systems*, vol. 51, no. 2, pp. 595–609, 2017. DOI: [10.1007/s10115-016-0986-0](https://doi.org/10.1007/s10115-016-0986-0). *(EFIM)*

**[45]** P. Fournier-Viger, S. Zida, J. C.-W. Lin, C.-W. Wu, and V. S. Tseng, "EFIM-Closed: Fast and memory efficient discovery of closed high-utility itemsets," in *Proc. 12th Int. Conf. on Machine Learning and Data Mining in Pattern Recognition (MLDM)*, 2016, pp. 199–213. DOI: [10.1007/978-3-319-41920-6_16](https://doi.org/10.1007/978-3-319-41920-6_16). *(EFIM-Closed)*

**[46]** C.-W. Wu, P. Fournier-Viger, J.-Y. Gu, and V. S. Tseng, "Mining closed high utility itemsets without candidate generation," in *Proc. IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC)*, 2015, pp. 199–204. DOI: [10.1109/SMC.2015.42](https://doi.org/10.1109/SMC.2015.42). *(CHUI-Miner)*

**[47]** V. S. Tseng, C.-W. Wu, B.-E. Shie, and P. S. Yu, "UP-Growth: An efficient algorithm for high utility itemset mining," in *Proc. 16th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2010, pp. 253–262. DOI: [10.1145/1835804.1835837](https://doi.org/10.1145/1835804.1835837). *(UP-Growth)*

**[48]** C. F. Ahmed, S. K. Tanbeer, B.-S. Jeong, and Y.-K. Lee, "Efficient tree structures for high utility pattern mining in incremental databases," *IEEE Transactions on Knowledge and Data Engineering*, vol. 21, no. 12, pp. 1708–1721, 2009. DOI: [10.1109/TKDE.2009.46](https://doi.org/10.1109/TKDE.2009.46). *(IHUP)*

**[49]** Z. Cheng, W. Fang, W. Shen, J. C.-W. Lin, and B. Yuan, "An efficient utility-list based high-utility itemset mining algorithm," *Applied Intelligence*, vol. 53, no. 6, pp. 6992–7006, 2023. DOI: [10.1007/s10489-022-03823-7](https://doi.org/10.1007/s10489-022-03823-7). *(HUIM-SU)*

**[50]** Q.-H. Duong, P. Fournier-Viger, H. Ramampiaro, K. Nørvåg, and T.-L. Dam, "Efficient high utility itemset mining using buffered utility-lists," *Applied Intelligence*, vol. 48, no. 5, pp. 1167–1187, 2018. DOI: [10.1007/s10489-017-1057-2](https://doi.org/10.1007/s10489-017-1057-2). *(ULB-Miner)*

**[51]** S. Dawar, V. Goyal, and D. Bera, "A hybrid framework for mining high-utility itemsets in a sparse transaction database," *Applied Intelligence*, vol. 47, no. 3, pp. 809–827, 2017. DOI: [10.1007/s10489-017-0932-1](https://doi.org/10.1007/s10489-017-0932-1). *(UFH)*

**[52]** J. Sahoo, A. K. Das, and A. Goswami, "An efficient approach for mining association rules from high utility itemsets," *Expert Systems with Applications*, vol. 42, no. 13, pp. 5754–5778, 2015. DOI: [10.1016/j.eswa.2015.03.018](https://doi.org/10.1016/j.eswa.2015.03.018). *(HUCI-Miner)*

**[53]** S. Dawar and V. Goyal, "UP-Hist tree: an efficient data structure for mining high utility patterns from transaction databases," in *Proc. 19th Int. Database Engineering & Applications Symp. (IDEAS)*, 2015, pp. 56–61. DOI: [10.1145/2783446.2783584](https://doi.org/10.1145/2783446.2783584). *(UP-Hist)*

**[54]** P. Sra and S. Chand, "A residual utility-based concept for high-utility itemset mining," *Knowledge and Information Systems*, vol. 66, no. 1, pp. 211–235, 2024. DOI: [10.1007/s10115-023-01948-w](https://doi.org/10.1007/s10115-023-01948-w). *(R-Miner)*

**[55]** P. Sra and S. Chand, "A reinduction-based approach for efficient high utility itemset mining from incremental datasets," *Data Science and Engineering*, vol. 9, no. 1, pp. 73–87, 2024. DOI: [10.1007/s41019-023-00229-4](https://doi.org/10.1007/s41019-023-00229-4). *(SUM)*

**[56]** P. Fournier-Viger, Y. Wang, J. C.-W. Lin, J. M. Luna, and S. Ventura, "Mining cross-level high utility itemsets," in *Proc. 33rd Int. Conf. on Industrial, Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE)*, 2020, pp. 1–13. DOI: [10.1007/978-3-030-55789-8_1](https://doi.org/10.1007/978-3-030-55789-8_1). *(CLH-Miner)*

**[57]** L. Cagliero, P. Garza, and E. Baralis, "Discovering high-utility itemsets at multiple abstraction levels," in *Proc. 21st East-European Conf. on Advances in Databases and Information Systems (ADBIS)*, 2017, pp. 201–215. DOI: [10.1007/978-3-319-66917-5_14](https://doi.org/10.1007/978-3-319-66917-5_14). *(MLHUI-Miner)*

**[58]** P. Fournier-Viger, J. C.-W. Lin, T. Dinh, and H. B. Le, "Mining correlated high-utility itemsets using the bond measure," in *Proc. 11th Int. Conf. on Hybrid Artificial Intelligence Systems (HAIS)*, 2016, pp. 108–120. DOI: [10.1007/978-3-319-32034-2_10](https://doi.org/10.1007/978-3-319-32034-2_10). *(FCHM)*

**[59]** P. Fournier-Viger, "FHN: Efficient mining of high-utility itemsets with negative unit profits," in *Proc. 11th Int. Conf. on Hybrid Artificial Intelligence Systems (HAIS)*, 2016, pp. 96–107. DOI: [10.1007/978-3-319-32034-2_9](https://doi.org/10.1007/978-3-319-32034-2_9). *(FHN)*

**[60]** P. Fournier-Viger and S. Zida, "FOSHU: Faster on-shelf high utility itemset mining – with or without negative unit profit," in *Proc. 30th Annual ACM Symp. on Applied Computing (SAC)*, 2015, pp. 845–850. DOI: [10.1145/2695664.2695786](https://doi.org/10.1145/2695664.2695786). *(FOSHU)*

**[61]** G.-C. Lan, T.-P. Hong, J.-P. Huang, and V. S. Tseng, "On-shelf utility mining with negative item values," *Expert Systems with Applications*, vol. 41, no. 7, pp. 3450–3459, 2014. DOI: [10.1016/j.eswa.2013.11.025](https://doi.org/10.1016/j.eswa.2013.11.025). *(TS-HOUN)*

**[62]** J. C.-W. Lin, W. Gan, T.-P. Hong, and J.-S. Pan, "Incrementally updating high-utility itemsets with transaction insertion," in *Proc. 10th Int. Conf. on Advanced Data Mining and Applications (ADMA)*, 2014, pp. 44–56. DOI: [10.1007/978-3-319-14717-8_4](https://doi.org/10.1007/978-3-319-14717-8_4). *(HUI-list-INS)*

**[63]** C.-W. Wu, B.-E. Shie, P. S. Yu, and V. S. Tseng, "Mining top-k high utility itemsets," in *Proc. 18th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2012, pp. 78–86. DOI: [10.1145/2339530.2339546](https://doi.org/10.1145/2339530.2339546). *(TKU-Miner)*

**[64]** S. Krishnamoorthy, "Mining top-k high utility itemsets with effective threshold raising strategies," *Expert Systems with Applications*, vol. 117, pp. 148–165, 2019. DOI: [10.1016/j.eswa.2018.09.028](https://doi.org/10.1016/j.eswa.2018.09.028). *(THUI)*

**[65]** W. Song, C. Zheng, C. Huang, and L. Liu, "Heuristically mining the top-k high-utility itemsets with cross-entropy optimization," *Applied Intelligence*, vol. 51, no. 12, pp. 8864–8881, 2021. DOI: [10.1007/s10489-021-02576-z](https://doi.org/10.1007/s10489-021-02576-z). *(TKU-CE / TKU-CE+)*

**[66]** S. Carstensen and J. C.-W. Lin, "TKU-PSO: An efficient particle swarm optimization model for top-k high-utility itemset mining," *International Journal of Interactive Multimedia and Artificial Intelligence*, vol. 9, no. 4, pp. 70–81, 2025. DOI: [10.9781/ijimai.2024.01.002](https://doi.org/10.9781/ijimai.2024.01.002). *(TKU-PSO)*

**[67]** X. Zhang, B. Liu, J. Chen, S. Yang, Z. Gu, Z. Liu, and S. Xu, "TOPKPHM: Periodic patterns mining of top-k high utility itemsets," in *Proc. 2025 Int. Conf. on Trustworthy Big Data and Artificial Intelligence (ICTBAI)*, 2025, pp. 39–44. DOI: [10.1109/ICTBAI68361.2025.00009](https://doi.org/10.1109/ICTBAI68361.2025.00009). *(TOPKPHM)*

**[68]** G. Kimura, Y. Hayamizu, R. U. Kiran, M. Kitsuregawa, and K. Goda, "DPHIM: Efficient parallel mining of high-utility itemsets on multicore processors and its evaluation," *IEEE Transactions on Knowledge and Data Engineering*, vol. 38, no. 5, pp. 2714–2730, 2026. DOI: [10.1109/TKDE.2026.3666851](https://doi.org/10.1109/TKDE.2026.3666851). *(DPHIM)*

**[69]** K. S. Sulanjari and C. Fatichah, "Optimized closed frequent high utility itemset mining using OSR, OWL, and MSU pruning on retail transaction data," *JUTI: Jurnal Ilmiah Teknologi Informasi*, vol. 24, no. 1, pp. 1–15, 2026. DOI: [10.12962/j24068535.v24i1.a1311](https://doi.org/10.12962/j24068535.v24i1.a1311). *(Closed-FHUIM-Kinana)*

**[70]** J. C.-W. Lin, T. Li, P. Fournier-Viger, T.-P. Hong, J. Zhan, and M. Voznak, "An efficient algorithm to mine high average-utility itemsets," *Advanced Engineering Informatics*, vol. 30, no. 2, pp. 233–243, 2016. DOI: [10.1016/j.aei.2016.03.003](https://doi.org/10.1016/j.aei.2016.03.003). *(HAUI-Miner)*

**[71]** J. C.-W. Lin, S. Ren, P. Fournier-Viger, and T.-P. Hong, "EHAUPM: Efficient high average-utility pattern mining with tighter upper bounds," *IEEE Access*, vol. 5, pp. 12927–12940, 2017. DOI: [10.1109/ACCESS.2017.2721950](https://doi.org/10.1109/ACCESS.2017.2721950). *(EHAUPM)*

**[72]** W. Song, L. Liu, and C. Huang, "Generalized maximal utility for mining high average-utility itemsets," *Knowledge and Information Systems*, vol. 63, no. 11, pp. 2947–2967, 2021. DOI: [10.1007/s10115-021-01614-z](https://doi.org/10.1007/s10115-021-01614-z). *(HAUIM-GMU)*

**[73]** J. C.-W. Lin, S. Ren, and P. Fournier-Viger, "MEMU: More efficient algorithm to mine high average-utility patterns with multiple minimum average-utility thresholds," *IEEE Access*, vol. 6, pp. 7593–7609, 2018. DOI: [10.1109/access.2018.2801261](https://doi.org/10.1109/access.2018.2801261). *(MEMU)*

**[74]** I. Yildirim, "Mining high-efficiency itemsets with negative utilities," *Mathematics*, vol. 13, no. 4, article 659, 2025. DOI: [10.3390/math13040659](https://doi.org/10.3390/math13040659). *(MHEINU)*

**[75]** C. H. Li, C.-W. Wu, and V. S. Tseng, "Efficient vertical mining of high utility quantitative itemsets," in *Proc. 2014 IEEE Int. Conf. on Granular Computing (GrC)*, 2014, pp. 155–160. DOI: [10.1109/GrC.2014.6982823](https://doi.org/10.1109/GrC.2014.6982823). *(VHUQI)*

**[76]** M. Nouioua, P. Fournier-Viger, C.-W. Wu, J. C.-W. Lin, and W. Gan, "FHUQI-Miner: Fast high utility quantitative itemset mining," *Applied Intelligence*, vol. 51, no. 9, pp. 6581–6603, 2021. DOI: [10.1007/s10489-021-02204-w](https://doi.org/10.1007/s10489-021-02204-w). *(FHUQI-Miner)*

**[77]** M. Nouioua, P. Fournier-Viger, W. Gan, Y. Wu, J. C.-W. Lin, and F. Nouioua, "TKQ: Top-K quantitative high utility itemset mining," in *Proc. 17th Int. Conf. on Advanced Data Mining and Applications (ADMA)*, 2021, pp. 15–28. DOI: [10.1007/978-3-030-95408-6_2](https://doi.org/10.1007/978-3-030-95408-6_2). *(TKQ)*

**[78]** Y.-I. Chang, P.-C. Chuang, Y.-H. Liao, P.-Y. Hu, and T.-W. Chen, "An efficient algorithm for mining top-k high-on-shelf-utility itemsets with positive/negative profits of local/global minimum count," *Engineering Proceedings*, vol. 108, no. 1, article 45, 2025. DOI: [10.3390/engproc2025108045](https://doi.org/10.3390/engproc2025108045). *(TIPN-HOUI-Miner)*

**[79]** S. Nawaz, J. C.-W. Lin, and P. Fournier-Viger, "Mining high utility itemsets with hill climbing and simulated annealing," *ACM Transactions on Management Information Systems*, vol. 12, no. 4, article 33, 2021. DOI: [10.1145/3462636](https://doi.org/10.1145/3462636). *(HUIM-HC / HUIM-SA)*

**[80]** J. C.-W. Lin, L. Yang, P. Fournier-Viger, T.-P. Hong, and M. Voznak, "A binary PSO approach to mine high-utility itemsets," *Soft Computing*, vol. 20, no. 7, pp. 2645–2655, 2016. DOI: [10.1007/s00500-016-2106-1](https://doi.org/10.1007/s00500-016-2106-1). *(HUIM-BPSO-tree)*

**[81]** W. Song and C. Huang, "Mining high utility itemsets using bio-inspired algorithms: A diverse optimal value framework," *IEEE Access*, vol. 6, pp. 19568–19582, 2018. DOI: [10.1109/access.2018.2819162](https://doi.org/10.1109/access.2018.2819162). *(Bio-HUIF)*

**[82]** W. Song, J. Li, and C. Huang, "Artificial fish swarm algorithm for mining high utility itemsets," in *Proc. 12th Int. Conf. on Advances in Swarm Intelligence (ICSI)*, 2021, pp. 407–419. DOI: [10.1007/978-3-030-78811-7_36](https://doi.org/10.1007/978-3-030-78811-7_36). *(HUIM-AFSA)*

**[83]** W. Song and C. Huang, "Discovering high-utility itemsets based on the artificial bee colony algorithm," in *Proc. 22nd Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD)*, 2018, pp. 240–252. DOI: [10.1007/978-3-319-93034-3_20](https://doi.org/10.1007/978-3-319-93034-3_20). *(HUIM-ABC)*

### High-Occupancy Itemset Mining (HOIM & HUOPM)
**[84]** Z.-H. Deng, "Mining high occupancy itemsets," *Future Generation Computer Systems*, vol. 102, pp. 222–229, 2020. DOI: [10.1016/j.future.2019.07.039](https://doi.org/10.1016/j.future.2019.07.039). *(HEP)*

**[85]** L. T. T. Nguyen, T. Mai, G.-H. Pham, U. Yun, and B. Vo, "An efficient method for mining high occupancy itemsets based on equivalence class and early pruning," *Knowledge-Based Systems*, vol. 270, article 110441, 2023. DOI: [10.1016/j.knosys.2023.110441](https://doi.org/10.1016/j.knosys.2023.110441). *(FHOI / DFHOI)*

**[86]** I. Yildirim, "Mining top-k high occupancy itemsets," *Black Sea Journal of Engineering and Science*, vol. 8, no. 6, pp. 1723–1730, 2025. DOI: [10.46570/bjd.1481169](https://doi.org/10.46570/bjd.1481169). *(TKHOIM)*

**[87]** S. Datta, K. Mali, and U. Ghosh, "High occupancy itemset mining with consideration of transaction occupancy," *Arabian Journal for Science and Engineering*, vol. 47, no. 2, pp. 2061–2075, 2022. DOI: [10.1007/s13369-021-06075-8](https://doi.org/10.1007/s13369-021-06075-8). *(HOIMTO)*

**[88]** T.-N. Tran, V. T. Hoang, T.-C. Truong, and M. Voznak, "A hierarchical set-enumeration tree enabling high occupancy item set mining and the use of an adaptive occupancy threshold," *Applied Intelligence*, vol. 55, no. 2, pp. 205–225, 2025. DOI: [10.1007/s10489-024-05891-3](https://doi.org/10.1007/s10489-024-05891-3). *(NAM-HEP)*

**[89]** B. Shen, Z. Wen, Y. Zhao, D. Zhou, and W. Zheng, "OCEAN: Fast discovery of high utility occupancy itemsets," in *Proc. 21st Int. Conf. on Database Systems for Advanced Applications (DASFAA)*, 2016, pp. 355–369. DOI: [10.1007/978-3-319-32025-0_22](https://doi.org/10.1007/978-3-319-32025-0_22). *(OCEAN)*

**[90]** W. Gan, J. C.-W. Lin, P. Fournier-Viger, H.-C. Chao, and P. S. Yu, "HUOPM: High-utility occupancy pattern mining," *IEEE Transactions on Cybernetics*, vol. 50, no. 3, pp. 1195–1208, 2020. DOI: [10.1109/tcyb.2019.2896267](https://doi.org/10.1109/tcyb.2019.2896267). *(HUOPM)*

**[91]** H. Kim, T. Ryu, C. Lee, S. Kim, B. Vo, J. C.-W. Lin, and U. Yun, "Efficient method for mining high utility occupancy patterns based on indexed list structure," *IEEE Access*, vol. 11, pp. 45053–45071, 2023. DOI: [10.1109/ACCESS.2023.3271864](https://doi.org/10.1109/ACCESS.2023.3271864). *(IL-HUOPM / HUOMIL)*

### Sequential & Episode Pattern Mining
**[92]** J. Pei, J. Han, B. Mortazavi-Asl, H. Pinto, Q. Chen, U. Dayal, and M.-C. Hsu, "PrefixSpan: Mining sequential patterns efficiently by prefix-projected pattern growth," in *Proc. 17th Int. Conf. on Data Engineering (ICDE)*, 2001, pp. 215–224. DOI: [10.1109/ICDE.2001.914830](https://doi.org/10.1109/ICDE.2001.914830). *(PrefixSpan)*

**[93]** M. J. Zaki, "SPADE: An efficient algorithm for mining frequent sequences," *Machine Learning*, vol. 42, no. 1–2, pp. 31–60, 2001. DOI: [10.1023/A:1007652502315](https://doi.org/10.1023/A:1007652502315). *(SPADE)*

**[94]** J. Yin, Z. Zheng, and L. Cao, "USpan: An efficient algorithm for high utility sequential pattern mining," in *Proc. 18th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2012, pp. 660–668. DOI: [10.1145/2339530.2339649](https://doi.org/10.1145/2339530.2339649). *(USpan)*

**[95]** M. Geng, Y. Wu, Y. Li, J. Liu, L. Guo, X. Zhu, and X. Wu, "Mining high average utility nonoverlapping patterns from sequential database," *ACM Transactions on Intelligent Systems and Technology*, vol. 17, no. 1, article 15, 2026. DOI: [10.1145/3773899](https://doi.org/10.1145/3773899). *(HUP-Miner)*

**[96]** R. A. Rizvee, C. F. Ahmed, and C. K. Leung, "A tree-based framework to mine top-K closed sequential patterns," *Applied Intelligence*, vol. 55, no. 3, article 221, 2025. DOI: [10.1007/s10489-024-06137-y](https://doi.org/10.1007/s10489-024-06137-y). *(KCloTreeMiner)*

**[97]** S. Wan, J. Chen, W. Gan, G. Chen, and V. Goyal, "THUE: Discovering top-K high utility episodes," in *Proc. 2021 IEEE Int. Conf. on Big Data (Big Data)*, 2021, pp. 583–590. DOI: [10.1109/BigData52589.2021.9671853](https://doi.org/10.1109/BigData52589.2021.9671853). *(THUE)*

**[98]** P. Fournier-Viger, Y. Chen, P. Nouioua, and J. C.-W. Lin, "Mining high-utility episodes in a complex event sequence," in *Proc. 15th Int. Conf. on Advanced Data Mining and Applications (ADMA)*, 2019, pp. 153–167. DOI: [10.1007/978-3-030-35231-8_12](https://doi.org/10.1007/978-3-030-35231-8_12). *(HUE-Span)*

**[99]** W. Gan, J. C.-W. Lin, H.-C. Chao, and P. S. Yu, "Utility mining across complex event sequences," *IEEE Transactions on Artificial Intelligence*, vol. 4, no. 5, pp. 1148–1159, 2023. DOI: [10.1109/TAI.2022.3223965](https://doi.org/10.1109/TAI.2022.3223965). *(UMEpi)*

### Rare Itemset & Data Stream Mining
**[100]** L. Szathmary, A. Napoli, and P. Valtchev, "Towards rare itemset mining," in *Proc. 19th IEEE Int. Conf. on Tools with Artificial Intelligence (ICTAI)*, 2007, vol. 1, pp. 305–312. DOI: [10.1109/ICTAI.2007.45](https://doi.org/10.1109/ICTAI.2007.45). *(Apriori-Rare)*

**[101]** Y. S. Koh and N. Rountree, "Finding sporadic rules using Apriori-Inverse," in *Proc. 9th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD)*, 2005, pp. 97–106. DOI: [10.1007/11430919_14](https://doi.org/10.1007/11430919_14). *(Apriori-Inverse)*

**[102]** S. Bouasker and S. Ben Yahia, "Key correlation mining by simultaneous monotone and anti-monotone constraints checking," in *Proc. 30th Annual ACM Symp. on Applied Computing (SAC)*, 2015, pp. 851–856. DOI: [10.1145/2695664.2695847](https://doi.org/10.1145/2695664.2695847). *(CORI)*

**[103]** S. Tsang, Y. S. Koh, and G. Dobbie, "RP-Tree: Rare pattern tree mining," in *Proc. 13th Int. Conf. on Data Warehousing and Knowledge Discovery (DaWaK)*, 2011, pp. 277–288. DOI: [10.1007/978-3-642-23544-3_21](https://doi.org/10.1007/978-3-642-23544-3_21). *(RP-Tree)*

**[104]** L. Szathmary, P. Valtchev, A. Napoli, and R. Godin, "Efficient vertical mining of minimal rare itemsets," in *Proc. 9th Int. Conf. on Concept Lattices and Their Applications (CLA)*, CEUR-WS vol. 972, 2012, pp. 269–280. [CEUR-WS](http://ceur-ws.org/Vol-972/paper23.pdf). *(Walky-G)*

**[105]** J. H. Chang and W. S. Lee, "Finding recent frequent itemsets adaptively over online data streams," in *Proc. 9th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2003, pp. 487–492. DOI: [10.1145/956750.956807](https://doi.org/10.1145/956750.956807). *(estDec)*

**[106]** S.-J. Yen, Y.-S. Lee, C.-W. Wu, and C.-L. Lin, "An efficient algorithm for maintaining frequent closed itemsets over data stream," in *Proc. 22nd Int. Conf. on Industrial Engineering and Other Applications of Applied Intelligent Systems (IEA/AIE)*, 2009, pp. 767–776. DOI: [10.1007/978-3-642-02568-6_79](https://doi.org/10.1007/978-3-642-02568-6_79). *(CloStream)*

**[107]** N. Jiang and L. Gruenwald, "CFI-Stream: Mining closed frequent itemsets in data streams," in *Proc. 12th ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining*, 2006, pp. 592–597. DOI: [10.1145/1150402.1150473](https://doi.org/10.1145/1150402.1150473). *(CFI-Stream)*

**[108]** S. Dawar, V. Sharma, and V. Goyal, "Mining top-k high-utility itemsets from a data stream under sliding window model," *Applied Intelligence*, vol. 47, no. 4, pp. 1240–1255, 2017. DOI: [10.1007/s10489-017-0939-7](https://doi.org/10.1007/s10489-017-0939-7). *(FHMDS)*

### Uncertain, Fuzzy & Classification Data Mining
**[109]** C.-K. Chui, B. Kao, and E. Hung, "Mining frequent itemsets from uncertain data," in *Proc. 11th Pacific-Asia Conf. on Knowledge Discovery and Data Mining (PAKDD)*, 2007, pp. 47–58. DOI: [10.1007/978-3-540-71701-0_8](https://doi.org/10.1007/978-3-540-71701-0_8). *(U-Apriori)*

**[110]** J. C.-W. Lin, T. Li, P. Fournier-Viger, and T.-P. Hong, "A fast algorithm for mining fuzzy frequent itemsets," *Journal of Intelligent & Fuzzy Systems*, vol. 29, no. 6, pp. 2373–2379, 2015. DOI: [10.3233/IFS-151708](https://doi.org/10.3233/IFS-151708). *(FFI-Miner)*

**[111]** J. C.-W. Lin, T.-P. Hong, T.-C. Lin, and S.-T. Pan, "An UBMFFP tree for mining multiple fuzzy frequent itemsets," *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, vol. 23, no. 6, pp. 861–879, 2015. DOI: [10.1142/s0218488515500385](https://doi.org/10.1142/s0218488515500385). *(UBMFFP-Tree)*

**[112]** T. Sousa, A. Silva, and A. Neves, "Particle swarm based data mining algorithms for classification tasks," *Parallel Computing*, vol. 30, no. 5–6, pp. 767–783, 2004. DOI: [10.1016/j.parco.2003.12.015](https://doi.org/10.1016/j.parco.2003.12.015). *(PSO Classifier)*

### Subword Tokenization Algorithms
**[113]** T. Reps, "Maximal-munch tokenization in linear time," *ACM Transactions on Programming Languages and Systems*, vol. 20, no. 2, pp. 259–273, 1998. DOI: [10.1145/276393.276394](https://doi.org/10.1145/276393.276394). *(Reps Maximal-Munch Scanner)*

**[114]** R. Sennrich, B. Haddow, and A. Birch, "Neural machine translation of rare words with subword units," in *Proc. 54th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2016, pp. 1715–1725. DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162). *(Sennrich BPE Subword)*

**[115]** T. Kudo, "Subword regularization: Improving neural network translation models with multiple subword candidates," in *Proc. 56th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2018, pp. 66–75. DOI: [10.18653/v1/P18-1007](https://doi.org/10.18653/v1/P18-1007). *(Unigram Subword Regularization)*

**[116]** T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing," in *Proc. 2018 Conf. on Empirical Methods in Natural Language Processing (EMNLP): System Demonstrations*, 2018, pp. 66–71. DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012). *(SentencePiece Lite)*

**[117]** I. Provilkov, D. Emelianenko, and E. Voita, "BPE-Dropout: Simple and effective subword regularization," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 1882–1892. DOI: [10.18653/v1/2020.acl-main.170](https://doi.org/10.18653/v1/2020.acl-main.170). *(BPE-Dropout)*

**[118]** X. Song, A. Salcianu, Y. Song, D. Dopson, and D. Zhou, "Fast WordPiece tokenization," in *Proc. 2021 Conf. on Empirical Methods in Natural Language Processing (EMNLP)*, 2021, pp. 2089–2103. DOI: [10.18653/v1/2021.emnlp-main.160](https://doi.org/10.18653/v1/2021.emnlp-main.160). *(Fast WordPiece / LinMaxMatch)*

**[119]** C. Xu, B. Zhou, T. Gan, Q. Zheng, and L. Li, "Vocabulary learning via optimal transport for neural machine translation," in *Proc. 59th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2021, pp. 7361–7373. DOI: [10.18653/v1/2021.acl-long.571](https://doi.org/10.18653/v1/2021.acl-long.571). *(VOLT)*

### Implementations verified against the original papers

The implementations below were checked against an independent exact oracle (brute-force enumeration on randomized databases, set-level comparison of the outputs) and, where the original papers publish result counts, against those counts. Choices that the papers leave open are listed.

| Algorithm | Paper | Verification | Choices not fixed by the paper |
|---|---|---|---|
| HEP (`hep`) | Deng 2020 [84] | Exact oracle: 8,073 cases (2,121 with an occupancy equal to the threshold), 0 mismatches. HOI counts equal Deng's Table 3 on retail, kosarak, T10I4D100K and accidents at 30% and 10%. | Exact (rational) comparison of occ(X) with the threshold; the early termination of the UBO scan (Section 3.2.3) is not implemented. |
| DFHOI (`dfhoi`) | Nguyen et al. 2023 [85] | Same oracle, 0 mismatches. HOI counts equal Tables 7 and 8 of the paper (mushroom, retail, T10I4D100K, chess, pumsb). | Items ordered by increasing support (the paper does not state an order); exact comparison as above. |
| Walky-G (`walky_g`) | Szathmary et al. 2012 [104] | Exact oracle: 5,294 cases on sparse, dense and equal-length random databases, 0 mismatches. MRI counts equal Hidouri et al. (IJCAI 2023), Table 2, on chess (\u03bb = 2500, 1000) and pumsb (\u03bb = 48000, 32000); on chess at \u03bb = 500, its count (1,353,317) agrees with an independent computation (maximal frequent itemsets by LCM, then minimal transversals by SHD) and differs from the published 1,353,344. | Items placed in increasing identifier order; rare candidates of support 0 are kept, as in Algorithm 3. |

Output of the verified miners can be written for checking: `DM_HOI_OUT` (HEP, DFHOI, AURA-HOI) and `DM_MRI_OUT` (Walky-G) name a file that receives one itemset per line. FPmax (`fpmax`) writes its maximal frequent itemsets to `DM_MFI_OUT`; its output matched a brute-force enumeration on 4,045 random cases, but its faithfulness to every step of the FPmax paper has not been reviewed.

Known issue: the output of `apriori_rare` does not match the exact oracle (it misses minimal rare itemsets in most random cases); use `walky_g` for minimal rare itemsets.


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
