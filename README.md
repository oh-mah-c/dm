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
29. **HUCI-Miner** - Mines high utility closed itemsets and their generators for non-redundant association rule mining.
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
47. **TIPN-HOUI-Miner** - Mines top-k high-on-shelf-utility itemsets with positive/negative utilities using TIPN/TIUL tables, interval occurrence bitmaps, TWUGC pruning, RLC pruning, TIO pruning, and RPRU/RRU threshold raising (Chang et al. 2025).
48. **TKU-CE** - Cross-Entropy Method for Mining Top-K High Utility Itemsets (Song et al. 2021).
49. **TKU-CE+** - Improved Cross-Entropy Method for Top-K HUIM with CUV pruning and smoothing mutation (Song et al. 2021).
50. **DPHIM** - Dynamic parallel high-utility itemset mining using utility-list subtasks and pthread workers (Kimura et al. 2026).

49. **HAUI-Miner** - Mining High Average-Utility Itemsets using AU-lists and transaction-maximum utility downward closure (Lin et al. 2016).
49. **EHAUPM** - Efficient High Average-Utility Pattern Mining with Tighter Upper Bounds and co-occurrence matrix (Lin et al. 2017).
50. **HAUIM-GMU** - Mining High Average-Utility Itemsets based on Generalized Maximal Utility and critical support count (Song et al. 2021).
51. **NAM-HEP** - Adaptive High Occupancy Itemset Mining using a hierarchical set-enumeration tree and median thresholds (Tran et al. 2025).
52. **MEMU** - Mining High Average-Utility Patterns with Multiple Thresholds using compact AU-lists (Lin et al. 2018).
53. **MHEINU** - Mining High-Efficiency Itemsets with Negative Utilities using ELNU lists, `uben`, and `ubeni` pruning (Yildirim 2025).
54. **Closed-FHUIM-Kinana** - Closed frequent high-utility itemset mining with OSR, OWL, and MSU pruning (Sulanjari & Fatichah 2026).

### Sequential Utility Pattern Mining
55. **USpan** - High utility sequential pattern mining using projected databases and sequence-weighted utility pruning.
56. **HUPSPM** - High utility-probability sequential pattern mining for uncertain sequence utility databases.
57. **HUP-Miner** - Mines high average utility nonoverlapping patterns from sequence utility databases using nonoverlapping SPC-style support, HUBP upper-bound pruning, and pattern join candidate generation (Geng et al. 2026).

### Top-K Closed Sequential Pattern Mining
58. **KCloTreeMiner** - Mines top-K closed sequential patterns over SPMF sequence datasets using SP-Tree-style projection, max-support candidate ordering, closed coverage, pattern absorption, and generic/group/redundancy-aware modes (Rizvee et al. 2025).

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
46. **FFI-Miner** - Fast Algorithm for mining fuzzy frequent itemsets from quantitative databases.
47. **UBMFFP-Tree** - Upper-bound Multiple Fuzzy Frequent Pattern Tree for mining multiple fuzzy frequent itemsets.

### Uncertain Data Mining
46. **U-Apriori** - Mining frequent itemsets from existential uncertain data using the expected support measure.

### Concise Frequent Itemset Representations
58. **RegularMine** - Mines frequent regular itemsets as an interpretable concise representation of frequent itemsets using closed classes, free sets, covering, and merging (Ruggieri 2010).

### Prompt Pattern Mining
59. **HUPP-Miner** - Mines High-Utility Prompt Patterns from prompt JSONL logs using semantic concept transactions, candidate optimizer banks, preserve masks, PTWO pruning, and AAUB alignment pruning.

### Closed High-Utility Occupancy Mining
60. **CHUO-Miner** - Mines Closed High-Utility Occupancy Itemsets from positive utility datasets using COU-lists, RUU utility pruning, OUB occupancy-envelope pruning, backward closure pruning, forward closure jumping, and CHUO-signatures.

### Classification Rule Mining
61. **PSO Classifier** - Particle Swarm based classification-rule discovery using continuous CPSO, TP/TN rule quality, rule pruning, covering, default rules, rule-set cleaning, and tenfold cross-validation.

---

## 🚀 Getting Started

### Prerequisites
- GCC Compiler (MinGW on Windows, or standard GCC on Linux/macOS)
- Make (optional, but recommended)

### Compilation
Build the executable from the source:
```bash
gcc -Iinclude -Iinclude/core -Wall -Wextra -O2 src/main.c src/core/*.c src/algorithms/*.c -o bin/dm.exe -lpsapi
```
*(On Linux, remove `-lpsapi` and use `-lm` if needed)*

### Usage
```bash
./bin/dm.exe <algorithm> <dataset_path> <format> <min_support> [min_io] [ins_threshold] [prn_threshold] [decay_base] [decay_life]
```
* **`<algorithm>`**: `ais`, `apriori`, `eclat`, `fpgrowth`, `tree_projection`, `aclose`, `close`, `closet`, `closetplus`, `fpclose`, `charm`, `dci_closed`, `lcm`, `lcmver2`, `nafcp`, `fcfia`, `max_miner`, `genmax`, `fpmax`, `mafia`, `hep`, `fhoi`, `dfhoi`, `tkhoim`, `hoimto`, `mfhoi`, `fhoi_miner`, `weak_mfhoi_miner`, `strong_mfhoi_miner`, `mhoui`, `houi_miner`, `weak_mhoui_miner`, `strong_mhoui_miner`, `direct_mhoui_miner`, `regular_mine`, `negfin`, `prepost`, `prepostplus`, `dic`, `ltm`, `sam`, `carpenter`, `dbv_miner`, `defme`, `talky_g`, `pascal`, `zart`, `apriori_rare`, `apriori_inverse`, `cori`, `rp_tree`, `estdec`, `clostream`, `cfi_stream`, `uapriori`, `msapriori`, `ffiminer`, `ubmffp`, `dfigrowth`, `sum`, `mlhui_miner`, `fchm`, `vhuqi`, `fhuqi_miner`, `tkq`, `mheinu`, `dphim`, `closed_fhuim_kinana`, `uspan`, `hupspm`, `hup_miner`.
* **`vhuqi`** uses `<min_support>` as the absolute minimum utility threshold and accepts optional `[qrc]`, defaulting to `3`.
* **`fhuqi_miner`** requires quantity format `4`: `./bin/dm.exe fhuqi_miner <quantity_dataset> 4 <theta> <qrc> <all|min|max> <profit_file>`.
* **`tkq`** requires quantity format `4`: `./bin/dm.exe tkq <quantity_dataset> 4 <k> <qrc> <all|min|max> <profit_file>`.
* **`mheinu`** requires utility format `1`: `./bin/dm.exe mheinu <utility_dataset> 1 <min_efficiency> [investment_file]`. If no investment file is supplied, the implementation follows the paper's experimental setup by deterministically generating positive per-item investment values with seed `42`.
* **`dphim`** requires utility format `1`: `./bin/dm.exe dphim <utility_dataset> 1 <min_utility> [threads]`. It parallelizes HUIM search with dynamic pthread task scheduling.
* **`closed_fhuim_kinana`** requires utility format `1`: `./bin/dm.exe closed_fhuim_kinana <utility_dataset> 1 <min_utility> <min_support> [min_owl]`.
* **`hup_miner`** requires sequence utility format `3`: `./bin/dm.exe hup_miner <sequence_utility_dataset> 3 <min_average_utility> [max_pattern_length] [max_candidates]`. The sequential files use SPMF notation such as `item[utility] -1 ... -2 SUtility:x`.
* **`regular_mine`** requires transactional format `0`: `./bin/dm.exe regular_mine <transactional_dataset> 0 <min_support>`.
* **`mfhoi`**, **`fhoi_miner`**, **`weak_mfhoi_miner`**, and **`strong_mfhoi_miner`** require transactional format `0`: `./bin/dm.exe strong_mfhoi_miner <transactional_dataset> 0 <min_support> <min_occupancy>`.
* **`mhoui`**, **`houi_miner`**, **`weak_mhoui_miner`**, **`strong_mhoui_miner`**, and **`direct_mhoui_miner`** require utility format `1`: `./bin/dm.exe mhoui <utility_dataset> 1 <min_support> <min_occupancy> <min_utility> [strong] [direct]`.
* **`hupp_miner`** is a standalone prompt-log miner: `./bin/hupp_miner --input <prompt_jsonl> --dataset-type auto|dolly|code_feedback --minsup <ratio|count> --minutil-ratio <value> --minalign <value>`. The companion script `scripts/run_hupp_experiments.sh` runs the full prompt benchmark and generates charts.
* **`chuo_miner`** is a standalone utility-dataset miner: `./bin/chuo_miner --input <utility_dataset> --minutil <value> --minsup <ratio|count> --minocc <value>`. The companion script `scripts/run_chuo_experiments.sh` runs Foodmart, Liquor, and Chainstore CHUO benchmarks and generates charts.
* **`pso_classifier`** is a standalone supervised classification-rule miner: `./bin/pso_classifier --input <spmf_class_folder> --particles <N> --threshold <T> --radius <R>`. The companion script `scripts/run_pso_classifier_experiments.sh` runs the malware-family classification benchmark and prints statistics only.
* **`tku_miner`** is a standalone top-k utility miner: `./bin/tku_miner --input <utility_dataset> --k <N> [--max-depth N] [--max-seconds S]`. The companion script `scripts/run_tku_experiments.sh` runs Foodmart and Liquor TKU benchmarks and prints statistics only.
* **`tku_pso_miner`** is a standalone heuristic top-k utility miner using TKU-PSO particle initialization, PEV checking, explored-particle caching, fitness estimation, pBest/gBest bit-difference updates, and minimum-solution-fitness threshold raising: `./bin/tku_pso_miner --input <utility_dataset> --k <N> --population <N> --iterations <N>`. The companion script `scripts/run_tku_pso_experiments.sh` runs statistics-only utility benchmarks.
* **`kclotree_miner`** is a standalone top-k closed sequential-pattern miner: `./bin/kclotree_miner --input <spmf_sequence_file_or_folder> --k <N> --type generic|group|redundancy_aware`. The companion script `scripts/run_kclotree_experiments.sh` runs the malware-sequence benchmark and prints statistics only.
* **`tipn_houi_miner`** is a standalone top-k high-on-shelf utility miner for positive/negative utility datasets: `./bin/tipn_houi_miner --input <negative_utility_dataset> --k <N> --intervals <N>`. The companion script `scripts/run_tipn_houi_experiments.sh` runs bounded Retail/Mushroom negative-utility benchmarks and prints statistics only.
* **`htk_miner`** is a standalone top-k frequent itemset miner using the paper's vertical BSN representation, top-k singleton initialization with ties, equivalence-class joins, and Q-Heap threshold raising: `./bin/htk_miner --input <transaction_dataset> --k <N> [--mode bsn]`. The companion script `scripts/run_htk_experiments.sh` runs transactional FIMI-style benchmarks and prints statistics only.
* **`topkphm_miner`** is a standalone top-k periodic high-utility itemset miner using periodic utility-lists, dynamic top-k utility thresholding, and EUSCS pruning: `./bin/topkphm_miner --input <utility_dataset> --k <N> --maxper <N> --maxavg <N>`. The companion script `scripts/run_topkphm_experiments.sh` runs statistics-only utility benchmarks.
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

**[97]** Q. Van, "MFHOI-Miner: An Efficient Method for Mining Maximal Frequent High-Occupancy Itemsets," local project paper, `docs/mfhoi.pdf`, 2026. *(MFHOI / Strong MFHOI-Miner)*

**[98]** Q. Van, "HUPP-Miner: High-Utility Prompt Pattern Mining for Cost-Aware and Accuracy-Preserving Generative AI Systems," local project paper, `docs/hupp.pdf`, 2026. *(HUPP-Miner)*

**[99]** Q. Van, "Closed High-Utility Occupancy Itemset Mining," local project paper, `docs/chuim.tex` and `docs/chuim.pdf`, 2026. *(CHUO-Miner)*

**[100]** Q. Van, "MHOUI-Miner: Mining High-Occupancy Utility Itemsets," local project paper, `docs/mhoui.tex` and `docs/mhoui.pdf`, 2026. *(MHOUI-Miner)*

**[101]** Q. Van, "VIFP: Verifiable/Privacy-Preserving Frequent Pattern Mining," local project paper, `docs/vifp.tex` and `docs/vifp.pdf`, 2026. *(VIFP-Miner)*

**[102]** T. Sousa, A. Silva, and A. Neves, "Particle Swarm based Data Mining Algorithms for classification tasks," *Parallel Computing*, vol. 30, no. 5-6, pp. 767-783, 2004. https://doi.org/10.1016/j.parco.2003.12.015 *(PSO Classifier)*

**[103]** C.-W. Wu, B.-E. Shie, P. S. Yu, and V. S. Tseng, "Mining top-k high utility itemsets," in *Proc. 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '12)*, 2012, pp. 78-86. https://doi.org/10.1145/2339530.2339546 *(TKU-Miner)*

**[104]** R. A. Rizvee, C. F. Ahmed, and C. K. Leung, "A tree-based framework to mine top-K closed sequential patterns," *Applied Intelligence*, vol. 55, article 221, 2025. https://doi.org/10.1007/s10489-024-06137-y *(KCloTreeMiner)*

**[105]** Y.-I. Chang, P.-C. Chuang, Y.-H. Liao, P.-Y. Hu, and T.-W. Chen, "An Efficient Algorithm for Mining Top-k High-On-Shelf-Utility Itemsets with Positive/Negative Profits of Local/Global Minimum Count," *Engineering Proceedings*, vol. 108, no. 1, article 45, 2025. https://doi.org/10.3390/engproc2025108045 *(TIPN-HOUI-Miner)*

**[106]** K. Malliaridis and S. Ougiaroglou, "Efficient techniques for retrieving top-K frequent itemsets," *Expert Systems With Applications*, vol. 311, article 131250, 2026. https://doi.org/10.1016/j.eswa.2026.131250 *(HTK-Miner / HTK-negFIN)*

**[107]** X. Zhang, B. Liu, J. Chen, S. Yang, Z. Gu, Z. Liu, and S. Xu, "TOPKPHM: Periodic Patterns Mining of Top-K High Utility Itemsets," in *Proc. 2025 International Conference on Trustworthy Big Data and Artificial Intelligence (ICTBAI)*, 2025. https://doi.org/10.1109/ICTBAI68361.2025.00009 *(TOPKPHM)*

**[108]** S. Carstensen and J. C.-W. Lin, "TKU-PSO: An Efficient Particle Swarm Optimization Model for Top-k High-Utility Itemset Mining," *International Journal of Interactive Multimedia and Artificial Intelligence*, vol. 9, no. 4, pp. 70-81, 2025. https://doi.org/10.9781/ijimai.2024.01.002 *(TKU-PSO)*

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
./bin/mfhoi_miner --input datasets/itemsets/retail.txt --algorithm strong_mfhoi --minsup 0.02 --minocc 0.6 --output results/patterns/out.txt
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
