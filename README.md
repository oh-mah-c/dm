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
2. **Apriori** - Level-wise search using candidate generation.
3. **Eclat** - Vertical data format search with depth-first traversal.
4. **FP-Growth** - Tree-based, projection-driven mining without candidate generation.
5. **TreeProjection** - Matrix-based counting via depth-first transactional tree projection.

### Frequent Closed Itemset Mining (FCIM)
6. **A-Close** - Uses frequent itemset generators to derive closed itemsets.
7. **CLOSET** - FP-tree based mining using frequent patterns.
8. **CLOSET+** - Extends CLOSET with hybrid tree traversals and pseudo-projection.
9. **FPclose** - Extends FP-growth with Array-based checking for closed itemsets.
10. **CHARM** - Vertical data format with diffsets for fast closed pattern extraction.
11. **DCI_CLOSED** - Highly optimized vertical bitset miner with order-preserving generators.
12. **LCM** - Linear time Closed itemset Miner using Prefix Preserving Closure Extension and Occurrence Deliver. (insanely fast for dense data).
13. **NAFCP** - An N-list-based Algorithm for mining Frequent Closed Patterns using the PPC-tree.
14. **FCFIA** - An Efficient Algorithm for Frequent Closed Itemsets Mining using PEP pruning and projection strategy.

### Maximal Frequent Itemset Mining (MFIM)
15. **Max-Miner** - Breadth/Depth search applying look-ahead superset pruning for long patterns.
16. **GenMax** - Backtrack search using Progressive Focusing and vertical bitsets for maximal itemsets.
17. **FPmax** - FP-growth extension with MFI-Tree for ultra-fast maximal itemset discovery.

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
./bin/dm.exe <algorithm> <dataset_path> <format> <min_support>
```
* **`<algorithm>`**: `ais`, `apriori`, `eclat`, `fpgrowth`, `tree_projection`, `aclose`, `closet`, `closetplus`, `fpclose`, `charm`, `dci_closed`, `lcm`, `nafcp`, `fcfia`, `max_miner`, `genmax`, `fpmax`.
* **`<dataset_path>`**: Path to your transactional dataset (e.g., `datasets/chess.txt`).
* **`<format>`**: Usually `0` for raw space-separated transactions.
* **`<min_support>`**: Support threshold as a fraction (e.g., `0.005` for 0.5%) or absolute count (e.g., `500`).

### Example Run
```bash
$ ./bin/dm.exe dci_closed datasets/chess.txt 0 0.8
```
```text
Loading dataset (0): datasets/chess.txt
Executing DCI_CLOSED Algorithm...
[DCI_CLOSED] Starting on 3196 transactions. Min Support: 2557
[DCI_CLOSED] Complete. Total frequent closed itemsets found: 5083

============================================================
                  DATA MINING BENCHMARK REPORT              
============================================================
 Algorithm   : DCI_CLOSED Algorithm
 Dataset     : datasets/chess.txt
------------------------------------------------------------
 [1] TIMING (High-Res)
     - I/O Load Data    :     17.446 ms
     - Algorithm Core   :      3.635 ms
     - Write Results    :      0.000 ms
     - TOTAL WALL TIME  :     21.587 ms
------------------------------------------------------------
 [3] MEMORY PROFILING
     - Peak RAM (VmHWM) :       3.79 MB  (3884 KB)
============================================================
```

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

**[14]** G. Grahne and J. Zhu, "High Performance Mining of Maximal Frequent Itemsets," in *6th International Workshop on High Performance Data Mining*, 2003. *(FPmax)*

**[15]** T. Uno, M. Kiyomi, and H. Arimura, "LCM ver. 2: Efficient Mining Algorithms for Frequent/Closed/Maximal Itemsets," in *Proc. IEEE ICDM Workshop on Frequent Itemset Mining Implementations (FIMI)*, 2004. *(LCM)*

**[16]** T. Le and B. Vo, "An N-list-based algorithm for mining frequent closed patterns," *Expert Systems with Applications*, vol. 42, no. 19, pp. 6748-6757, 2015. *(NAFCP)*

**[17]** L. Ma and Y. Qi, "An Efficient Algorithm for Frequent Closed Itemsets Mining," in *Proc. 2008 International Conference on Computer Science and Software Engineering*, 2008, pp. 260-262. *(FCFIA)*

---
<div align="center">
  <p>Made with ❤️ for Data Science & C Programming.</p>
</div>
