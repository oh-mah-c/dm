<div align="center">
  <h1>🚀 C-DataMiner Framework</h1>
  <p><strong>A Highly Optimized, Blazing-Fast Open-Source Data Mining Framework built purely in C</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Language: C](https://img.shields.io/badge/Language-C99%2FC%2B%2B17-orange.svg)](https://en.wikipedia.org/wiki/C99)
  [![Build: CMake](https://img.shields.io/badge/Build-CMake-success.svg)](#)
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

### Quantum / Scientific Models
75. **PauliNet** — Hermann, Schätzle & Noé, arXiv:1909.08423v5, *Nature Chemistry* 2020 **[143]**. Deep neural network solution of the electronic Schrödinger equation via variational quantum Monte Carlo (VMC). Wave function ansatz (Eq. 1): ψ_θ(r) = e^{J(r)+γ(r)} · Σ_p c_p · det[φ̃↑_{θ,p}(r)] · det[φ̃↓_{θ,p}(r)] — a multi-determinant Slater–Jastrow–backflow form where both the **Jastrow factor** J and **backflow** f_i are deep networks. **Architecture** (Figure 2): (1) **Distance featurisation** (Eqs. 12–14): cuspless RBF e_k(r)=r²·exp(-(r-μ_k)²/σ_k²), q_k uniform in (0,1); (2) **SchNet-like interaction** (Eq. 11): iteratively refines per-electron feature vectors x_i^(n) through three message channels — same-spin z^+, opposite-spin z^−, and nucleus z^n (each with independent w_θ, h_θ, g_θ MLPs); nuclear embeddings Y_{θ,I} are trainable per-nucleus vectors; (3) **Jastrow** J=η_θ(Σ_i x_i^(L)) — MLP on summed features (invariant); (4) **Backflow** f_i=κ_θ(x_i^(L)) — per-electron MLP multiplicatively modifying HF orbitals φ̃=φ·f (equivariant); (5) **Electronic cusp** γ(r) (Eq. 9): γ=-Σ_{i<j} c_{ij}/(1+|r_i−r_j|), c_{ij}=½ (same spin), ¼ (opposite) — analytically enforces electron–electron cusp conditions (Eq. 10). **Training** via variational Monte Carlo (Eqs. 6–8): minimise E[ψ]=E_{r~|ψ|²}[E_loc[ψ](r)] where E_loc=Ĥψ/ψ; gradient ∇_θL=2E[(E_loc−Ē)·∇_θ ln|ψ|]; Metropolis–Hastings MCMC generates electron positions; local energies clipped at 5× median-absolute-deviance. Optimizer: AdamW (lr=0.01, decay period t₀=200). **Hyperparameters** (Table 2): dim_e=32, dim_x=128, dim_z=64, L=3 interaction layers, 2000 walkers, 7000 steps, batch=10000. Recovers 99.99% of H₂ correlation energy, 97.3% for boron (B), 98.0% for H₁₀ with 36 determinants vs thousands needed by multi-determinant QMC methods (Table 1). Train with `./build/paulinet_train --system h2 --n-det 1 --steps 7000`.
76. **FermiNet** — Pfau, Spencer, Matthews & Foulkes, arXiv:1909.02487v3 **[144]**. Implements the antisymmetric FermiNet wave-function ansatz for ab-initio electronic structure with VMC training. The model uses single-electron and two-electron streams, permutation-equivariant spin-channel pooling, multi-determinant Slater matrices with exponential nuclear envelopes, Metropolis-Hastings sampling, differentiable local-energy evaluation, MAD-clipped VMC gradients, and an Adam-based training path. Includes H, He, H2, LiH, and carbon configs plus structural, differentiability, local-energy, MCMC, and one-step VMC training tests. Train with `./build/ferminet_train --system h2 --n-det 16 --steps 200000`.
77. **SchNet** — Schutt, Kindermans, Sauceda, Chmiela, Tkatchenko & Muller, NeurIPS 2017 **[145]**. Implements continuous-filter convolution for atomistic quantum interactions: atom-type embeddings, radial-basis distance expansion, filter-generating networks, residual interaction blocks, shifted-softplus smooth activations, atom-wise energy contributions with sum pooling, and forces as negative energy gradients. Defaults follow the paper: F=64 hidden features, three interaction blocks, RBF centers from 0 to 30 Angstrom at 0.1 Angstrom spacing, gamma=10, Adam lr=1e-3, batch=32, and energy/force loss with rho=0.01. Includes invariance/equivariance, force-gradient, loss, and train CLI smoke tests. Train with `./build/schnet_train --steps 100000`.
78. **PhysNet** — Unke & Meuwly, arXiv:1902.08408v2 **[146]**. Implements the message-passing HDNN for energies, forces, dipoles and partial charges: atom embeddings, five modular blocks by default, gated interaction updates, exponential-distance RBF attention masks, pre-activation residual blocks, module-wise output hierarchy, element-specific scale/shift, charge correction, damped long-range Coulomb energy, dipole prediction, and forces from analytic energy gradients. Defaults follow Table 1: F=128, K=64, Nmodule=5, atomic residual=2, interaction residual=3, output residual=1, rcut=10 Angstrom. The optional DFT-D3 term referenced by the paper is left disabled because the paper does not specify a complete standalone D3 parameterization. Train with `./build/physnet_train --steps 100000`.
79. **Neural Quantum States / VMC** — Medvidovic & Robledo Moreno, arXiv:2402.11014v2 **[147]**. This review paper is not a CNN-style primitive, so the implementation lives under quantum models rather than the vendored PyTorch core. It implements the explicitly specified RBM neural quantum-state ansatz ψθ(s)=exp(s·a)∏h 2cosh((sW+b)h), probability pθ(s)∝|ψθ(s)|², Metropolis-Hastings single-spin-flip sampling, transverse-field Ising local energy E_loc(s)=⟨s|H|ψθ⟩/⟨s|ψθ⟩, exact small-system energy enumeration, and an Adam train CLI for deterministic VMC smoke tests. Train with `./build/nqs_train --spins 8 --hidden 16 --steps 1000`.
80. **GA Schrödinger Solver** — Lahoz-Beltra, *Computers* 2022 **[148]**. Implements the paper's standard genetic algorithm for one-dimensional stationary Schrödinger problems: chromosomes store sampled wave-function values, residual fitness is F=exp(-Z), selection uses roulette-wheel/tournament steps, crossover is one-point, and mutation perturbs wave-function genes. Includes particle-in-a-box, harmonic-oscillator, and simplified hydrogen-radial potentials from the paper plus structural tests and a CLI. Run with `./build/ga_schrodinger_train --system box --generations 3200`.
81. **Clifford Group Equivariant Neural Networks** — Brandstetter, Ruhe & Forré, NeurIPS 2023 **[149]**. Implements the paper's core multivector machinery for Clifford group equivariant models: canonical blade-basis multivectors, diagonal-metric geometric products, grade projections, extended quadratic forms, grade-wise equivariant linear layers (Eq. 13), second-order geometric-product layers (Eqs. 14-15), grade normalization (Eq. 16), gated nonlinearities, and a small CGENN stack with 2D rotation-equivariance tests. Train smoke test with `./build/clifford_cgenn_train --dim 3 --channels 2 --hidden 4 --depth 2`.

### LibTorch Core Primitives (`dm::prim`)
82. **QuanvNd** — Henderson et al., arXiv:1904.04767 **[150]**. Quantum-Inspired Convolutional Layer (`dm::prim::QuanvNdImpl<D>`, D∈{1,2,3}), implemented directly inside the vendored LibTorch source tree at `torch/csrc/api/src/nn/modules/quanv_nd.cpp` and exposed via `<torch/nn/modules/quanv_nd.h>` — addressable as any native layer (Conv1d, Conv2d, etc.). Encapsulates a **Virtual Quantum Circuit (VQC)** operating inside a non-overlapping sliding window (kernel=2, stride=2) across 1-D, 2-D, or 3-D spatial grids. Three VQC phases: (1) **Quantum Embedding** — RY rotation maps each patch value x_i to a qubit state |ψ_i⟩=cos(x_i)|0⟩+sin(x_i)|1⟩ (amplitude pair α_i=cos(θ_i), β_i=sin(θ_i)); inputs pre-normalised via π·tanh(x); (2) **Quantum Entanglement** — parameterised Givens rotation in the |1⟩ subspace mixes adjacent qubit β amplitudes: [β'_i, β'_j]=R(δθ)[β_i, β_j] where R is a 2×2 rotation matrix and δθ∈ℝ is a learned per-(channel,qubit) parameter registered with autograd; D=1 applies one layer (pair 0,1); D=2 applies two layers (horizontal pairs 0,1/2,3 then vertical 0,2/1,3 covering all plaquette edges); D=3 applies three cascade layers covering all 12 edges of the 2×2×2 cube (x/y/z-axis strides 1/2/4); (3) **Quantum Measurement** — Born-rule projection P_i=|β'_i|²=β'_i² collapses the complex amplitude to a real-valued feature probability ∈[0,1], differentiable end-to-end through δθ. **Template meta-programming**: single `template<size_t D>` class, all per-dimension branching via `if constexpr`; no code duplication. **Memory contract**: window extraction uses chained `torch::Tensor::unfold()` (zero-copy strided views) then one `.contiguous()` on the hot path; no `std::vector` resizing inside `forward()`. Output channels: C_out=C_in×2^D. Registered in `modules.h` and `caffe2/CMakeLists.txt` alongside Conv1d/Conv2d. Use: `dm::prim::QuanvNd<2> qconv(3); auto y = qconv(x);` for x:[B,3,H,W] → y:[B,12,H/2,W/2].

### Vision Models
75. **CCNet** — Huang, Yuan, Guo, Zhang, Li & Wang, IEEE TPAMI 2020 **[153]**. Criss-Cross Network for Semantic Segmentation. Achieves full-image context with ~11× less GPU memory than non-local means by attending only along the **criss-cross path** (H+W-1 positions) of each pixel rather than all N positions. **CrissCrossAttention** (Eqs. 1-2, Section 3.2, `dm::prim`): for pixel u, query Q_u·(key Ω_{i,u})^T yields affinity D_{i,u}; softmax over the H+W-1 criss-cross positions gives attention weights A; aggregation H′_u = Σ_i A_{i,u}·V_{i,u} + H_u. **RCCA** (Section 3.3): applies CrissCrossAttention R=2 times with shared parameters; after 2 loops every pixel can gather context from every other position (horizontal+vertical overlap), recovering full-image context at O(N√N) cost. **Architecture** (Fig. 2): backbone (ResNet-101 dilated, stride 8) → 1×1 reduction → RCCA(R=2) → concat(H″,X) → head (Conv3×3-BN-ReLU-Dropout-Conv1×1) → logits. **Category Consistent Loss** (CCL, Section 3.4, Eqs. 3-7): auxiliary loss on RCCA features enforcing intra-class compactness (l_var, piecewise φ_var with δ_v) and inter-class separation (l_dis, piecewise φ_dis with 2δ_d), plus L2 centre regularisation (l_reg). Total: l_seg + α·l_var + β·l_dis + γ·l_reg (defaults α=β=1, γ=0.001, δ_v=0.5, δ_d=1.5). Cityscapes mIoU: 81.4% (ResNet-101) vs 78.6% non-local baseline. ADE20K: 45.22% mIoU. Training: SGD poly LR (base=0.01, power=0.9, 40K iters), momentum=0.9, weight-decay=1e-4, crop 769×769. Train with `./build/ccnet_train --rcca-ch 512 --key-ch 64 --num-classes 19 --epochs 40 --ccl`.
76. **DenseNet** (DenseNet-121 / 169 / 201 / 264, CIFAR variants) — Huang, Liu, van der Maaten & Weinberger, CVPR 2017 **[152]**. Densely Connected Convolutional Networks. Each layer receives the feature maps of all preceding layers within its dense block as input and passes its own feature maps to all subsequent layers (Eq. 2: x_ℓ = H_ℓ([x₀, x₁, ..., x_{ℓ-1}])). Unlike ResNets (addition), dense blocks **concatenate** features, eliminating redundant learning and enabling direct gradient flow to every layer (implicit deep supervision). **Composite function H_ℓ** (plain): BN→ReLU→Conv(3×3,k); (DenseNet-B bottleneck): BN→ReLU→Conv(1×1,4k)→BN→ReLU→Conv(3×3,k). **Growth rate k**: each layer appends exactly k new feature maps to the shared "collective knowledge" tensor; k=12 or k=32 suffice because every prior layer is reachable. **Transition layers** (between dense blocks): BN→ReLU→Conv(1×1)→AvgPool(2×2,stride 2); compression factor θ=0.5 halves channels (DenseNet-C). **DenseNet-BC** uses both bottleneck + compression. **ImageNet architectures** (Table 1, k=32, θ=0.5): DenseNet-121 [6,12,24,16] ~8M params; DenseNet-169 [6,12,32,32] ~14M; DenseNet-201 [6,12,48,32] ~20M; DenseNet-264 [6,12,64,48] ~34M. DenseNet-201 (20M params) matches ResNet-101 (40M params) validation error. **CIFAR** (3 equal dense blocks, 3×3 stem, no maxpool): DenseNet(L=40,k=12)=7.00% C10 / 27.55% C100; DenseNet-BC(L=100,k=12)=4.51%/22.27% C10+/C100+ with 0.8M params (vs ResNet-1001 10.2M params at comparable accuracy). Training: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4, Nesterov, cosine LR with 1/10 decay at 50%/75% of epochs, batch=64 (CIFAR) or 256 (ImageNet). Train with `./build/densenet_train --model densenet121 --epochs 90` or `./build/densenet_train --model cifar-L40-k12 --epochs 300`.
76. **VICReg** (ResNet-50 backbone + MLP encoder) — Bardes, Ponce & LeCun, arXiv:2105.04906v3, ICLR 2022 **[142]**. Variance-Invariance-Covariance Regularization for Self-Supervised Learning. A joint-embedding SSL method that prevents representational collapse without contrastive samples, memory banks, stop-gradient, or normalization tricks. Architecture (Figure 1, Section 4): each branch is an **encoder** f_θ (backbone, 2048-dim) followed by an **expander** h_φ (3-layer MLP: Linear→BN→ReLU → Linear→BN→ReLU → Linear, size 8192). Both branches are regularized independently. **Loss** (Eq. 6): ℓ(Z,Z′) = λ·s(Z,Z′) + μ·[v(Z)+v(Z′)] + ν·[c(Z)+c(Z′)] — weighted sum of three terms: (1) **Invariance** s (Eq. 5): mean squared Euclidean distance (1/n)·Σᵢ ||zᵢ−zᵢ′||²; (2) **Variance** v (Eqs. 1–2): hinge loss (1/d)·Σⱼ max(0, γ−√(Var(zʲ)+ε)) maintaining std-dev of each embedding dim ≥ γ=1 (prevents uniform collapse); (3) **Covariance** c (Eqs. 3–4): (1/d)·Σᵢ≠ⱼ [C(Z)]²ᵢⱼ penalising off-diagonal covariance (prevents informational/dimensional collapse). Default coefficients (Section 4.2): λ=μ=25, ν=1. Unlike BYOL/SimSiam, **no weight sharing, stop-gradient, or predictor is required**; unlike Barlow Twins, each branch is regularized independently enabling **multi-modal setups** (Table 3: VICReg outperforms Barlow Twins on MS-COCO image-text retrieval). Achieves 73.2% ImageNet Top-1 linear (Table 1) with ResNet-50. Training: LARS (lr=batch/256×0.2, cosine→0.002, warmup 10 epochs, weight-decay=1e-6, 1000 epochs, batch=2048). Train with `./build/vicreg_train --repr-dim 2048 --exp-dim 8192 --lambda 25 --mu 25 --nu 1`.
76. **I-JEPA** (ViT-B/16, ViT-L/16, ViT-H/14) — Assran et al., arXiv:2301.08243v3, CVPR 2023 **[141]**. Image-based Joint-Embedding Predictive Architecture. Self-supervised pretraining without hand-crafted data augmentations. Architecture (Figure 3): (1) **context-encoder** f_θ — standard ViT (no [CLS]) processes visible context patches; (2) **target-encoder** f_θ̄ — identical ViT whose weights are an EMA of f_θ (τ=0.996→1.0 linearly, Appendix A); (3) **predictor** g_φ — narrow ViT (embed_dim=384, depth=6 for ViT-B, depth=12 for ViT-L/H) conditioned on positional mask tokens. **Multi-block masking** (Section 3, Fig. 4): M=4 possibly overlapping target blocks sampled with scale∈[0.15,0.20], aspect∈[0.75,1.5]; 1 context block with scale∈[0.85,1.0]; overlapping patches removed from context. Loss = (1/M)·Σᵢ Σⱼ∈Bᵢ ||ŝ_yⱼ − s_yⱼ||² in representation space (not pixel space — critical for semantic quality, Table 7). Target representations s_y are computed by f_θ̄ on the full image (masking applied to the *output*, not the input). Predictor takes context tokens s_x + positional mask tokens; predicts target representations for each target block. Context-encoder and predictor trained with AdamW (lr=1e-4→1e-3 warmup 15 epochs, cosine decay to 1e-6, weight-decay 0.04→0.4, batch=2048). Model sizes: ViT-B/16 — embed=768, depth=12, pred_depth=6; ViT-L/16 — embed=1024, depth=24, pred_depth=12; ViT-H/14 — embed=1280, depth=32, patch=14, pred_depth=12. ViT-H/14 achieves 81.1% ImageNet linear-probe (Table 1) trained in <72 GPU-hours on 16×A100. Train with `./build/ijepa_train --model vit_b16 --img-size 224 --epochs 300`.
76. **ResNet** (ResNet-18 / 34 / 50 / 101 / 152) — He et al., CVPR 2016 **[126]**. Deep Residual Learning for image classification. Implements all five variants from Table 1 using BasicBlock (2-layer) and Bottleneck (3-layer) residual blocks with identity/projection shortcuts. Backed by LibTorch with Vulkan GPU support. Train with `./build/resnet_train --model resnet50 --epochs 90`.
76. **VGGNet** (VGG-A / B / C / VGG-16 / VGG-19) — Simonyan & Zisserman, ICLR 2015 **[127]**. Very Deep Convolutional Networks. Implements all 5 configurations from Table 1 (11–19 weight layers) using stacked 3×3 conv filters, 5 max-pooling layers, and three FC layers with Dropout(0.5). Config C includes 1×1 conv layers; no LRN used. Backed by LibTorch. Train with `./build/vgg_train --model vgg16 --epochs 74`.
77. **YOLO v1** — Redmon et al., CVPR 2016 **[128]**. You Only Look Once: Unified, Real-Time Object Detection. Single-pass grid-based detector (S=7, B=2, C=20) with 24-layer backbone, Leaky ReLU activations, and multi-part SSE loss (λ_coord=5, λ_noobj=0.5) with √(w·h) trick. Includes NMS post-processing and SGD training. Train with `./build/yolo_train --S 7 --B 2 --C 20 --epochs 135`.
78. **VAE** (Variational Autoencoder) — Kingma & Welling, ICLR 2014 **[129]**. Auto-Encoding Variational Bayes. Implements encoder (tanh MLP → μ, log σ²), reparameterisation trick (z = μ + ε·σ), and decoder (tanh MLP → sigmoid output). ELBO loss: reconstruction BCE + KL divergence D_KL(q(z|x) ‖ p(z)). Train with `./build/vae_train --latent 20 --epochs 50`.
79. **MobileNet v1** — Howard et al., arXiv:1704.04861v1 **[130]**. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. Factorises standard convolutions into depthwise (groups=in_ch) + pointwise (1×1) blocks, reducing computation by ≈8–9×. Supports width multiplier α ∈ {1.0, 0.75, 0.5, 0.25} scaling from 4.2M to 0.5M parameters. RMSprop training. Train with `./build/mobilenet_train --alpha 1.0 --classes 1000 --epochs 100`.

### NLP / Sequence Models
80. **Word2Vec** (CBOW + Skip-gram) — Mikolov et al., arXiv:1301.3781v3 **[131]**. Efficient Estimation of Word Representations in Vector Space. Implements both CBOW (context → centre) and Skip-gram (centre → context) with negative sampling using unigram^(3/4) noise distribution and SGD. Supports analogy arithmetic (king − man + woman ≈ queen) and cosine-similarity nearest-neighbour retrieval. Train with `./build/word2vec_train --mode skipgram --dim 300 --window 5 --neg 5 --epochs 5`.
81. **Whisper** (Tiny / Base / Small / Medium / Large) — Radford et al., ICML 2023 **[132]**. Robust Speech Recognition via Large-Scale Weak Supervision. Encoder-decoder Transformer for multilingual ASR: Conv1D stem (stride=2) on 80-ch log-mel spectrograms, sinusoidal PE, pre-norm Transformer encoder; learned PE + masked self-attention + cross-attention decoder with tied output projection. Multitask tokens: SOT, language tags (99 languages), TRANSCRIBE/TRANSLATE, NOSPEECH, NOTIMESTAMPS, timestamps, EOT. AdamW with linear warmup. Train with `./build/whisper_train --size base --epochs 10`.
82. **Llama 2** (stories110k / 7B / 13B / 70B) — Touvron et al., arXiv:2307.09288, 2023 **[133]**. Open Foundation and Fine-Tuned Chat Models. Decoder-only Transformer with RMSNorm (pre-norm), RoPE positional embeddings, Grouped-Query Attention (GQA, n_kv_heads ≤ n_heads), and SwiGLU feed-forward (FFN = W2(SiLU(W1·x) ⊙ W3·x)). Tied input/output embedding weights. KV-cache for efficient autoregressive inference with greedy / temperature / top-p sampling. AdamW with weight-decay separation (2-D params decay, 1-D norms/biases do not) and cosine LR schedule with linear warmup. Native C++ port of karpathy/llama2.c. Train with `./build/llama2_train --size stories110k --epochs 10`.
83. **Transformer-XL** (tiny / base / large / xl / small) — Dai et al., ACL 2019 **[134]**. Attentive Language Models Beyond a Fixed-Length Context. Introduces two key innovations for unbounded context: (1) segment-level recurrence — previous segment hidden states are cached (stop-gradient) and prepended as extended context for key/value attention, enabling O(N×L) effective dependency; (2) relative positional encoding — sinusoidal R matrix injected per-layer via four attention score terms (content+content, content+position, global-content-bias, global-position-bias) with learnable u/v bias vectors and separate W_k,E / W_k,R projections; relative shift applied in O(q×k) via the Appendix B padding trick. Pre-norm LayerNorm + residual. Tied embeddings. Adam with gradient clipping 0.25. Memory expandable at eval for longer context (up to 1,800× speedup over vanilla Transformer eval). Train with `./build/transformer_xl_train --size base --epochs 10 --seglen 512 --memlen 512`.
84. **Mamba** (tiny / 130M / 370M / 790M / 1.4B) — Gu & Dao, arXiv:2312.00752v2, 2023 **[135]**. Linear-Time Sequence Modeling with Selective State Spaces. Selective SSM (S6): SSM parameters B, C are functions of the input (B=Linear_N(x), C=Linear_N(x)); Δ=softplus(Parameter+Broadcast_D(Linear_1(x))) is input-dependent, breaking LTI and enabling content-based selection. A remains a fixed diagonal parameter (init: A[i]=-(i+1), always negative for stability). Mamba block: in_proj(d_model→2×d_inner) splits into x and gate z; x passes through depthwise Conv1d(kernel=4) + SiLU then selective SSM scan; output = SSM(x) ⊙ SiLU(z) → out_proj. Stacked with RMSNorm (pre-norm) + residual. Constant-time O(1) autoregressive inference via step() method (no KV-cache growth). AdamW (β=(0.9,0.95), clip=1.0, wd=0.1), cosine LR decay. Mamba-3B matches Transformer 6B quality with 5× generation throughput. Train with `./build/mamba_train --size 130m --epochs 10 --seqlen 2048`.
85. **RWKV** (tiny / 169M / 430M / 1.5B / 3B / 7B / 14B) — Peng et al., arXiv:2305.13048v2, 2023 **[136]**. Reinventing RNNs for the Transformer Era. Combines Transformer parallelism (train) with RNN O(1)-per-token inference (deploy). Core operator: WKV — channel-wise AFT-style linear attention with learnable time-decay w (non-negative) and bonus u (direct-token weight). Numerically stable RNN recurrence (eqs.19–28, App.D) maintains state (a′,b′,p) per layer. Token-shift (§3.1.1): all R/K/V inputs are linearly interpolated with x_{t-1} via learned μ vectors (layer-depth initialisation, App.E). Channel-mixing FFN (eq.18) uses squared-ReLU. Small-init embedding U(±1e-4) + post-embed LayerNorm (§3.4). Named after learnable per-block scalars: **R**eceptance, **W**eight, **K**ey, **V**alue. Adam (β=(0.9,0.99), no weight decay, §4.1), exponential LR decay. 169M RWKV matches GPT-3-size perplexity with ≈1/6 inference memory. Train with `./build/rwkv_train --size 169m --epochs 10 --seqlen 1024`.
86. **LoRA** (rank 1/2/4/8/16) — Hu et al., arXiv:2106.09685v2, 2021 **[137]**. Low-Rank Adaptation of Large Language Models. Freezes pre-trained weights W₀ and injects trainable rank-decomposition matrices: h = W₀x + ΔWx = W₀x + BAx, where A ∈ ℝ^{r×k} (Kaiming init) and B ∈ ℝ^{d×r} (zero init, so ΔW=0 at training start). Scaling α/r applied to ΔW (§4.1). Supports LoRAEmbedding with reversed init (A=0, B~N). merge() folds BA into W₀ for zero-inference-latency deployment; unmerge() restores original weights. inject_lora() replaces named nn::Linear sub-modules in any torch::nn::Module via replace_module(). LoRAEmbedding supports low-rank adaptation of embedding tables. Can reduce trainable params by 10,000× vs full fine-tuning (r=4 on {Wq,Wv} in GPT-3 175B = 4.7M vs 175B). Adam, no weight decay, cosine LR decay. Train with `./build/lora_train --rank 4 --alpha 4 --layers 4 --dim 128`.
88. **CMA-ES** — N. Hansen, arXiv:1604.00772v2, 2023 **[140]**. The (μ/μ_W, λ)-Covariance Matrix Adaptation Evolution Strategy. A stochastic, derivative-free optimizer for non-linear, non-convex continuous functions. Maintains a multivariate Gaussian search distribution N(**m**, σ²**C**) and adapts it each generation using: (1) **mean update** (Eq. 42): weighted recombination of the top-μ offspring; (2) **cumulative step-size adaptation (CSA)** (Eqs. 43–44): evolution path **p**_σ drives σ via exponential update to match ||N(0,I)|| in expectation; (3) **covariance matrix adaptation** (Eqs. 45–47): rank-one + rank-μ update of **C** using the evolution path **p**_c and all λ offspring with active (negative) weights (aCMA-ES, 2016 default). Eigendecomposition **C** = **B** **D**² **B**ᵀ recomputed every max(1, 1/(10n(c₁+c_μ))) generations for O(n²) amortised cost (Sec. B.2). Default parameters from Table 1: λ=4+⌊3 ln n⌋, μ=⌊λ/2⌋, weights by Eqs. 49–53, c_σ/d_σ by Eq. 55, c_c/c₁/c_μ by Eqs. 56–58. Termination criteria: ftol, xtol (TolX), condition(C)>10¹⁴, NoEffectAxis, Stagnation. Call `cmaes_run()` directly with any `CMAES_ObjectiveFn`.

89. **Llama 3** (tiny / 8B / 70B / 405B) — Llama Team, AI @ Meta, arXiv:2407.21783v3, 2024 **[138]**. The Llama 3 Herd of Models. Dense decoder-only Transformer pre-trained on 15.6T multilingual tokens. Three key improvements over Llama 2: (1) **GQA on all sizes** — 8 KV heads for 8B, 70B, and 405B (§3.2); (2) **RoPE θ=500,000** (vs 10,000), enabling context up to 128K tokens via incremental long-context pre-training (§3.2, §3.4.2); (3) **128K-token vocabulary** combining 100K tiktoken BPE tokens + 28K non-English tokens (§3.2). SwiGLU FFN (W2(SiLU(W1·x)⊙W3·x)), RMSNorm (pre-norm), no weight tying between embedding and lm-head. Separate output projection. 8B: layers=32, dim=4096, ffn=14336; 70B: layers=80, dim=8192, ffn=28672; 405B: layers=126, dim=16384, ffn=53248. Post-training: SFT + DPO rejection sampling in 6 rounds (§4). AdamW (β=(0.9,0.95), ε=1e-5, wd=0.1), cosine LR with 8000-step warmup, grad clip=1.0. KV-cache for O(1)-per-step autoregressive inference with temperature/top-p sampling. Train with `./build/llama3_train --size tiny --epochs 5`.
89. **BitNet a4.8** (700M / 1.3B / 3B / 7B) — Wang et al., arXiv:2411.04965v1, 2024 **[139]**. 4-bit Activations for 1-bit LLMs. Extends BitNet b1.58 (1.58-bit ternary weights {-1,0,+1}, α·RoundClip(W/(α+ε),-1,1)) with a **hybrid quantization + sparsification** architecture that reduces activation precision to ≈4 bits. Three activation regimes (Figure 1): (1) Attention QKV inputs and FFN Up/Gate inputs → **INT4 absmean**: Q_INT4(X) = β/√7·RoundClip(√7/(β+ε)·X, -8, 7), β=mean(|X|); (2) Attention Out-projection input → **INT8 absmax + TopK 50% sparsification**: Q_INT8(X) = γ/127·Round(127/γ·X), γ=max(|X|), mask=Top50%(|X|); (3) FFN Down-projection input → **INT8 absmax** (naturally >80% sparse via ReLU²GLU). **ReLU²GLU** replaces SwiGLU: FFN(X) = (XW_up^T) ⊙ ReLU²(XW_gate^T), where ReLU²=(max(0,·))²; RoPE positional encoding; RMSNorm (pre-norm). 2-stage training: stage 1 — INT8 + ReLU²GLU, 95B tokens; stage 2 — 4-bit + sparse, 5B tokens. STE gradients. AdamW β=(0.9,0.95), warmup 375 steps, cosine LR decay, weight-decay 0.1→0. Achieves parity with b1.58 while significantly reducing inference activation footprint. Train with `./build/bitnet_a4_8_train --size 700m --data tokens.bin`.
90. **OhmC1** (tiny / small / medium / large) — oh-mah-c, dm/OhmC1, 2026 **[151]**. A dm-native autoregressive decoder-only LLM family designed from scratch for this repository. Four sizes: tiny (<10M params, seq_len=128, for unit tests), small (~145M, seq_len=2048), medium (~1.7B, seq_len=4096), large (~7.1B, seq_len=8192). Six architectural distinctions from Llama 2/3: (1) **Sandwich-norm blocks** — each transformer block applies four RMSNorm layers: pre_attn_norm before attention, post_attn_norm on the attention output before the residual add, and equivalently for FFN. Formula: h = x + post_attn_norm(Attention(pre_attn_norm(x))); out = h + post_ffn_norm(FFN(pre_ffn_norm(h))). Prevents gradient magnitude explosion in deep networks without DeepNorm weight scaling. (2) **QKNorm** (Query-Key normalization) — after Q and K projections are reshaped to [B, T, n_heads, head_dim], a per-head RMS normalization with a learned per-(head,dim) gain parameter is applied to both Q and K before RoPE. Stabilizes attention logit scale as a function of depth, preventing attention entropy collapse (inspired by Gemma 2). (3) **Tiered GQA** — n_kv_heads is 1 (MQA), 2, 4, 8 for tiny/small/medium/large, giving repetition ratios 4x/6x/4x/4x. Tiny uses true MQA for maximum memory efficiency in testing. (4) **Size-scaled RoPE theta** — base frequency theta = 10,000 / 50,000 / 200,000 / 500,000 for tiny/small/medium/large, enabling each size to exploit an appropriate frequency range for its context window. (5) **64K vocabulary** — 64,000 BPE tokens (between Llama 2's 32K and Llama 3's 128K). Includes an optional `OhmC1BPETokenizer` C++ class that reads tiktoken-compatible vocab.json + merges.txt files at runtime with no external dependencies; all model methods work with raw int64_t IDs without it. (6) **Untied embeddings for all sizes** — tok_embeddings and lm_head are always separate parameter tensors. norm_eps=1e-6 (tighter than Llama's 1e-5). Training: AdamW (beta=(0.9,0.95), eps=1e-8, wd=0.1 on 2-D params), cosine LR with linear warmup, grad-clip 1.0. Two AdamW param groups: 2-D weight matrices (decay) and 1-D norms/gains (no decay). KV-cache for O(1)-per-step autoregressive inference with temperature/top-p nucleus sampling. Train with `./build/ohmc1_train --size tiny --epochs 5`.
91. **KAN** (Kolmogorov-Arnold Networks) — Liu, Wang, Vaidya, Ruehle, Halverson, Solja&#269;i&#263;, Hou & Tegmark, ICLR 2025 **[154]**. KAN: Kolmogorov-Arnold Networks. Replaces the fixed activation functions on nodes (as in MLPs) with learnable univariate activation functions on edges, grounded in the Kolmogorov-Arnold representation theorem. **KAN layer** (Eq. 5): x_{l+1,j} = sum_i phi_{l,j,i}(x_{l,i}); **edge activation** (Eq. 15): phi(x) = w_b * silu(x) + w_s * spline(x), where silu(x) = x/(1+e^{-x}) is a fixed residual basis and spline(x) = sum_m c_m B_m(x) is a learnable B-spline of order k on G uniform intervals (G+k trainable coefficients per edge). **B-spline basis** (Cox-de Boor recurrence, Eq. 17): B_{i,0}=1[t_i<=x<t_{i+1}]; B_{i,p}=(x-t_i)/(t_{i+p}-t_i) B_{i,p-1} + (t_{i+p+1}-x)/(t_{i+p+1}-t_{i+1}) B_{i+1,p-1}; knot vector: k repeated left/right endpoints (clamped) + G+1 interior knots = G+2k+1 total. **Adaptive grid** (Section 2.5): each forward pass (training mode) updates the grid for each input feature to span [min-eps*range, max+eps*range] of the batch activations, ensuring splines track the actual data distribution. **Grid extension** (Eq. 18, Appendix L): coarsen-to-fine refinement via least-squares projection -- given old coefficients, solve argmin ||sum_j c'_j B'_j(x) - sum_i c_i B_i(x)||^2 over a finer grid, enabling a schedule G=5->10->20->50 without restarting training. **Sparsity regularisation** (Eqs. 19-22, Section 2.4): l_total = l_pred + lambda*(mu1*|Phi|_1 + mu2*S(Phi)), where |phi_{j,i}|_1 = (1/N_p) sum_s |phi(x^s)|, |Phi|_1 = sum_{i,j} |phi_{j,i}|_1, and entropy S(Phi) = -sum_{i,j} (|phi_{j,i}|_1/|Phi|_1) log(...) encourages edge pruning and function interpretability. **KANLinear** (`dm::prim`, `torch/nn/modules/kan_linear.h`): header-only reusable layer with parameters spline_weight [n_out,n_in,G+k], w_b [n_out,n_in], w_s [n_out,n_in] and non-trainable grid buffer [n_in,G+2k+1]; free function `kan_linear_sparsity()` computes the per-layer regularisation term. Architecture notation [n_0,...,n_L]; paper examples: [2,5,1] (regression), [2,1,1] (physics formula discovery), [784,100,10] (MNIST), [4,2,1,1] (4-variable PDE). Optimisers: Adam (lr=1e-3) or L-BFGS (lr=0.1-1.0). Paper demonstrates exact symbolic formula recovery (sin, exp, x^2) via grid extension + sparsification + pruning, outperforming MLPs on symbolic regression at fixed parameter count. Train with `./build/kan_train --widths 2,5,1 --G 5 --k 3 --optimizer adam --steps 2000`.
92. **NEAT** (NeuroEvolution of Augmenting Topologies) — Stanley & Miikkulainen, *Evolutionary Computation* 10(2):99-127, 2002 **[155]**. Evolves both neural-network topology and weights simultaneously via three innovations: (1) **Historical markings** (Section 3.2) — a global innovation counter assigns unique innovation numbers to new connection genes, enabling meaningful gene-level crossover alignment across networks of different topologies, analogous to chromosome crossover with homologous genes; (2) **Speciation** (Section 3.3) — compatible genomes (compatibility distance delta = c1*E/N + c2*D/N + c3*W_bar, Eq. 1, where E=excess genes, D=disjoint genes, W_bar=avg weight diff of matching genes) are grouped into protected species using threshold delta_t=3.0; fitness sharing (Eq. 2) f'_i = f_i / sum_j sh(delta(i,j)) gives each species a proportional offspring allocation based on mean adjusted fitness, preventing one species from dominating; stagnant species (no improvement for stagnation_limit=15 gens) are culled; (3) **Minimal initial structure** (Section 3.4) — population starts with no hidden nodes (fully connected input-to-output only), growing topological complexity only when beneficial. **Genetic encoding** (Section 3.1, Figure 2): NodeGene (node id, type: sensor/hidden/output/bias); ConnGene (in_node, out_node, weight, enabled, innovation_number). **Structural mutations** (Figure 3): add_connection (new ConnGene with next global innovation number); add_node (disable existing connection, insert new hidden node with two new connections: weight-1 in, old-weight out). **Crossover** (Figure 4): align parent genes by innovation number; matching genes inherited randomly; disjoint/excess genes always from the more fit parent. **Activation** (Section 4.1): modified sigmoid phi(x) = 1/(1+exp(-4.9*x)) applied in topological order (Kahn's algorithm ensures hidden nodes are evaluated before the output nodes they feed). **Phenotype decoding** (`neat_decode`): produces a `NEAT_Network` with pre-built parallel conn_in/conn_out/conn_w arrays and pinned bias activation = 1.0. Default hyperparameters (Section 4.1): pop=150, c1=c2=1.0, c3=0.4, delta_t=3.0, p_weight_mut=0.80, p_weight_perturb=0.90, p_add_conn=0.05, p_add_node=0.03, p_no_crossover=0.25, interspecies_rate=0.001, survival_rate=0.20, stagnation_limit=15. XOR benchmark (Section 4.2): `neat_run()` with n_inputs=2, n_outputs=1, pop=150 solves XOR (fitness>=15.9 out of 16.0) in approximately 50-200 generations; solution requires at least 1 hidden node since XOR is not linearly separable. Located in `include/algorithms/neat.h` + `src/algorithms/neat.c`.

---

## 🚀 Getting Started

### Prerequisites
- CMake ≥ 3.18, Ninja
- GCC / Clang with C99 + C++17 support
- Vulkan SDK (`libvulkan-dev vulkan-headers glslc`)
- ICU (`libicu-dev`), OpenBLAS (`libopenblas-dev`)
- Python 3 + NumPy (required by LibTorch build system only)

> **Full step-by-step install:** see [docs/HOWTO/INSTALL.md](docs/HOWTO/INSTALL.md)

### Build
```bash
# 1. Build LibTorch (first time only, ~30 min)
cd src/core/pytorch && mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON -DUSE_CUDA=OFF -DUSE_VULKAN=ON \
         -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF \
         -DGLSLC_EXECUTABLE=/usr/bin/glslc \
         -DCMAKE_INSTALL_PREFIX=../dist
cmake --build . --parallel 4 && cmake --install .
cd ../../../../

# 2. Build dm
mkdir build
cmake -S . -B build -GNinja
cmake --build build --parallel 4
```

### Run
```bash
./build/dm <algorithm> <dataset_path> <format> <min_support>
./build/dm_tokenizer          # tokenizer binary
./build/resnet_train --help   # ResNet training binary
```

### Usage
```bash
./bin/dm.exe <algorithm> <dataset_path> <format> <min_support> [min_io] [ins_threshold] [prn_threshold] [decay_base] [decay_life]
```
* **`<algorithm>` / direct subcommand**: `ais`, `apriori`, `eclat`, `fpgrowth`, `tree_projection`, `aclose`, `close`, `closet`, `closetplus`, `fpclose`, `charm`, `dci_closed`, `lcm`, `lcmver2`, `nafcp`, `fcfia`, `max_miner`, `genmax`, `fpmax`, `mafia`, `hep`, `fhoi`, `dfhoi`, `tkhoim`, `hoimto`, `mfhoi`, `fhoi_miner`, `weak_mfhoi_miner`, `strong_mfhoi_miner`, `mhoui`, `houi_miner`, `weak_mhoui_miner`, `strong_mhoui_miner`, `direct_mhoui_miner`, `regular_mine`, `negfin`, `prepost`, `prepostplus`, `dic`, `ltm`, `sam`, `carpenter`, `dbv_miner`, `defme`, `talky_g`, `pascal`, `zart`, `apriori_rare`, `apriori_inverse`, `cori`, `rp_tree`, `estdec`, `clostream`, `cfi_stream`, `uapriori`, `msapriori`, `ffiminer`, `ubmffp`, `dfigrowth`, `sum`, `mlhui_miner`, `fchm`, `vhuqi`, `fhuqi_miner`, `tkq`, `thue`, `mheinu`, `dphim`, `closed_fhuim_kinana`, `uspan`, `hupspm`, `hup_miner`, `hiep`, `maximal_munch`, `bpe`, `bpe_dropout`, `unigram`, `sentencepiece`, `tokenizer_lab`, `gpe`, `parity_bpe`, `fast_wordpiece`.
* **`vhuqi`** uses `<min_support>` as the absolute minimum utility threshold and accepts optional `[qrc]`, defaulting to `3`.
* **`fhuqi_miner`** requires quantity format `4`: `./bin/dm.exe fhuqi_miner <quantity_dataset> 4 <theta> <qrc> <all|min|max> <profit_file>`.
* **`tkq`** requires quantity format `4`: `./bin/dm.exe tkq <quantity_dataset> 4 <k> <qrc> <all|min|max> <profit_file>`.
* **`thue`** mines top-k high-utility episodes from utility event sequences: `./bin/dm.exe thue <utility_dataset> 1 <k> [MTD]`. With `MTD=0`, each timestamp is a simultaneous event set, so the output count and top-k threshold can be compared against top-k high-utility itemset miners such as TKU-Miner.
* **`mheinu`** requires utility format `1`: `./bin/dm.exe mheinu <utility_dataset> 1 <min_efficiency> [investment_file]`. If no investment file is supplied, the implementation follows the paper's experimental setup by deterministically generating positive per-item investment values with seed `42`.
* **`dphim`** requires utility format `1`: `./bin/dm.exe dphim <utility_dataset> 1 <min_utility> [threads]`. It parallelizes HUIM search with dynamic pthread task scheduling.
* **`closed_fhuim_kinana`** requires utility format `1`: `./bin/dm.exe closed_fhuim_kinana <utility_dataset> 1 <min_utility> <min_support> [min_owl]`.
* **`hup_miner`** requires sequence utility format `3`: `./bin/dm.exe hup_miner <sequence_utility_dataset> 3 <min_average_utility> [max_pattern_length] [max_candidates]`. The sequential files use SPMF notation such as `item[utility] -1 ... -2 SUtility:x`.
* **`regular_mine`** requires transactional format `0`: `./bin/dm.exe regular_mine <transactional_dataset> 0 <min_support>`.
* **`mfhoi`**, **`fhoi_miner`**, **`weak_mfhoi_miner`**, and **`strong_mfhoi_miner`** require transactional format `0`: `./bin/dm.exe strong_mfhoi_miner <transactional_dataset> 0 <min_support> <min_occupancy>`.
* **`cloe_hoi`** requires transactional format `0`: `./bin/dm.exe cloe_hoi <transactional_dataset> 0 <min_occupancy> [min_support] [max_seconds]`. If `min_support` is omitted, it defaults to `ceil(min_occupancy * transaction_count)` for comparable HOI threshold sweeps.
* **`aura_hoi`** requires transactional format `0`: `./bin/dm.exe aura_hoi <transactional_dataset> 0 <min_occupancy> [min_support] [max_seconds] [avg|sum] [closed|raw]`. If `min_support` is omitted or `0`, it defaults to `ceil(min_occupancy * transaction_count)`. Use `sum raw` to match the raw fullset summed-occupancy semantics used by the repository HEP/DFHOI implementations.
* **`mhoui`**, **`houi_miner`**, **`weak_mhoui_miner`**, **`strong_mhoui_miner`**, and **`direct_mhoui_miner`** require utility format `1`: `./bin/dm.exe mhoui <utility_dataset> 1 <min_support> <min_occupancy> <min_utility> [strong] [direct]`.
* **`hupp_miner`** is a standalone prompt-log miner: `./bin/hupp_miner --input <prompt_jsonl> --dataset-type auto|dolly|code_feedback --minsup <ratio|count> --minutil-ratio <value> --minalign <value>`. The companion script `scripts/run_hupp_experiments.sh` runs the full prompt benchmark and generates charts.
* **`hiep`** runs HIEP-Miner through the unified `dm.exe` algorithm registry: `./bin/dm.exe hiep --input <text_or_transaction_file> --input-type text|transactions --mode itemset|sequence --minsup <ratio|count> --theta-ratio <value> --window <N> --stride <N> --tokenizer faro`. HIEP text input uses `include/tokenizer/tokenizer.h`, so new tokenizer algorithms can be exposed through `dm_tokenizer_create()` and selected with `--tokenizer`. The companion scripts `scripts/run_hiep_experiments.sh` and `scripts/run_hiep_experiments.ps1` run Retail, Accidents, Chess, Dolly text, Zipfian planted-signal, ablation, and Apriori/Eclat/FP-Growth external baseline experiments through `dm.exe`, then write Q1-style charts to `results/hiep_q1/`.
* **`bpe`** runs the Sennrich-Haddow-Birch ACL 2016 BPE subword tokenizer through native C99 in `dm.exe`: `./bin/dm.exe bpe learn-bpe -i <corpus...> -m <merges> -o codes.bpe`, then `./bin/dm.exe bpe apply-bpe -c codes.bpe -i <corpus> -o corpus.bpe`. Pass multiple input corpora to `learn-bpe` for joint BPE, matching the paper's source-target vocabulary-union setup.
* **`bpe_dropout`** runs Provilkov et al.'s ACL 2020 BPE-Dropout through native C99 in `dm.exe`: first learn ordinary BPE codes with `./bin/dm.exe bpe learn-bpe ...`, then sample training-time segmentations with `./bin/dm.exe bpe_dropout -c codes.bpe -p 0.1 --seed 7 segment -i corpus.txt`. Use `sample-word` to inspect stochastic alternatives and `stats` to measure segmentation diversity. Setting `-p 0` recovers deterministic BPE; setting `-p 1` leaves character-level pieces.
* **`unigram`** runs the Kudo ACL 2018 unigram LM subword regularization tokenizer through native C99 in `dm.exe`: `./bin/dm.exe unigram train -i <corpus...> -o unigram.model --vocab-size <N>`, then `./bin/dm.exe unigram encode -m unigram.model -i <corpus> --mode viterbi|sample --alpha <value> --seed <N>`. `--mode sample` uses Forward-Filtering Backward-Sampling over all segmentations for on-the-fly subword regularization.
* **`sentencepiece`** runs a native C99 SentencePiece-style raw-text tokenizer/detokenizer from EMNLP 2018 D18-2012 through `dm.exe`: `./bin/dm.exe sentencepiece train --input raw.txt --model-type bpe|unigram --vocab-size <N> -o spm.model`, then `./bin/dm.exe sentencepiece encode --model spm.model --input raw.txt --output-format piece|id` and `./bin/dm.exe sentencepiece decode --model spm.model --input pieces.txt --input-format piece|id`. It also supports literal `--text`, `--add-bos`, `--add-eos`, `vocab`, and self-contained models with vocabulary/id mapping, BPE merges or unigram probabilities, and the U+2581 whitespace escape needed for lossless detokenization.
* **`tokenizer_lab`** runs the Dagan/Synnaeve/Roziere tokenizer-domain-adaptation toolkit through native C99 in `dm.exe`: `./bin/dm.exe tokenizer_lab train-bpe -i <corpus...> --pretokenizer gpt4|punct|identity --vocab-size <N> -o tok.json`, then `./bin/dm.exe tokenizer_lab evaluate -m tok.json -i <eval...> --baseline base.json`. It reports NSL, bytes-per-token, observed vocabulary, and Renyi entropy, plus `vocab-tradeoff` for memory/inference vocabulary-size estimates.
* **`gpe`** runs Grapheme Pair Encoding through native C99 in `dm.exe`: `./bin/dm.exe gpe train -i <corpus...> --unit grapheme --pretokenizer whitespace --vocab-size <N> -o gpe.json`, then `./bin/dm.exe gpe encode -m gpe.json -i <text>`. It also supports `pretoken-eval` for CRmax/Tokenization Parity and `units` to compare UTF-8 bytes, Unicode codepoints, and grapheme clusters.
* **`parity_bpe`** runs Parity-aware BPE from arXiv:2508.04796v2 through native C99 in `dm.exe`: `./bin/dm.exe parity_bpe train --lang-corpus en=en.txt --lang-corpus ta=ta.txt --dev-corpus en=en_dev.txt --dev-corpus ta=ta_dev.txt --merges <K> --strategy parity -o pbpe.json`, then `./bin/dm.exe parity_bpe encode -m pbpe.json -i text.txt`. It also supports `classic`, `hybrid`, and `window` strategies and `evaluate` reports per-language compression rates, tokenizer-fairness Gini, vocabulary utilization, TTR, and Renyi entropy.
* **`fast_wordpiece`** runs Song et al.'s EMNLP 2021 LinMaxMatch WordPiece tokenizer through native C99 in `dm.exe`: `./bin/dm.exe fast_wordpiece word -v vocab.txt johanson`, or `./bin/dm.exe fast_wordpiece encode -v vocab.txt -i text.txt --ids`. It builds the trie, failure links, and failure pops from the WordPiece vocabulary and supports BERT-style `##` suffix tokens, `[UNK]`, punctuation splitting, and numeric token-id output.
* **`volt`** runs VOLT (Vocabulary Learning via Optimal Transport, ACL 2021) through the `dm` dispatcher: first learn BPE candidates with `dm --train --algo bpe -i corpus.txt -m 10000 -o bpe.txt`, then find the optimal vocabulary with `dm --train --algo volt -i corpus.txt --bpe-model bpe.txt --S-min 1000 --S-max 10000 --S-step 1000 -o vocab.json --stats`. The dispatcher calls `dm_volt.exe` directly without a subcommand; the Python reference implementation at `scripts/volt.py` accepts identical flags.
* **`maximal_munch`** runs Reps' TOPLAS 1998 linear-time maximal-munch scanner through `dm.exe`: `./bin/dm.exe maximal_munch --dfa spec.dfa --input text.txt`, or add `--stats` to report tokens, errors, transitions, backtracks, failed-table hits, failed-table marks, and optimized `Tab` states. The DFA spec uses `states N`, `start Q`, `final Q TOKEN_ID NAME`, and `trans FROM SYMBOL TO`; symbols may be `a`, `'a'`, `0x61`, `0x30-0x39`, or `ANY`.
* **`huciminer`** implements Sahoo, Das, and Goswami's HUCI-Miner flow: utility-list HUI mining, level-wise high utility closed itemset/generator derivation, and HGB rule-basis statistics. Use `./bin/dm.exe huciminer <utility_dataset> 1 <min_utility> [min_uconf]` or `./bin/dm_run --algorithm huciminer --input <utility_dataset> --minutil <value> --minconf <value>`.
* **`chuo_miner`** is a standalone utility-dataset miner: `./bin/chuo_miner --input <utility_dataset> --minutil <value> --minsup <ratio|count> --minocc <value>`. The companion script `scripts/run_chuo_experiments.sh` runs Foodmart, Liquor, and Chainstore CHUO benchmarks and generates charts.
* **`pso_classifier`** is a standalone supervised classification-rule miner: `./bin/pso_classifier --input <spmf_class_folder> --particles <N> --threshold <T> --radius <R>`. The companion script `scripts/run_pso_classifier_experiments.sh` runs the malware-family classification benchmark and prints statistics only.
* **`tku_miner`** is a standalone top-k utility miner: `./bin/tku_miner --input <utility_dataset> --k <N> [--max-depth N] [--max-seconds S]`. The companion script `scripts/run_tku_experiments.sh` runs Foodmart and Liquor TKU benchmarks and prints statistics only.
* **`tku_pso_miner`** is a standalone heuristic top-k utility miner using TKU-PSO particle initialization, PEV checking, explored-particle caching, fitness estimation, pBest/gBest bit-difference updates, and minimum-solution-fitness threshold raising: `./bin/tku_pso_miner --input <utility_dataset> --k <N> --population <N> --iterations <N>`. The companion script `scripts/run_tku_pso_experiments.sh` runs statistics-only utility benchmarks.
* **`kclotree_miner`** is a standalone top-k closed sequential-pattern miner: `./bin/kclotree_miner --input <spmf_sequence_file_or_folder> --k <N> --type generic|group|redundancy_aware`. The companion script `scripts/run_kclotree_experiments.sh` runs the malware-sequence benchmark and prints statistics only.
* **`tipn_houi_miner`** is a standalone top-k high-on-shelf utility miner for positive/negative utility datasets: `./bin/tipn_houi_miner --input <negative_utility_dataset> --k <N> --intervals <N>`. The companion script `scripts/run_tipn_houi_experiments.sh` runs bounded Retail/Mushroom negative-utility benchmarks and prints statistics only.
* **`htk_miner`** is a standalone top-k frequent itemset miner using the paper's vertical BSN representation, top-k singleton initialization with ties, equivalence-class joins, and Q-Heap threshold raising: `./bin/htk_miner --input <transaction_dataset> --k <N> [--mode bsn]`. The companion script `scripts/run_htk_experiments.sh` runs transactional FIMI-style benchmarks and prints statistics only.
* **`topkphm_miner`** is a standalone top-k periodic high-utility itemset miner using periodic utility-lists, dynamic top-k utility thresholding, and EUSCS pruning: `./bin/topkphm_miner --input <utility_dataset> --k <N> --maxper <N> --maxavg <N>`. The companion script `scripts/run_topkphm_experiments.sh` runs statistics-only utility benchmarks.
* **`laga`** is a self-contained synthetic data generator: `./bin/dm.exe laga --input <dataset> --output <synthetic_file> --schema auto|transaction|sequence|text|tabular|utility [--target-size N] [--minsup ratio|count] [--tau-copy V] [--iterations N]`. It writes the synthetic dataset plus `<synthetic_file>.stats`.
* **`dm_connect`** is the universal connector smoke-test tool for the next-generation core: `./bin/dm_connect --input <path> --connector spmf|text|graph`. It uses mmap input and an arena-backed flat `uint32_t[] + row_offsets[]` representation without writing intermediate files.
* **`dm_run`** is the unified connector runner: `./bin/dm_run --algorithm <id> --input <path> --connector spmf|text|graph`. It first builds the arena-backed `DM_FlatDataset` when the algorithm can use flat item ids, then runs either a native flat plugin or a legacy `DM_Algorithm` through a flat-to-family adapter (`transactional`, `utility`, `sequence utility`, `quantity`, or `matrix`). Algorithms whose paper format carries extra semantics, such as HUCI utility values or PSO class folders, are exposed as raw plugins through the same command. Built-in adapters include `flat_stats`, `topk_items`, `chuo`, `huciminer`, `tku_miner`, `tku_pso`, `htk_miner`, `hupp`, `kclotree_miner`, `tipn_houi`, `topkphm`, and `pso_classifier`.
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

**[126]** K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90 *(ResNet-18 / 34 / 50 / 101 / 152)*

**[152]** G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4700–4708. arXiv:1608.06993v5. https://arxiv.org/abs/1608.06993 *(DenseNet-121 / 169 / 201 / 264; DenseNet-BC CIFAR variants L=40/100/190/250, k=12/24/40; growth rate k; bottleneck 1×1→3×3; θ=0.5 compression; L(L+1)/2 dense connections per block)*

**[153]** L. Huang, Y. Yuan, J. Guo, C. Zhang, X. Chen, and J. Wang, "CCNet: Criss-Cross Attention for Semantic Segmentation," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 43, no. 6, pp. 2193–2205, 2021. arXiv:1811.11721v2. https://arxiv.org/abs/1811.11721 *(CCNet; CrissCrossAttention primitive; RCCA R=2 shared params; Category Consistent Loss Eqs. 3-7; Cityscapes 81.4% mIoU; ADE20K 45.22% mIoU)*

**[154]** Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Y. Hou, and M. Tegmark, "KAN: Kolmogorov-Arnold Networks," in *Proc. International Conference on Learning Representations (ICLR)*, 2025. arXiv:2404.19756. https://arxiv.org/abs/2404.19756 *(KAN; KANLinear primitive; B-spline edge activations Eq. 15-17; Cox-de Boor recurrence; adaptive grid Section 2.5; grid extension Eq. 18; sparsity regularisation Eqs. 19-22; symbolic regression; [2,5,1]/[784,100,10] architectures)*

**[155]** K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2, pp. 99–127, 2002. Technical Report TR-AI-01-290, UT Austin, 2001. *(NEAT; historical markings / innovation numbers Section 3.2; speciation + fitness sharing Eqs. 1-2 Section 3.3; minimal initial structure Section 3.4; structural mutations Figure 3; crossover Figure 4; modified sigmoid phi(x)=1/(1+exp(-4.9*x)) Section 4.1; XOR benchmark Section 4.2; topological activation order; neat_run / neat_decode / neat_activate API)*

**[127]** K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in *Proc. International Conference on Learning Representations (ICLR)*, 2015. arXiv:1409.1556v6. https://arxiv.org/abs/1409.1556 *(VGGNet — VGG-A / B / C / VGG-16 / VGG-19)*

**[128]** J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788. https://doi.org/10.1109/CVPR.2016.91 *(YOLO v1)*

**[129]** D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," in *Proc. 2nd International Conference on Learning Representations (ICLR)*, 2014. arXiv:1312.6114v11. https://arxiv.org/abs/1312.6114 *(VAE)*

**[130]** A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861v1, 2017. https://arxiv.org/abs/1704.04861 *(MobileNet v1 — α ∈ {1.0, 0.75, 0.5, 0.25})*

**[131]** T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient Estimation of Word Representations in Vector Space," arXiv:1301.3781v3, 2013. https://arxiv.org/abs/1301.3781 *(Word2Vec — CBOW / Skip-gram with Negative Sampling)*

**[132]** A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. 40th International Conference on Machine Learning (ICML)*, 2023, pp. 28492–28518. https://proceedings.mlr.press/v202/radford23a *(Whisper — Tiny / Base / Small / Medium / Large)*

**[133]** H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom, "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288 *(Llama 2 — stories110k / 7B / 13B / 70B; native C++ port of karpathy/llama2.c)*

**[134]** Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov, "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context," in *Proc. 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, Florence, Italy, pp. 2978–2988, 2019. https://aclanthology.org/P19-1285 *(Transformer-XL — tiny / base / large / xl)*

**[135]** A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752v2, 2023. https://arxiv.org/abs/2312.00752 *(Mamba — tiny / 130M / 370M / 790M / 1.4B)*

**[136]** B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, H. Cao, X. Cheng, M. Chung, M. Grella, K. K. GV, X. He, H. Hou, P. Kazienko, J. Kocoń, J. Kong, B. Koptyra, H. Lau, K. S. I. Mantri, F. Mom, A. Saito, X. Tang, B. Wang, J.-X. Wang, S. Wong, and R. Zhu, "RWKV: Reinventing RNNs for the Transformer Era," arXiv:2305.13048v2, 2023. https://arxiv.org/abs/2305.13048 *(RWKV — tiny / 169M / 430M / 1.5B / 3B / 7B / 14B)*

**[137]** E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685v2, 2021. https://arxiv.org/abs/2106.09685 *(LoRA — rank 1/2/4/8/16, LoRALinear, LoRAEmbedding, inject\_lora)*

**[138]** Llama Team, AI @ Meta, "The Llama 3 Herd of Models," arXiv:2407.21783v3, 2024. https://arxiv.org/abs/2407.21783 *(Llama 3 — tiny / 8B / 70B / 405B; GQA all sizes; RoPE θ=500k; 128K vocab)*

**[139]** H. Wang, S. Ma, R. Wang, and F. Wei, "BitNet a4.8: 4-bit Activations for 1-bit LLMs," arXiv:2411.04965v1, 2024. https://arxiv.org/abs/2411.04965 *(BitNet a4.8 — 700M / 1.3B / 3B / 7B; INT4 absmean QKV/FFN-up, INT8+TopK out-proj, ReLU²GLU down-proj)*

**[140]** N. Hansen, "The CMA Evolution Strategy: A Tutorial," arXiv:1604.00772v2, 2023. https://arxiv.org/abs/1604.00772 *(CMA-ES — (μ/μ_W, λ)-CMA-ES with active CMA; Jacobi eigendecomposition; CSA step-size control; default parameters Table 1)*

**[141]** M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. arXiv:2301.08243v3. https://arxiv.org/abs/2301.08243 *(I-JEPA — ViT-B/16 / ViT-L/16 / ViT-H/14; multi-block masking; EMA target encoder; representation-space L2 loss; 81.1% ImageNet linear-probe)*

**[142]** A. Bardes, J. Ponce, and Y. LeCun, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning," in *Proc. International Conference on Learning Representations (ICLR)*, 2022. arXiv:2105.04906v3. https://arxiv.org/abs/2105.04906 *(VICReg — joint-embedding SSL; variance hinge Eq. 1–2, covariance penalty Eq. 3–4, invariance MSE Eq. 5; weighted loss Eq. 6 λ=μ=25 ν=1; 73.2% ImageNet Top-1 with ResNet-50; multi-modal image-text retrieval)*

**[143]** J. Hermann, Z. Schätzle, and F. Noé, "Deep-neural-network solution of the electronic Schrödinger equation," *Nature Chemistry*, 12, 891–897, 2020. arXiv:1909.08423v5. https://arxiv.org/abs/1909.08423 *(PauliNet — Slater–Jastrow–backflow DNN wave function; SchNet electron interaction Eq. 11; cuspless RBF Eqs. 12–14; electron cusp Eq. 9; VMC gradient Eq. 8; 99.99% H₂ correlation energy; N⁴ scaling)*

**[144]** D. Pfau, J. S. Spencer, A. G. D. G. Matthews, and W. M. C. Foulkes, "Ab-Initio Solution of the Many-Electron Schrödinger Equation with Deep Neural Networks," arXiv:1909.02487v3, 2021. https://arxiv.org/abs/1909.02487 *(FermiNet — antisymmetric neural wave function; one- and two-electron streams; multi-determinant Slater expansion; exponential envelopes; VMC local-energy optimization)*

**[145]** K. T. Schutt, P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, and K.-R. Muller, "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions," in *Advances in Neural Information Processing Systems 30*, 2017. https://papers.nips.cc/paper_files/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html *(SchNet — continuous-filter convolution; radial filters; smooth energy model; force prediction as negative energy gradient; QM9, MD17, ISO17)*

**[146]** O. T. Unke and M. Meuwly, "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges," arXiv:1902.08408v2, 2019. https://arxiv.org/abs/1902.08408 *(PhysNet — gated message-passing HDNN; exponential-distance RBF attention; charge correction; damped electrostatics; energies, forces, dipoles, partial charges)*

**[147]** M. Medvidovic and J. Robledo Moreno, "Neural-network quantum states for many-body physics," arXiv:2402.11014v2, 2024. https://arxiv.org/abs/2402.11014 *(NQS/VMC review — variational neural wave functions; RBM ansatz; local observables; Metropolis-Hastings sampling; spin Hamiltonians; transverse-field Ising model)*

**[148]** R. Lahoz-Beltra, "Solving the Schrödinger Equation with Genetic Algorithms: A Practical Approach," *Computers*, 11(12), 169, 2022. https://doi.org/10.3390/computers11120169 *(GA Schrödinger solver — chromosomes as wave-function samples; residual fitness exp(-Z); particle in a box; harmonic oscillator; simplified hydrogen radial equation; quantum neuron and circuit toy models)*

**[149]** J. Brandstetter, D. Ruhe, and P. Forré, "Clifford Group Equivariant Neural Networks," in *Advances in Neural Information Processing Systems 36*, 2023. https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks *(CGENN — Clifford algebra multivectors; geometric product equivariant layers; grade projections; O(n)/E(n)-equivariant parameterizations)*

**[150]** M. Henderson, R. Shakya, S. Pradhan, and T. Cook, "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits," *Quantum Machine Intelligence*, vol. 2, no. 1, p. 2, 2020. https://doi.org/10.1007/s42484-020-00012-y *(QuanvNd — virtual quantum circuit in sliding window; RY embedding; parameterised entanglement gates; Born-rule measurement; 1-D/2-D/3-D template; LibTorch core primitive dm::prim)*

**[151]** oh-mah-c, "OhmC1: A dm-Native Autoregressive Language Model Family with QK-Normalization and Sandwich-Norm Blocks," dm repository, 2026. https://github.com/oh-mah-c/dm/tree/main/src/models/nlp/ohmc1 *(OhmC1 — tiny/small/medium/large; GQA tiered KV-heads 1/2/4/8; QKNorm per-head Q/K RMS normalization before RoPE; SwiGLU FFN; sandwich-norm blocks 4 RMSNorm per block; RoPE theta size-scaled 10k/50k/200k/500k; 64K vocabulary; untied embeddings; optional C++ BPE tokenizer)*

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
