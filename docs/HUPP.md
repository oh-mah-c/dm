# High-Utility Prompt Pattern Mining for Generative AI Systems

## Research Landscape and the Unsolved Gap

The current optimization stack around LLM prompting is split into several largely independent research traditions. Prompt-compression work focuses on shrinking a **single** prompt while trying to preserve output quality. Selective Context prunes redundant context to cut memory and latency while maintaining comparable downstream performance; LLMLingua introduced coarse-to-fine prompt compression with a budget controller and iterative token selection, reporting up to 20× compression with little performance loss; LongLLMLingua targeted long-context settings and reported better long-context performance together with substantial token-cost and latency reductions; and LLMLingua-2 reframed compression as token classification and reported 3×–6× faster compression than prior prompt-compression approaches together with 1.6×–2.9× end-to-end latency gains. These papers are important because they show that prompt length is economically and computationally consequential, but they all optimize prompt instances one-by-one rather than mining recurring high-value patterns across historical prompt logs. citeturn5view3turn5view0turn6view0turn6view1

A second strand optimizes **serving reuse** for LLMs and RAG pipelines. Prompt Cache reuses attention states for recurring prompt modules and reports large time-to-first-token gains for prompts with overlapping segments, while RAGCache caches intermediate states of retrieved knowledge in a knowledge tree and reports up to 4× lower time-to-first-token and up to 2.1× higher throughput. On the semantic-caching side, GPTCache stores embeddings of prompts and responses for similarity-based reuse, GPT Semantic Cache reports API-call reductions up to 68.8%, and MeanCache shows that privacy-aware federated semantic caching can reduce storage by 83% while improving cache-hit decision quality over GPTCache. Newer work such as CacheRAG extends caching from answers and KV states to retrieval planning itself. Yet this whole line is fundamentally a **systems reuse** line: it caches or replays previous work, but it does not discover semantically meaningful prompt-concept sets whose repeated occurrence predicts a favorable cost–accuracy trade-off. citeturn6view3turn6view2turn5view6turn0search1turn5view7turn0search2

A third strand treats prompt construction as an optimization problem over full prompt strings or pipelines. Automatic Prompt Engineer (APE) searches over instruction candidates; Promptbreeder evolves prompts through self-referential mutation; DSPy compiles declarative LM pipelines into self-improving prompt programs; and recent work on logged-bandit prompt optimization treats prompts as actions and optimizes a prompt policy from partial user feedback. These methods are powerful, but their search object is the **whole prompt** or the **prompt policy**, not recurring semantic subsets within prompt logs. The logged-bandit work is particularly revealing: it explicitly notes that treating each prompt independently produces a very large action space and high-variance off-policy learning, which is almost the opposite of what itemset mining is good at. citeturn6view6turn8view0turn5view8turn15view0

On the data-mining side, HUIM has a mature algorithmic core. Two-Phase introduced transaction-weighted utilization as a safe anti-monotone upper bound; HUI-Miner introduced utility-lists to remove candidate generation; FHM reduced join cost with estimated utility co-occurrence pruning; EFIM added local/subtree utility bounds, fast array-based counting, projection, and transaction merging; and HUOPM / HUOPM+ extended utility mining toward frequency–utility–occupancy settings using utility-occupancy lists and length-aware upper bounds. These algorithms are efficient because their utility model is known **before** mining and can be summarized transaction by transaction in static numeric structures. citeturn12search3turn11search1turn10view0turn5view10turn6view4turn6view5

There is now also early evidence that prompt logs themselves can be mined structurally. A 2026 HICSS paper mines hidden prompt-engineering patterns with formal concept analysis and association rules, showing that prompt properties and output quality can indeed be analyzed as a pattern-mining problem. But that work operates on hand-coded prompt properties and quality relations, not on semantic concept itemsets, not on token-cost/latency/accuracy utility, and not with HUIM-style exact upper-bound mining. citeturn14view0

That combination of facts defines the gap. To the best of the literature reviewed here, there is still no formulation that treats **prompt logs as a transactional database of semantic concepts**, defines a **dynamic utility** over token saving, latency saving, and response alignment, and then mines **reusable prompt templates** with **exact HUIM-style pruning**. Existing prompt-compression papers are intra-prompt, existing cache papers are reuse systems, existing prompt-optimization papers search over full prompts or policies, and existing HUIM papers assume additive, pre-specified utilities that do not depend on which rewrite/compression plan is selected for the whole prompt. citeturn5view0turn6view0turn6view1turn6view2turn6view3turn5view7turn6view6turn8view0turn5view8turn15view0turn12search3turn11search1turn10view0turn6view4turn14view0

## Formalizing High-Utility Prompt Pattern Mining

I propose the following new problem family: **High-Utility Prompt Pattern Mining**, abbreviated **HUPPM**. The basic insight is that a prompt log can be formalized as a database of transactions, where each transaction contains semantic concepts extracted from a user request, while the utility of a pattern is not a static “price × quantity” term but the **best achievable optimization gain** among compression and retrieval plans that preserve that pattern’s semantics.

### Data Model

Let the prompt-log database be

\[
D=\{T_1,T_2,\dots,T_n\}.
\]

Each transaction \(T_q\) corresponds to one prompt request (and optionally its associated RAG context, system prompt, and execution metadata). After semantic canonicalization, each prompt becomes an ordered semantic stream

\[
\mathbf{s}_q=\langle i_{q1},i_{q2},\dots,i_{q\ell_q}\rangle,
\]

and an itemset view

\[
X_q=\{i_{q1},i_{q2},\dots,i_{q\ell_q}\}\subseteq I,
\]

where

\[
I=\{i_1,i_2,\dots,i_m\}
\]

is the finite universe of semantic items. In practice, items may encode intent labels, entities, constraints, tool requirements, output-format requirements, persona markers, safety constraints, retrieval-scope markers, or other canonical semantic concepts.

For any pattern \(X\subseteq I\), define the supporting transaction set

\[
\Gamma(X)=\{T_q\in D \mid X\subseteq X_q\},
\]

and support

\[
\operatorname{sup}(X)=|\Gamma(X)|.
\]

This is the usual transactional support notion.

### Candidate Optimizer Family

For each transaction \(T_q\), let there be a finite set of candidate prompt optimizers or rewrites

\[
\Pi_q=\{\pi_{q1},\pi_{q2},\dots,\pi_{qK_q}\}.
\]

A candidate optimizer may be produced by historical successful rewrites, a local compression operator bank, a lightweight proposer model, or a RAG-context compactor. Each candidate \(\pi\in\Pi_q\) has the following attributes:

- a preserved semantic subset \(P_q(\pi)\subseteq X_q\),
- compressed prompt length \(L_q(\pi)\),
- execution latency \(\tau_q(\pi)\),
- response output \(o_q(\pi)\),
- accuracy/alignment score \(A_q(\pi)\in[0,1]\),
- and optionally a cache or reuse score \(H_q(\pi)\in[0,1]\).

Let the original prompt length and latency be \(L_q\) and \(\tau_q\). Define normalized improvements

\[
s_q(\pi)=\frac{L_q-L_q(\pi)}{L_q},
\qquad
\lambda_q(\pi)=\frac{\tau_q-\tau_q(\pi)}{\tau_q},
\qquad
a_q(\pi)=A_q(\pi).
\]

If cache-aware optimization is desired, include

\[
h_q(\pi)=H_q(\pi).
\]

The multi-objective gain vector is then

\[
\mathbf{b}_q(\pi)=
\bigl(
s_q(\pi),\,\lambda_q(\pi),\,a_q(\pi),\,h_q(\pi)
\bigr).
\]

A deployment chooses a nonnegative weight vector

\[
\mathbf{w}=
(w_s,w_\lambda,w_a,w_h),
\qquad
\sum_j w_j=1,
\]

and forms a scalarized contextual gain

\[
g_q(\pi)=
w_s\,s_q(\pi)+
w_\lambda\,\lambda_q(\pi)+
w_a\,a_q(\pi)+
w_h\,h_q(\pi).
\]

If only token-saving and accuracy are considered, set \(w_\lambda=w_h=0\).

### External Utility and Semantic Specificity

Classical HUIM uses a fixed external utility per item. In HUPPM, I keep a static, item-level notion of **semantic specificity** to reward patterns that are semantically informative rather than overly generic. Let

\[
df(i)=|\{q \mid i\in X_q\}|
\]

and define the item specificity weight

\[
\omega(i)=\log\frac{n+1}{df(i)+1}.
\]

Then the specificity of an itemset is

\[
\Omega(X)=\sum_{i\in X}\omega(i).
\]

This plays the role of a static external utility analogous to item importance, while the dynamic optimization gain \(g_q(\pi)\) captures the deployment-specific cost/accuracy benefit.

### Pattern-Conditioned Utility

For a pattern \(X\subseteq X_q\), define the feasible optimizer family

\[
\Pi_q(X)=\{\pi\in\Pi_q \mid X\subseteq P_q(\pi)\}.
\]

This means that every candidate in \(\Pi_q(X)\) preserves all semantic concepts in \(X\). Define the best achievable gain for pattern \(X\) in transaction \(q\) as

\[
M_q(X)=
\max_{\pi\in\Pi_q(X)} g_q(\pi),
\]

with \(M_q(X)=0\) if \(\Pi_q(X)=\emptyset\).

The transaction-level utility of pattern \(X\) is then

\[
u_q(X)=\Omega(X)\cdot M_q(X).
\]

The database-level utility is

\[
U(X)=\sum_{T_q\in \Gamma(X)}u_q(X)
=
\Omega(X)\sum_{T_q\in\Gamma(X)}M_q(X).
\]

For the gain-maximizing optimizer

\[
\pi_q^*(X)=\arg\max_{\pi\in\Pi_q(X)} g_q(\pi),
\]

define the average alignment of a pattern by

\[
\overline{A}(X)=
\frac{1}{\operatorname{sup}(X)}
\sum_{T_q\in\Gamma(X)}a_q\bigl(\pi_q^*(X)\bigr).
\]

### The HUPPM Problem

Given thresholds \(\sigma\) (minimum support), \(\theta\) (minimum utility), and optionally \(\alpha_{\min}\) (minimum average alignment), the **High-Utility Prompt Pattern Mining** problem is to find all patterns \(X\subseteq I\) such that

\[
\operatorname{sup}(X)\ge \sigma,
\qquad
U(X)\ge \theta,
\qquad
\overline{A}(X)\ge \alpha_{\min}.
\]

The output patterns are not merely descriptive frequent concepts: they are recurring semantic concept sets that admit high-quality prompt compression or cache-aware reuse with favorable operational benefit.

## Why Classical HUIM Bounds Do Not Transfer Directly

Standard HUIM algorithms assume that itemset utility is numerically available from item-level terms before mining, typically by summing item utilities within each transaction and across the database. Two-Phase, HUI-Miner, FHM, and EFIM all exploit this structure, and HUOPM / HUOPM+ extend it to utility-occupancy settings with carefully engineered list or tree summaries. In HUPPM, however, the reward assigned to a pattern is not simply the sum of per-item utilities: it is the best score of a **whole-prompt rewrite or cache-aware plan** chosen from a feasible family that depends on the preserved semantic subset. That is a different mathematical object. citeturn12search3turn11search1turn10view0turn5view10turn6view4turn6view5

This difference creates four concrete incompatibilities with classical HUIM. First, the utility signal is **contextual**: token saving, latency, and alignment depend on the model, the prompt instance, the retrieval context, and the chosen optimizer plan, not on a fixed item profit. Second, the utility is **non-additive** over items: the value of preserving `{"JSON output", "citations", "legal domain"}` is not the sum of three item profits; it is the score of a rewritten prompt that preserves those semantics jointly. Third, the optimization is **multi-objective** by nature, because shorter prompts are not always better if they reduce answer quality. Fourth, the value of a canonical template is partly **cross-transactional** because canonicalization improves semantic-cache and prompt-module reuse, which caching papers explicitly show to matter operationally. citeturn5view0turn6view0turn6view1turn6view2turn6view3turn5view7turn5view8turn15view0

The right replacement for classical downward closure is therefore not a direct reuse of TWU, EUCP, or local/subtree utility. The right object is a **pattern-conditioned optimizer envelope**. The key monotonic fact is:

\[
X\subseteq Y \Longrightarrow \Pi_q(Y)\subseteq \Pi_q(X).
\]

Therefore,

\[
M_q(Y)\le M_q(X).
\]

This is the fundamental property that makes HUPPM mineable.

### Prompt Transaction-Weighted Opportunity

At a search node with prefix \(X\) and suffix candidate set \(E_X\), define the remaining specificity available in transaction \(q\) as

\[
R_q(X)=\sum_{i\in E_X\cap X_q}\omega(i).
\]

For any descendant \(Y\supseteq X\) generated from the suffix,

\[
\Omega(Y)\le \Omega(X)+R_q(X),
\]

and by feasible-family shrinkage,

\[
M_q(Y)\le M_q(X).
\]

Hence for any descendant \(Y\),

\[
u_q(Y)
=
\Omega(Y)\,M_q(Y)
\le
M_q(X)\bigl(\Omega(X)+R_q(X)\bigr).
\]

Summing over all supporting transactions of \(X\) gives the new safe upper bound

\[
PTWO(X)=
\sum_{T_q\in\Gamma(X)}
M_q(X)\bigl(\Omega(X)+R_q(X)\bigr),
\]

where **PTWO** stands for **Prompt Transaction-Weighted Opportunity**.

This yields the pruning rule:

\[
PTWO(X)<\theta
\Longrightarrow
\text{no descendant of }X\text{ can satisfy }U(\cdot)\ge\theta.
\]

That is the HUPPM analogue of TWU plus remaining utility.

### Accuracy-Aware Upper Bound

If a hard alignment floor \(\alpha_{\min}\) is enforced, define

\[
A_q^{+}(X)=
\max_{\pi\in\Pi_q(X)} a_q(\pi).
\]

Sort the values \(A_q^{+}(X)\) over \(q\in\Gamma(X)\) in descending order:

\[
A^{+}_{(1)}(X)\ge A^{+}_{(2)}(X)\ge \cdots.
\]

Any descendant must still have support at least \(\sigma\), so its best possible average alignment is bounded by the top-\(\sigma\) average

\[
AAUB(X)=
\frac{1}{\sigma}\sum_{j=1}^{\sigma} A^{+}_{(j)}(X).
\]

Thus

\[
AAUB(X)<\alpha_{\min}
\Longrightarrow
\text{no descendant of }X\text{ can satisfy the alignment floor.}
\]

Together, support anti-monotonicity, feasible-family shrinkage, PTWO, and AAUB provide the pruning regime that standard HUIM lacks for this dynamic prompt-utility setting.

## HUPP-Miner

The algorithm I recommend is a new vertical, array-oriented miner named **HUPP-Miner**. It is much closer to EFIM/HUI-Miner than to classic FP-Growth, because the utility state of a pattern depends on the feasible optimizer family and cannot be represented by simple node counters in an FP-tree. A pointer-heavy tree would also be undesirable in a low-level production implementation because each node would need dynamic optimizer-state payloads. EFIM-style vertical arrays are the better fit for this utility model. citeturn11search1turn10view0turn5view10

### Core Data Structures

The first structure is a **Compressed Concept Bitmap** for singleton support filtering. Each concept \(i\) has a compressed TID bitmap \(B_i\), used for fast support counting and fast singleton / shallow-prefix intersections.

The second structure is the central one: a **High-Utility Prompt Pattern List**, or **HUPP-list**. For a pattern \(X\), each entry stores one supporting transaction and the exact optimizer-envelope state needed for utility and pruning:

```text
HUPPEntry {
    uint32_t tid;          // transaction id
    uint16_t pos;          // last-item position in the ordered concept stream
    uint64_t feas_mask;    // bitmask of feasible optimizers preserving X, K_q <= 64
    float    best_gain;    // M_q(X)
    float    exact_acc;    // alignment of argmax-gain optimizer
    float    acc_ub;       // A_q^+(X)
    float    rem_ic;       // remaining specificity R_q(X)
}
```

The third structure is the **Preserve-Mask Table**. For each transaction \(q\) and each concept \(i\in X_q\), store a bitmask

\[
keep_q(i)\in\{0,1\}^{K_q}
\]

indicating which candidate optimizers preserve that concept. Then for any pattern \(X\),

\[
feas\_mask_q(X)=\bigcap_{i\in X} keep_q(i),
\]

implemented as bitwise AND. This is the main trick that makes dynamic utility practical: instead of storing a huge prompt-template state, the miner stores a tiny feasible-optimizer mask per transaction.

The fourth structure is a **Suffix Specificity Array**. Each transaction stream is stored as a sorted concept array with suffix sums of \(\omega(i)\). This makes \(R_q(X)\) available in \(O(1)\) once the new last position is known.

In a C99/C++ implementation, these structures should be stored in contiguous arenas with structure-of-arrays layout for scan-heavy kernels. The goal is to avoid pointer chasing entirely.

### Offline Preparation

HUPP-Miner assumes an offline preparation stage:

1. Parse each prompt into semantic concepts and canonicalize them into item IDs.
2. Generate a small candidate optimizer bank \(\Pi_q\) for each transaction. This bank can come from historical prompt rewrites, deterministic rewrite schemas, context compaction operators, or a local proposer model.
3. Pareto-prune \(\Pi_q\) so only non-dominated candidates remain in \((s,\lambda,a,h)\)-space. This keeps \(K_q\) small.
4. Compute \(g_q(\pi)\), \(a_q(\pi)\), and the preserve mask \(P_q(\pi)\) for every candidate.
5. Build singleton bitmaps, Preserve-Mask Tables, and singleton HUPP-lists.

This stage is expensive only because candidate generation and evaluation exist, but it is done once offline. The **mining step itself never calls the target LLM**.

### Mining Logic

With a global item order \(\prec\) chosen as ascending support and then descending specificity, singleton HUPP-lists are built first. A singleton list for concept \(i\) contains one entry for every transaction that contains \(i\), with

\[
feas\_mask_q(\{i\})=keep_q(i),
\qquad
best\_gain=M_q(\{i\}),
\qquad
acc\_ub=A^+_q(\{i\}).
\]

The recursive mining procedure is then:

```text
HUPP-MINER(prefix X, list UL(X), suffix E)
    s  <- |UL(X)|
    if s < sigma: return

    exactU <- Omega(X) * sum_{e in UL(X)} e.best_gain
    avgA   <- (1/s) * sum_{e in UL(X)} e.exact_acc

    if exactU >= theta and avgA >= alpha_min:
        emit X as a High-Utility Prompt Pattern

    ubU <- sum_{e in UL(X)} e.best_gain * (Omega(X) + e.rem_ic)
    if ubU < theta:
        return

    accUB <- top-sigma-average of e.acc_ub over UL(X)
    if accUB < alpha_min:
        return

    for each item j in E:
        UL(Y) <- JOIN(UL(X), j)
        if UL(Y) not empty:
            HUPP-MINER(X U {j}, UL(Y), items after j)
```

The join operator is the key step. For each entry \(e\) in \(UL(X)\) whose transaction contains extension item \(j\), compute

\[
feas\_mask' = e.feas\_mask \,\&\, keep_q(j).
\]

If \(feas\_mask'=0\), then no candidate optimizer preserves \(X\cup\{j\}\) in that transaction, so that entry disappears. Otherwise, retrieve

\[
best\_gain' = \max_{\pi\in feas\_mask'} g_q(\pi),
\quad
exact\_acc' = a_q(\pi_q^*(X\cup\{j\})),
\quad
acc\_ub' = \max_{\pi\in feas\_mask'} a_q(\pi),
\]

and the new \(rem\_ic\) from the suffix-specificity array. The result is an exact HUPP-list for \(Y=X\cup\{j\}\).

This makes the pattern-conditioned dynamic utility cheap: after offline candidate evaluation, every extension step uses only sorted-list merges, bitwise AND, and tiny max lookups.

### What the Output Actually Is

The mined object is an itemset \(X\), but the deployable artifact is a **High-Utility Prompt Template**:

\[
\mathcal{T}(X)=\bigl(X,\hat{\pi}(X)\bigr),
\]

where \(\hat{\pi}(X)\) is the consensus optimizer schema induced by the gain-maximizing candidates \(\{\pi_q^*(X)\}_{q\in\Gamma(X)}\). In practice, \(\hat{\pi}(X)\) is not a literal fixed prompt string; it is a slot-based rewrite schema plus a canonical compression recipe, for example:

- remove discourse filler,
- collapse redundant instruction clauses,
- keep entity and constraint slots,
- compact format directives,
- reduce retrieved context to the learned minimal evidence profile,
- attach canonical prompt modules for reusable segments.

That distinction matters because it makes the mined patterns directly actionable in a production optimizer.

## Theoretical Properties and Complexity

### Correctness

HUPP-Miner is exact **with respect to the prepared candidate bank** \(\Pi_q\). If the candidate bank is the finite search space of allowed prompt rewrites, then for every pattern \(X\), the algorithm computes the exact

\[
M_q(X)=\max_{\pi\in\Pi_q(X)}g_q(\pi)
\]

for every supporting transaction.

The pruning rules are safe for three reasons. First, support is anti-monotone:

\[
X\subseteq Y \Rightarrow \operatorname{sup}(Y)\le \operatorname{sup}(X).
\]

Second, the feasible optimizer family shrinks monotonically:

\[
X\subseteq Y \Rightarrow \Pi_q(Y)\subseteq \Pi_q(X) \Rightarrow M_q(Y)\le M_q(X).
\]

Third, the PTWO and AAUB rules upper-bound the utility and achievable alignment of every descendant. Therefore, no qualifying pattern can be pruned incorrectly.

### Time Complexity

Let

- \(n\) be the number of prompt transactions,
- \(m\) the number of semantic items,
- \(\bar d\) the average number of semantic items per transaction,
- \(K=\max_q K_q\) the maximum number of candidate optimizers retained per transaction after Pareto pruning,
- and \(\mathcal{V}\) the set of visited search nodes.

For singleton support filtering with compressed bitmaps, counting costs

\[
O\!\left(\sum_{i=1}^{m}|B_i|\right)
\]

in compressed-bitmap time, or \(O(m\lceil n/w\rceil)\) in packed-word form.

For a node \(X\) and extension \(j\), the join cost is linear in the participating list sizes:

\[
O\bigl(|UL(X)| + |UL(j)|\bigr),
\]

or \(O(|UL(X)|)\) if singleton extension access is indexed by transaction position. The dynamic-utility update inside each matched transaction is

\[
O(\lceil K/64\rceil)
\]

for the mask intersection plus either \(O(1)\) with a tiny subset-max lookup table or \(O(\operatorname{popcount}(feas\_mask'))\) with a short branchless scan. Because \(K\) is deliberately kept small after local Pareto pruning, this term behaves like a constant.

Hence the core mining cost is

\[
O\!\left(
\sum_{X\in\mathcal{V}}
\sum_{Y\in \text{children}(X)}
|UL(Y)|
\right),
\]

which is the standard vertical-miner profile: exponential in the worst case, but sharply reduced in practice by support, PTWO, and AAUB pruning.

### Memory Complexity

A HUPP-entry stores one transaction-level envelope state for one visited pattern occurrence. Thus the active-memory footprint is

\[
O\!\left(\sum_{X\in\text{active frontier}} |UL(X)|\right)
\]

times a very small constant entry size. With a 32-bit TID, 16-bit position, 64-bit feasible-optimizer mask, and four floats, the constant is modest. The Preserve-Mask Tables require

\[
O\!\left(\sum_{q=1}^{n}|X_q|\cdot \lceil K/64\rceil\right)
\]

bits plus candidate metadata

\[
O\!\left(\sum_{q=1}^{n}K_q\right).
\]

This is substantially better than a naïve alternative that would materialize optimizer-specific prompt states under every tree node.

### Why This Is a Meaningful Leap

The scientific leap is not that HUPP-Miner is “another utility-list miner.” The leap is that it converts a previously unstructured LLM systems problem into a mineable one through the notion of a **pattern-conditioned feasible optimizer family**. That move gives the field three things classical prompt engineering and classical HUIM both lacked:

1. **An exact utility definition for prompt patterns** that is faithful to cost–accuracy trade-offs rather than static item profits.
2. **A safe upper bound**—PTWO—that plays the role of TWU in a dynamic, model-conditioned environment.
3. **A practical low-level implementation path** in which dynamic prompt utility is updated by tiny mask operations rather than repeated LLM evaluation.

That is why HUPP-Miner is not a cosmetic adaptation of EFIM or FP-Growth. It introduces a new mining object, a new utility semantics, and a new envelope bound.

## Deployment as an On-Premise Prompt Compressor and Optimizer

The natural deployment target is an on-premise gateway that sits in front of a cloud API or a local inference engine. The gateway continuously logs prompts, retrieval context, execution latency, cache events, and quality feedback; HUPP-Miner periodically mines **High-Utility Prompt Templates** from those logs; and the online path uses those templates to canonicalize new prompts before they reach the actual model.

The runtime path is simple. An incoming prompt is parsed into semantic concepts and mapped to a compact concept bitset. The gateway then queries a mined-template index—implemented as a subset trie or a compressed inverted index—to find the highest-utility pattern \(X\subseteq X_{\text{new}}\) that also satisfies the deployment’s alignment floor. It instantiates the associated template schema \(\hat{\pi}(X)\) with the current prompt’s entity, constraint, and formatting slots, thereby removing redundant tokens while preserving the semantics known to matter. If desired, a small local verifier checks that the generated compressed prompt preserves required fields or satisfies a learned risk bound; otherwise the system falls back to the original prompt.

This architecture has three concrete downstream effects. First, it directly reduces prompt length and therefore API cost or local prefill cost, which is exactly what the prompt-compression literature leverages. Second, because it **canonicalizes** semantically equivalent requests into recurring template families, it raises the effectiveness of semantic caches, prompt-module caches, and RAG-side reuse systems that benefit from repeated structure. Third, because the optimization happens on-premise, the verbose raw prompt never needs to leave the local boundary if the system chooses to transmit only the compressed version or a template-plus-slots representation. These advantages are aligned with the gains already observed for prompt compression and cache reuse in the serving literature, but HUPPM adds a missing learning layer: the templates themselves are mined from history instead of being handcrafted. citeturn5view0turn6view0turn6view1turn6view3turn6view2turn5view6turn5view7

The most important strategic implication is that HUPPM reframes LLM optimization from **online heuristic token deletion** to **offline discovery of semantically stable, economically valuable prompt templates**. In other words, instead of asking only “Which tokens can I drop in this prompt?”, HUPP-Miner asks the more operationally meaningful question: “Which recurring semantic concept sets, across my entire prompt history, admit a high-value canonical prompt form that reliably preserves answer quality?” That is the new research domain.