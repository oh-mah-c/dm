# FLCHUSPM re-implementation notes

Paper: T. Truong, H. Duong, B. Le, P. Fournier-Viger, U. Yun, "Mining interesting sequences with
low average cost and high average utility", Applied Intelligence 52 (2022) 7136-7157,
doi:10.1007/s10489-021-02505-0 (official version; kept outside the repository for copyright reasons, /home/hutech/self/papers_thirdparty/docs/).
No reference implementation was found (the paper reports a Java 1.8 implementation; it is not
in SPMF 2.67 and the paper gives no link).

Code: src/algorithms/flchuspm.c, include/algorithms/flchuspm.h, CLI in src/main.c:

    ./build/bin/dm.exe flchuspm <file> 0 <mins> <minu> <maxc> [pct]
    env: DM_FLCHUS_OUT=<file> (patterns), FLCHUS_TRACE=1 (CULs, CMAPs, per-node bounds),
         FLCHUS_DPS=0|1, FLCHUS_WPS=0|1 (ablations of Section 6.1)

## Choices where the paper leaves a detail open

1. Input format (the paper's QSDBs are not published): one q-sequence per line, q-items
   `item:cost:utility`, `-1` closes an itemset, `-2` closes the sequence.
2. Definition 4: u(alpha, Psi) and c(alpha, Psi) are maxima over occurrences, taken
   independently (confirmed by the running example: c(d->c, Psi4) = max{5, 4} = 5).
3. m = mins is used as an integer count (Definition 5 sums the m largest values); in the
   percentage form m = ceil(mins% / 100 * |D|); minu% and maxc% are taken w.r.t. maxUSeq and
   maxCSeq of the original database.
4. cf of extensions follows Section 5.1.3 literally: i-extension, t(alpha).c_end + t(y).c_end of
   the first pair with equal ends; s-extension, the first t(alpha) plus the first t(y) that has a
   t(alpha) before it.
5. RedS (Fig. 3, lines 1-3) is applied once: S is computed on the original database, items not
   in S are removed, then the ISU vector and the CULs are recomputed (not iterated to a fixpoint).
6. The item order (the paper's lexicographic order) is the integer order of item identifiers.
7. An item appears at most once in an itemset (as in the paper's definition of itemsets).
8. Ablation flags (Section 6.1 "WPS", "DPS", "All"): WPS=0 drops every AMUB test (S, RedS,
   CMAPs, I/S candidate sets) but keeps the support tests; DPS=0 drops Fig. 4 line 1.

## Verification against the paper (running example, Table 1, minu = 40, maxc = 6.6, mins = 3)

- Output FLCHUS = {d->c (au 84.33, ac 5.67, supp 3), d->f (au 69, ac 6.5, supp 4)}: matches.
- ISU after RedS = (132, 23, 168, 159): matches Section 5.1.1.
- CUL of a, c, d, e, f: identical to Fig. 1, tuple by tuple.
- iCMAP(c) = {e}, sCMAP(c) = {a, d, f}, iCMAP(d) = {}, sCMAP(d) = {a, c, f}: matches Section 4.3.1.
- The 14 visited nodes are the nodes of Fig. 5; AMUB, FUB, Flb, au, ac, supp match for 12 of
  them. Differences, all on pruned nodes (no effect on the output) and consistent with typos
  in the figure: c->a Flb 10.0 here vs 16.25 in Fig. 5 (16.25 is ac(c->a)); f Flb 1.67 here vs
  12.25 in Fig. 5 (12.25 is ac(e)); the superscript of d->a in Fig. 5 reads "148.33;44;9.333",
  here AMUB 153, FUB 148.33, Flb 9.333, au 44.
- Text typo in Section 5.1.3: "t(gamma).ru_end = t(c).ru_end = 36" for end 4, while the tuple
  (and Fig. 2b) gives 50.

## Brute-force oracle (oracle_flchus.py, Definitions 1-4, 500 random QSDBs, seed 7)

| configuration | oracle patterns | mismatching cases | missing | extra | wrong values |
|---|---|---|---|---|---|
| as in the paper (DPS + WPS) | 21,807 | 2 | 2 | 0 | 0 |
| DPS disabled | 21,807 | 0 | 0 | 0 | 0 |

## Finding: Theorem 2b does not hold, so FLCHUSPM can miss FLCHU sequences

Theorem 2b states Flb(beta) >= Flb(alpha) for every forward extension beta of alpha. For an
i-extension, the first occurrence of beta may end at a later itemset whose alpha-part is cheaper
than the first occurrence of alpha. Minimal example (data/cx_flb.txt), one q-sequence
(a, cost 100, utility 10) -> (a, 1, 10)(b, 1, 10), mins = 1, minu = 0, maxc = 10:
Flb(a) = 100 > maxc, so DPS prunes branch(a), but ac(ab) = 2 <= maxc and au(ab) = 20 >= minu:
FLCHUS = {b, ab}, FLCHUSPM outputs {b}. The two oracle mismatches above are of this kind.
Analysis (not yet proved formally): for s-extensions the cf rule of Section 5.1.3 adds a
non-negative cost to the first tuple of alpha, whose cost is >= cf(alpha), so the violation
seems specific to i-extensions; with DPS disabled the oracle finds no other discrepancy.

## Proposed algorithm FLCHUS-X (src/algorithms/flchus_x.c): bounds and proofs

Notation: CUL tuples of alpha in Psi are (e; c_e, cm_e; u_e, ru_e) where c_e (cm_e) is the
maximum (minimum) cost of the occurrences of alpha ending at itemset e; e_1 is the first end.

**MLB (cheapest occurrence).** lbmin(alpha, Psi) = min_e cm_e; MLB(alpha) = mean of the m
smallest lbmin over rho(alpha). Every occurrence of a super-sequence beta contains an occurrence
of alpha, so c(beta, Psi) >= lbmin(alpha, Psi); with rho(beta) subset of rho(alpha) the mean of
the m smallest values gives ac(beta) >= MLB(beta) >= MLB(alpha) for EVERY super-sequence beta
with supp >= m. Used for depth pruning of alpha and its subtree, item removal (RedS, repeated to
a fixpoint), iCMAP/sCMAP and the candidate sets I/S.

**MDB (descendant bound, Lemma D).** For Psi in rho(alpha) let
  lbD(alpha, Psi) = min( c_{e_1} if an itemset follows e_1 in Psi,
                         min { c_e : the itemset e contains an item greater than last(alpha) } ),
(infinite if neither exists), and MDB(alpha) = mean of the m smallest finite lbD (infinite if
fewer than m). Claim: for every proper forward-extension descendant beta of alpha (any node of
the subtree of alpha other than alpha) with supp(beta) >= m, ac(beta) >= MDB(alpha).
Proof. Let Psi in rho(beta) and beta' any occurrence of beta in Psi; its alpha-part ends at some
e, and the remaining q-items of beta' are, by the definition of forward extensions, either in
later itemsets (the first extension step after alpha is an s-extension) or in itemset e and
greater than last(alpha) (the first step is an i-extension).
 - s-extension first: replace the alpha-part by an occurrence of alpha ending at e_1 <= e with
   cost c_{e_1} (the maximum at e_1); the remaining q-items are after e >= e_1, so the result is
   an occurrence of beta and c(beta, Psi) >= c_{e_1} (c(beta, Psi) is a MAXIMUM over occurrences).
   An itemset follows e_1 because one follows e.
 - i-extension first: replace the alpha-part by the most expensive occurrence of alpha ending at
   the same e (cost c_e); the q-items at e greater than last(alpha) are unchanged, so the result is
   an occurrence of beta and c(beta, Psi) >= c_e, where itemset e contains an item greater than
   last(alpha).
In both cases c(beta, Psi) >= lbD(alpha, Psi) (finite), and rho(beta) is a subset of the
sequences with finite lbD; the mean-of-m-smallest argument gives ac(beta) >= MDB(alpha). QED
MDB is only valid for the subtree (forward extensions), not for all super-sequences, so it is
used only to stop the expansion of a node AFTER the node itself has been tested for output.
Since c_e >= cm_e, lbD >= lbmin and MDB >= MLB.
Relation to Flb: for s-extension descendants lbD equals the first-occurrence cost used by Flb
(maximum at the first end); the i-extension term is what makes it sound.

Verification (oracle_flchus.py): with MDB, 500 + 1,000 random QSDBs (seeds 7, 8) and 400 long
repetitive QSDBs (LONG=1, seed 9), 0 differences; each component disabled in turn (DESC, COSTW,
FIX, SEL), 500 QSDBs each, 0 differences (results/oracle_ablation_mdb.log).
