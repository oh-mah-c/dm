# Revision notes (prepared 2026-10-21, apply at the first revision)

## 1. Credit DFHOI for the uniform-length fact
The statement "if all transactions have the same length, UBO(X) = supp(X)" is Theorem 3 of
L.T.T. Nguyen, T. Mai, G.-H. Pham et al., "An efficient method for mining high occupancy itemsets
based on equivalence class and early pruning", Knowledge-Based Systems 267 (2023) 110441
(checked in the full text). Cite it at the proposition ("[DFHOI, Theorem 3]") and, in the abstract and
introduction, present it as known ("as shown by Nguyen et al.") rather than as an observation of
this paper. The contribution remains the consequence on dense data and the new bound.

## 2. Cite the most recent related work
T.-K. Ngo, H.-S. Nguyen, W. Pedrycz, B. Vo, "Efficient algorithms for mining top-k high occupancy
itemsets", Information Sciences 741 (2026) 123226, doi:10.1016/j.ins.2026.123226. Top-k HOI
mining; uses the UBO of HEP/DFHOI, skips its computation when the supporting transactions have
equal length, early termination of the UBO loop, pair matrix EOCS. No bound tighter than UBO.

## 3. Cross-cite the companion manuscript
AURA-HOI (HOIs, length-capped residual bound) and MFHOI (maximal HOIs, size-coupled bound) are
under review at different journals. Both bounds start from occ(Y) = |Y| * sum_{q in T(Y)} 1/|t_q|
and couple the size of a descendant with the transactions that can contain it. Suggested text:
"A companion manuscript [ref, under review] studies maximal HOIs with a bound that, unlike the
present one, uses for each supporting transaction the number of tail items it contains
(AURA-HOI uses one cap K per branch); the problems, algorithms and experiments differ."
Consider posting both as preprints first (check each journal's preprint policy) so that they can
be cited, and tell the editors if the submission form asked about related manuscripts.
