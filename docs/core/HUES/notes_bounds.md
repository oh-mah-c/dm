# Upper bounds for redefined HUEM (HUE-Span, ADMA 2019, Definitions 12-14) — working notes

Status labels: **proved** (complete argument below), **measured** (script + result file),
**open** (not established).

## Setting

Complex event sequence t = 1..n, tSE_t = events at t, u(x, t) > 0, u(tSE_t) = sum. Episode
alpha = <SE_1..SE_k>. Minimal occurrences (MO) with te - ts + 1 <= maxDur. Utility of an MO
(Def. 12, reading I1): max over matchings ts = t_1 < ... < t_k = te of sum_i u(SE_i, t_i).
u(alpha) = sum over MOs. Search tree: alpha = <SE_1..SE_k>, children are simultaneous
extensions <SE_1..SE_{k-1}, SE_k + x> (x after every event of SE_k in a fixed total order) and
serial extensions <SE_1..SE_k, {x}>. Descendant = transitive closure.

## Fact 1 (measured): ERU is not an upper bound under Definition 12

Sequence (maxDur = 5): t1 A(1), t2 B(1), t3 B(9) C(1), t4 C(1), t5 D(1); total 14.
alpha = <(A)(B)(C)>: only MO [1,3], u = 3. ERU(alpha) = 3 + 0 + u(t4) + u(t5) = 5.
beta = <(A)(B)(C)(D)> (serial descendant): MO [1,5], best matching A1 B3 C4 D5, u = 12 > 5.
With minUtil = 0.5 (7), beta is a HUE and is missed by the SPMF jar (2026-09-24 build), by the
port, and by the corrected-utility variant (HUE_SPAN_FIXED=1) with ERU, with ERU without EEUCP,
and with EWU (the EWU path also misses <(A)(B)(D)>, u = 11). Oracle: oracle_huespan.py.
Reason: beta may match alpha's sets at time points different from those of alpha's MO
(here B at t3 = te(alpha)), while ERU only adds the events of te after SE_k and the time points
after te.

## Lemma 1 (proved): MO starts of a descendant are MO starts of the ancestor

Let beta = <B_1..B_m> be a descendant of alpha = <B_1..B_{j-1}, P>, P a non-empty initial
segment of B_j. If [s, e] is an MO of beta with e - s + 1 <= maxDur, then alpha has an MO
[s, te] with te - s + 1 <= maxDur.

Proof. Take any matching s = t_1 < ... < t_m = e of beta. (t_1..t_j) matches alpha (P is a
subset of B_j), so alpha has an occurrence starting at s; let te be its earliest end, te <= t_j <= e,
hence te - s + 1 <= maxDur. Suppose [s, te] is not minimal: there is an occurrence [a, b] of
alpha with [a, b] a proper sub-interval of [s, te]; since te is the earliest end from s, a > s.
If j = 1, alpha = <P> and te = s, so no such [a, b] exists. If j >= 2, take a matching
a = u_1 < ... < u_j = b of alpha; then u_1 < ... < u_{j-1} < b <= te <= t_j < t_{j+1} < ... < t_m
is a matching of beta (B_1..B_{j-1} at u_1..u_{j-1}, B_j at t_j, the rest unchanged), i.e.
[a, e] is an occurrence of beta with a > s, contradicting the minimality of [s, e]. QED.
MOs have pairwise distinct starts, so the map MO(beta) -> MO(alpha), [s, e] -> [s, te(s)],
is injective.

## Proposition 2 (proved): start-window utility is an upper bound

SWU(alpha) = sum over MOs [s, te] of alpha with te - s + 1 <= maxDur of
             sum_{t = s}^{min(n, s + maxDur - 1)} u(tSE_t).
For every descendant beta of alpha (and alpha itself), u(beta) <= SWU(alpha).

Proof. By Lemma 1 every counted MO [s, e] of beta maps injectively to a counted MO of alpha
with the same start. A matching uses each time point at most once and only time points in
[s, e], subset of [s, s + maxDur - 1], and u(B_i, t_i) <= u(tSE_{t_i}). QED.

SWU is looser than ERU (it ignores the utility already fixed by alpha's matching), so it prunes
less; see Proposition 3 for a tighter valid bound.

## Proposition 3 (proved): MERU, the ERU taken over every feasible end, is an upper bound

For an MO start s of alpha = <SE_1..SE_k> (last set P = SE_k), let lim = min(n, s + maxDur - 1)
and, for s <= t' <= lim, U(s, t') = max over matchings s = t_1 < ... < t_k = t' of alpha of
sum_i u(SE_i, t_i) (undefined if none), r(t') = sum of u(x, t') over x in tSE_t' with x greater
than every event of P, W(a, b) = sum_{t=a}^{b} u(tSE_t). Define
  MERU(alpha) = sum over MOs [s, te] of alpha of  max_{t'} [ U(s, t') + r(t') + W(t' + 1, lim) ].
For alpha and every descendant beta: u(beta) <= MERU(alpha) <= SWU(alpha).

Proof. Let [s, e] be a counted MO of beta = <B_1..B_m> with best matching t_1..t_m, alpha =
<B_1..B_{j-1}, P>. By Lemma 1, s is the start of a counted MO of alpha (injectively). Then
(t_1..t_j) is a matching of alpha ending at t' = t_j <= e <= lim, so
sum_{i<j} u(B_i, t_i) + u(P, t_j) <= U(s, t'). B_j \ P contains only events greater than every
event of P (descendants only add larger events to the last set, then append sets), so
u(B_j \ P, t_j) <= r(t'). The remaining sets lie at distinct time points of (t', e], so their
utility is <= W(t' + 1, lim). Summing gives u(beta, [s, e]) <= the term of s. The second
inequality holds because U(s, t') + r(t') <= W(t', t') + W(s, t' - 1). QED.
The ERU of the paper keeps only t' = te(s) (with U replaced by the MO utility); Fact 1 is a
case where the maximum is attained at t' > te.

## Open items

- Does the AWU single-event pruning remain valid under Definition 12? (not proved)
- Does EEUCP (co-occurrence pruning) remain valid? (not proved)
- Measured pruning power and runtime of MERU versus SWU and versus (invalid) ERU.
- Lemma 1 relies on the search tree order; the reference implementation also restricts serial
  extensions to t <= min(ts + maxDur - 1, te of the next occurrence) (Algorithm 3, line 4);
  completeness of that restriction under Definition 12 is not proved.
