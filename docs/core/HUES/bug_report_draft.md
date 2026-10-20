Subject: HUE-Span (SPMF): over-counted utilities in getMaximalUtility, and a question on the ERU bound

Dear Prof. Fournier-Viger,

I am an independent researcher working on high-utility episode mining. While re-implementing
HUE-Span (Fournier-Viger, Yang, Lin, Yun, ADMA 2019) and checking it against a brute-force
implementation of Definitions 12-14 of the paper, I found two issues that I would like to report.
All files mentioned below are available on request.

1. Over-counted utilities in ComplexSequence.getMaximalUtility (SPMF, spmf.jar built 2026-09-24)

FiniteStateMachine.scan(pairs) adds the utility of the matched event set to the FSM
(`this.utility += utilityOfitemset`) whenever it succeeds. In getMaximalUtility, scan is also
called on the previous FSM inside the duplicate-state test

    if (j >= 1 && fsaList.get(j).isSame(fsaList.get(j - 1)) && !fsaList.get(j - 1).scan(pairs))

When this scan succeeds, the utility is added although no transition is made, so the same time
point is counted again later.

Minimal example (input file, one time point per line, maxDuration = 4, minutil ratio 0.0001):

    1:2:2
    2:3:3
    2:4:4
    1:3:3

The episode <(1),(2),(2),(1)> has a single minimal occurrence [t1, t4] and a single matching
(t1, t2, t3, t4), so its utility is 2 + 3 + 4 + 3 = 12. SPMF outputs
`1 -1 2 -1 2 -1 1 -1 #UTIL: 16` (the utility at t3 is counted twice).

Effect on the experiments of the paper: with retail_utility_spmf.txt and kosarak_utility_spmf.txt
from the SPMF website, the reference implementation reproduces every #HUE* value of Table 2
exactly (e.g. 179 for Retail, minUtil 2%, maxDur 5), and the traditional-utility mode reproduces
every #HUE value. Recomputing the utility of each reported episode from the input with an
independent verifier gives the following numbers of reported episodes whose utility is below
minUtil:

    Retail   1%/5: 373 of 1,556    1.5%/5: 96 of 523    2%/5: 8 of 179
             2%/6: 286 of 730      2%/7: 995 of 2,084
    Kosarak 10%/5: 32 of 105      15%/5: 5 of 27       20%/5: 1 of 3
            20%/6: 13 of 21       20%/7: 53 of 81

With the utility computed as the maximum over matchings (a small dynamic program replacing
getMaximalUtility), the counts become 1,183 / 427 / 171 / 444 / 1,089 (Retail) and
73 / 22 / 2 / 8 / 28 (Kosarak).

A second, smaller point: when the events of a line are not sorted, ties in the AWU sort keep the
input order, so the order of events differs between time points and some HUEs are missed. Sorting
ties by event identifier removes this.

2. ERU and Definition 12

With the maximum-utility semantics of Definition 12, the ERU of an episode does not seem to bound
the utility of all its extensions. Example, maxDuration = 5, minutil = 7 (total utility 14):

    t1: A(1)   t2: B(1)   t3: B(9) C(1)   t4: C(1)   t5: D(1)

alpha = <(A),(B),(C)> has one minimal occurrence [t1, t3] with utility 3, and
ERU(alpha) = 3 + 0 + u(t4) + u(t5) = 5 < 7. Its extension <(A),(B),(C),(D)> has the minimal
occurrence [t1, t5] whose best matching is (t1, t3, t4, t5) with utility 1 + 9 + 1 + 1 = 12 >= 7,
so it is a HUE but is pruned (SPMF, and also the corrected utility above). The reason is that an
extension can match the prefix at time points other than those of the prefix's own minimal
occurrence. On Retail and Kosarak this did not change any result in the settings above.

I hope this is useful. I would be glad to send the corrected code, the test programs and the
scripts that reproduce the numbers above.

Best regards,
[name]
Independent researcher
