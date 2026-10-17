#!/usr/bin/env python3
"""Exact verification of MHOUI-Miner and baselines against a Brute-Force Oracle.

For random quantitative utility databases, this oracle:
  1. Exhaustively enumerates every candidate itemset X contained in some transaction.
  2. Evaluates exact support supp(X), exact database utility u(X), and exact average occupancy aocc(X).
  3. Identifies qualified High Utility Occupancy Itemsets (HOUI):
       supp(X) >= sigma, aocc(X) >= beta, u(X) >= mu.
  4. Computes the exact Pareto-maximal frontier (Strong & Weak Pareto Superset Dominance):
       X is maximal iff not exists Y proper superset of X such that
       aocc(Y) >= aocc(X) and u(Y) >= u(X).

A test case PASSES if and only if the mining algorithm discovers the exact identical
family of maximal patterns, with 0 false positives and 0 false negatives.

Run from repository root:
  python docs/core/MHOUI/oracle_test.py
"""

import csv
import itertools
import math
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

BIN_BENCH = "./bin/mhoui_bench.exe"
OUT_CSV = Path("docs/core/MHOUI/results/oracle.csv")

MODES = (
    "direct_strong_mhoui",
    "houi_filter",
    "ocean_filter",
    "huopm_filter",
    "il_huopm_filter",
    "ablation_no_dom",
    "ablation_no_oub",
    "ablation_no_uub",
)

SUITES = [
    # (name, seed, databases, items_range, trans_range, max_qty, max_profit)
    ("dense-small",  42, 10, (4, 8),   (10, 25), 5, 10),
    ("sparse-mixed", 84, 10, (6, 12),  (15, 35), 4, 8),
    ("skewed-noise", 126, 10, (5, 10), (12, 30), 8, 15),
]


def brute_force_oracle(transactions, sigma, beta, mu, occ_mode="physical", strong=True):
    """Computes exact ground-truth maximal patterns via brute-force enumeration.
    Supports both:
      - 'physical': Deng physical basket occupancy (|X| / |D_X|) * sum(1 / |T|)
      - 'utility': Shen utility occupancy (1 / |D_X|) * sum(u(X, T) / TU(T))
    """
    cands = set()
    for t in transactions:
        t_items = sorted(t["items"].keys())
        for k in range(1, len(t_items) + 1):
            cands.update(frozenset(comb) for comb in itertools.combinations(t_items, k))

    houi = []
    for X in cands:
        sup_t = [t for t in transactions if X.issubset(t["items"].keys())]
        supp_count = len(sup_t)
        if supp_count < sigma:
            continue

        if occ_mode == "physical":
            occ_sum = sum(Fraction(1, len(t["items"])) for t in sup_t)
            aocc = float(Fraction(len(X), supp_count) * occ_sum)
        else:
            uocc_sum = sum(sum(t["items"][i] for i in X) / sum(t["items"].values()) for t in sup_t)
            aocc = uocc_sum / supp_count

        # Total database utility u(X) = sum_{T in D_X} u(X, T)
        util = sum(t["items"][i] for t in sup_t for i in X)

        if aocc + 1e-9 >= beta and util + 1e-9 >= mu:
            houi.append({"items": X, "supp": supp_count, "aocc": aocc, "util": util})

    # Pareto Maximality Filter
    maximal = []
    for x in houi:
        dominated = False
        for y in houi:
            if x["items"] < y["items"]: # y is proper superset of x
                ge_occ = (y["aocc"] + 1e-9 >= x["aocc"])
                ge_util = (y["util"] + 1e-9 >= x["util"])
                if strong:
                    if ge_occ and ge_util:
                        dominated = True
                        break
                else:
                    strict = (y["aocc"] > x["aocc"] + 1e-9) or (y["util"] > x["util"] + 1e-9)
                    if ge_occ and ge_util and strict:
                        dominated = True
                        break
        if not dominated:
            maximal.append(x["items"])

    return set(maximal)


def write_spmf_utility_dataset(path, transactions):
    """Writes dataset in SPMF standard utility format:
    item1 item2 ...:total_utility:util1 util2 ...
    """
    with open(path, "w") as f:
        for t in transactions:
            items_sorted = sorted(t["items"].keys())
            items_str = " ".join(map(str, items_sorted))
            total_u = sum(t["items"][i] for i in items_sorted)
            utils_str = " ".join(str(t["items"][i]) for i in items_sorted)
            f.write(f"{items_str}:{total_u}:{utils_str}\n")


def run_miner(algo, dataset_path, sigma, beta, mu, strong=1, dump_path=None):
    cmd = [
        BIN_BENCH,
        "--input", dataset_path,
        "--algorithm", algo,
        "--minsup", str(sigma),
        "--minocc", str(beta),
        "--minutil", str(mu),
        "--strong", str(strong)
    ]
    if dump_path:
        cmd.extend(["--dump-patterns", dump_path])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nOutput: {proc.stderr}")

    patterns = set()
    if dump_path and os.path.exists(dump_path):
        with open(dump_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    patterns.add(frozenset(map(int, line.split())))
    return patterns


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 85)
    print("MHOUI MATHEMATICAL ORACLE & PATTERN SET EQUALITY VERIFICATION SUITE")
    print("=" * 85)

    rows = []
    total_cases = 0
    total_mismatches = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_patterns_evaluated = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for suite_name, seed, num_dbs, (lo_items, hi_items), (lo_trans, hi_trans), max_qty, max_prof in SUITES:
            rng = random.Random(seed)
            suite_cases = 0
            suite_mismatches = 0
            suite_fp = 0
            suite_fn = 0
            suite_pats = 0

            print(f"\n[Suite: {suite_name}] (Dbs: {num_dbs}, Items: {lo_items}-{hi_items}, Trans: {lo_trans}-{hi_trans})")

            for db_idx in range(num_dbs):
                num_items = rng.randint(lo_items, hi_items)
                num_trans = rng.randint(lo_trans, hi_trans)
                profits = {i: rng.randint(1, max_prof) for i in range(1, num_items + 1)}

                transactions = []
                for _ in range(num_trans):
                    tx_len = rng.randint(1, num_items)
                    items_in_tx = rng.sample(range(1, num_items + 1), tx_len)
                    tx_items = {i: rng.randint(1, max_qty) * profits[i] for i in items_in_tx}
                    transactions.append({"items": tx_items})

                db_path = os.path.join(tmp_dir, f"toy_db_{db_idx}.txt")
                write_spmf_utility_dataset(db_path, transactions)

                dbu = sum(sum(t["items"].values()) for t in transactions)

                # Grid of thresholds
                test_grids = [
                    (max(2, int(math.ceil(0.10 * num_trans))), 0.20, math.ceil(0.01 * dbu)),
                    (max(2, int(math.ceil(0.15 * num_trans))), 0.30, math.ceil(0.02 * dbu)),
                    (max(2, int(math.ceil(0.20 * num_trans))), 0.25, math.ceil(0.05 * dbu)),
                ]

                for sigma, beta, mu in test_grids:
                    expected_physical = brute_force_oracle(transactions, sigma, beta, mu, occ_mode="physical", strong=True)
                    expected_utility = brute_force_oracle(transactions, sigma, beta, mu, occ_mode="utility", strong=True)
                    suite_cases += 1
                    total_cases += 1

                    for mode in MODES:
                        expected = expected_utility if mode in ("ocean_filter", "huopm_filter", "il_huopm_filter") else expected_physical
                        dump_file = os.path.join(tmp_dir, f"dump_{mode}.txt")
                        mined = run_miner(mode, db_path, sigma, beta, mu, strong=1, dump_path=dump_file)

                        suite_pats += len(expected)
                        total_patterns_evaluated += len(expected)

                        fp_set = mined - expected
                        fn_set = expected - mined

                        if fp_set or fn_set:
                            suite_mismatches += 1
                            total_mismatches += 1
                            suite_fp += len(fp_set)
                            total_false_positives += len(fp_set)
                            suite_fn += len(fn_set)
                            total_false_negatives += len(fn_set)
                            print(f"  FAILED: db={db_idx} algo={mode} sigma={sigma} beta={beta} mu={mu} "
                                  f"| Expected: {len(expected)}, Mined: {len(mined)}, FP: {len(fp_set)}, FN: {len(fn_set)}")

            status = "PASSED (100% SOUND & COMPLETE)" if suite_mismatches == 0 else "FAILED"
            print(f"  -> Result: {status} | Cases: {suite_cases}, Pats Evaluated: {suite_pats}, FP: {suite_fp}, FN: {suite_fn}")
            rows.append({
                "suite": suite_name,
                "databases": num_dbs,
                "tested_cases": suite_cases,
                "modes_checked": len(MODES),
                "patterns_evaluated": suite_pats,
                "false_positives": suite_fp,
                "false_negatives": suite_fn,
                "mismatches": suite_mismatches,
                "status": status
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 85)
    print(f"VERIFICATION SUMMARY: {total_cases} test cases executed across {len(MODES)} modes.")
    print(f"Total Patterns Evaluated: {total_patterns_evaluated}")
    print(f"False Positives (Spurious Patterns): {total_false_positives}")
    print(f"False Negatives (Missed Patterns):   {total_false_negatives}")
    print(f"Total Mismatches:                    {total_mismatches}")
    if total_mismatches == 0:
        print(">> ALL TESTS PASSED: EXACT 100% MATHEMATICAL SET EQUALITY CONFIRMED! <<")
    print(f"Report saved to: {OUT_CSV}")
    print("=" * 85)


if __name__ == "__main__":
    main()

