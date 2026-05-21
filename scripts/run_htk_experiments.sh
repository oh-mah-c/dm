#!/usr/bin/env bash
set -u

OUT_DIR="${1:-results/htk_compare_full}"
mkdir -p "$OUT_DIR"

make bin/htk_miner >/dev/null

run_dataset() {
  local name="$1"
  local path="$2"
  shift 2
  local outfile="$OUT_DIR/htk_miner_${name}.txt"
  : > "$outfile"
  for k in "$@"; do
    {
      echo "=== htk_miner dataset=$name k=$k mode=bsn ==="
      bin/htk_miner --input "$path" --k "$k" --mode bsn --max-seconds 180 --max-candidates 3000000
      echo
    } >> "$outfile" 2>&1
  done
}

run_dataset mushrooms datasets/itemsets/mushrooms.txt 50 100 500 1000
run_dataset chess datasets/itemsets/chess.txt 50 100 500 1000
run_dataset foodmartFIM datasets/itemsets/foodmartFIM.txt 50 100 500 1000
run_dataset retail datasets/itemsets/retail.txt 50 100 500

echo "HTK-Miner statistics written to $OUT_DIR"
