#!/usr/bin/env bash
set -u

OUT_DIR="${1:-results/mfhoi_q1}"
TIMEOUT_SECONDS="${MFHOI_Q1_TIMEOUT:-45}"
MAX_SECONDS="${MFHOI_Q1_MAX_SECONDS:-12}"
MAX_PATTERNS="${MFHOI_Q1_MAX_PATTERNS:-75000}"
MEM_MB="${MFHOI_Q1_MEM_MB:-1536}"

SUPPORTS=(0.10 0.05 0.02 0.01)
OCCS=(0.2 0.4 0.6 0.8)
MFHOI_ALGOS=(apriori mfi_baseline fhoi weak_mfhoi strong_mfhoi)
LEGACY_SUPPORT_ALGOS=(fpgrowth eclat mafia fpmax)
LEGACY_OCCUPANCY_ALGOS=(dfhoi hep)
if [[ -n "${MFHOI_Q1_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$MFHOI_Q1_DATASETS"
else
  DATASETS=(
    datasets/itemsets/mushrooms.txt
    datasets/itemsets/retail.txt
    datasets/itemsets/chess.txt
    datasets/itemsets/connect.txt
    datasets/itemsets/pumsb.txt
    datasets/itemsets/T10I4D100K.txt
    datasets/itemsets/t20i6d100k.txt
    datasets/itemsets/accidents.txt
    datasets/itemsets/foodmartFIM.txt
    datasets/synthetic/syn_sparse.txt
    datasets/synthetic/syn_dense.txt
    datasets/synthetic/syn_long.txt
    datasets/synthetic/syn_correlated.txt
  )
fi

mkdir -p "$OUT_DIR" results/patterns

dataset_name() {
  basename "$1" .txt | tr ' /.' '___'
}

run_mfhoi_algo_dataset() {
  local algo="$1"
  local dataset="$2"
  local dname
  dname="$(dataset_name "$dataset")"
  local report="$OUT_DIR/${algo}_${dname}.txt"

  : > "$report"
  {
    echo "MFHOI Q1 Benchmark"
    echo "algorithm=$algo"
    echo "dataset=$dataset"
    echo "supports=${SUPPORTS[*]}"
    echo "occupancies=${OCCS[*]}"
    echo "timeout_seconds=$TIMEOUT_SECONDS"
    echo "max_seconds=$MAX_SECONDS"
    echo "max_patterns=$MAX_PATTERNS"
    echo "mem_mb=$MEM_MB"
    echo
  } >> "$report"

  for sup in "${SUPPORTS[@]}"; do
    if [[ "$algo" == "apriori" || "$algo" == "mfi_baseline" ]]; then
      local pattern_file="results/patterns/q1_${algo}_${dname}_${sup}.txt"
      {
        echo "================================================================"
        echo "minsup=$sup"
        echo "minocc=NA"
      } >> "$report"
      (
        ulimit -v $((MEM_MB * 1024))
        exec timeout "$TIMEOUT_SECONDS" ./bin/mfhoi_miner \
          --input "$dataset" \
          --algorithm "$algo" \
          --minsup "$sup" \
          --minocc 0.2 \
          --output "$pattern_file" \
          --max-seconds "$MAX_SECONDS" \
          --max-patterns "$MAX_PATTERNS"
      ) >> "$report" 2>&1
      echo "exit_code=$?" >> "$report"
    else
      for occ in "${OCCS[@]}"; do
        local pattern_file="results/patterns/q1_${algo}_${dname}_${sup}_${occ}.txt"
        {
          echo "================================================================"
          echo "minsup=$sup"
          echo "minocc=$occ"
        } >> "$report"
        (
          ulimit -v $((MEM_MB * 1024))
          exec timeout "$TIMEOUT_SECONDS" ./bin/mfhoi_miner \
            --input "$dataset" \
            --algorithm "$algo" \
            --minsup "$sup" \
            --minocc "$occ" \
            --output "$pattern_file" \
            --max-seconds "$MAX_SECONDS" \
            --max-patterns "$MAX_PATTERNS"
        ) >> "$report" 2>&1
        echo "exit_code=$?" >> "$report"
      done
    fi
  done
}

run_legacy_baseline_dataset() {
  local algo="$1"
  local dataset="$2"
  local thresholds="$3"
  local dname
  dname="$(dataset_name "$dataset")"
  ./bin/itemset_mining_bench "$algo" 0 "$thresholds" "$dataset" \
    --out "$OUT_DIR/${algo}_${dname}.txt" \
    --timeout "$TIMEOUT_SECONDS" \
    --mem-mb "$MEM_MB" \
    --pause 0 \
    --replace
}

for dataset in "${DATASETS[@]}"; do
  if [[ ! -f "$dataset" ]]; then
    continue
  fi
  for algo in "${MFHOI_ALGOS[@]}"; do
    echo "[q1] $algo $dataset"
    run_mfhoi_algo_dataset "$algo" "$dataset"
  done
  for algo in "${LEGACY_SUPPORT_ALGOS[@]}"; do
    echo "[q1] $algo $dataset"
    run_legacy_baseline_dataset "$algo" "$dataset" "${SUPPORTS[*]}"
  done
  for algo in "${LEGACY_OCCUPANCY_ALGOS[@]}"; do
    echo "[q1] $algo $dataset"
    run_legacy_baseline_dataset "$algo" "$dataset" "${OCCS[*]}"
  done
done

echo "Done. TXT files are under $OUT_DIR"
