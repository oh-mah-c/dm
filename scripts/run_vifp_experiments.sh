#!/usr/bin/env bash
set -u

OUT_DIR="${1:-results/vifp_compare_full}"
TIMEOUT_SECONDS="${VIFP_TIMEOUT:-90}"
MAX_SECONDS="${VIFP_MAX_SECONDS:-30}"
MAX_ITEMSETS="${VIFP_MAX_ITEMSETS:-500000}"
MEM_MB="${VIFP_MEM_MB:-2048}"
RUN_BASELINES="${VIFP_RUN_BASELINES:-1}"
VERBOSE="${VIFP_VERBOSE:-0}"

if [[ -n "${VIFP_MODES_OVERRIDE:-}" ]]; then
  read -r -a VIFP_MODES <<< "$VIFP_MODES_OVERRIDE"
else
  VIFP_MODES=(plaintext smpc fhe)
fi

if [[ -n "${VIFP_BASELINES_OVERRIDE:-}" ]]; then
  read -r -a BASELINES <<< "$VIFP_BASELINES_OVERRIDE"
else
  BASELINES=(fpgrowth eclat apriori fpmax)
fi

if [[ -n "${VIFP_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$VIFP_DATASETS"
else
  DATASETS=(
    datasets/itemsets/mushrooms.txt
    datasets/itemsets/chess.txt
    datasets/itemsets/foodmartFIM.txt
    datasets/itemsets/retail.txt
    datasets/synthetic/syn_sparse.txt
    datasets/synthetic/syn_dense.txt
    datasets/synthetic/syn_long.txt
  )
fi

mkdir -p "$OUT_DIR/logs"

log_run() {
  if [[ "$VERBOSE" == "1" ]]; then
    echo "$1"
  fi
}

dataset_name() {
  basename "$1" .txt | tr ' /.' '___'
}

dataset_thresholds() {
  local dname="$1"
  case "$dname" in
    mushrooms)
      SUPPORTS=(0.80 0.60 0.40)
      ;;
    chess)
      SUPPORTS=(0.90 0.80 0.70)
      ;;
    foodmartFIM)
      SUPPORTS=(0.08 0.05 0.03)
      ;;
    retail)
      SUPPORTS=(0.03 0.02 0.01)
      ;;
    syn_dense)
      SUPPORTS=(0.70 0.60 0.50)
      ;;
    syn_long)
      SUPPORTS=(0.60 0.50 0.40)
      ;;
    *)
      SUPPORTS=(0.10 0.05 0.02)
      ;;
  esac
  if [[ -n "${VIFP_SUPPORTS:-}" ]]; then read -r -a SUPPORTS <<< "$VIFP_SUPPORTS"; fi
}

csv_get() {
  local key="$1"
  local file="$2"
  awk -F= -v k="$key" '$1 == k { print $2; found=1 } END { if (!found) print "" }' "$file"
}

report_path() {
  local algo="$1"
  local dname="$2"
  echo "$OUT_DIR/${algo}_${dname}.txt"
}

ensure_report_header() {
  local report="$1"
  local algo="$2"
  local dataset="$3"
  local dname="$4"
  if [[ -f "$report" ]]; then return; fi
  {
    echo "VIFP Comparison Benchmark"
    echo "algorithm=$algo"
    echo "dataset=$dataset"
    echo "dataset_name=$dname"
    echo "timeout_seconds=$TIMEOUT_SECONDS"
    echo "max_seconds=$MAX_SECONDS"
    echo "max_itemsets=$MAX_ITEMSETS"
    echo "mem_mb=$MEM_MB"
    echo
  } > "$report"
}

append_run_block() {
  local report="$1"
  local run_id="$2"
  local sup="$3"
  local status="$4"
  local tmp="$5"
  {
    echo "================================================================"
    echo "run_id=$run_id"
    echo "minsup=$sup"
    echo "status=$status"
    echo "----------------------------------------------------------------"
    cat "$tmp"
    echo
  } >> "$report"
}

append_vifp_summary() {
  local run_id="$1"
  local mode="$2"
  local dname="$3"
  local sup="$4"
  local log="$5"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "vifp_${mode}" "$dname" "$sup" "$(csv_get minsup_count "$log")" "$(csv_get status "$log")" \
    "$(csv_get runtime_sec "$log")" "$(csv_get peak_ram_mb "$log")" "$(csv_get output_count "$log")" \
    "$(csv_get occurrence_records "$log")" "$(csv_get hpa_records "$log")" "$(csv_get projected_states "$log")" \
    "$(csv_get cpb_records "$log")" "$(csv_get predecessor_fetches "$log")" "$(csv_get histogram_updates "$log")" \
    "$(csv_get secure_comparisons "$log")" "$(csv_get stable_partitions "$log")" "$(csv_get oblivious_sorts "$log")" \
    "$(csv_get estimated_comm_bytes "$log")" "$(csv_get estimated_ciphertext_bytes "$log")" "$(csv_get estimated_bootstraps "$log")" \
    "$(csv_get result_disk_est_bytes "$log")" >> "$OUT_DIR/run_summary.csv"
}

append_baseline_summary() {
  local run_id="$1"
  local algo="$2"
  local dname="$3"
  local sup="$4"
  local minsup_count="$5"
  local status="$6"
  local log="$7"
  local runtime peak output_count disk_bytes
  runtime="$(awk -F: '/Algorithm Core/ { gsub(/ ms.*/, "", $2); gsub(/[[:space:]]/, "", $2); printf "%.6f", $2 / 1000.0 }' "$log")"
  peak="$(awk -F: '/Peak RAM/ { gsub(/ MB.*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2 }' "$log")"
  output_count="$(awk -F: '/Frequent Itemsets/ { gsub(/[[:space:]]/, "", $2); print $2 }' "$log")"
  disk_bytes="$(awk -F'[()]' '/Est. Disk/ { gsub(/[^0-9]/, "", $2); print $2 }' "$log")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,,,,,,,,,%s\n' \
    "$run_id" "$algo" "$dname" "$sup" "$minsup_count" "$status" "${runtime:-NA}" "${peak:-NA}" "${output_count:-NA}" "${disk_bytes:-NA}" >> "$OUT_DIR/run_summary.csv"
}

run_vifp() {
  local mode="$1"
  local dataset="$2"
  local dname="$3"
  local sup="$4"
  local run_id="${dname}_vifp_${mode}_s${sup}"
  run_id="${run_id//./p}"
  local report
  report="$(report_path "vifp_${mode}" "$dname")"
  local log="$OUT_DIR/logs/${run_id}.log"
  ensure_report_header "$report" "vifp_${mode}" "$dataset" "$dname"
  log_run "[vifp] $run_id"
  (
    ulimit -v $((MEM_MB * 1024))
    exec timeout "$TIMEOUT_SECONDS" ./bin/vifp_miner \
      --input "$dataset" \
      --minsup "$sup" \
      --mode "$mode" \
      --max-seconds "$MAX_SECONDS" \
      --max-itemsets "$MAX_ITEMSETS"
  ) > "$log" 2>&1
  local code=$?
  local status="OK"
  [[ $code -eq 124 ]] && status="TIMEOUT"
  [[ $code -ne 0 && $code -ne 124 && $code -ne 2 ]] && status="FAILED"
  [[ $code -eq 2 ]] && status="LIMITED"
  if [[ $code -eq 124 ]]; then echo "status=TIMEOUT" >> "$log"; fi
  append_run_block "$report" "$run_id" "$sup" "$status" "$log"
  append_vifp_summary "$run_id" "$mode" "$dname" "$sup" "$log"
}

run_baseline() {
  local algo="$1"
  local dataset="$2"
  local dname="$3"
  local sup="$4"
  local run_id="${dname}_${algo}_s${sup}"
  run_id="${run_id//./p}"
  local report
  report="$(report_path "$algo" "$dname")"
  local log="$OUT_DIR/logs/${run_id}.log"
  ensure_report_header "$report" "$algo" "$dataset" "$dname"
  log_run "[baseline] $run_id"
  (
    ulimit -v $((MEM_MB * 1024))
    exec timeout "$TIMEOUT_SECONDS" ./bin/dm.exe "$algo" "$dataset" 0 "$sup"
  ) > "$log" 2>&1
  local code=$?
  local status="OK"
  [[ $code -eq 124 ]] && status="TIMEOUT"
  [[ $code -ne 0 && $code -ne 124 ]] && status="FAILED"
  local minsup_count
  minsup_count="$(awk '/Min Support:/ { print $NF; exit }' "$log")"
  append_run_block "$report" "$run_id" "$sup" "$status" "$log"
  append_baseline_summary "$run_id" "$algo" "$dname" "$sup" "${minsup_count:-NA}" "$status" "$log"
}

cat > "$OUT_DIR/run_summary.csv" <<'CSV'
run_id,algorithm,dataset,minsup_ratio,minsup_count,status,runtime_sec,peak_ram_mb,output_count,occurrence_records,hpa_records,projected_states,cpb_records,predecessor_fetches,histogram_updates,secure_comparisons,stable_partitions,oblivious_sorts,estimated_comm_bytes,estimated_ciphertext_bytes,estimated_bootstraps,result_disk_est_bytes
CSV

make bin/vifp_miner bin/dm.exe >/dev/null || exit 1

for dataset in "${DATASETS[@]}"; do
  [[ -f "$dataset" ]] || { echo "Missing dataset: $dataset" >&2; continue; }
  dname="$(dataset_name "$dataset")"
  dataset_thresholds "$dname"
  mkdir -p "$OUT_DIR/$dname"
  for sup in "${SUPPORTS[@]}"; do
    for mode in "${VIFP_MODES[@]}"; do
      run_vifp "$mode" "$dataset" "$dname" "$sup"
    done
    if [[ "$RUN_BASELINES" == "1" ]]; then
      for algo in "${BASELINES[@]}"; do
        run_baseline "$algo" "$dataset" "$dname" "$sup"
      done
    fi
  done
done

find "$OUT_DIR" -maxdepth 1 -type f -name '*.txt' | while read -r file; do
  dname="$(basename "$file" .txt)"
  dname="${dname##*_}"
done

echo "VIFP benchmark complete"
echo "summary_csv=$OUT_DIR/run_summary.csv"
echo "reports_dir=$OUT_DIR"
echo "logs_dir=$OUT_DIR/logs"
echo "status_counts:"
awk -F, 'NR>1{c[$6]++} END{for (s in c) print "  " s "=" c[s]}' "$OUT_DIR/run_summary.csv" | sort
echo "algorithm_counts:"
awk -F, 'NR>1{c[$2]++} END{for (a in c) print "  " a "=" c[a]}' "$OUT_DIR/run_summary.csv" | sort
echo "dataset_counts:"
awk -F, 'NR>1{c[$3]++} END{for (d in c) print "  " d "=" c[d]}' "$OUT_DIR/run_summary.csv" | sort
