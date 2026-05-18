#!/usr/bin/env bash
set -u

OUT_DIR="${1:-results/mhoui_full}"
TIMEOUT_SECONDS="${MHOUI_TIMEOUT:-90}"
MAX_SECONDS="${MHOUI_MAX_SECONDS:-30}"
MAX_PATTERNS="${MHOUI_MAX_PATTERNS:-200000}"
MEM_MB="${MHOUI_MEM_MB:-2048}"
RUN_BASELINES="${MHOUI_RUN_BASELINES:-1}"

if [[ -n "${MHOUI_ALGOS_OVERRIDE:-}" ]]; then
  read -r -a MHOUI_ALGOS <<< "$MHOUI_ALGOS_OVERRIDE"
else
  MHOUI_ALGOS=(houi weak_mhoui strong_mhoui direct_strong_mhoui)
fi

if [[ -n "${MHOUI_UTILITY_BASELINES:-}" ]]; then
  read -r -a UTILITY_BASELINES <<< "$MHOUI_UTILITY_BASELINES"
else
  UTILITY_BASELINES=(twophase fhm efim huiminer upgrowth)
fi

if [[ -n "${MHOUI_CONCISE_BASELINES:-}" ]]; then
  read -r -a CONCISE_UTILITY_BASELINES <<< "$MHOUI_CONCISE_BASELINES"
else
  CONCISE_UTILITY_BASELINES=(closed_fhuim_kinana chui_miner)
fi

if [[ -n "${MHOUI_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$MHOUI_DATASETS"
else
  DATASETS=(
    datasets/utilities/foodmart.txt
    datasets/utilities/liquor_11.txt
    datasets/utilities/fruithut_utility.txt
    datasets/utilities/chainstore.txt
  )
fi

mkdir -p "$OUT_DIR"

dataset_name() {
  basename "$1" .txt | tr ' /.' '___'
}

dataset_thresholds() {
  local dname="$1"
  case "$dname" in
    foodmart)
      SUPPORTS=(0.02 0.01 0.005)
      OCCS=(0.10 0.20 0.40)
      UTIL_RATIOS=(0.001 0.0005 0.0001)
      ;;
    liquor_11)
      SUPPORTS=(0.05 0.02 0.01)
      OCCS=(0.20 0.40 0.60)
      UTIL_RATIOS=(0.005 0.001 0.0005)
      ;;
    fruithut_utility)
      SUPPORTS=(0.05 0.02 0.01)
      OCCS=(0.20 0.40 0.60)
      UTIL_RATIOS=(0.005 0.001 0.0005)
      ;;
    chainstore)
      SUPPORTS=(0.05 0.02)
      OCCS=(0.20 0.40)
      UTIL_RATIOS=(0.005 0.001)
      ;;
    *)
      SUPPORTS=(0.05 0.02)
      OCCS=(0.20 0.40)
      UTIL_RATIOS=(0.001 0.0005)
      ;;
  esac
  if [[ -n "${MHOUI_SUPPORTS:-}" ]]; then read -r -a SUPPORTS <<< "$MHOUI_SUPPORTS"; fi
  if [[ -n "${MHOUI_OCCS:-}" ]]; then read -r -a OCCS <<< "$MHOUI_OCCS"; fi
  if [[ -n "${MHOUI_UTIL_RATIOS:-}" ]]; then read -r -a UTIL_RATIOS <<< "$MHOUI_UTIL_RATIOS"; fi
}

total_utility() {
  awk -F: '
    /^[[:space:]]*($|#|%|@)/ { next }
    NF >= 2 { s += $2 }
    END { printf "%.10f", s }
  ' "$1"
}

ceil_mul() {
  awk -v a="$1" -v b="$2" 'BEGIN { x = a * b; y = int(x); if (x > y) y++; print y }'
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
    echo "MHOUI Comparison Benchmark"
    echo "algorithm=$algo"
    echo "dataset=$dataset"
    echo "dataset_name=$dname"
    echo "timeout_seconds=$TIMEOUT_SECONDS"
    echo "max_seconds=$MAX_SECONDS"
    echo "max_patterns=$MAX_PATTERNS"
    echo "mem_mb=$MEM_MB"
    echo
  } > "$report"
}

append_run_block() {
  local report="$1"
  local run_id="$2"
  local sup="$3"
  local occ="$4"
  local ur="$5"
  local minutil_abs="$6"
  local status="$7"
  local tmp="$8"
  {
    echo "================================================================"
    echo "run_id=$run_id"
    echo "minsup=${sup:-NA}"
    echo "minocc=${occ:-NA}"
    echo "minutil_ratio=${ur:-NA}"
    echo "minutil_abs=${minutil_abs:-NA}"
    echo "status=$status"
    echo "----------------------------------------------------------------"
    cat "$tmp"
    echo
  } >> "$report"
}

append_mhoui_summary() {
  local run_id="$1"
  local algo="$2"
  local dataset="$3"
  local minsup_ratio="$4"
  local minocc="$5"
  local minutil_ratio="$6"
  local log="$7"
  local output_file="$8"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "$algo" "$dataset" "$minsup_ratio" "$(csv_get minsup_count "$log")" "$minocc" \
    "$minutil_ratio" "$(csv_get minutil "$log")" "$(csv_get status "$log")" \
    "$(csv_get runtime_sec "$log")" "$(csv_get peak_ram_mb "$log")" "$(csv_get output_count "$log")" \
    "$(csv_get visited_nodes "$log")" "$(csv_get candidates "$log")" "$(csv_get pruned_support "$log")" \
    "$(csv_get pruned_twu "$log")" "$(csv_get pruned_uub "$log")" "$(csv_get pruned_oub1 "$log")" \
    "$(csv_get pruned_oub2 "$log")" "$(csv_get pruned_dom "$log")" "$(csv_get num_houi "$log")" \
    "$(csv_get num_weak_mhoui "$log")" "$(csv_get num_strong_mhoui "$log")" "$(csv_get avg_len "$log")" \
    "$(csv_get avg_occupancy "$log")" "$output_file" >> "$OUT_DIR/run_summary.csv"
}

append_dm_summary() {
  local run_id="$1"
  local algo="$2"
  local dataset="$3"
  local minsup_ratio="$4"
  local minsup_count="$5"
  local minocc="$6"
  local minutil_ratio="$7"
  local minutil_abs="$8"
  local status="$9"
  local log="${10}"
  local runtime peak output_count
  runtime="$(awk -F: '/Algorithm Core/ { gsub(/ ms.*/, "", $2); gsub(/[[:space:]]/, "", $2); printf "%.6f", $2 / 1000.0 }' "$log")"
  peak="$(awk -F: '/Peak RAM/ { gsub(/ MB.*/, "", $2); gsub(/[[:space:]]/, "", $2); print $2 }' "$log")"
  output_count="$(awk -F: '/Frequent Itemsets/ { gsub(/[[:space:]]/, "", $2); print $2 }' "$log")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,,,,,,,,,,,,,\n' \
    "$run_id" "$algo" "$dataset" "$minsup_ratio" "$minsup_count" "$minocc" "$minutil_ratio" "$minutil_abs" \
    "$status" "${runtime:-NA}" "${peak:-NA}" "${output_count:-NA}" >> "$OUT_DIR/run_summary.csv"
}

run_mhoui() {
  local algo="$1"
  local dataset="$2"
  local dname="$3"
  local sup="$4"
  local occ="$5"
  local ur="$6"
  local run_id="${dname}_${algo}_s${sup}_o${occ}_u${ur}"
  run_id="${run_id//./p}"
  local report
  report="$(report_path "$algo" "$dname")"
  local log="$OUT_DIR/.${run_id}.tmp"
  local pattern_file="/tmp/${run_id}.csv"
  ensure_report_header "$report" "$algo" "$dataset" "$dname"
  echo "[mhoui] $run_id"
  (
    ulimit -v $((MEM_MB * 1024))
    exec timeout "$TIMEOUT_SECONDS" ./bin/mhoui_miner \
      --input "$dataset" \
      --algorithm "$algo" \
      --minsup "$sup" \
      --minocc "$occ" \
      --minutil-ratio "$ur" \
      --output "$pattern_file" \
      --max-seconds "$MAX_SECONDS" \
      --max-patterns "$MAX_PATTERNS"
  ) > "$log" 2>&1
  local code=$?
  local status="OK"
  [[ $code -eq 124 ]] && status="TIMEOUT"
  [[ $code -ne 0 && $code -ne 124 ]] && status="FAILED"
  if [[ $code -eq 124 ]]; then echo "status=TIMEOUT" >> "$log"; fi
  append_run_block "$report" "$run_id" "$sup" "$occ" "$ur" "$(csv_get minutil "$log")" "$status" "$log"
  append_mhoui_summary "$run_id" "$algo" "$dname" "$sup" "$occ" "$ur" "$log" "discarded:/tmp/${run_id}.csv"
  rm -f "$pattern_file"
  rm -f "$log"
}

run_utility_baseline() {
  local algo="$1"
  local dataset="$2"
  local dname="$3"
  local ur="$4"
  local dbu="$5"
  local minutil_abs
  minutil_abs="$(ceil_mul "$dbu" "$ur")"
  local run_id="${dname}_${algo}_u${ur}"
  run_id="${run_id//./p}"
  local report
  report="$(report_path "$algo" "$dname")"
  local log="$OUT_DIR/.${run_id}.tmp"
  ensure_report_header "$report" "$algo" "$dataset" "$dname"
  echo "[baseline] $run_id"
  (
    ulimit -v $((MEM_MB * 1024))
    exec timeout "$TIMEOUT_SECONDS" ./bin/dm.exe "$algo" "$dataset" 1 "$minutil_abs"
  ) > "$log" 2>&1
  local code=$?
  local status="OK"
  [[ $code -eq 124 ]] && status="TIMEOUT"
  [[ $code -ne 0 && $code -ne 124 ]] && status="FAILED"
  append_run_block "$report" "$run_id" "NA" "NA" "$ur" "$minutil_abs" "$status" "$log"
  append_dm_summary "$run_id" "$algo" "$dname" "NA" "NA" "NA" "$ur" "$minutil_abs" "$status" "$log"
  rm -f "$log"
}

run_concise_baseline() {
  local algo="$1"
  local dataset="$2"
  local dname="$3"
  local sup="$4"
  local ur="$5"
  local dbu="$6"
  local minutil_abs
  minutil_abs="$(ceil_mul "$dbu" "$ur")"
  local minsup_count
  minsup_count="$(awk -v s="$sup" -v n="$7" 'BEGIN { x=s*n; y=int(x); if (x>y) y++; print y }')"
  local run_id="${dname}_${algo}_s${sup}_u${ur}"
  run_id="${run_id//./p}"
  local report
  report="$(report_path "$algo" "$dname")"
  local log="$OUT_DIR/.${run_id}.tmp"
  ensure_report_header "$report" "$algo" "$dataset" "$dname"
  echo "[concise] $run_id"
  if [[ "$algo" == "closed_fhuim_kinana" ]]; then
    (
      ulimit -v $((MEM_MB * 1024))
      exec timeout "$TIMEOUT_SECONDS" ./bin/dm.exe "$algo" "$dataset" 1 "$minutil_abs" "$minsup_count"
    ) > "$log" 2>&1
  else
    (
      ulimit -v $((MEM_MB * 1024))
      exec timeout "$TIMEOUT_SECONDS" ./bin/dm.exe "$algo" "$dataset" 1 "$minutil_abs"
    ) > "$log" 2>&1
  fi
  local code=$?
  local status="OK"
  [[ $code -eq 124 ]] && status="TIMEOUT"
  [[ $code -ne 0 && $code -ne 124 ]] && status="FAILED"
  append_run_block "$report" "$run_id" "$sup" "NA" "$ur" "$minutil_abs" "$status" "$log"
  append_dm_summary "$run_id" "$algo" "$dname" "$sup" "$minsup_count" "NA" "$ur" "$minutil_abs" "$status" "$log"
  rm -f "$log"
}

{
  echo "run_id,algorithm,dataset,minsup_ratio,minsup_count,minocc,minutil_ratio,minutil_abs,status,runtime_sec,peak_ram_mb,output_count,visited_nodes,candidates,pruned_support,pruned_twu,pruned_uub,pruned_oub1,pruned_oub2,pruned_dom,num_houi,num_weak_mhoui,num_strong_mhoui,avg_len,avg_occupancy,output_file"
} > "$OUT_DIR/run_summary.csv"

for dataset in "${DATASETS[@]}"; do
  [[ -f "$dataset" ]] || continue
  dname="$(dataset_name "$dataset")"
  dataset_thresholds "$dname"
  dbu="$(total_utility "$dataset")"
  ntrans="$(grep -cv '^[[:space:]]*[@#%]' "$dataset")"
  echo "[dataset] $dname total_utility=$dbu transactions=$ntrans"

  for sup in "${SUPPORTS[@]}"; do
    for occ in "${OCCS[@]}"; do
      for ur in "${UTIL_RATIOS[@]}"; do
        for algo in "${MHOUI_ALGOS[@]}"; do
          run_mhoui "$algo" "$dataset" "$dname" "$sup" "$occ" "$ur"
        done
      done
    done
  done

  if [[ "$RUN_BASELINES" == "1" ]]; then
    for ur in "${UTIL_RATIOS[@]}"; do
      for algo in "${UTILITY_BASELINES[@]}"; do
        run_utility_baseline "$algo" "$dataset" "$dname" "$ur" "$dbu"
      done
    done
    for sup in "${SUPPORTS[@]}"; do
      for ur in "${UTIL_RATIOS[@]}"; do
        for algo in "${CONCISE_UTILITY_BASELINES[@]}"; do
          run_concise_baseline "$algo" "$dataset" "$dname" "$sup" "$ur" "$dbu" "$ntrans"
        done
      done
    done
  fi
done

awk -F, '
  BEGIN { OFS=","; print "dataset,minsup_ratio,minocc,minutil_ratio,houi,weak_mhoui,strong_mhoui,invariant" }
  NR > 1 && $2 == "strong_mhoui" {
    ok = ($21 + 0 >= $22 + 0 && $22 + 0 >= $23 + 0) ? "PASS" : "FAIL";
    print $3,$4,$6,$7,$21,$22,$23,ok
  }
' "$OUT_DIR/run_summary.csv" > "$OUT_DIR/invariant_summary.csv"

echo "Done. Statistics are under $OUT_DIR"
