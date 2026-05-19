#!/usr/bin/env python3
import csv
import re
import subprocess
import sys
from pathlib import Path


REAL_CASES = [
    ("retail", "datasets/itemsets/retail.txt", 12000, "0.015"),
    ("accidents", "datasets/itemsets/accidents.txt", 3500, "0.03"),
    ("chess", "datasets/itemsets/chess.txt", 3196, "0.08"),
]

BASELINES = ["apriori", "eclat", "fpgrowth"]


def parse_report(text: str):
    def number(pattern, scale=1.0):
        m = re.search(pattern, text)
        return float(m.group(1)) / scale if m else ""

    return {
        "runtime_sec": number(r"Algorithm Core\s*:\s*([0-9.]+)\s*ms", 1000.0),
        "total_sec": number(r"TOTAL WALL TIME\s*:\s*([0-9.]+)\s*ms", 1000.0),
        "peak_ram_mb": number(r"Peak RAM \(VmHWM\)\s*:\s*([0-9.]+)\s*MB"),
        "output_count": number(r"Frequent Itemsets:\s*([0-9]+)"),
        "total_output_items": number(r"Total Items\s*:\s*([0-9]+)"),
    }


def write_truncated(src: Path, dst: Path, limit: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="ignore") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break
            fout.write(line)


def run_baseline(bin_path: Path, algo: str, data_path: Path, min_support: str, timeout_sec: int):
    cmd = [str(bin_path), algo, str(data_path), "0", min_support]
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        metrics = parse_report(proc.stdout)
        metrics.update({
            "status": "OK" if proc.returncode == 0 else "ERROR",
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
        })
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        metrics = {
            "runtime_sec": timeout_sec,
            "total_sec": timeout_sec,
            "peak_ram_mb": "",
            "output_count": "",
            "total_output_items": "",
            "status": "TIMEOUT",
            "exit_code": 124,
            "stdout": stdout + f"\n[TIMEOUT after {timeout_sec}s]\n",
        }
    return cmd, metrics


def load_hiep_representatives(out_dir: Path):
    summary = out_dir / "hiep_summary.csv"
    if not summary.exists():
        return []
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    reps = []
    for name, _, _, _ in REAL_CASES:
        candidates = [
            r for r in rows
            if r.get("case_group") == "real_itemset"
            and r.get("case_name") == name
            and abs(float(r.get("theta_ratio", 0) or 0) - 0.50) < 1e-9
        ]
        if not candidates:
            continue
        r = candidates[0]
        reps.append({
            "dataset": name,
            "algorithm": "hiep",
            "min_support": r.get("minsup_ratio", ""),
            "status": r.get("status", "OK"),
            "runtime_sec": r.get("runtime_sec", ""),
            "total_sec": r.get("total_sec", ""),
            "peak_ram_mb": r.get("peak_ram_mb", ""),
            "output_count": r.get("output_count", ""),
            "total_output_items": r.get("total_output_items", ""),
            "exit_code": "0" if r.get("status") == "OK" else "2",
            "notes": "HIEP theta_ratio=0.50, max_depth=3",
        })
    return reps


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row if k != "stdout"})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "stdout"} for row in rows)


def svg_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_runtime_svg(path: Path, rows, timeout_sec: int):
    datasets = [r[0] for r in REAL_CASES]
    algos = ["hiep", *BASELINES]
    colors = {
        "hiep": "#245c73",
        "apriori": "#b45f43",
        "eclat": "#4f8a5b",
        "fpgrowth": "#7a5c9e",
    }
    lookup = {(r["dataset"], r["algorithm"]): r for r in rows}
    width, height = 980, 540
    left, right, top, bottom = 78, 30, 58, 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(timeout_sec, max(to_float(r.get("runtime_sec")) for r in rows))
    slot = plot_w / len(datasets)
    bar_w = slot * 0.72 / len(algos)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700">External baseline runtime comparison</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for i in range(5):
        y = top + i * plot_h / 4
        value = max_val - i * max_val / 4
        out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#ddd"/>')
        out.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>')
    for di, dataset in enumerate(datasets):
        x0 = left + di * slot
        for ai, algo in enumerate(algos):
            row = lookup.get((dataset, algo))
            if not row:
                continue
            val = to_float(row.get("runtime_sec"))
            h = val / max_val * plot_h
            bx = x0 + slot * 0.14 + ai * bar_w
            fill = colors[algo]
            out.append(f'<rect x="{bx:.2f}" y="{height-bottom-h:.2f}" width="{bar_w*0.86:.2f}" height="{h:.2f}" fill="{fill}"/>')
            if row.get("status") == "TIMEOUT":
                out.append(f'<text x="{bx+bar_w*0.43:.2f}" y="{height-bottom-h-5:.2f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#9b1c1c">TO</text>')
        out.append(f'<text x="{x0+slot/2:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="13">{svg_escape(dataset)}</text>')
    out.append(f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">Runtime seconds</text>')
    lx, ly = left + 12, top + 18
    for i, algo in enumerate(algos):
        out.append(f'<rect x="{lx}" y="{ly+i*20-10}" width="12" height="12" fill="{colors[algo]}"/>')
        out.append(f'<text x="{lx+18}" y="{ly+i*20}" font-family="Arial" font-size="13">{svg_escape(algo)}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    root = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else Path(__file__).resolve().parents[1]
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "results" / "hiep_q1"
    bin_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "bin" / "dm.exe"
    timeout_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 35
    baseline_dir = out_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    log_path = out_dir / "hiep_external_baselines.txt"
    log_path.write_text("", encoding="utf-8")

    for dataset, rel_path, limit, minsup in REAL_CASES:
        sliced = baseline_dir / f"{dataset}_{limit}.txt"
        write_truncated(root / rel_path, sliced, limit)
        for algo in BASELINES:
            cmd, metrics = run_baseline(bin_path, algo, sliced, minsup, timeout_sec)
            row = {
                "dataset": dataset,
                "algorithm": algo,
                "input": str(sliced),
                "transactions": limit,
                "min_support": minsup,
                "timeout_sec": timeout_sec,
                "notes": "External support-mining baseline through dm.exe",
                **{k: v for k, v in metrics.items() if k != "stdout"},
            }
            rows.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"===== {' '.join(cmd)} =====\n")
                for k, v in row.items():
                    if k not in {"stdout"}:
                        f.write(f"{k}={v}\n")
                f.write(metrics["stdout"])
                f.write("\n\n")

    comparison_rows = [*load_hiep_representatives(out_dir), *rows]
    write_csv(out_dir / "hiep_external_baselines.csv", rows)
    write_csv(out_dir / "hiep_baseline_comparison.csv", comparison_rows)
    write_runtime_svg(out_dir / "hiep_external_baseline_runtime.svg", comparison_rows, timeout_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
