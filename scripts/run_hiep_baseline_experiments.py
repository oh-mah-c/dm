#!/usr/bin/env python3
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False


REAL_CASES = [
    ("retail", "datasets/itemsets/retail.txt", 12000, "0.015"),
    ("accidents", "datasets/itemsets/accidents.txt", 3500, "0.03"),
    ("chess", "datasets/itemsets/chess.txt", 3196, "0.08"),
]

BASELINES = ["apriori", "eclat", "fpgrowth"]


def parse_report(text: str, is_hiep: bool = False):
    def number(pattern, scale=1.0):
        m = re.search(pattern, text)
        return float(m.group(1)) / scale if m else ""

    if is_hiep:
        return {
            "runtime_sec": number(r"runtime_sec=([0-9.]+)"),
            "total_sec": number(r"total_sec=([0-9.]+)"),
            "peak_ram_mb": number(r"peak_ram_mb=([0-9.]+)"),
            "output_count": number(r"output_count=([0-9]+)"),
            "total_output_items": number(r"total_output_items=([0-9]+)"),
        }
    else:
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


def run_algo(bin_path: Path, algo: str, data_path: Path, min_support: str, timeout_sec: int):
    if algo == "hiep":
        cmd = [
            str(bin_path), "hiep",
            "--input", str(data_path),
            "--input-type", "transactions",
            "--mode", "itemset",
            "--minsup", min_support,
            "--theta-ratio", "0.50",
            "--max-depth", "3",
            "--max-patterns", "25000",
            "--max-seconds", str(timeout_sec)
        ]
    else:
        cmd = [str(bin_path), algo, str(data_path), "0", min_support]
    
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
        )
        metrics = parse_report(proc.stdout, is_hiep=(algo == "hiep"))
        metrics.update({
            "status": "OK" if proc.returncode in (0, 2) else "ERROR",
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
        })
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        metrics = {
            "runtime_sec": float(timeout_sec),
            "total_sec": float(timeout_sec),
            "peak_ram_mb": "",
            "output_count": "",
            "total_output_items": "",
            "status": "TIMEOUT",
            "exit_code": 124,
            "stdout": stdout + f"\n[TIMEOUT after {timeout_sec}s]\n",
        }
    return cmd, metrics


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
        if value == "" or value is None:
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


def plot_runtime_comparison_bars(rows, out_dir: Path, timeout_sec: int):
    if not HAS_MPL:
        return
    datasets = [r[0] for r in REAL_CASES]
    algos = ["hiep", *BASELINES]
    colors = {
        "hiep": "#245c73",
        "apriori": "#b45f43",
        "eclat": "#4f8a5b",
        "fpgrowth": "#7a5c9e",
    }
    lookup = {(r["dataset"], r["algorithm"]): r for r in rows}
    
    plt.figure(figsize=(7.2, 4.8))
    x = list(range(len(datasets)))
    width = 0.18
    
    for idx, algo in enumerate(algos):
        runtimes = []
        for dataset in datasets:
            row = lookup.get((dataset, algo))
            val = to_float(row.get("runtime_sec")) if row else 0.0
            runtimes.append(val)
        pos = [i - 1.5 * width + idx * width for i in x]
        plt.bar(pos, runtimes, width, label=algo, color=colors[algo])
        
        for dataset_idx, dataset in enumerate(datasets):
            row = lookup.get((dataset, algo))
            if row and row.get("status") == "TIMEOUT":
                plt.text(pos[dataset_idx], runtimes[dataset_idx] + 0.5, "TO", 
                         ha="center", va="bottom", color="#9b1c1c", fontweight="bold", fontsize=9)
                         
    plt.ylabel("Runtime (seconds)", fontsize=11)
    plt.title("External Baseline Runtime Comparison", fontsize=13, fontweight="bold")
    plt.xticks(x, [d.capitalize() for d in datasets], fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.ylim(0, timeout_sec + 4)
    plt.tight_layout()
    plt.savefig(out_dir / "hiep_external_baseline_runtime.png", dpi=180)
    plt.close()


def write_svg_line(path: Path, title: str, xlabel: str, ylabel: str, series, timeout_sec: int):
    width, height = 900, 520
    left, right, top, bottom = 82, 28, 54, 78
    
    xs_all = [x for _, xs, _, _ in series for x in xs]
    ys_all = [y for _, _, ys, _ in series for y in ys]
    if not xs_all:
        return
    xmin, xmax = min(xs_all), max(xs_all)
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
        
    ymin = 0.0
    ymax = max(ys_all)
    if ymax >= timeout_sec - 1e-3:
        ymax = timeout_sec + 3.0
    else:
        ymax = ymax * 1.1 if ymax > 0 else 1.0

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
    ]
    
    # y-axis grid
    for i in range(5):
        y = top + i * (height - top - bottom) / 4
        value = ymax - i * (ymax - ymin) / 4
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3g}</text>')
        
    # x-axis grid/labels
    for i in range(5):
        x = left + i * (width - left - right) / 4
        value = xmin + i * (xmax - xmin) / 4
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#eee"/>')
        lines.append(f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3g}</text>')
        
    lines.append(f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(xlabel)}</text>')
    lines.append(f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(ylabel)}</text>')
    
    # Timeout reference line if applicable
    if ymax >= timeout_sec:
        ty = sy(timeout_sec)
        lines.append(f'<line x1="{left}" y1="{ty:.2f}" x2="{width-right}" y2="{ty:.2f}" stroke="#d32f2f" stroke-dasharray="4 4" stroke-width="1.5"/>')
        lines.append(f'<text x="{width-right-5}" y="{ty-6:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#d32f2f" font-weight="bold">Timeout ({timeout_sec}s)</text>')

    legend_x = left + 12
    legend_y = top + 18
    for idx, (label, xs, ys, color) in enumerate(series):
        pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y in zip(xs, ys):
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}"/>')
        ly = legend_y + idx * 20
        lines.append(f'<rect x="{legend_x}" y="{ly-10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+18}" y="{ly}" font-family="Arial" font-size="13">{svg_escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_support_sweeps(sweeps_rows, out_dir: Path, timeout_sec: int):
    datasets = ["retail", "accidents", "chess"]
    algos = ["hiep", "apriori", "eclat", "fpgrowth"]
    colors = {
        "hiep": "#245c73",
        "apriori": "#b45f43",
        "eclat": "#4f8a5b",
        "fpgrowth": "#7a5c9e",
    }
    markers = {
        "hiep": "o",
        "apriori": "s",
        "eclat": "^",
        "fpgrowth": "D",
    }

    for dataset in datasets:
        ds_rows = [r for r in sweeps_rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
            
        series_data = []
        for algo in algos:
            algo_rows = sorted(
                [r for r in ds_rows if r["algorithm"] == algo],
                key=lambda r: to_float(r["min_support"])
            )
            xs = [to_float(r["min_support"]) for r in algo_rows]
            ys = [to_float(r["runtime_sec"]) for r in algo_rows]
            if xs:
                series_data.append((algo, xs, ys, colors[algo]))
                
        # Generate SVG
        svg_path = out_dir / f"hiep_runtime_vs_support_{dataset}.svg"
        write_svg_line(
            svg_path,
            f"Runtime vs. Support Threshold ({dataset.capitalize()})",
            "Minimum Support Ratio",
            "Runtime (seconds)",
            series_data,
            timeout_sec
        )
        
        # Generate PNG if Matplotlib is available
        if HAS_MPL:
            plt.figure(figsize=(7.2, 4.8))
            for algo, xs, ys, color in series_data:
                plt.plot(xs, ys, marker=markers[algo], label=algo, color=color, linewidth=2.0, markersize=6)
                
            plt.axhline(y=timeout_sec, color="#d32f2f", linestyle="--", alpha=0.7, label="Timeout" if dataset=="retail" else None)
            
            # y limits
            ymax = max(y for _, _, ys, _ in series_data for y in ys)
            if ymax >= timeout_sec - 1e-3:
                plt.ylim(-1, timeout_sec + 3)
            else:
                plt.ylim(-0.01 * ymax, ymax * 1.1)
                
            plt.xlabel("Minimum Support Ratio", fontsize=11)
            plt.ylabel("Runtime (seconds)", fontsize=11)
            plt.title(f"Runtime vs. Support Threshold ({dataset.capitalize()})", fontsize=13, fontweight="bold")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(loc="upper right", framealpha=0.9)
            plt.tight_layout()
            
            png_path = out_dir / f"hiep_runtime_vs_support_{dataset}.png"
            plt.savefig(png_path, dpi=180)
            plt.close()


def main():
    root = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else Path(__file__).resolve().parents[1]
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "results" / "hiep_q1"
    bin_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "bin" / "dm.exe"
    timeout_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 35
    baseline_dir = out_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    sweeps_rows = []
    log_path = out_dir / "hiep_external_baselines.txt"
    log_path.write_text("", encoding="utf-8")

    REAL_CASES_DICT = {
        "retail": (root / "datasets/itemsets/retail.txt", 12000),
        "accidents": (root / "datasets/itemsets/accidents.txt", 3500),
        "chess": (root / "datasets/itemsets/chess.txt", 3196),
    }

    dataset_sweeps = {
        "retail": ["0.005", "0.008", "0.010", "0.015", "0.020", "0.030"],
        "accidents": ["0.010", "0.015", "0.020", "0.025", "0.030", "0.040", "0.050"],
        "chess": ["0.050", "0.060", "0.070", "0.080", "0.090", "0.100", "0.120"]
    }

    algos = ["hiep", "apriori", "eclat", "fpgrowth"]

    print("Running support sweeps for all datasets...")
    for dataset, (rel_path, limit) in REAL_CASES_DICT.items():
        sliced = baseline_dir / f"{dataset}_{limit}.txt"
        write_truncated(rel_path, sliced, limit)
        
        minsup_list = dataset_sweeps[dataset]
        for minsup in minsup_list:
            for algo in algos:
                print(f"  Running {algo} on {dataset} with support {minsup}...")
                cmd, metrics = run_algo(bin_path, algo, sliced, minsup, timeout_sec)
                row = {
                    "dataset": dataset,
                    "algorithm": algo,
                    "input": str(sliced),
                    "transactions": limit,
                    "min_support": minsup,
                    "timeout_sec": timeout_sec,
                    "notes": "Sweep support-mining baseline through dm.exe",
                    **{k: v for k, v in metrics.items() if k != "stdout"},
                }
                sweeps_rows.append(row)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"===== {' '.join(cmd)} =====\n")
                    for k, v in row.items():
                        if k not in {"stdout"}:
                            f.write(f"{k}={v}\n")
                    f.write(metrics["stdout"])
                    f.write("\n\n")

    # Extract comparison rows at default support values for original bar chart
    comparison_rows = []
    defaults = {
        "retail": 0.015,
        "accidents": 0.030,
        "chess": 0.080,
    }
    for row in sweeps_rows:
        dataset = row["dataset"]
        minsup_val = float(row["min_support"])
        if abs(minsup_val - defaults[dataset]) < 1e-9:
            comparison_rows.append(row)

    write_csv(out_dir / "hiep_baseline_runtime_vs_support.csv", sweeps_rows)
    write_csv(out_dir / "hiep_external_baselines.csv", [r for r in comparison_rows if r["algorithm"] != "hiep"])
    write_csv(out_dir / "hiep_baseline_comparison.csv", comparison_rows)
    write_runtime_svg(out_dir / "hiep_external_baseline_runtime.svg", comparison_rows, timeout_sec)
    plot_runtime_comparison_bars(comparison_rows, out_dir, timeout_sec)
    
    print("Generating line charts...")
    plot_support_sweeps(sweeps_rows, out_dir, timeout_sec)
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
