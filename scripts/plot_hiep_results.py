#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ModuleNotFoundError:
    HAS_MPL = False


NUMERIC_KEYS = {
    "window_length", "stride", "transactions", "token_stream_length", "input_bytes",
    "nnz", "vocabulary_size", "minsup_ratio", "minsup_count", "theta",
    "theta_ratio", "alpha", "gamma", "disable_tiub", "disable_iwru",
    "uniform_weights", "disable_compactness", "runtime_sec", "total_sec",
    "peak_ram_mb", "throughput_mb_s", "throughput_tok_s", "surviving_items",
    "singleton_occurrences", "visited_nodes", "generated_children", "joins",
    "joined_entries", "pruned_support", "pruned_tiub", "pruned_iwru",
    "output_count", "total_output_items", "avg_output_length", "avg_support",
    "avg_utility", "avg_pattern_weight", "best_utility",
    "information_density_optimization", "noise_filtering_efficiency",
    "signal_recall", "avg_utility_list_length", "max_utility_list_length",
    "max_depth", "result_ram_bytes", "result_disk_est_bytes", "zipf_skew",
    "exit_code",
}


def parse_file(path: Path):
    rows = []
    current = {}
    config = ""
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            if current:
                current["source_file"] = path.name
                current["config"] = config
                rows.append(current)
                current = {}
            continue
        if line.startswith("====="):
            config = line.strip("= ")
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key in NUMERIC_KEYS:
            try:
                current[key] = float(value)
            except ValueError:
                current[key] = value
        else:
            current[key] = value
    if current:
        current["source_file"] = path.name
        current["config"] = config
        rows.append(current)
    return rows


def save_csv(rows, out_dir: Path):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with (out_dir / "hiep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_bounds(values, log_scale=False):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    if log_scale:
        vals = [math.log10(max(v, 1e-9)) for v in vals]
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
        lo = min(0.0, lo)
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def write_svg_line(path: Path, title: str, xlabel: str, ylabel: str, series, log_y=False):
    width, height = 900, 520
    left, right, top, bottom = 82, 28, 54, 78
    xs_all = [x for _, xs, _, _ in series for x in xs]
    ys_all = [y for _, _, ys, _ in series for y in ys]
    if not xs_all:
        return
    xmin, xmax = min(xs_all), max(xs_all)
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    ymin, ymax = nice_bounds(ys_all, log_y)

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y):
        val = math.log10(max(y, 1e-9)) if log_y else y
        return top + (ymax - val) / (ymax - ymin) * (height - top - bottom)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for i in range(5):
        y = top + i * (height - top - bottom) / 4
        value = ymax - i * (ymax - ymin) / 4
        shown = 10 ** value if log_y else value
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{shown:.3g}</text>')
    for i in range(5):
        x = left + i * (width - left - right) / 4
        value = xmin + i * (xmax - xmin) / 4
        lines.append(f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3g}</text>')
    lines.append(f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(xlabel)}</text>')
    lines.append(f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(ylabel)}</text>')
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


def write_svg_bars(path: Path, title: str, labels, series, ylabel: str, stacked=False):
    width, height = 980, 540
    left, right, top, bottom = 86, 28, 56, 118
    colors = ["#536878", "#b45f43", "#4f8a5b", "#7a5c9e", "#c9823b"]
    if stacked:
        max_val = max([sum(vals[i] for _, vals in series) for i in range(len(labels))] or [1.0])
    else:
        max_val = max([v for _, vals in series for v in vals] or [1.0])
    if max_val <= 0:
        max_val = 1.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    slot = plot_w / max(len(labels), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for i in range(5):
        y = top + i * plot_h / 4
        value = max_val - i * max_val / 4
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3g}</text>')
    for i, label in enumerate(labels):
        x0 = left + i * slot
        if stacked:
            y_base = height - bottom
            for si, (_, vals) in enumerate(series):
                h = vals[i] / max_val * plot_h
                lines.append(f'<rect x="{x0+slot*0.18:.2f}" y="{y_base-h:.2f}" width="{slot*0.64:.2f}" height="{h:.2f}" fill="{colors[si % len(colors)]}"/>')
                y_base -= h
        else:
            bar_w = slot * 0.72 / max(len(series), 1)
            for si, (_, vals) in enumerate(series):
                h = vals[i] / max_val * plot_h
                bx = x0 + slot * 0.14 + si * bar_w
                lines.append(f'<rect x="{bx:.2f}" y="{height-bottom-h:.2f}" width="{bar_w*0.88:.2f}" height="{h:.2f}" fill="{colors[si % len(colors)]}"/>')
        lines.append(f'<text transform="translate({x0+slot/2:.2f} {height-bottom+16}) rotate(35)" text-anchor="start" font-family="Arial" font-size="12">{svg_escape(label)}</text>')
    lines.append(f'<text transform="translate(22 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(ylabel)}</text>')
    lx, ly = left + 8, top + 18
    for si, (name, _) in enumerate(series):
        lines.append(f'<rect x="{lx}" y="{ly+si*20-10}" width="12" height="12" fill="{colors[si % len(colors)]}"/>')
        lines.append(f'<text x="{lx+18}" y="{ly+si*20}" font-family="Arial" font-size="13">{svg_escape(name)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def by_case(rows):
    groups = {}
    for row in rows:
        name = str(row.get("case_name") or Path(str(row.get("input", "dataset"))).stem)
        groups.setdefault(name, []).append(row)
    return groups


def sorted_by_theta(rows):
    return sorted(rows, key=lambda r: (float(r.get("theta_ratio", 0.0)), float(r.get("theta", 0.0))))


def plot_metric(rows, out_dir: Path, metric: str, title: str, ylabel: str):
    for case, case_rows in by_case(rows).items():
        usable = [r for r in sorted_by_theta(case_rows) if metric in r]
        if len(usable) < 2:
            continue
        xs = [float(r.get("theta_ratio", 0.0)) for r in usable]
        ys = [float(r.get(metric, 0.0)) for r in usable]
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(xs, ys, marker="o", linewidth=2.0, color="#245c73")
        plt.xlabel("Theta ratio")
        plt.ylabel(ylabel)
        plt.title(f"{title}: {case}")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"{case}_{metric}_vs_theta.png", dpi=180)
        plt.close()


def representative_rows(rows):
    chosen = {}
    for row in rows:
        case = str(row.get("case_name") or Path(str(row.get("input", "dataset"))).stem)
        prev = chosen.get(case)
        if prev is None or abs(float(row.get("theta_ratio", 0.0)) - 0.35) < abs(float(prev.get("theta_ratio", 0.0)) - 0.35):
            chosen[case] = row
    return list(chosen.values())


def plot_pruning(rows, out_dir: Path):
    reps = representative_rows([r for r in rows if str(r.get("case_group", "")) != "ablation"])
    if not reps:
        return
    labels = [str(r.get("case_name", "case")) for r in reps]
    support = [float(r.get("pruned_support", 0.0)) for r in reps]
    tiub = [float(r.get("pruned_tiub", 0.0)) for r in reps]
    iwru = [float(r.get("pruned_iwru", 0.0)) for r in reps]
    xs = range(len(reps))
    plt.figure(figsize=(max(8.0, len(reps) * 0.8), 4.8))
    plt.bar(xs, support, label="Support", color="#536878")
    plt.bar(xs, tiub, bottom=support, label="TIUB", color="#b45f43")
    bottom = [a + b for a, b in zip(support, tiub)]
    plt.bar(xs, iwru, bottom=bottom, label="IWRU", color="#4f8a5b")
    plt.xticks(list(xs), labels, rotation=30, ha="right")
    plt.ylabel("Pruned branches")
    plt.title("HIEP-Miner pruning breakdown")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "hiep_pruning_breakdown.png", dpi=180)
    plt.close()


def plot_throughput(rows, out_dir: Path):
    reps = representative_rows(rows)
    if not reps:
        return
    labels = [str(r.get("case_name", "case")) for r in reps]
    mb = [float(r.get("throughput_mb_s", 0.0)) for r in reps]
    tok = [float(r.get("throughput_tok_s", 0.0)) / 1000000.0 for r in reps]
    xs = list(range(len(reps)))
    width = 0.38
    plt.figure(figsize=(max(8.0, len(reps) * 0.8), 4.8))
    plt.bar([x - width / 2 for x in xs], mb, width=width, label="MB/s", color="#246a73")
    plt.bar([x + width / 2 for x in xs], tok, width=width, label="Million tokens/s", color="#c9823b")
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("Throughput")
    plt.title("Embedded tokenizer and miner throughput")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "hiep_throughput.png", dpi=180)
    plt.close()


def plot_utility_lists(rows, out_dir: Path):
    reps = representative_rows(rows)
    if not reps:
        return
    labels = [str(r.get("case_name", "case")) for r in reps]
    avg = [float(r.get("avg_utility_list_length", 0.0)) for r in reps]
    mx = [float(r.get("max_utility_list_length", 0.0)) for r in reps]
    xs = list(range(len(reps)))
    plt.figure(figsize=(max(8.0, len(reps) * 0.8), 4.8))
    plt.plot(xs, avg, marker="o", label="Average", color="#245c73")
    plt.plot(xs, mx, marker="s", label="Maximum", color="#9b4e3d")
    plt.yscale("log")
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("Utility-list entries, log scale")
    plt.title("Average and maximum utility-list length")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "hiep_utility_list_lengths.png", dpi=180)
    plt.close()


def plot_zipf_effect(rows, out_dir: Path):
    syn = [r for r in rows if str(r.get("case_group", "")) == "synthetic_zipf" and abs(float(r.get("theta_ratio", 0.0)) - 0.35) < 1e-9]
    syn = sorted(syn, key=lambda r: float(r.get("zipf_skew", 0.0)))
    if len(syn) < 2:
        return
    xs = [float(r.get("zipf_skew", 0.0)) for r in syn]
    recall = [float(r.get("signal_recall", 0.0)) for r in syn]
    nfe = [float(r.get("noise_filtering_efficiency", 0.0)) for r in syn]
    ido = [float(r.get("information_density_optimization", 0.0)) for r in syn]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(xs, recall, marker="o", label="Signal recall", color="#245c73")
    ax1.plot(xs, nfe, marker="s", label="Noise filtering", color="#4f8a5b")
    ax1.set_xlabel("Zipf skew s")
    ax1.set_ylabel("Quality score")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(xs, ido, marker="^", label="IDO", color="#b45f43")
    ax2.set_ylabel("Information density optimization")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title("Effect of Zipf skew on HIEP output quality")
    fig.tight_layout()
    plt.savefig(out_dir / "hiep_zipf_skew_quality.png", dpi=180)
    plt.close(fig)


def plot_ablation(rows, out_dir: Path):
    abl = [r for r in rows if str(r.get("case_group", "")) == "ablation"]
    if not abl:
        return
    labels = [str(r.get("ablation", r.get("case_name", "case"))) for r in abl]
    metrics = [
        ("runtime_sec", "Runtime seconds", "#245c73"),
        ("visited_nodes", "Visited nodes", "#b45f43"),
        ("signal_recall", "Signal recall", "#4f8a5b"),
        ("information_density_optimization", "IDO", "#7a5c9e"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, (metric, title, color) in zip(axes.ravel(), metrics):
        vals = [float(r.get(metric, 0.0)) for r in abl]
        ax.bar(range(len(abl)), vals, color=color)
        ax.set_title(title)
        ax.set_xticks(range(len(abl)), labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("HIEP-Miner ablation study")
    fig.tight_layout()
    plt.savefig(out_dir / "hiep_ablation_study.png", dpi=180)
    plt.close(fig)


def plot_svg_fallback(rows, out_dir: Path):
    metric_specs = [
        ("runtime_sec", "Runtime versus threshold", "Theta ratio", "Seconds"),
        ("peak_ram_mb", "Peak memory versus threshold", "Theta ratio", "MB"),
        ("output_count", "Output pattern count versus threshold", "Theta ratio", "Patterns"),
        ("noise_filtering_efficiency", "Noise-filtering efficiency versus threshold", "Theta ratio", "NFE"),
        ("signal_recall", "Signal recall versus threshold", "Theta ratio", "Recall"),
        ("information_density_optimization", "Information density versus threshold", "Theta ratio", "IDO"),
    ]
    for metric, title, xlabel, ylabel in metric_specs:
        for case, case_rows in by_case(rows).items():
            usable = [r for r in sorted_by_theta(case_rows) if metric in r]
            if len(usable) < 2:
                continue
            xs = [float(r.get("theta_ratio", 0.0)) for r in usable]
            ys = [float(r.get(metric, 0.0)) for r in usable]
            write_svg_line(
                out_dir / f"{case}_{metric}_vs_theta.svg",
                f"{title}: {case}",
                xlabel,
                ylabel,
                [(metric, xs, ys, "#245c73")],
            )

    reps = representative_rows([r for r in rows if str(r.get("case_group", "")) != "ablation"])
    if reps:
        labels = [str(r.get("case_name", "case")) for r in reps]
        write_svg_bars(
            out_dir / "hiep_pruning_breakdown.svg",
            "HIEP-Miner pruning breakdown",
            labels,
            [
                ("Support", [float(r.get("pruned_support", 0.0)) for r in reps]),
                ("TIUB", [float(r.get("pruned_tiub", 0.0)) for r in reps]),
                ("IWRU", [float(r.get("pruned_iwru", 0.0)) for r in reps]),
            ],
            "Pruned branches",
            stacked=True,
        )
        write_svg_bars(
            out_dir / "hiep_throughput.svg",
            "Embedded tokenizer and miner throughput",
            labels,
            [
                ("MB/s", [float(r.get("throughput_mb_s", 0.0)) for r in reps]),
                ("Million tokens/s", [float(r.get("throughput_tok_s", 0.0)) / 1000000.0 for r in reps]),
            ],
            "Throughput",
            stacked=False,
        )
        xs = list(range(len(reps)))
        write_svg_line(
            out_dir / "hiep_utility_list_lengths.svg",
            "Average and maximum utility-list length",
            "Representative dataset index",
            "Utility-list entries, log scale",
            [
                ("Average", xs, [float(r.get("avg_utility_list_length", 0.0)) for r in reps], "#245c73"),
                ("Maximum", xs, [float(r.get("max_utility_list_length", 0.0)) for r in reps], "#b45f43"),
            ],
            log_y=True,
        )

    syn = [r for r in rows if str(r.get("case_group", "")) == "synthetic_zipf" and abs(float(r.get("theta_ratio", 0.0)) - 0.35) < 1e-9]
    syn = sorted(syn, key=lambda r: float(r.get("zipf_skew", 0.0)))
    if len(syn) >= 2:
        xs = [float(r.get("zipf_skew", 0.0)) for r in syn]
        write_svg_line(
            out_dir / "hiep_zipf_skew_quality.svg",
            "Effect of Zipf skew on HIEP output quality",
            "Zipf skew s",
            "Score",
            [
                ("Signal recall", xs, [float(r.get("signal_recall", 0.0)) for r in syn], "#245c73"),
                ("Noise filtering", xs, [float(r.get("noise_filtering_efficiency", 0.0)) for r in syn], "#4f8a5b"),
                ("IDO scaled", xs, [float(r.get("information_density_optimization", 0.0)) / 1000.0 for r in syn], "#b45f43"),
            ],
        )

    abl = [r for r in rows if str(r.get("case_group", "")) == "ablation"]
    if abl:
        labels = [str(r.get("ablation", r.get("case_name", "case"))) for r in abl]
        write_svg_bars(
            out_dir / "hiep_ablation_study.svg",
            "HIEP-Miner ablation study",
            labels,
            [
                ("Runtime sec", [float(r.get("runtime_sec", 0.0)) for r in abl]),
                ("Visited nodes / 1k", [float(r.get("visited_nodes", 0.0)) / 1000.0 for r in abl]),
                ("Signal recall", [float(r.get("signal_recall", 0.0)) for r in abl]),
            ],
            "Scaled value",
            stacked=False,
        )


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/hiep_q1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(out_dir.glob("hiep_*.txt")):
        rows.extend(parse_file(path))
    save_csv(rows, out_dir)
    if not rows:
        return 0
    if not HAS_MPL:
        plot_svg_fallback(rows, out_dir)
        return 0

    for metric, title, ylabel in [
        ("runtime_sec", "Runtime versus threshold", "Seconds"),
        ("peak_ram_mb", "Peak memory versus threshold", "MB"),
        ("output_count", "Output pattern count versus threshold", "Patterns"),
        ("noise_filtering_efficiency", "Noise-filtering efficiency versus threshold", "NFE"),
        ("signal_recall", "Signal recall versus threshold", "Recall"),
        ("information_density_optimization", "Information density versus threshold", "IDO"),
    ]:
        plot_metric(rows, out_dir, metric, title, ylabel)
    plot_pruning(rows, out_dir)
    plot_throughput(rows, out_dir)
    plot_utility_lists(rows, out_dir)
    plot_zipf_effect(rows, out_dir)
    plot_ablation(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
