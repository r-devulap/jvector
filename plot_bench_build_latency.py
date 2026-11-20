#!/usr/bin/env python3
"""Plot the results of bench_build_latency.sh: single-thread full-precision index construction time and mean
latency vs recall@10, for each branch (baseline vs. branch) and vectorization provider (native, panama).

Usage: python3 plot_bench_build_latency.py [results-dir (default: bench-build-latency-results)]
                                           [--baseline main] [--branch panama-threshold] [--threads 1]
                                           [--machine "x8i.4xlarge (Intel GNR)"] [--out-dir DIR]
Writes build_time.png and latency_vs_recall.png (default: into the results directory). Needs matplotlib.
Values are means over reps; the bars carry a +/- std dev error bar (zero with one rep).
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (light, dark) per mode: light = baseline, dark = branch
COLORS = {"native": ("#87b4f0", "#10417f"), "panama": ("#f09c74", "#a14010")}
MODES = ["native", "panama"]


def metric_name(ds):
    if "dot" in ds:
        return "dot product"
    if "euclidean" in ds:
        return "Euclidean"
    return "cosine"  # angular and the e5 embedding sets


def load_dims(results_dir):
    dims = {}
    for path in glob.glob(os.path.join(results_dir, "*.log")):
        with open(path, errors="replace") as f:
            for _, line in zip(range(200), f):
                m = re.match(r"(\S+): \d+ base and \d+ query vectors.*dimensions (\d+)", line)
                if m:
                    dims[m.group(1)] = int(m.group(2))
                    break
    return dims


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="bench-build-latency-results")
    ap.add_argument("--baseline", default="main")
    ap.add_argument("--branch", default="panama-threshold")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--machine", default="")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or a.results_dir
    os.makedirs(out_dir, exist_ok=True)
    dims = load_dims(a.results_dir)

    builds = defaultdict(list)  # (ds, branch, mode) -> [seconds]
    for r in read_csv(os.path.join(a.results_dir, "build.csv")):
        try:
            builds[(r["dataset"], r["branch"], r["mode"])].append(float(r["build_s"]))
        except (ValueError, KeyError):
            continue
    queries = defaultdict(list)  # (ds, branch, mode, overquery) -> [(latency, recall)]
    for r in read_csv(os.path.join(a.results_dir, "query.csv")):
        try:
            if int(r["topk"]) != 10:
                continue
            queries[(r["dataset"], r["branch"], r["mode"], float(r["overquery"]))].append(
                (float(r["mean_latency_ms"]), float(r["recall"])))
        except (ValueError, KeyError):
            continue

    datasets = sorted({k[0] for k in builds} | {k[0] for k in queries}, key=lambda d: (dims.get(d, 1 << 30), d))
    if not datasets:
        raise SystemExit("no results found in " + a.results_dir)
    series = [(m, br) for m in MODES for br in (a.baseline, a.branch)]
    series_label = {(m, br): f"{m} / {br}" for m, br in series}
    series_color = {(m, br): COLORS[m][0 if br == a.baseline else 1] for m, br in series}
    # legend order as in the reference plots: native/main, native/branch, panama/main, panama/branch
    handles_fig = {}

    def title(ds):
        d = f"{dims[ds]}-d, " if ds in dims else ""
        return f"{ds} ({d}{metric_name(ds)})"

    ncols = 2
    nrows = (len(datasets) + ncols - 1) // ncols
    sub = f"Full-precision build · {a.threads} thread" + ("s" if a.threads != "1" else "")
    if a.machine:
        sub += f" · benchmarked on {a.machine}"

    # ---- build time
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows), squeeze=False)
    width = 0.2
    for ax, ds in zip(axes.flat, datasets):
        for i, s in enumerate(series):
            v = builds.get((ds, s[1], s[0]))
            if not v:
                continue
            m = statistics.mean(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            x = (i - 1.5) * (width + 0.01)
            h = ax.bar(x, m, width, yerr=sd, capsize=2, color=series_color[s], label=series_label[s],
                       error_kw={"elinewidth": 1, "ecolor": "#333"})
            handles_fig[s] = h
        ax.set_xticks([0])
        ax.set_xticklabels([f"{a.threads} thread" + ("s" if a.threads != "1" else "")])
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylabel("Index build time (s)")
        ax.set_title(title(ds), fontsize=11)
        ax.grid(axis="y", color="#e0e0e0")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)
    fig.suptitle(f"Index construction time — {a.branch} vs {a.baseline} (mean over reps, ± std dev)\n{sub}")
    ordered = [s for s in series if s in handles_fig]
    fig.legend([handles_fig[s] for s in ordered], [series_label[s] for s in ordered], loc="lower center",
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "build_time.png"), dpi=140)
    plt.close(fig)

    # ---- latency vs recall
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows), squeeze=False)
    handles_fig = {}
    for ax, ds in zip(axes.flat, datasets):
        for s in series:
            mode, br = s
            oqs = sorted({k[3] for k in queries if k[:3] == (ds, br, mode)})
            pts = []
            for oq in oqs:
                v = queries[(ds, br, mode, oq)]
                pts.append((statistics.mean(x[1] for x in v), statistics.mean(x[0] for x in v)))
            if not pts:
                continue
            base = br == a.baseline
            (line,) = ax.plot([p[0] for p in pts], [p[1] for p in pts], "--" if base else "-",
                              marker="s" if base else "o", color=series_color[s], linewidth=2, markersize=6,
                              markeredgecolor="white", label=series_label[s])
            handles_fig[s] = line
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("Mean latency (ms)")
        ax.set_title(title(ds), fontsize=11)
        ax.grid(color="#e0e0e0")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)
    fig.suptitle(f"Mean latency vs recall@10 — {a.branch} vs {a.baseline} (points = overquery; index built "
                 f"with {a.threads} thread{'s' if a.threads != '1' else ''}; mean over reps)\n"
                 f"Full-precision build" + (f" · benchmarked on {a.machine}" if a.machine else ""))
    ordered = [s for s in series if s in handles_fig]
    fig.legend([handles_fig[s] for s in ordered], [series_label[s] for s in ordered], loc="lower center",
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "latency_vs_recall.png"), dpi=140)
    plt.close(fig)
    print("wrote", os.path.join(out_dir, "build_time.png"), "and", os.path.join(out_dir, "latency_vs_recall.png"))


if __name__ == "__main__":
    main()
