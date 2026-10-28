#!/usr/bin/env python3
"""Plot results produced by bench_threads.sh.

Reads <results-dir>/results.csv (build times) and <results-dir>/query.csv (latency / recall@10) and writes:
  build_time.png                 index construction time, one subplot per dataset, grouped vertical
                                 bars per thread count, one bar per (branch, mode, bulk) combination
  latency_recall_<dataset>.png   mean latency vs recall@10, one subplot per build thread count,
                                 one line per (branch, mode, bulk) combination (points = overquery values)

Encoding: color = branch (+ bulk setting), panama = solid / circle markers, native = hatched bars /
dashed lines / square markers. Values are means over reps; build-time error bars are the std dev.

Usage: python3 plot_bench_threads.py [results-dir] [--out-dir DIR] [--branches "main mybranch"]
                                     [--datasets "a b"] [--threads "1 16"] [--logy]
Requires matplotlib only.
"""
import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODE_ORDER = ["panama", "native"]
MODE_STYLE = {
    "panama": dict(linestyle="-", marker="o", hatch=None),
    "native": dict(linestyle="--", marker="s", hatch="//"),
}
TEXT = "#3d3d3a"
GRID = "#e4e3dd"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def split_arg(s):
    return s.split() if s else None


def ordered_unique(values, preferred=None):
    seen = list(dict.fromkeys(values))
    if preferred:
        return [v for v in preferred if v in seen] + [v for v in seen if v not in preferred]
    return seen


def filter_rows(rows, branches, datasets, threads):
    out = []
    for r in rows:
        if branches and r["branch"] not in branches:
            continue
        if datasets and r["dataset"] not in datasets:
            continue
        if threads and r["threads"] not in threads:
            continue
        out.append(r)
    return out


class Series:
    """Assigns a stable color per (branch, bulk) and a style per mode, based on all rows seen."""

    def __init__(self, rows, branch_order):
        branches = ordered_unique([r["branch"] for r in rows], branch_order)
        bulks_by_branch = defaultdict(set)
        for r in rows:
            bulks_by_branch[r["branch"]].add(r["bulk"])
        self.groups = []  # (branch, bulk) in display order
        for br in branches:
            for b in ("false", "true"):
                if b in bulks_by_branch[br]:
                    self.groups.append((br, b))
        if len(self.groups) > len(PALETTE):
            raise SystemExit(f"Too many branch/bulk groups ({len(self.groups)}); filter with --branches")
        self.color = {g: PALETTE[i] for i, g in enumerate(self.groups)}
        self.multi_bulk = {br for br, s in bulks_by_branch.items() if len(s) > 1}
        modes = ordered_unique([r["mode"] for r in rows], MODE_ORDER)
        self.keys = [(br, m, b) for (br, b) in self.groups for m in modes]

    def label(self, key):
        br, mode, bulk = key
        s = f"{br} / {mode}"
        if br in self.multi_bulk:
            s += " / bulk" if bulk == "true" else " / no-bulk"
        return s


def style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8b7ae")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def grid_shape(n, max_cols):
    cols = min(n, max_cols)
    return math.ceil(n / cols), cols


def plot_build_times(rows, series, out_dir):
    data = defaultdict(list)  # (dataset, key, threads) -> [build_time]
    for r in rows:
        v = to_float(r["build_time_s"])
        if v is not None:
            data[(r["dataset"], (r["branch"], r["mode"], r["bulk"]), r["threads"])].append(v)
    if not data:
        print("No build times to plot")
        return

    datasets = ordered_unique([d for d, _, _ in data])
    nrows, ncols = grid_shape(len(datasets), 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.6 * nrows), squeeze=False)
    handles = {}
    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        keys = [k for k in series.keys if any(d == ds and kk == k for d, kk, _ in data)]
        threads = sorted({t for d, _, t in data if d == ds}, key=int)
        n = len(keys)
        width = 0.8 / max(n, 1)
        for j, key in enumerate(keys):
            xs, means, errs = [], [], []
            for ti, t in enumerate(threads):
                vals = data.get((ds, key, t))
                if not vals:
                    continue
                xs.append(ti - 0.4 + width * (j + 0.5))
                means.append(statistics.mean(vals))
                errs.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            br, mode, bulk = key
            st = MODE_STYLE.get(mode, MODE_STYLE["panama"])
            color = series.color[(br, bulk)]
            bars = ax.bar(xs, means, width=width, color=color if not st["hatch"] else "white",
                          edgecolor=color if st["hatch"] else "white", hatch=st["hatch"], linewidth=1.2,
                          yerr=errs, error_kw=dict(ecolor=TEXT, elinewidth=0.8, capsize=2))
            handles.setdefault(key, bars)
        ax.set_xticks(range(len(threads)))
        ax.set_xticklabels([f"{t} thread{'s' if t != '1' else ''}" for t in threads])
        ax.set_ylabel("Index build time (s)")
        ax.set_title(ds, fontsize=11)
        style_axes(ax)
    for k in range(len(datasets), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    keys = [k for k in series.keys if k in handles]
    fig.legend([handles[k] for k in keys], [series.label(k) for k in keys], loc="lower center",
               ncol=min(len(keys), 4), frameon=False, fontsize=9)
    fig.suptitle("Index construction time (mean over reps, ± std dev)", color=TEXT)
    fig.tight_layout(rect=(0, 0.04 + 0.03 * math.ceil(len(keys) / 4), 1, 0.97))
    path = os.path.join(out_dir, "build_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_latency_recall(rows, series, out_dir, logy):
    data = defaultdict(lambda: defaultdict(list))  # (dataset, key, threads) -> overquery -> [(lat, rec)]
    for r in rows:
        lat, rec, oq = to_float(r["mean_latency_ms"]), to_float(r["recall_at_10"]), to_float(r["overquery"])
        if None in (lat, rec, oq):
            continue
        data[(r["dataset"], (r["branch"], r["mode"], r["bulk"]), r["threads"])][oq].append((lat, rec))
    if not data:
        print("No query results to plot")
        return

    for ds in ordered_unique([d for d, _, _ in data]):
        threads = sorted({t for d, _, t in data if d == ds}, key=int)
        nrows, ncols = grid_shape(len(threads), 3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.4 * nrows), squeeze=False)
        handles = {}
        for i, t in enumerate(threads):
            ax = axes[i // ncols][i % ncols]
            for key in series.keys:
                pts = data.get((ds, key, t))
                if not pts:
                    continue
                xs, ys = [], []
                for oq in sorted(pts):
                    xs.append(statistics.mean(p[1] for p in pts[oq]))
                    ys.append(statistics.mean(p[0] for p in pts[oq]))
                br, mode, bulk = key
                st = MODE_STYLE.get(mode, MODE_STYLE["panama"])
                (line,) = ax.plot(xs, ys, color=series.color[(br, bulk)], linestyle=st["linestyle"],
                                  marker=st["marker"], markersize=6, linewidth=2,
                                  markeredgecolor="white", markeredgewidth=1)
                handles.setdefault(key, line)
            ax.set_xlabel("Recall@10")
            ax.set_ylabel("Mean latency (ms)")
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"built with {t} thread{'s' if t != '1' else ''}", fontsize=10)
            style_axes(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
        for k in range(len(threads), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)
        keys = [k for k in series.keys if k in handles]
        fig.legend([handles[k] for k in keys], [series.label(k) for k in keys], loc="lower center",
                   ncol=min(len(keys), 3), frameon=False, fontsize=9)
        fig.suptitle(f"{ds}: mean latency vs recall@10 (points = overquery, mean over reps)", color=TEXT)
        fig.tight_layout(rect=(0, 0.05 + 0.04 * math.ceil(len(keys) / 3), 1, 0.95))
        path = os.path.join(out_dir, f"latency_recall_{ds}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default="bench-threads-results")
    ap.add_argument("--out-dir", help="default: <results-dir>/plots")
    ap.add_argument("--branches", help="space-separated branches to include, in display order (default: all)")
    ap.add_argument("--datasets", help="space-separated datasets to include (default: all)")
    ap.add_argument("--threads", help="space-separated thread counts to include (default: all)")
    ap.add_argument("--logy", action="store_true", help="log-scale latency axis")
    args = ap.parse_args()

    branches, datasets, threads = split_arg(args.branches), split_arg(args.datasets), split_arg(args.threads)
    build = filter_rows(read_csv(os.path.join(args.results_dir, "results.csv")), branches, datasets, threads)
    query = filter_rows(read_csv(os.path.join(args.results_dir, "query.csv")), branches, datasets, threads)
    if not build and not query:
        raise SystemExit(f"No results found in {args.results_dir}")

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    # main first by default so it reads as the baseline
    series = Series(build + query, branches or ["main"])
    plot_build_times(build, series, out_dir)
    plot_latency_recall(query, series, out_dir, args.logy)


if __name__ == "__main__":
    main()
