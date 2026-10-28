#!/usr/bin/env python3
"""Plot results produced by bench_threads.sh.

Reads <results-dir>/results.csv (build times) and <results-dir>/query.csv (latency / recall@10) and writes:
  build_time.png       index construction time, one subplot per dataset, grouped vertical bars per
                       thread count, one bar per (mode, branch, bulk) combination
  latency_recall.png   mean latency vs recall@10, one subplot per dataset (points = overquery values),
                       for the index built with --query-threads build threads

Encoding: hue = mode (native = blue, panama = orange), so all native series read as one family.
Within a mode, lightness = variant (main lightest, then this branch without bulk, then with bulk);
lines also use linestyle per variant (dotted / dashed / solid) and marker per mode (square / circle).
Bars within a cluster are ordered native variants first, then panama variants.
Values are means over reps; build-time error bars are the std dev.

Usage: python3 plot_bench_threads.py [results-dir] [--out-dir DIR] [--branches "main mybranch"]
                                     [--datasets "a b"] [--threads "1 16"] [--query-threads N] [--logy]
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

MODE_ORDER = ["native", "panama"]
# One-hue ordinal ramps (light -> dark), validated: monotone lightness, light end >= 2:1 on white.
RAMPS = {
    "native": ["#86b6ef", "#2a78d6", "#104281"],
    "panama": ["#ef9a72", "#eb6834", "#a3400f"],
}
MARKERS = {"native": "s", "panama": "o"}
VARIANT_LINESTYLES = [":", "--", "-"]
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
    return [r for r in rows
            if (not branches or r["branch"] in branches)
            and (not datasets or r["dataset"] in datasets)
            and (not threads or r["threads"] in threads)]


class Series:
    """Series key = (branch, mode, bulk). A 'variant' is a (branch, bulk) pair; it picks the ramp step."""

    def __init__(self, rows, branch_order):
        branches = ordered_unique([r["branch"] for r in rows], branch_order)
        bulks_by_branch = defaultdict(set)
        for r in rows:
            bulks_by_branch[r["branch"]].add(r["bulk"])
        self.variants = [(br, b) for br in branches for b in ("false", "true") if b in bulks_by_branch[br]]
        if len(self.variants) > len(VARIANT_LINESTYLES):
            raise SystemExit(f"{len(self.variants)} branch/bulk variants found {self.variants}; at most "
                             f"{len(VARIANT_LINESTYLES)} can be told apart. Filter with --branches.")
        self.multi_bulk = {br for br, s in bulks_by_branch.items() if len(s) > 1}
        modes = ordered_unique([r["mode"] for r in rows], MODE_ORDER)
        # mode-major so native series sit next to each other (bars) and are listed together (legend)
        self.keys = [(br, m, b) for m in modes for (br, b) in self.variants]

    def _variant(self, key):
        br, _, bulk = key
        return self.variants.index((br, bulk))

    def color(self, key):
        ramp = RAMPS.get(key[1], RAMPS["native"])
        # spread variants over the ramp: 2 variants -> light/dark ends, 3 -> all steps
        n = len(self.variants)
        if n == 1:
            return ramp[-1]
        return ramp[round(self._variant(key) * (len(ramp) - 1) / (n - 1))]

    def linestyle(self, key):
        n = len(self.variants)
        return VARIANT_LINESTYLES[len(VARIANT_LINESTYLES) - n + self._variant(key)]

    def marker(self, key):
        return MARKERS.get(key[1], "o")

    def label(self, key):
        br, mode, bulk = key
        s = f"{mode} / {br}"
        if br in self.multi_bulk:
            s += " / bulk" if bulk == "true" else " / no-bulk"
        return s


def style_axes(ax, xgrid=False):
    ax.grid(axis="both" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8b7ae")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def subplot_grid(n, figsize_per):
    cols = 1 if n == 1 else 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def add_legend(fig, series, handles):
    # keys are mode-major and matplotlib fills legend columns top-to-bottom, so with one column per
    # mode the native variants form one column and panama the other, row-aligned by variant
    keys = [k for k in series.keys if k in handles]
    ncol = max(len({k[1] for k in keys}), 1)
    fig.legend([handles[k] for k in keys], [series.label(k) for k in keys], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False, fontsize=10)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    # bbox_inches="tight" grows the canvas to include the legend placed below the axes
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


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
    fig, axes = subplot_grid(len(datasets), (7.5, 4.6))
    handles = {}
    for ax, ds in zip(axes, datasets):
        keys = [k for k in series.keys if any(d == ds and kk == k for d, kk, _ in data)]
        threads = sorted({t for d, _, t in data if d == ds}, key=int)
        width = 0.84 / max(len(keys), 1)
        for j, key in enumerate(keys):
            xs, means, errs = [], [], []
            for ti, t in enumerate(threads):
                vals = data.get((ds, key, t))
                if not vals:
                    continue
                xs.append(ti - 0.42 + width * (j + 0.5))
                means.append(statistics.mean(vals))
                errs.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            bars = ax.bar(xs, means, width=width, color=series.color(key), edgecolor="white", linewidth=1,
                          yerr=errs, error_kw=dict(ecolor=TEXT, elinewidth=0.8, capsize=2))
            handles.setdefault(key, bars)
        ax.set_xticks(range(len(threads)))
        ax.set_xticklabels([f"{t} thread{'s' if t != '1' else ''}" for t in threads])
        ax.set_ylabel("Index build time (s)")
        ax.set_title(ds, fontsize=11)
        style_axes(ax)
    fig.suptitle("Index construction time (mean over reps, ± std dev)", color=TEXT)
    fig.tight_layout()
    add_legend(fig, series, handles)
    save(fig, out_dir, "build_time.png")


def plot_latency_recall(rows, series, out_dir, query_threads, logy):
    if not rows:
        print("No query results to plot")
        return
    if query_threads is None:
        query_threads = max((r["threads"] for r in rows), key=int)
    data = defaultdict(lambda: defaultdict(list))  # (dataset, key) -> overquery -> [(lat, rec)]
    for r in rows:
        if r["threads"] != str(query_threads):
            continue
        lat, rec, oq = to_float(r["mean_latency_ms"]), to_float(r["recall_at_10"]), to_float(r["overquery"])
        if None in (lat, rec, oq):
            continue
        data[(r["dataset"], (r["branch"], r["mode"], r["bulk"]))][oq].append((lat, rec))
    if not data:
        print(f"No query results for indexes built with {query_threads} threads")
        return

    datasets = ordered_unique([d for d, _ in data])
    fig, axes = subplot_grid(len(datasets), (6.5, 4.8))
    handles = {}
    for ax, ds in zip(axes, datasets):
        for key in series.keys:
            pts = data.get((ds, key))
            if not pts:
                continue
            xs = [statistics.mean(p[1] for p in pts[oq]) for oq in sorted(pts)]
            ys = [statistics.mean(p[0] for p in pts[oq]) for oq in sorted(pts)]
            (line,) = ax.plot(xs, ys, color=series.color(key), linestyle=series.linestyle(key),
                              marker=series.marker(key), markersize=7, linewidth=2,
                              markeredgecolor="white", markeredgewidth=1)
            handles.setdefault(key, line)
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("Mean latency (ms)")
        if logy:
            ax.set_yscale("log")
        ax.set_title(ds, fontsize=11)
        style_axes(ax, xgrid=True)
    fig.suptitle(f"Mean latency vs recall@10 (points = overquery; index built with {query_threads} threads; "
                 f"mean over reps)", color=TEXT)
    fig.tight_layout()
    add_legend(fig, series, handles)
    save(fig, out_dir, "latency_recall.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default="bench-threads-results")
    ap.add_argument("--out-dir", help="default: <results-dir>/plots")
    ap.add_argument("--branches", help="space-separated branches to include, in display order (default: all)")
    ap.add_argument("--datasets", help="space-separated datasets to include (default: all)")
    ap.add_argument("--threads", help="space-separated build thread counts to include (default: all)")
    ap.add_argument("--query-threads", type=int,
                    help="build thread count whose index is used for latency/recall (default: the largest)")
    ap.add_argument("--logy", action="store_true", help="log-scale latency axis")
    args = ap.parse_args()

    branches, datasets, threads = split_arg(args.branches), split_arg(args.datasets), split_arg(args.threads)
    build = filter_rows(read_csv(os.path.join(args.results_dir, "results.csv")), branches, datasets, threads)
    query = filter_rows(read_csv(os.path.join(args.results_dir, "query.csv")), branches, datasets, threads)
    if not build and not query:
        raise SystemExit(f"No results found in {args.results_dir}")

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    # main first by default so it reads as the baseline (lightest step)
    series = Series(build + query, branches or ["main"])
    plot_build_times(build, series, out_dir)
    plot_latency_recall(query, series, out_dir, args.query_threads, args.logy)


if __name__ == "__main__":
    main()
