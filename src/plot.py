"""Plot a run's results: per-benchmark accuracy (reference vs our best proven program
vs Herbie), benchmarks ordered by reference size. One chart for average accuracy
(Herbie's bits-of-error metric), one for worst case (FPTaylor bounds; split into a
relative-ulps panel and an absolute-error panel, which are different units).

Reads <run>/herbie.json (written by herbie.py); writes PNGs to <run>/plots/.

    uv run src/plot.py [--run NAME]
"""

import argparse
import json

from base import benchmarks, paths, regions

# dataviz reference palette (light mode): neutral baseline + series 1-2
ORIGINAL, OURS, HERBIE = "#aeaca4", "#2a78d6", "#1baf7a"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SERIES = (("original", ORIGINAL), ("ours (best proven)", OURS), ("herbie", HERBIE))


def op_count(node) -> int:
    if isinstance(node, str):
        return 0
    return 1 + sum(op_count(c) for c in node[1:])


def load_rows(run) -> list[dict]:
    rows = []
    for r in json.loads((run / "herbie.json").read_text()):
        b = r["benchmark"]
        try:
            ref = benchmarks.read_reference(b)
        except FileNotFoundError:
            continue
        w = lambda k, m: (r.get(k) or {}).get(m)
        rows.append({
            "benchmark": b, "ops": op_count(regions.body_of(ref)),
            "avg": (r.get("reference_bits"), r.get("ours_bits"), r.get("herbie_bits")),
            "ulps": tuple(w(k, "rel_err_ulps") for k in
                          ("reference_worst", "ours_worst", "herbie_worst")),
            "abs": tuple(w(k, "abs_err") for k in
                         ("reference_worst", "ours_worst", "herbie_worst")),
        })
    rows.sort(key=lambda r: (r["ops"], r["benchmark"]))
    return rows


def grouped_bars(ax, rows, key, log=False, floor_min=0.0):
    import math
    n = len(rows)
    vals = [r[key] for r in rows]
    floor = 0.0
    positive = [v for triple in vals for v in triple if v]
    if log and positive:
        floor = max(10 ** math.floor(math.log10(min(positive))) / 2, floor_min)
    for j, (label, color) in enumerate(SERIES):
        xs = [i + (j - 1) * 0.28 for i in range(n)]
        ax.bar(xs, [v if v is not None else 0 for v in (t[j] for t in vals)],
               width=0.26, color=color, label=label, zorder=3)
        for x, v in zip(xs, (t[j] for t in vals)):
            if v is None:
                ax.text(x, floor if log else 0, "×", ha="center", va="bottom",
                        fontsize=5.5, color=INK2, zorder=4)
    if log and positive:  # scale AFTER drawing: set_ylim first would freeze the top
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor, top=max(positive) * 3)
    ax.set_xticks(range(n),
                  [f"{r['benchmark']}  [{r['ops']}]" for r in rows],
                  rotation=90, fontsize=6.8, color=INK)
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis="y", color="#e8e7e2", lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_facecolor(SURFACE)


def plot_average(rows, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 5.2), facecolor=SURFACE)
    # log scale: values span 0.01..58 bits; bars under the floor are ~exact anyway
    grouped_bars(ax, rows, "avg", log=True, floor_min=0.005)
    ax.set_ylabel("average bits of error (log; lower is better)", color=INK2, fontsize=9)
    ax.set_title("Average accuracy — benchmarks ordered by reference size [ops]",
                 color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_worst(rows, out):
    import matplotlib.pyplot as plt
    with_ulps = [r for r in rows if r["ulps"][0] is not None]
    abs_only = [r for r in rows if r["ulps"][0] is None and any(r["abs"])]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9.5), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [max(len(with_ulps), 1), max(len(abs_only), 1)]})
    grouped_bars(ax1, with_ulps, "ulps", log=True)
    ax1.set_ylabel("worst-case relative error (ulps, log)", color=INK2, fontsize=9)
    ax1.set_title("Worst case — benchmarks ordered by reference size [ops]   "
                  "(× = no bound)", color=INK, fontsize=12, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    grouped_bars(ax2, abs_only, "abs", log=True)
    ax2.set_ylabel("worst-case absolute error (log)", color=INK2, fontsize=9)
    ax2.set_title("benchmarks where relative error is undefined over the box",
                  color=INK2, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=None, metavar="NAME")
    args = p.parse_args()
    run = paths.run_dir(args.run) if args.run else paths.latest_run()
    if run is None or not (run / "herbie.json").exists():
        p.error("run has no herbie.json (run herbie.py first)")

    rows = load_rows(run)
    out = run / "plots"
    out.mkdir(exist_ok=True)
    plot_average(rows, out / "average.png")
    plot_worst(rows, out / "worst_case.png")
    print(f"wrote {out / 'average.png'} and {out / 'worst_case.png'}")


if __name__ == "__main__":
    main()
