"""Aggregate one run's results into a console table and results/<run>/summary.md."""

import json

from base import paths


def _benchmark_row(b, run):
    esrc = paths.equivalents_path(run, b)
    if not esrc.exists():
        return None
    data = json.loads(esrc.read_text())
    programs = data.get("programs", [])
    row = {"benchmark": b,
           "samples": len(data.get("attempts") or programs),
           "best_ulp": None, "metric": None,
           "verdict": "unmeasurable" if programs else "no-valid"}
    fsrc = paths.fptaylor_path(run, b)
    if fsrc.exists() and programs:
        fd = json.loads(fsrc.read_text())
        results = fd.get("results", [])
        reference = fd.get("reference_result") or {}
        row["best_ulp"] = next((r["rel_err_ulps"] for r in results
                                if r.get("rel_err_ulps") is not None), None)
        # rel error first; absolute where rel is undefined
        for key, label in (("rel_err_ulps", "rel"), ("abs_err", "abs")):
            cands = [r[key] for r in results if r.get(key) is not None]
            if cands and reference.get(key) is not None:
                best, ref = min(cands), reference[key]
                row["metric"] = label
                row["verdict"] = ("improved" if best < ref * 0.99 else
                                  "worse" if best > ref * 1.01 else "no-change")
                break
    return row


def summarize(run) -> None:
    rows = [r for r in (_benchmark_row(b, run) for b in paths.benchmarks_in(run)) if r]
    if not rows:
        print(f"no results found in {run}")
        return

    tot = sum(r["samples"] for r in rows) or 1
    nb = len(rows)
    verd = {k: sum(r["verdict"] == k for r in rows)
            for k in ("improved", "no-change", "worse", "unmeasurable", "no-valid")}

    def pct(n, d):
        return f"{100 * n / d:.1f}%"

    lines = [
        f"# Benchmark run summary — {run.name}", "",
        f"Benchmarks with results: **{nb}**   |   programs evaluated: **{tot}**", "",
        "## Benchmarks",
        f"- accuracy improved over reference: **{verd['improved']}/{nb} ({pct(verd['improved'], nb)})**",
        f"- valid rewrite but no accuracy gain: {verd['no-change']}/{nb}",
        f"- best rewrite was worse than reference: {verd['worse']}/{nb}",
        f"- unmeasurable (box straddles zero / singularity): {verd['unmeasurable']}/{nb}",
        f"- no program sampled: {verd['no-valid']}/{nb}",
        "",
        "## Per-benchmark",
        "(abs) = rel error undefined over the box; compared on absolute error instead",
        "",
        "| benchmark | n | best rel (ulp) | vs reference |",
        "|---|--:|--:|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["verdict"] != "improved", r["benchmark"])):
        ulp = f"{r['best_ulp']:.1f}" if r["best_ulp"] is not None else "-"
        vs = r["verdict"] + (" (abs)" if r["metric"] == "abs" else "")
        lines.append(f"| {r['benchmark']} | {r['samples']} | {ulp} | {vs} |")

    text = "\n".join(lines) + "\n"
    out = run / "summary.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
