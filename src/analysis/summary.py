"""Aggregate one run's results into a console table and results/<run>/summary.md."""

import json

from base import paths


def _compare(best: float, other: float) -> str:
    return ("improved" if best < other * 0.99 else
            "worse" if best > other * 1.01 else "no-change")


def _benchmark_row(b, run):
    """Stats from a benchmark's equivalents (+ fptaylor) file in `run`, or None."""
    esrc = paths.equivalents_path(run, b)
    if not esrc.exists():
        return None
    data = json.loads(esrc.read_text())
    programs = data.get("programs", [])
    row = {"benchmark": b,
           "samples": len(data.get("attempts") or programs),  # attempts: legacy runs
           "best_ulp": None, "best_cost": None, "metric": None,
           "extraction_ulp": None, "extraction_cost": None, "vs_extraction": None,
           "verdict": "unmeasurable" if programs else "no-valid"}
    fsrc = paths.fptaylor_path(run, b)
    if fsrc.exists() and programs:
        fd = json.loads(fsrc.read_text())
        results = fd.get("results", [])
        reference = fd.get("reference_result") or {}
        extraction = fd.get("extraction_result") or {}
        row["best_ulp"] = next((r["rel_err_ulps"] for r in results
                                if r.get("rel_err_ulps") is not None), None)
        row["extraction_ulp"] = extraction.get("rel_err_ulps")
        row["extraction_cost"] = extraction.get("cost")
        # compare on relative error; fall back to absolute where rel is undefined
        for key, label in (("rel_err_ulps", "rel"), ("abs_err", "abs")):
            cands = [r for r in results if r.get(key) is not None]
            if cands and reference.get(key) is not None:
                best = min(cands, key=lambda r: r[key])
                row["metric"] = label
                row["best_cost"] = best.get("cost")
                row["verdict"] = _compare(best[key], reference[key])
                if extraction.get(key) is not None:
                    row["vs_extraction"] = _compare(best[key], extraction[key])
                break
    return row


def summarize(run) -> None:
    """`run` is a results/<name> directory."""
    rows = [r for r in (_benchmark_row(b, run) for b in paths.benchmarks_in(run)) if r]
    if not rows:
        print(f"no results found in {run}")
        return

    tot = sum(r["samples"] for r in rows) or 1
    nb = len(rows)
    verd = {k: sum(r["verdict"] == k for r in rows)
            for k in ("improved", "no-change", "worse", "unmeasurable", "no-valid")}
    vsx = {k: sum(r["vs_extraction"] == k for r in rows)
           for k in ("improved", "no-change", "worse")}
    with_x = sum(v for v in vsx.values())

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
    ] + ([
        "",
        "## LLM search vs min-cost e-graph extraction "
        f"({with_x} benchmarks with both measurable)",
        f"- LLM best more accurate than extraction: **{vsx['improved']}/{with_x}**",
        f"- tie (within 1%): {vsx['no-change']}/{with_x}",
        f"- extraction more accurate: {vsx['worse']}/{with_x}",
    ] if with_x else []) + [
        "",
        "## Per-benchmark",
        "(abs) = rel error undefined over the box; compared on absolute error instead",
        "",
        "| benchmark | n | best rel (ulp) | cost | extr rel (ulp) | extr cost | vs reference | vs extraction |",
        "|---|--:|--:|--:|--:|--:|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["verdict"] != "improved", r["benchmark"])):
        ulp = lambda v: f"{v:.1f}" if v is not None else "-"
        num = lambda v: str(v) if v is not None else "-"
        vs = r["verdict"] + (" (abs)" if r["metric"] == "abs" else "")
        lines.append(
            f"| {r['benchmark']} | {r['samples']} | {ulp(r['best_ulp'])} "
            f"| {num(r['best_cost'])} | {ulp(r['extraction_ulp'])} "
            f"| {num(r['extraction_cost'])} | {vs} | {num(r['vs_extraction'])} |")

    text = "\n".join(lines) + "\n"
    out = run / "summary.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
