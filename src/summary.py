"""Aggregate one run's results into a console table and equivalents/<run>/summary.md."""

import json

import paths


def _metric(r):
    """Comparable error for a result: relative ulps if defined, else absolute error."""
    if r and r.get("rel_err_ulps") is not None:
        return ("rel_ulp", r["rel_err_ulps"])
    if r and r.get("abs_err") is not None:
        return ("abs", r["abs_err"])
    return (None, None)


def _benchmark_row(b, run):
    """Stats from a benchmark's equivalents (+ fptaylor) file in `run`, or None."""
    esrc = paths.equivalents_path(run, b)
    if not esrc.exists():
        return None
    attempts = json.loads(esrc.read_text()).get("attempts", [])
    v = lambda pred: sum(1 for a in attempts if pred(a))
    verd = lambda a: a.get("numeric", {}).get("verdict")
    unproven = lambda a: not a["proven_equivalent"]
    valid = v(lambda a: a["proven_equivalent"])
    row = {"benchmark": b, "candidates": len(attempts), "valid": valid,
           "missing_rule": v(lambda a: unproven(a) and verd(a) == "equal"),
           "non_equiv": v(lambda a: unproven(a) and verd(a) == "different"),
           "indeterminate": v(lambda a: unproven(a) and verd(a) == "indeterminate"),
           "best_ulp": None, "verdict": "unmeasurable" if valid else "no-valid"}
    fsrc = paths.fptaylor_path(run, b)
    if fsrc.exists():
        fd = json.loads(fsrc.read_text())
        results = fd.get("results", [])
        row["best_ulp"] = next((r["rel_err_ulps"] for r in results
                                if r.get("rel_err_ulps") is not None), None)
        if row["valid"]:
            best_k, best_m = None, None
            for r in results:
                k, m = _metric(r)
                if m is not None and (best_m is None or m < best_m):
                    best_k, best_m = k, m
            rk, rm = _metric(fd.get("reference_result"))
            if rm is None or best_m is None or rk != best_k:
                row["verdict"] = "unmeasurable"
            elif best_m < rm * 0.99:
                row["verdict"] = "improved"
            elif best_m > rm * 1.01:
                row["verdict"] = "worse"
            else:
                row["verdict"] = "no-change"
    return row


def summarize(run) -> None:
    """`run` is a results/<name> directory."""
    rows = [r for r in (_benchmark_row(b, run) for b in paths.benchmarks_in(run)) if r]
    if not rows:
        print(f"no results found in {run}")
        return

    tot = sum(r["candidates"] for r in rows) or 1
    prog = {k: sum(r[k] for r in rows)
            for k in ("valid", "missing_rule", "non_equiv", "indeterminate")}
    nb = len(rows)
    with_valid = sum(r["valid"] > 0 for r in rows)
    with_missing = sum(r["missing_rule"] > 0 for r in rows)
    verd = {k: sum(r["verdict"] == k for r in rows)
            for k in ("improved", "no-change", "worse", "unmeasurable", "no-valid")}

    def pct(n, d):
        return f"{100 * n / d:.1f}%"

    lines = [
        f"# Benchmark run summary — {run.name}", "",
        f"Benchmarks with results: **{nb}**   |   candidates evaluated: **{tot}**", "",
        "## Programs (all candidates the model produced)",
        f"- valid (proven equivalent): **{prog['valid']} ({pct(prog['valid'], tot)})**",
        f"- invalid — missing e-graph rule (numerically equal, unproven): {prog['missing_rule']} ({pct(prog['missing_rule'], tot)})",
        f"- invalid — non-equivalent (model error): {prog['non_equiv']} ({pct(prog['non_equiv'], tot)})",
        f"- indeterminate (no finite sample point): {prog['indeterminate']} ({pct(prog['indeterminate'], tot)})",
        "",
        "## Benchmarks",
        f"- produced >=1 valid rewrite: **{with_valid}/{nb} ({pct(with_valid, nb)})**",
        f"- accuracy improved over reference: **{verd['improved']}/{nb} ({pct(verd['improved'], nb)})**",
        f"- valid rewrite but no accuracy gain: {verd['no-change']}/{nb}",
        f"- best valid rewrite was worse than reference: {verd['worse']}/{nb}",
        f"- unmeasurable (box straddles zero / singularity): {verd['unmeasurable']}/{nb}",
        f"- no valid rewrite found: {verd['no-valid']}/{nb}",
        f"- had a missing-rule candidate: {with_missing}/{nb} ({pct(with_missing, nb)})",
        "",
        "## Per-benchmark",
        "| benchmark | candidates | valid | best rel (ulp) | vs reference |",
        "|---|--:|--:|--:|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["verdict"] != "improved", r["benchmark"])):
        ulp = f"{r['best_ulp']:.1f}" if r["best_ulp"] is not None else "-"
        lines.append(f"| {r['benchmark']} | {r['candidates']} | {r['valid']} | {ulp} | {r['verdict']} |")

    text = "\n".join(lines) + "\n"
    out = run / "summary.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
