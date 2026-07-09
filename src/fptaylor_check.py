"""Bound each harvested program's IEEE-754 double rounding error with FPTaylor
(needs the `fptaylor` binary on PATH). A branching program is split on `if` and each
branch bounded over the sub-interval where it applies.

    uv run src/fptaylor_check.py x_by_xy [--run N]
"""

import argparse
import json
import os
import re
import subprocess
import tempfile

import paths
import regions

EPS = 2.0 ** -52  # double-precision ulp
CONFIG = "abs-error = true\nrel-error = true\n"
TIMEOUT = 120

def _imported_intervals() -> dict:
    f = paths.BENCHMARKS / "fpbench_intervals.json"
    return json.loads(f.read_text()) if f.exists() else {}


INTERVALS = _imported_intervals()


def fmt(v):
    return f"{v:.3e}" if v is not None else "n/a"


def rank_key(r: dict):
    rel = r.get("rel_err")
    return (rel is None, rel if rel is not None else 0.0)


def to_fptaylor(n):
    if isinstance(n, str):
        return n
    head = n[0]
    if head == "FPCore":
        return to_fptaylor(n[2])
    if head == "sqrt":
        return f"sqrt({to_fptaylor(n[1])})"
    if head == "-" and len(n) == 2:  # unary minus
        return f"(-({to_fptaylor(n[1])}))"
    return f"({to_fptaylor(n[1])} {head} {to_fptaylor(n[2])})"


def build_input(expr: str, box: dict) -> str:
    decls = "\n".join(f"  float64 {v} in [{iv.strip('[]')}];" for v, iv in box.items())
    return f"Variables\n{decls}\n\nExpressions\n  result rnd64= {expr};\n"


def run_fptaylor(input_text: str, cfg_path: str):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(input_text)
        in_path = f.name
    try:
        res = subprocess.run(["fptaylor", in_path, "-c", cfg_path],
                             capture_output=True, text=True, timeout=TIMEOUT)
        return res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(in_path)


def grab(label: str, out: str):
    for kind in ("exact", "approximate"):
        m = re.search(rf"{label} \({kind}\):\s*([0-9.eE+-]+)", out)
        if m:
            return float(m.group(1))
    return None


def bounds(out: str):
    m = re.search(r"Bounds \(without rounding\):\s*\[([^,]+),\s*([^\]]+)\]", out)
    return [float(m.group(1)), float(m.group(2))] if m else []


def analyze(expr: str, box: dict, cfg_path: str) -> dict:
    out = run_fptaylor(build_input(expr, box), cfg_path)
    if out is None:
        return {"fptaylor_expr": expr, "enclosure": [], "abs_err": None,
                "rel_err": None, "rel_err_ulps": None, "timeout": True}
    abs_err, rel_err, encl = grab("Absolute error", out), grab("Relative error", out), bounds(out)
    # FPTaylor often omits rel error through divisions; derive abs_err / min|value|.
    derived = False
    if rel_err is None and abs_err is not None and encl and (encl[0] > 0 or encl[1] < 0):
        rel_err = abs_err / min(abs(encl[0]), abs(encl[1]))
        derived = True
    return {"fptaylor_expr": expr, "enclosure": encl, "abs_err": abs_err,
            "rel_err": rel_err, "rel_err_derived": derived,
            "rel_err_ulps": (rel_err / EPS) if rel_err is not None else None}


def _combine(branches: list) -> dict:
    """One program result from its branches: worst error across them."""
    def worst(key):
        vals = [b[key] for b in branches if b.get(key) is not None]
        return max(vals) if vals else None

    abs_err, rel_err = worst("abs_err"), worst("rel_err")
    if any(b.get("rel_err") is None and not b.get("timeout") for b in branches):
        rel_err = None  # a rel bound holds only if every non-timeout branch had one
    return {"fptaylor_expr": None, "abs_err": abs_err, "rel_err": rel_err,
            "rel_err_derived": any(b.get("rel_err_derived") for b in branches),
            "rel_err_ulps": (rel_err / EPS) if rel_err is not None else None,
            "timeout": any(b.get("timeout") for b in branches), "branches": branches}


def analyze_program(ast, box: dict, cfg_path: str) -> dict:
    """Analyze one program, splitting on `if` and bounding each branch over its box."""
    leaves = list(regions.split_branches(regions.body_of(ast)))
    if len(leaves) == 1:
        return analyze(to_fptaylor(body), box, cfg_path)
    branches = []
    for conds, expr in leaves:
        region = regions.narrow_box(box, conds)
        if region is None:  # branch unreachable in the box
            continue
        b = analyze(to_fptaylor(expr), region, cfg_path)
        b["condition"] = " and ".join(f"{l} {op} {r}" for op, l, r in conds)
        b["region"] = region
        branches.append(b)
    return _combine(branches)


def check(benchmark: str, run: int | None = None):
    """Bound one equivalents run and write fptaylor/<benchmark>-NNN.json."""
    if run is not None:
        n, src = run, paths.path_for(paths.EQUIVALENTS, benchmark, run)
    else:
        n, src = paths.latest(paths.EQUIVALENTS, benchmark)
    if src is None or not src.exists():
        raise FileNotFoundError(f"no equivalents file for {benchmark!r} in {paths.EQUIVALENTS}")
    box = INTERVALS.get(benchmark)
    if box is None:
        raise KeyError(f"no interval box for {benchmark!r} (have: {', '.join(sorted(INTERVALS))})")

    data = json.loads(src.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
        f.write(CONFIG)
        cfg_path = f.name
    try:
        # bound the reference too, so "did we improve on it?" is answerable offline
        reference_result = None
        if data.get("reference"):
            reference_result = analyze_program(
                regions.parse(regions.tokenize(data["reference"])), box, cfg_path)
            reference_result["program"] = data["reference"]
        results = []
        for i, p in enumerate(data["programs"]):
            r = analyze_program(regions.parse(regions.tokenize(p)), box, cfg_path)
            r["program"] = p
            results.append(r)
            if r.get("timeout"):
                status = f"timed out (> {TIMEOUT}s)"
            elif r["rel_err_ulps"] is not None:
                status = f"~{r['rel_err_ulps']:.0f} ulp{' (derived)' if r.get('rel_err_derived') else ''}"
            else:
                status = "no rel err: range straddles zero"
            print(f"[{i:2d}] abs={fmt(r['abs_err'])}  rel={fmt(r['rel_err'])}  ({status})")
            for b in r.get("branches", ()):
                print(f"       {b['condition'] or 'all inputs'}: "
                      f"abs={fmt(b['abs_err'])}  rel={fmt(b['rel_err'])}")
    finally:
        os.unlink(cfg_path)

    ranked = sorted(results, key=rank_key)
    print(f"\n{'=' * 70}\nranked by relative error, best first\n{'=' * 70}")
    for rank, r in enumerate(ranked):
        if r.get("timeout"):
            label = "timeout"
        elif r["rel_err_ulps"] is not None:
            label = f"{r['rel_err_ulps']:.1f} ulp{'*' if r.get('rel_err_derived') else ''}"
        else:
            label = "unbounded"
        print(f"{rank:2d}. {label:>13}  abs={fmt(r['abs_err'])}  {r['program']}")

    paths.FPTAYLOR.mkdir(parents=True, exist_ok=True)
    dst = paths.path_for(paths.FPTAYLOR, benchmark, n)
    dst.write_text(json.dumps({
        "benchmark": benchmark, "reference": data.get("reference"),
        "config": data.get("config"), "intervals": box,
        "note": "worst-case double-rounding bounds over this box; a branching "
                "program's error is the worst over its branches (see `branches`).",
        "reference_result": reference_result, "results": ranked,
    }, indent=2))
    print(f"\nread {src}\nwrote {len(results)} results to {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark")
    ap.add_argument("--run", type=int, default=None)
    args = ap.parse_args()
    try:
        check(args.benchmark, args.run)
    except (FileNotFoundError, KeyError) as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
