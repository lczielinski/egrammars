"""Bound the rounding error of egrammar-harvested equivalent programs with FPTaylor.
Requires the `fptaylor` binary on PATH.

Inputs are treated as exact doubles in the given interval box, every operation
is rounded to IEEE-754 double (round-to-nearest, via rnd64=), and we report the
worst-case error of the rounded program against the ideal real-valued one over
the box.

Usage:
    uv run src/fptaylor_check.py quadratic
    uv run src/fptaylor_check.py quadratic --run 2
"""

import argparse
import json
import os
import re
import subprocess
import tempfile

import paths

EPS = 2.0 ** -52  # double-precision ulp
CONFIG = "abs-error = true\nrel-error = true\n"

# Per-benchmark input interval boxes. Boxes may be wide, but every sqrt argument
# must stay >= 0 and every denominator clear of 0 across the box, else the bound
# is +inf. Domain constraint per reference expression:
#
#   quadratic   (-b + sqrt(b*b - 4ac)) / (2a)   need b*b > 4ac, a != 0
#   sqrtminus   sqrt(x*x + 1) - x               defined for all x
#   randexpr    ... sqrt(x*z), z/sqrt(z) ...     need x, y, z > 0
#   subfrac     1/(x+1) - 1/x                    need x != 0, -1
#   sqrtshift   sqrt(x + 4) - 2                  need x > -4; keep x > 0 (rel err)
#   sqrtquad    sqrt(x*x + x) - x                need x >= 0
#   recipsqrt   1/(x + sqrt(x)) - 1/x            need x > 0
#   recipback   1/(x - 1) - 1/x                  need x != 0, 1
INTERVALS = {
    "quadratic": {"a": "[1,2]", "b": "[10,12]", "c": "[1,3]"},
    "sqrtminus": {"x": "[1,100]"},
    "randexpr": {"x": "[1,4]", "y": "[1,4]", "z": "[1,4]"},
    "subfrac": {"x": "[1,100]"},
    "sqrtshift": {"x": "[0.01,0.5]"},     # cancellation as x -> 0 (sqrt(x+4) -> 2)
    "sqrtquad": {"x": "[100,2000]"},      # cancellation as x grows (sqrt(x*x+x) -> x)
    "recipsqrt": {"x": "[1,2000]"},       # cancellation as x grows (both terms -> 1/x)
    "recipback": {"x": "[100,2000]"},     # cancellation as x grows (both terms -> 1/x)
}


def tokenize(s):
    return re.findall(r"\(|\)|[^\s()]+", s)


def parse(toks):
    t = toks.pop(0)
    if t == "(":
        lst = []
        while toks[0] != ")":
            lst.append(parse(toks))
        toks.pop(0)
        return lst
    return t


def fmt(v):
    return f"{v:.3e}" if v is not None else "n/a"


def to_fptaylor(n):
    """FPCore s-expression -> FPTaylor infix string."""
    if isinstance(n, str):
        return n
    head = n[0]
    if head == "FPCore":
        return to_fptaylor(n[2])  # (FPCore (args) <expr>)
    if head == "sqrt":
        return f"sqrt({to_fptaylor(n[1])})"
    if head == "-" and len(n) == 2:  # unary minus
        return f"(-({to_fptaylor(n[1])}))"
    return f"({to_fptaylor(n[1])} {head} {to_fptaylor(n[2])})"


def build_input(expr: str, box: dict) -> str:
    """An FPTaylor input file: exact float64 inputs, every op rounded (rnd64=)."""
    decls = "\n".join(
        f"  float64 {v} in [{iv.strip('[]')}];" for v, iv in box.items()
    )
    return f"Variables\n{decls}\n\nExpressions\n  result rnd64= {expr};\n"


def run_fptaylor(input_text: str, cfg_path: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(input_text)
        in_path = f.name
    try:
        res = subprocess.run(
            ["fptaylor", in_path, "-c", cfg_path], capture_output=True, text=True
        )
        return res.stdout + res.stderr
    finally:
        os.unlink(in_path)


def grab(label: str, out: str):
    """The numeric value FPTaylor printed for `label`, preferring 'exact' over
    'approximate', or None if neither is present."""
    for kind in ("exact", "approximate"):
        m = re.search(rf"{label} \({kind}\):\s*([0-9.eE+-]+)", out)
        if m:
            return float(m.group(1))
    return None


def bounds(out: str):
    """The real-valued '[lo, hi]' range FPTaylor reports (or [])."""
    m = re.search(r"Bounds \(without rounding\):\s*\[([^,]+),\s*([^\]]+)\]", out)
    return [float(m.group(1)), float(m.group(2))] if m else []


def analyze(expr: str, box: dict, cfg_path: str) -> dict:
    out = run_fptaylor(build_input(expr, box), cfg_path)
    abs_err = grab("Absolute error", out)
    rel_err = grab("Relative error", out)
    return {
        "fptaylor_expr": expr,
        "enclosure": bounds(out),
        "abs_err": abs_err,
        "rel_err": rel_err,
        "rel_err_ulps": (rel_err / EPS) if rel_err is not None else None,
    }


def check(benchmark: str, run: int = None):
    """Bound the rounding error of one equivalents run and write the results.

    Returns the output path. Raises FileNotFoundError if the equivalents file is
    missing and KeyError if no interval box is configured for the benchmark.
    """
    if run is not None:
        n, src = run, paths.path_for(paths.EQUIVALENTS, benchmark, run)
    else:
        n, src = paths.latest(paths.EQUIVALENTS, benchmark)
    if src is None or not src.exists():
        raise FileNotFoundError(
            f"no equivalents file for {benchmark!r} in {paths.EQUIVALENTS}")
    box = INTERVALS.get(benchmark)
    if box is None:
        raise KeyError(f"no interval box configured for {benchmark!r}; add one to "
                       f"INTERVALS (have: {', '.join(sorted(INTERVALS))})")

    data = json.loads(src.read_text())
    programs = data["programs"]

    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
        f.write(CONFIG)
        cfg_path = f.name
    try:
        results = []
        for i, p in enumerate(programs):
            expr = to_fptaylor(parse(tokenize(p)))
            r = analyze(expr, box, cfg_path)
            r["program"] = p
            results.append(r)
            ulp = (f"~{r['rel_err_ulps']:.0f} ulp" if r["rel_err_ulps"] is not None
                   else "fptaylor could not bound rel err (range may straddle zero)")
            print(f"[{i:2d}] abs={fmt(r['abs_err'])}  rel={fmt(r['rel_err'])}  ({ulp})")
    finally:
        os.unlink(cfg_path)

    paths.FPTAYLOR.mkdir(parents=True, exist_ok=True)
    dst = paths.path_for(paths.FPTAYLOR, benchmark, n)
    dst.write_text(json.dumps({
        "benchmark": benchmark,
        "reference": data.get("reference"),
        "model": data.get("model"),
        "rounding": "float64, round-to-nearest (rnd64=)",
        "intervals": box,
        "note": "Certified worst-case bounds over this interval box; valid only "
                "within it. The accuracy ranking can reorder in other regions.",
        "results": results,
    }, indent=2))
    print(f"\nread {src}\nwrote {len(results)} results to {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", nargs="?", default="quadratic")
    ap.add_argument("--run", type=int, default=None,
                    help="equivalents run number to analyze (default: latest)")
    args = ap.parse_args()
    try:
        check(args.benchmark, args.run)
    except (FileNotFoundError, KeyError) as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
