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
TIMEOUT = 120

# Per-benchmark input interval boxes. Branching lets one program stay accurate
# across a whole regime, so these span from the ill-conditioned region into the
# well-conditioned one (each branch is checked only over its own sub-interval); the
# reference must stay real and finite across the box (no sqrt of a negative, no
# division by zero).
#   quadratic   (-b + sqrt(b*b - 4ac)) / (2a)    need b*b > 4ac, a != 0
#   sqrtminus   sqrt(x*x + 1) - x                defined for all x
#   randexpr    ... sqrt(x*z), z/sqrt(z) ...     need x, y, z > 0
#   subfrac     1/(x+1) - 1/x                    need x != 0, -1
#   sqrtshift   sqrt(x + 4) - 2                  need x > -4
#   sqrtquad    sqrt(x*x + x) - x                need x >= 0
#   recipsqrt   1/(x + sqrt(x)) - 1/x            need x > 0
#   recipback   1/(x - 1) - 1/x                  need x != 0, 1
INTERVALS = {
    "quadratic": {"a": "[1,2]", "b": "[20,100]", "c": "[1,10]"},  # -b+sqrt cancels, b>0
    "sqrtminus": {"x": "[1,1000]"},                # cancellation grows with x
    "randexpr": {"x": "[1,100]", "y": "[1,100]", "z": "[1,100]"},
    "subfrac": {"x": "[1,1000]"},                  # cancellation grows with x
    "sqrtshift": {"x": "[0.01,100]"},              # cancels as x -> 0, fine for large x
    "sqrtquad": {"x": "[1,100000]"},               # cancels as x grows
    "recipsqrt": {"x": "[1,100000]"},              # both terms -> 1/x as x grows
    "recipback": {"x": "[2,100000]"},              # both terms -> 1/x; stay clear of x=1
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


def rank_key(r: dict):
    rel = r.get("rel_err")
    return (rel is None, rel if rel is not None else 0.0)


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


def run_fptaylor(input_text: str, cfg_path: str):
    """FPTaylor's combined output, or None if it exceeds TIMEOUT."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(input_text)
        in_path = f.name
    try:
        res = subprocess.run(
            ["fptaylor", in_path, "-c", cfg_path],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        return res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return None
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
    if out is None:
        return {"fptaylor_expr": expr, "enclosure": [], "abs_err": None,
                "rel_err": None, "rel_err_ulps": None, "timeout": True}
    abs_err = grab("Absolute error", out)
    rel_err = grab("Relative error", out)
    encl = bounds(out)
    # FPTaylor often omits relative error through divisions even when the value
    # is far from zero. When it does, fall back to abs_err / min|value| from the
    # value range -- a sound (looser) bound, valid only when the range avoids 0.
    derived = False
    if rel_err is None and abs_err is not None and encl and (encl[0] > 0 or encl[1] < 0):
        rel_err = abs_err / min(abs(encl[0]), abs(encl[1]))
        derived = True
    return {
        "fptaylor_expr": expr,
        "enclosure": encl,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "rel_err_derived": derived,
        "rel_err_ulps": (rel_err / EPS) if rel_err is not None else None,
    }


NEGATE = {"<": ">=", ">": "<=", "<=": ">", ">=": "<"}


def split_branches(node):
    """Yield (conditions, expr) for each leaf of an if-tree, where conditions is
    the list of (op, lhs, rhs) comparisons that hold along the path to that leaf.
    A leaf is an if-free expression."""
    if isinstance(node, list) and node and node[0] == "if":
        _, (op, lhs, rhs), then, els = node
        for conds, expr in split_branches(then):
            yield [(op, lhs, rhs), *conds], expr
        for conds, expr in split_branches(els):
            yield [(NEGATE[op], lhs, rhs), *conds], expr
    else:
        yield [], node


def _as_float(tok: str):
    try:
        return float(tok)
    except ValueError:
        return None


def narrow_box(box: dict, conds) -> dict | None:
    """The interval box restricted to where all `conds` hold, or None if that region
    is empty. A comparison between two variables can't be drawn as a box, so it is
    ignored (the leaf is then analyzed over a sound superset of its region)."""
    iv = {v: list(map(float, s.strip("[]").split(","))) for v, s in box.items()}
    for op, lhs, rhs in conds:
        lo_bound = op in (">", ">=")  # tightens a lower bound on the variable side
        if lhs in iv and (n := _as_float(rhs)) is not None:
            iv[lhs][0 if lo_bound else 1] = (max if lo_bound else min)(
                iv[lhs][0 if lo_bound else 1], n)
        elif rhs in iv and (n := _as_float(lhs)) is not None:
            # `n op var` flips: n < var means var > n, tightening var's lower bound.
            iv[rhs][1 if lo_bound else 0] = (min if lo_bound else max)(
                iv[rhs][1 if lo_bound else 0], n)
    if any(lo > hi for lo, hi in iv.values()):
        return None
    return {v: f"[{lo},{hi}]" for v, (lo, hi) in iv.items()}


def _combine(branches: list) -> dict:
    """Roll per-leaf results into one program result: worst error across leaves."""
    def worst(key):
        vals = [b[key] for b in branches if b.get(key) is not None]
        return max(vals) if vals else None

    abs_err, rel_err = worst("abs_err"), worst("rel_err")
    # A program-wide relative bound holds only if every (non-timeout) leaf bounded it.
    if any(b.get("rel_err") is None and not b.get("timeout") for b in branches):
        rel_err = None
    return {
        "fptaylor_expr": None,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "rel_err_derived": any(b.get("rel_err_derived") for b in branches),
        "rel_err_ulps": (rel_err / EPS) if rel_err is not None else None,
        "timeout": any(b.get("timeout") for b in branches),
        "branches": branches,
    }


def analyze_program(ast, box: dict, cfg_path: str) -> dict:
    """Analyze one FPCore program, splitting on `if` and analyzing each branch over
    the sub-interval where it applies."""
    body = ast[2] if isinstance(ast, list) and ast and ast[0] == "FPCore" else ast
    leaves = list(split_branches(body))
    if len(leaves) == 1:  # no `if`: a single expression over the whole box
        return analyze(to_fptaylor(body), box, cfg_path)
    branches = []
    for conds, expr in leaves:
        region = narrow_box(box, conds)
        if region is None:  # this branch is unreachable within the interval box
            continue
        b = analyze(to_fptaylor(expr), region, cfg_path)
        b["condition"] = " and ".join(f"{l} {op} {r}" for op, l, r in conds)
        b["region"] = region
        branches.append(b)
    return _combine(branches)


def check(benchmark: str, run: int | None = None):
    """Bound the rounding error of one equivalents run and write the results."""
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
            r = analyze_program(parse(tokenize(p)), box, cfg_path)
            r["program"] = p
            results.append(r)
            if r.get("timeout"):
                status = f"timed out (> {TIMEOUT}s)"
            elif r["rel_err_ulps"] is not None:
                tag = " (from abs/range)" if r.get("rel_err_derived") else ""
                status = f"~{r['rel_err_ulps']:.0f} ulp{tag}"
            else:
                status = "no rel err: value range straddles zero"
            print(f"[{i:2d}] abs={fmt(r['abs_err'])}  rel={fmt(r['rel_err'])}  ({status})")
            for b in r.get("branches", ()):
                print(f"       where {b['condition'] or 'all inputs'}: "
                      f"abs={fmt(b['abs_err'])}  rel={fmt(b['rel_err'])}")
    finally:
        os.unlink(cfg_path)

    ranked = sorted(results, key=rank_key)
    print(f"\n{'=' * 70}\nranked by relative error, best first "
          f"(* = derived from value range)\n{'=' * 70}")
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
        "benchmark": benchmark,
        "reference": data.get("reference"),
        "model": data.get("model"),
        "rounding": "float64, round-to-nearest (rnd64=)",
        "intervals": box,
        "note": "Certified worst-case bounds over this interval box; valid only "
                "within it. The accuracy ranking can reorder in other regions. "
                "Results are ordered best-first by relative error. A branching "
                "program's error is the worst over its branches, each bounded over "
                "the sub-interval where that branch applies (see `branches`).",
        "results": ranked,
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
