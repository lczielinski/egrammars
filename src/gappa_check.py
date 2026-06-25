"""Bound the rounding error of egrammar-harvested equivalent programs with Gappa.
Requires the `gappa` binary on PATH.

Usage:
    uv run src/gappa_check.py quadratic
    uv run src/gappa_check.py quadratic --run 2
    uv run src/gappa_check.py sqrtminus --subdiv 64
"""

import argparse
import json
import re
import subprocess

import paths

EPS = 2.0 ** -52  # double-precision ulp, for the ulp estimate

INTERVALS = {
    "quadratic": {"a": "[1,1.01]", "b": "[10,10.01]", "c": "[6,6.01]"},
    "sqrtminus": {"x": "[1,2]"},
    "randexpr": {"x": "[1,1.01]", "y": "[1,1.01]", "z": "[1,1.01]"},
    "subfrac": {"x": "[1,1.01]"},
    "sqrtshift": {"x": "[0.01,0.02]"},    # cancellation as x -> 0 (sqrt(x+4) -> 2)
    "sqrtquad": {"x": "[1000,1000.01]"},  # cancellation as x grows (sqrt(x*x+x) -> x)
    "recipsqrt": {"x": "[1000,1000.01]"}, # cancellation as x grows (both terms -> 1/x)
    "recipback": {"x": "[1000,1000.01]"}, # cancellation as x grows (both terms -> 1/x)
}


def tokenize(s):
    return re.findall(r"\(|\)|[^\s()]+", s)


def parse(toks):
    t = toks.pop(0)
    if t == "(":
        lst = []
        while toks[0] != ")":
            lst.append(parse(toks))
        toks.pop(0)  # ")"
        return lst
    return t


def to_gappa(n):
    """FPCore s-expression -> Gappa infix string."""
    if isinstance(n, str):
        return n  # number or variable
    head = n[0]
    if head == "FPCore":
        return to_gappa(n[2])  # (FPCore (args) <expr>)
    if head == "sqrt":
        return f"sqrt({to_gappa(n[1])})"
    if head == "-" and len(n) == 2:  # unary minus
        return f"(-({to_gappa(n[1])}))"
    return f"({to_gappa(n[1])} {head} {to_gappa(n[2])})"


def run_gappa(script: str) -> str:
    res = subprocess.run(["gappa"], input=script, capture_output=True, text=True)
    return (res.stdout + res.stderr).strip()


def enclosure_bounds(out: str):
    """Decimal bounds of the '[lo, hi]' enclosure in Gappa output (or [])."""
    m = re.search(r"in (\[[^\]]*\])", out)
    if not m:
        return []
    return [float(x) for x in re.findall(r"\{(-?[0-9.eE+-]+)", m.group(1))]


def worst_magnitude(decs):
    return max(abs(x) for x in decs) if decs else None


def fmt(v):
    return f"{v:.3e}" if v is not None else "n/a"


def analyze(expr: str, hyp: str, hint: str = "") -> dict:
    rounded = f"@rnd = float<ieee_64, ne>;\nMe rnd= {expr};\nRe = {expr};\n"
    encl_dec = enclosure_bounds(run_gappa(f"Re = {expr};\n{{ {hyp} -> Re in ? }}\n{hint}"))
    abs_dec = enclosure_bounds(run_gappa(rounded + f"{{ {hyp} -> (Me - Re) in ? }}\n{hint}"))
    rel_dec = enclosure_bounds(run_gappa(rounded + f"{{ {hyp} -> (Me - Re) / Re in ? }}\n{hint}"))
    abs_err = worst_magnitude(abs_dec)
    rel_err = worst_magnitude(rel_dec)
    return {
        "gappa_expr": expr,
        "enclosure": encl_dec,  # [lo, hi] as decimals
        "abs_err": abs_err,
        "rel_err": rel_err,
        "rel_err_ulps": (rel_err / EPS) if rel_err is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", nargs="?", default="quadratic")
    ap.add_argument("--run", type=int, default=None,
                    help="equivalents run number to analyze (default: latest)")
    ap.add_argument("--subdiv", type=int, default=0, metavar="N",
                    help="subdivide each interval variable into N pieces (Gappa "
                         "bisection hint). Needed for wide boxes where cancellation "
                         "makes a denominator/value straddle zero under naive "
                         "interval arithmetic. Cost scales as N^(#vars). Default 0.")
    args = ap.parse_args()

    if args.run is not None:
        n, src = args.run, paths.path_for(paths.EQUIVALENTS, args.benchmark, args.run)
    else:
        n, src = paths.latest(paths.EQUIVALENTS, args.benchmark)
    if src is None or not src.exists():
        ap.error(f"no equivalents file for {args.benchmark!r} in {paths.EQUIVALENTS}")
    box = INTERVALS.get(args.benchmark)
    if box is None:
        ap.error(f"no interval box configured for {args.benchmark!r}; add one to "
                 f"INTERVALS (have: {', '.join(sorted(INTERVALS))})")
    data = json.loads(src.read_text())
    programs = data["programs"]
    hyp = " /\\ ".join(f"{v} in {iv}" for v, iv in box.items())
    hint = "".join(f"$ {v} in {args.subdiv};\n" for v in box) if args.subdiv else ""

    results = []
    for i, p in enumerate(programs):
        expr = to_gappa(parse(tokenize(p)))
        r = analyze(expr, hyp, hint)
        r["program"] = p
        results.append(r)
        ulp = (f"~{r['rel_err_ulps']:.0f} ulp" if r["rel_err_ulps"] is not None
               else "gappa could not bound rel err (value not provably nonzero)")
        print(f"[{i:2d}] abs={fmt(r['abs_err'])}  rel={fmt(r['rel_err'])}  ({ulp})")

    paths.GAPPA.mkdir(parents=True, exist_ok=True)
    dst = paths.path_for(paths.GAPPA, args.benchmark, n)
    dst.write_text(json.dumps({
        "benchmark": args.benchmark,
        "reference": data.get("reference"),
        "model": data.get("model"),
        "rounding": "ieee_64, ne (round-to-nearest double)",
        "intervals": box,
        "subdivisions": args.subdiv,
        "note": "Certified worst-case bounds over this interval box; valid only "
                "within it. The accuracy ranking can reorder in other regions.",
        "results": results,
    }, indent=2))
    print(f"\nread {src}\nwrote {len(results)} results to {dst}")


if __name__ == "__main__":
    main()
