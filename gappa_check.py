"""Run Gappa on egrammar-harvested equivalent programs.

Reads out/<benchmark>.equivalents.json, converts each FPCore s-expression to
Gappa's infix syntax, and over a fixed interval box computes, per program:

  - enclosure:  the exact real-valued range of the expression
  - abs_err:    certified worst-case |rounded - exact| in IEEE-754 double (ne)
  - rel_err:    certified worst-case |(rounded - exact) / exact|

Reads out/equivalents/<benchmark>-NNN.json (latest run by default, or --run N)
and writes out/gappa/<benchmark>-NNN.json, reusing the same run number.

Each benchmark has its own interval box (INTERVALS below). The boxes are
deliberately NARROW: Gappa's interval arithmetic loses variable correlations (it
cannot, e.g., prove b - sqrt(b*b - 4ac) > 0 for a wide b, and sees the denominator
straddle zero), so wide ranges make programs fail with an undischargeable division.
Narrow ranges keep every denominator provably bounded away from zero. Bounds are
certified only within the chosen box, and the accuracy ranking can reorder in other
regions (e.g. the sign of b for quadratic, or large x for sqrtminus), so pick a
representative regime per benchmark. For a wider box, pass --subdiv N: Gappa then
bisects each variable into N pieces, recovering the lost correlations (at N^(#vars)
cost). Bounds it cannot prove are reported as "n/a" rather than crashing.

Requires the `gappa` binary on PATH. No Python deps beyond the stdlib, so it runs
in the base (egglog-only) environment — no `--extra cars` needed.

Usage:
    uv run gappa_check.py quadratic
    uv run gappa_check.py quadratic --run 2
    uv run gappa_check.py sqrtminus --subdiv 64   # wide box; subdivide to bound it
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import runpaths

HERE = Path(__file__).resolve().parent
EPS = 2.0 ** -52  # double-precision ulp, for the ulp estimate

# Per-benchmark interval boxes; see module docstring for why they are narrow and
# regime-specific. Add an entry (variable -> Gappa interval) for each new benchmark.
INTERVALS = {
    "quadratic": {"a": "[1,1.01]", "b": "[10,10.01]", "c": "[6,6.01]"},
    "sqrtminus": {"x": "[1,2]"},
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


def enclosure(out: str):
    """Pull the '[lo, hi]' enclosure and its decimal bounds out of Gappa output."""
    m = re.search(r"in (\[[^\]]*\])", out)
    if not m:
        return None, None
    raw = m.group(1)
    decs = [float(x) for x in re.findall(r"\{(-?[0-9.eE+-]+)", raw)]
    return raw, decs


def worst_magnitude(decs):
    return max(abs(x) for x in decs) if decs else None


def analyze(expr: str, hyp: str, hint: str = "") -> dict:
    rounded = f"@rnd = float<ieee_64, ne>;\nMe rnd= {expr};\nRe = {expr};\n"
    _, encl_dec = enclosure(run_gappa(f"Re = {expr};\n{{ {hyp} -> Re in ? }}\n{hint}"))
    _, abs_dec = enclosure(run_gappa(rounded + f"{{ {hyp} -> (Me - Re) in ? }}\n{hint}"))
    _, rel_dec = enclosure(run_gappa(rounded + f"{{ {hyp} -> (Me - Re) / Re in ? }}\n{hint}"))
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
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()

    equiv_dir, gappa_dir = args.out / "equivalents", args.out / "gappa"
    if args.run is not None:
        n, src = args.run, runpaths.path_for(equiv_dir, args.benchmark, args.run)
    else:
        n, src = runpaths.latest(equiv_dir, args.benchmark)
    if src is None or not src.exists():
        ap.error(f"no equivalents file for {args.benchmark!r} in {equiv_dir}")
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
        fmt = lambda v, u="": f"{v:.3e}{u}" if v is not None else "n/a"
        ulp = (f"~{r['rel_err_ulps']:.0f} ulp" if r["rel_err_ulps"] is not None
               else "gappa could not bound rel err (value not provably nonzero)")
        print(f"[{i:2d}] abs={fmt(r['abs_err'])}  rel={fmt(r['rel_err'])}  ({ulp})")

    gappa_dir.mkdir(parents=True, exist_ok=True)
    dst = runpaths.path_for(gappa_dir, args.benchmark, n)
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
