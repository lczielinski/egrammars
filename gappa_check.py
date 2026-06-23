"""Run Gappa on egrammar-harvested equivalent programs.

Reads out/<benchmark>.equivalents.json, converts each FPCore s-expression to
Gappa's infix syntax, and over a fixed interval box computes, per program:

  - enclosure:  the exact real-valued range of the expression
  - abs_err:    certified worst-case |rounded - exact| in IEEE-754 double (ne)
  - rel_err:    certified worst-case |(rounded - exact) / exact|

Results are written to out/<benchmark>.gappa.json.

The default interval box is deliberately NARROW: Gappa's interval arithmetic
cannot prove b - sqrt(b*b - 4ac) > 0 for a wide b (it loses the correlation and
sees the denominator straddle zero), which makes several programs fail with an
undischargeable division. Narrow ranges keep every denominator provably bounded
away from zero. The accuracy ranking is only valid within this box (b > 0, no
genuine cancellation); other regions can reorder the programs.

Usage:
    python3 gappa_check.py quadratic
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPS = 2.0 ** -52  # double-precision ulp, for the ulp estimate

# easy interval box; see module docstring for why it is narrow.
INTERVALS = {"a": "[1,1.01]", "b": "[10,10.01]", "c": "[6,6.01]"}


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


def analyze(expr: str, hyp: str) -> dict:
    rounded = f"@rnd = float<ieee_64, ne>;\nMe rnd= {expr};\nRe = {expr};\n"
    encl_raw, encl_dec = enclosure(run_gappa(f"Re = {expr};\n{{ {hyp} -> Re in ? }}\n"))
    _, abs_dec = enclosure(run_gappa(rounded + f"{{ {hyp} -> (Me - Re) in ? }}\n"))
    _, rel_dec = enclosure(run_gappa(rounded + f"{{ {hyp} -> (Me - Re) / Re in ? }}\n"))
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
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args()

    src = args.out / f"{args.benchmark}.equivalents.json"
    data = json.loads(src.read_text())
    programs = data["programs"]
    hyp = " /\\ ".join(f"{v} in {iv}" for v, iv in INTERVALS.items())

    results = []
    for i, p in enumerate(programs):
        expr = to_gappa(parse(tokenize(p)))
        r = analyze(expr, hyp)
        r["program"] = p
        results.append(r)
        print(f"[{i:2d}] abs={r['abs_err']:.3e}  rel={r['rel_err']:.3e}  "
              f"(~{r['rel_err_ulps']:.0f} ulp)")

    dst = args.out / f"{args.benchmark}.gappa.json"
    dst.write_text(json.dumps({
        "benchmark": args.benchmark,
        "reference": data.get("reference"),
        "model": data.get("model"),
        "rounding": "ieee_64, ne (round-to-nearest double)",
        "intervals": INTERVALS,
        "note": "Certified worst-case bounds over the interval box; valid only "
                "within it (b > 0, no genuine cancellation).",
        "results": results,
    }, indent=2))
    print(f"\nwrote {len(results)} results to {dst}")


if __name__ == "__main__":
    main()
