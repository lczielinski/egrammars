"""Import FPCore benchmarks (from the vendored Herbie/FPBench sources) into the tool's
egglog form.

Parses an `.fpcore`, inlines `let`/`let*`, keeps only the tool's subset (`+ - * / sqrt`,
unary minus, integer-valued literals, a single branch-free body), folds variadic
operators left-associatively, and emits `benchmarks/egglog/<slug>.egglog` plus a box in
`benchmarks/intervals.json`.

The curated set of Herbie imports lives in `SPEC` below (each entry also records its
source core, per-variable box provenance, and a note on why it is improvable). Re-running
is idempotent: it regenerates each listed benchmark and merges its box into
`intervals.json` (existing entries for other benchmarks are preserved).

    uv run scripts/import_fpcore.py            # (re)generate every benchmark in SPEC
    uv run scripts/import_fpcore.py --list     # list source cores matching the subset
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
HERBIE = BENCH / "sources" / "herbie"
EGGLOG = BENCH / "egglog"

# --------------------------------------------------------------------------- parser
def tokenize(s: str):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == ";":
            while i < n and s[i] != "\n":
                i += 1
        elif c in "()[]":
            out.append("(" if c in "([" else ")"); i += 1
        elif c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 1
            out.append(s[i:j + 1]); i = j + 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and not s[j].isspace() and s[j] not in '()[];"':
                j += 1
            out.append(s[i:j]); i = j
    return out


def parse_all(tokens):
    pos = 0

    def parse():
        nonlocal pos
        t = tokens[pos]; pos += 1
        if t == "(":
            lst = []
            while tokens[pos] != ")":
                lst.append(parse())
            pos += 1
            return lst
        return t

    forms = []
    while pos < len(tokens):
        forms.append(parse())
    return forms


def split_core(core):
    """['FPCore', name?, (args), prop*, body] -> (args, props, body)."""
    assert core[0] == "FPCore"
    i = 1
    if isinstance(core[i], str):          # optional symbol name
        i += 1
    args = core[i]; i += 1
    props = {}
    while i < len(core) - 1:
        key = core[i]
        if isinstance(key, str) and key.startswith(":"):
            props[key] = core[i + 1]; i += 2
        else:
            break
    return args, props, core[-1]


def inline(node, env):
    if isinstance(node, str):
        return env.get(node, node)
    if not node:
        return node
    head = node[0]
    if head in ("let", "let*"):
        newenv = dict(env)
        for b in node[1]:
            newenv[b[0]] = inline(b[1], newenv if head == "let*" else env)
        return inline(node[2], newenv)
    return [head] + [inline(x, env) for x in node[1:]]


# ------------------------------------------------------------------- subset + output
ALLOWED = {"+", "-", "*", "/", "sqrt"}
CTOR = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div", "sqrt": "Sqrt"}


class Unsupported(Exception):
    pass


def as_int_literal(tok):
    if not isinstance(tok, str):
        return None
    try:
        f = Fraction(tok)
    except (ValueError, ZeroDivisionError):
        return None
    return int(f) if f.denominator == 1 else None


def is_number(tok):
    if not isinstance(tok, str):
        return False
    try:
        Fraction(tok); return True
    except (ValueError, ZeroDivisionError):
        return False


def foldl(head, ops):
    node = ops[0]
    for o in ops[1:]:
        node = [head, node, o]
    return node


def to_math(node, variables):
    if isinstance(node, str):
        lit = as_int_literal(node)
        if lit is not None:
            return f"(Num {lit})"
        if is_number(node):
            raise Unsupported(f"non-integer literal {node!r}")
        if node in variables:
            return f'(Var "{node}")'
        raise Unsupported(f"unknown symbol/constant {node!r}")
    if not node or not isinstance(node[0], str):
        raise Unsupported("malformed node")
    head, ops = node[0], node[1:]
    if head not in ALLOWED:
        raise Unsupported(f"op {head!r}")
    if head == "sqrt":
        if len(ops) != 1:
            raise Unsupported("sqrt arity")
        return f"(Sqrt {to_math(ops[0], variables)})"
    if head == "-" and len(ops) == 1:
        return f"(Neg {to_math(ops[0], variables)})"
    if len(ops) < 2:
        raise Unsupported(f"{head} arity {len(ops)}")
    if len(ops) > 2:
        ops = foldl(head, ops)[1:]
    return f"({CTOR[head]} {to_math(ops[0], variables)} {to_math(ops[1], variables)})"


def to_fpcore(node):
    if isinstance(node, str):
        lit = as_int_literal(node)
        return str(lit) if lit is not None else node
    head, ops = node[0], node[1:]
    if head == "sqrt":
        return f"(sqrt {to_fpcore(ops[0])})"
    if head == "-" and len(ops) == 1:
        return f"(- {to_fpcore(ops[0])})"
    if len(ops) > 2:
        ops = foldl(head, ops)[1:]
    return f"({head} {to_fpcore(ops[0])} {to_fpcore(ops[1])})"


def convert(body, variables):
    """Return (math_term, fpcore_string) for an already-inlined subset body."""
    return to_math(body, set(variables)), to_fpcore(body)


def egglog_text(variables, fpcore_str, math_term) -> str:
    header = f";; (FPCore ({' '.join(variables)}) {fpcore_str})\n\n"
    lets = "".join(f'(let {v} (Var "{v}"))\n' for v in variables)
    return f"{header}{lets}\n(let start {math_term})\n"


# ------------------------------------------------------------------ core index / spec
def index_sources() -> dict:
    """{name -> record} for every subset-expressible core across the vendored sources."""
    idx = {}
    for f in sorted(HERBIE.rglob("*.fpcore")):
        for form in parse_all(tokenize(f.read_text())):
            if not (isinstance(form, list) and form and form[0] == "FPCore"):
                continue
            try:
                args, props, body = split_core(form)
            except Exception:
                continue
            name = props.get(":name")
            if isinstance(name, str):
                name = name.strip('"')
            variables = [a for a in args if isinstance(a, str)]
            if len(variables) != len(args) or not variables:
                continue
            try:
                inlined = inline(body, {})
                math, fps = convert(inlined, variables)
            except (Unsupported, Exception):
                continue
            idx.setdefault((f.relative_to(HERBIE).as_posix(), name),
                           {"vars": variables, "math": math, "fpcore": fps})
    return idx


# Curated Herbie imports: every entry is a numerically non-trivial program the tool's
# subset can express AND (per Herbie) admits a more accurate equivalent form. `box`
# gives the input interval per variable; `prov` records where each bound came from:
#   "pre"  = read from the core's :pre        "pre-hi"/"pre-lo" = one side from :pre
#   "lit"  = standard interval from the benchmark's source literature
#   "cur"  = curated to expose the intended difficulty (no :pre in the source)
SPEC = [
  # ---- backed by an explicit :pre in the source ----
  dict(slug="kahan_p9", src="numerics/great-debate.fpcore", name="Kahan p9 Example",
       box={"x": "[0.001,1]", "y": "[-1,1]"},
       prov={"x": "pre-hi,cur-lo", "y": "pre-hi,cur-lo"},
       note="(x-y)(x+y)/(x^2+y^2): the numerator is x^2-y^2 in disguise. :pre is "
            "0<x<1, y<1; x_lo nudged 0->0.001 off the x=y=0 singularity, y_lo defaulted."),
  dict(slug="conte_x_minus_sqrt", src="numerics/conte.fpcore",
       name="ENA, Section 1.4, Exercise 4d", box={"x": "[2,1000]", "eps": "[0,1]"},
       prov={"x": "cur", "eps": "pre-hi,cur-lo"},
       note="x-sqrt(x^2-eps): cancellation for eps<<x^2. x is [2,1000] (the source :pre "
            "[0,1e9] both hits radicand 0 at x=0 and is too wide for FPTaylor to "
            "optimize); eps_lo raised -1->0 to keep the radicand >0."),
  dict(slug="conte_near_pole", src="numerics/conte.fpcore",
       name="ENA, Section 1.4, Mentioned, B", box={"x": "[1.0001,1.001]"},
       prov={"x": "pre-hi,cur-lo"},
       note="10/(1-x^2) just above the pole at x=1; :pre is [0.999,1.001] but that "
            "straddles the pole (1-x^2=0), so the box is the sliver of :pre above the "
            "pole (x_hi=1.001 from :pre, x_lo curated just past 1)."),
  dict(slug="pbrt_cone_z", src="graphics/pbrt.fpcore", name="UniformSampleCone, z",
       box={"ux": "[2.328306437e-10,1]", "maxCos": "[0,1]"},
       prov={"ux": "pre", "maxCos": "pre"},
       note="(1-ux)+ux*maxCos = 1-ux*(1-maxCos); unused variable uy dropped."),
  dict(slug="excel_x0", src="mathematics/excel.fpcore",
       name="(- (/ x0 (- 1 x1)) x0)", box={"x0": "[1,3]", "x1": "[0.0001,0.02]"},
       prov={"x0": "cur", "x1": "cur"},
       note="x0/(1-x1) - x0 = x0*x1/(1-x1): cancellation for small x1. Box brackets the "
            "two :pre test points (x0~1.85..3, x1~2e-4..0.02)."),
  dict(slug="beta_a", src="mathematics/beta-distribution.fpcore",
       name="a parameter of renormalized beta distribution",
       box={"m": "[0.4,0.6]", "v": "[0.2,0.25]"},
       prov={"m": "cur", "v": "cur"},
       note="(m(1-m)/v - 1)*m -> (m(1-m)-v)*m/v: cancellation when v~m(1-m). Box sits "
            "in the valid beta domain (:pre 0<m, 0<v<1/4) where m(1-m)~=v."),
  dict(slug="beta_b", src="mathematics/beta-distribution.fpcore",
       name="b parameter of renormalized beta distribution",
       box={"m": "[0.4,0.6]", "v": "[0.2,0.25]"},
       prov={"m": "cur", "v": "cur"},
       note="(m(1-m)/v - 1)*(1-m); companion of beta_a, same cancellation."),
  dict(slug="martel_p6", src="numerics/martel.fpcore", name="Expression, p6",
       box={"a": "[-14,-13]", "b": "[-3,-2]", "c": "[3,3.5]", "d": "[12.5,13.5]"},
       prov={"a": "pre", "b": "pre", "c": "pre", "d": "pre"},
       note="2*(a+b+c+d): the four terms nearly cancel, so summation order matters."),
  # ---- famous cores whose interesting domain the source leaves implicit ----
  dict(slug="expand_square", src="tutorial.fpcore", name="Expanding a square",
       box={"x": "[-1,1]"}, prov={"x": "cur"},
       note="(x+1)^2-1 = x^2+2x = x(x+2): cancellation near x=0 (Herbie tutorial)."),
  dict(slug="complex_square_real", src="libraries/mathjs/arithmetic.fpcore",
       name="math.square on complex, real part",
       box={"re": "[1,2]", "im": "[1,2]"}, prov={"re": "cur", "im": "cur"},
       note="re^2-im^2 = (re-im)(re+im): cancellation when re~im."),
  dict(slug="asymptote_c", src="mathematics/sarnoff.fpcore", name="Asymptote C",
       box={"x": "[2,100]"}, prov={"x": "cur"},
       note="x/(x+1) - (x+1)/(x-1) = -(3x+1)/(x^2-1); box avoids the pole at x=1."),
  dict(slug="kahan_p13", src="numerics/great-debate.fpcore", name="Kahan p13 Example 1",
       box={"t": "[1,100]"}, prov={"t": "cur"},
       note="(1+u^2)/(2+u^2) with u=2t/(1+t): Kahan's cancellation example."),
  dict(slug="som_setup_w", src="proj/som.fpcore", name="setup-w",
       box={"es": "[0,0.1]", "ca": "[-1,1]", "rone_es": "[1,1.05]"},
       prov={"es": "cur", "ca": "cur", "rone_es": "cur"},
       note="A^2-1 = (A-1)(A+1) with A=(1-es*ca^2)*rone_es; box keeps A near 1."),
  dict(slug="fastmath_dist4", src="libraries/fast-math.fpcore", name="FastMath dist4",
       box={"d1": "[1,100]", "d2": "[1,100]", "d3": "[1,100]", "d4": "[1,100]"},
       prov={"d1": "cur", "d2": "cur", "d3": "cur", "d4": "cur"},
       note="d1*d2 - d1*d3 + d4*d1 - d1*d1 = d1*(d2-d3+d4-d1): factor out d1."),
]


def generate(only: set[str] | None = None):
    idx = index_sources()
    EGGLOG.mkdir(parents=True, exist_ok=True)
    intervals = json.loads((BENCH / "intervals.json").read_text())
    written = []
    for e in SPEC:
        if only and e["slug"] not in only:
            continue
        rec = idx.get((e["src"], e["name"]))
        if rec is None:
            raise SystemExit(f"core not found: {e['name']!r} in {e['src']}")
        variables = list(e["box"])                       # box order = FPCore arg order
        missing = [v for v in variables if v not in rec["vars"]]
        extra = [v for v in rec["vars"] if v not in variables]
        if missing:
            raise SystemExit(f"{e['slug']}: box names {missing} absent from core vars "
                             f"{rec['vars']}")
        if extra:  # e.g. an unused argument we deliberately omit from the box
            print(f"note {e['slug']}: core var(s) {extra} not in box (unused/dropped)")
        text = egglog_text(variables, rec["fpcore"], rec["math"])
        (EGGLOG / f"{e['slug']}.egglog").write_text(text)
        intervals[e["slug"]] = e["box"]
        written.append(e["slug"])
    (BENCH / "intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")
    print(f"wrote {len(written)} benchmarks: {', '.join(written)}")


def list_matching():
    idx = index_sources()
    for (src, name), rec in sorted(idx.items()):
        print(f"{src:40s} {str(name)[:40]:40s} {rec['fpcore'][:60]}")
    print(f"\n{len(idx)} subset-expressible cores across vendored sources")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list every subset-expressible source core and exit")
    ap.add_argument("--only", nargs="*", help="only (re)generate these slugs")
    args = ap.parse_args()
    if args.list:
        list_matching()
    else:
        generate(set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
