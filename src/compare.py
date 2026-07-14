"""Compare a run against Herbie, entirely inside Herbie's own harness.

One `herbie report` call scores everything on the same sampled points with Herbie's
Rival ground truth and bits-of-error metric: each benchmark becomes an FPCore whose
body is the reference (`start` bits), with every proven program attached as an `:alt`
(`target` bits), improved by Herbie itself (`end` bits). By default Herbie is
restricted to this tool's operators (herbie_platform.rkt: + - * / sqrt, if) so the
comparison is search-vs-search, not vocabulary; pass --platform default to lift that.
When `fptaylor` is on PATH, a second table bounds the same three programs' WORST-case
error over the box. Writes <run>/herbie.json and <run>/herbie.md.

    uv run src/compare.py              # latest run
    uv run src/compare.py --run NAME
"""

import argparse
import json
import re
import shutil
import subprocess

import benchmarks
import fptaylor_check
import paths
import regions

HERBIE_SECONDS = 300  # per-benchmark budget
PLATFORM = paths.ROOT / "herbie_platform.rkt"


def herbie_cmd() -> list[str] | None:
    if shutil.which("herbie"):
        return ["herbie"]
    if shutil.which("racket"):
        return ["racket", "-l", "herbie", "--"]
    return None


def body_of_text(program: str) -> str:
    """The body of a single-line `(FPCore (vars) body)` string."""
    return program[program.index(") ") + 2:-1]


def to_fpcore(benchmark: str, alts: list[str]) -> str:
    """The reference as an FPCore: box as :pre, each proven program as an :alt."""
    ref = benchmarks.read_reference(benchmark)
    box = benchmarks.INTERVALS.get(benchmark) or {}
    variables = " ".join(regions.parse(regions.tokenize(ref))[1])
    pres = [f"(<= {lo} {v} {hi})" for v, (lo, hi) in box.items()]
    lines = [f"(FPCore ({variables})", f' :name "{benchmark}"']
    if pres:
        lines.append(f" :pre {pres[0] if len(pres) == 1 else '(and ' + ' '.join(pres) + ')'}")
    lines += [f" :alt {body_of_text(p)}" for p in alts]
    lines.append(f" {body_of_text(ref)})")
    return "\n".join(lines)


def herbie_to_ast(output: str):
    """Herbie's typed output expression -> our AST. Raises ValueError on an operator
    outside the subset (possible when --platform default is used)."""
    output = re.sub(r"#s\(literal ([^ ]+) \w+\)", r"\1", output)

    def convert(node):
        if isinstance(node, str):
            if "/" in node and re.fullmatch(r"-?\d+/\d+", node):  # rational literal
                a, b = node.split("/")
                return ["/", a, b]
            return node.removesuffix(".f64")
        head = node[0].removesuffix(".f64")
        args = [convert(a) for a in node[1:]]
        if head == "neg":
            return ["-", args[0]]
        if head in ("+", "-", "*", "/", "sqrt", "if", "<", ">", "<=", ">=") :
            return [head, *args]
        raise ValueError(f"operator {head!r} outside the subset")

    return convert(regions.parse(regions.tokenize(output)))


def run_report(run, cores: list[str], timeout_each: int, platform: str) -> list[dict]:
    """One `herbie report` over all cores; returns results.json's tests."""
    hdir = run / "herbie"
    hdir.mkdir(parents=True, exist_ok=True)
    (hdir / "input.fpcore").write_text("\n\n".join(cores) + "\n")
    cmd = herbie_cmd() + ["report", "--platform", platform, "--seed", "1",
                          "--timeout", str(timeout_each),
                          str(hdir / "input.fpcore"), str(hdir / "report")]
    subprocess.run(cmd, timeout=(len(cores) + 2) * timeout_each)  # streams progress
    return json.loads((hdir / "report" / "results.json").read_text())["tests"]


def _worst(run, row) -> None:
    """Attach worst-case FPTaylor bounds for reference / ours / herbie's program.
    `ours` here is the run's worst-case CHAMPION (best bound among proven programs),
    which may be a different program than the average-bits column's."""
    b = row["benchmark"]
    box = benchmarks.INTERVALS[b]
    fsrc = paths.fptaylor_path(run, b)
    if fsrc.exists():  # reference + ours were already bounded during the run
        fd = json.loads(fsrc.read_text())
        pick = lambda r: {"abs_err": r.get("abs_err"), "rel_err_ulps": r.get("rel_err_ulps")}
        if fd.get("reference_result"):
            row["reference_worst"] = pick(fd["reference_result"])
        for key in ("rel_err_ulps", "abs_err"):  # champion: best rel, else best abs
            scored = [r for r in fd.get("results", []) if r.get(key) is not None]
            if scored:
                champ = min(scored, key=lambda r: r[key])
                row["ours_worst"] = pick(champ)
                row["ours_worst_program"] = champ.get("program")
                break
    if row["herbie_program"]:
        try:
            ast = herbie_to_ast(row["herbie_program"])
            r = fptaylor_check.bound_program(ast, box)
            row["herbie_worst"] = {"abs_err": r.get("abs_err"),
                                   "rel_err_ulps": r.get("rel_err_ulps")}
        except ValueError as e:
            row["herbie_worst_error"] = str(e)


def rows_from_tests(tests: list[dict], alts_by_name: dict[str, list[str]]) -> list[dict]:
    rows = []
    for t in tests:
        start, end = t.get("start"), t.get("end")
        targets = [x[1] for x in (t.get("target") or [])]  # [cost, bits] per :alt
        programs = alts_by_name.get(t["name"], [])
        scored = sorted(zip(targets, programs)) if len(targets) == len(programs) else []
        rows.append({
            "benchmark": t["name"], "status": t.get("status"),
            "reference_bits": start if isinstance(start, (int, float)) else None,
            "herbie_bits": end if isinstance(end, (int, float)) else None,
            "herbie_program": t.get("output"),
            "ours_bits": scored[0][0] if scored else None,
            "ours_program": scored[0][1] if scored else None,
            "ours_all": [{"program": p, "bits": b} for b, p in scored],
            "reference_worst": None, "ours_worst": None, "ours_worst_program": None,
            "herbie_worst": None,
        })
    return rows


def _winner(ours, herb, margin) -> str:
    if ours is None or herb is None:
        return "-"
    if herb < ours - margin:
        return "herbie"
    if ours < herb - margin:
        return "ours"
    return "tie"


def avg_table(rows: list[dict]) -> str:
    f = lambda x: f"{x:.2f}" if x is not None else "-"
    lines = ["| benchmark | reference | ours (best) | herbie | winner |",
             "|---|--:|--:|--:|---|"]
    for r in sorted(rows, key=lambda r: r["benchmark"]):
        lines.append(f"| {r['benchmark']} | {f(r['reference_bits'])} | {f(r['ours_bits'])} "
                     f"| {f(r['herbie_bits'])} | {_winner(r['ours_bits'], r['herbie_bits'], 0.1)} |")
    return "\n".join(lines)


def worst_table(rows: list[dict]) -> str:
    def cell(w):
        if not w:
            return "-"
        if w["rel_err_ulps"] is not None:
            return f"{w['rel_err_ulps']:.1f} ulp"
        if w["abs_err"] is not None:
            return f"{w['abs_err']:.1e} abs"
        return "-"

    def metric(a, b):  # comparable pair: rel if both have it, else abs
        for key in ("rel_err_ulps", "abs_err"):
            if a and b and a.get(key) is not None and b.get(key) is not None:
                return a[key], b[key]
        return None, None

    lines = ["| benchmark | reference | ours (best) | herbie | winner |",
             "|---|--:|--:|--:|---|"]
    for r in sorted(rows, key=lambda r: r["benchmark"]):
        o, h = metric(r["ours_worst"], r["herbie_worst"])
        win = _winner(o, h, 0.0) if o is not None else "-"
        lines.append(f"| {r['benchmark']} | {cell(r['reference_worst'])} "
                     f"| {cell(r['ours_worst'])} | {cell(r['herbie_worst'])} | {win} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, metavar="NAME",
                   help="results/<NAME> (default: the latest run)")
    p.add_argument("--timeout", type=int, default=HERBIE_SECONDS, metavar="SECONDS",
                   help="herbie budget per benchmark")
    p.add_argument("--platform", default=str(PLATFORM),
                   help="herbie platform: a platform file or a built-in name "
                        "(default: the repo's arithmetic-only platform; pass "
                        "`default` to let Herbie use its full operator set)")
    args = p.parse_args()

    run = paths.run_dir(args.run) if args.run else paths.latest_run()
    if run is None or not run.exists():
        p.error(f"no run found in {paths.RESULTS}")
    if herbie_cmd() is None:
        p.error("herbie not found: install it, or racket with the herbie package")
    fptaylor = shutil.which("fptaylor") is not None
    if not fptaylor:
        print("`fptaylor` not on PATH -- skipping the worst-case table")

    names = paths.benchmarks_in(run) or benchmarks.suite()
    cores, alts_by_name = [], {}
    for b in names:
        if benchmarks.INTERVALS.get(b) is None:
            continue
        esrc = paths.equivalents_path(run, b)
        alts = json.loads(esrc.read_text()).get("programs", []) if esrc.exists() else []
        alts_by_name[b] = list(dict.fromkeys(alts))
        cores.append(to_fpcore(b, alts_by_name[b]))

    print(f"herbie report on {len(cores)} benchmarks (this runs Herbie's full search)")
    tests = run_report(run, cores, args.timeout, args.platform)
    rows = rows_from_tests(tests, alts_by_name)
    if fptaylor:
        for r in rows:
            _worst(run, r)

    md = (f"# Herbie comparison — {run.name}\n\n"
          "## Average bits of error (Herbie's metric, same sampled points)\n\n"
          "Lower is better; `ours` = best proven program, scored via `:alt`.\n\n"
          + avg_table(rows) + "\n")
    if fptaylor:
        md += ("\n## Worst-case error over the box (FPTaylor)\n\n"
               "`ours` = the run's worst-case champion (best bound among proven "
               "programs; may differ from the average-bits column's program). Winner "
               "compares relative ulps when both sides have them, else absolute "
               "error.\n\n" + worst_table(rows) + "\n")
    (run / "herbie.json").write_text(json.dumps(rows, indent=2))
    (run / "herbie.md").write_text(md)
    print("\n" + md)
    print(f"wrote {run / 'herbie.json'} and {run / 'herbie.md'}")


if __name__ == "__main__":
    main()
