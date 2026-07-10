"""Bound the *reference* program of every benchmark with FPTaylor and record the
result in `benchmarks/reference_bounds.json`.

This is the baseline each synthesized rewrite is measured against: the worst-case
double-rounding error of the untouched reference over its interval box. Needs the
`fptaylor` binary on PATH (run with the opam env active).

    uv run scripts/reference_bounds.py            # bound all 47
    uv run scripts/reference_bounds.py kepler0 …  # bound only the named benchmarks
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fptaylor_check as fc  # noqa: E402
import paths  # noqa: E402
import regions  # noqa: E402

OUT = paths.BENCHMARKS / "reference_bounds.json"


def reference_fpcore(name: str) -> str:
    """The `(FPCore …)` reference lives in the first `;;` comment of each egglog file."""
    first = (paths.EGGLOG / f"{name}.egglog").read_text().splitlines()[0]
    return first.removeprefix(";; ").strip()


def main() -> None:
    intervals = json.loads(paths.INTERVALS_FILE.read_text())
    only = set(sys.argv[1:])
    names = sorted(n for n in intervals if not only or n in only)
    if only - set(intervals):
        raise SystemExit(f"unknown benchmark(s): {', '.join(sorted(only - set(intervals)))}")

    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
        f.write(fc.CONFIG)
        cfg = f.name
    try:
        out = json.loads(OUT.read_text()) if OUT.exists() else {}
        for name in names:
            box = intervals[name]
            fpcore = reference_fpcore(name)
            r = fc.analyze_program(regions.parse(regions.tokenize(fpcore)), box, cfg)
            ulp = r.get("rel_err_ulps")
            out[name] = {
                "reference": fpcore,
                "box": box,
                "abs_err": r.get("abs_err"),
                "rel_err": r.get("rel_err"),
                "rel_err_ulps": ulp,
                "rel_err_derived": bool(r.get("rel_err_derived")),
                "enclosure": r.get("enclosure"),
                "timeout": bool(r.get("timeout")),
            }
            if r.get("timeout"):
                status = "TIMEOUT"
            elif ulp is not None:
                status = f"{ulp:,.1f} ulp{' (derived)' if r.get('rel_err_derived') else ''}"
            else:
                status = "no rel err (enclosure straddles 0)"
            print(f"  {name:28s} abs={fc.fmt(r.get('abs_err')):>10}  {status}", flush=True)
            # write after each benchmark so progress persists / is resumable
            OUT.write_text(json.dumps({k: out[k] for k in sorted(out)}, indent=2) + "\n")
    finally:
        os.unlink(cfg)

    print(f"\nwrote {len(names)} of {len(out)} reference bounds to {OUT.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
