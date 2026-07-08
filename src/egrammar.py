"""Check whether a candidate FPCore program is equivalent to a benchmark's reference
over an input box, via egglog congruence with the interval analysis seeded from the box."""

from __future__ import annotations

import contextlib
import os
import sys

import paths

SATURATION_RUNS = 6
CONSTRUCTOR = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div", "sqrt": "Sqrt"}


@contextlib.contextmanager
def _quiet_stderr():
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def read_reference(benchmark: str) -> str:
    return (paths.BENCHMARKS / f"{benchmark}.egglog").read_text().splitlines()[0].removeprefix(";; ")


def _seeds(box: dict[str, tuple[float, float]]) -> str:
    return "".join(f"(set (lo {v}) {lo}) (set (hi {v}) {hi})\n" for v, (lo, hi) in box.items())


def fpcore_to_math(node) -> str:
    """Render a branch-free FPCore AST as an egglog `Math` term. Raises ValueError on a
    non-integer literal or unsupported form (`Num` is i64)."""
    if isinstance(node, str):
        try:
            value = float(node)
        except ValueError:
            return f'(Var "{node}")'
        if value != int(value):
            raise ValueError(f"non-integer literal {node!r}")
        return f"(Num {int(value)})"
    if not node:
        raise ValueError("empty node")
    head, operands = node[0], node[1:]
    if head == "-" and len(operands) == 1:
        return f"(Neg {fpcore_to_math(operands[0])})"
    ctor = CONSTRUCTOR.get(head)
    if ctor is None or len(operands) != (1 if head == "sqrt" else 2):
        raise ValueError(f"unsupported form {node!r}")
    return f"({ctor} " + " ".join(fpcore_to_math(o) for o in operands) + ")"


def equivalent(benchmark: str, box: dict[str, tuple[float, float]] | None,
               body, runs: int = SATURATION_RUNS) -> bool:
    """True if branch-free FPCore AST `body` is provably equal to the reference over
    `box`: add it to the reference's e-graph, saturate with the box seeded into the
    interval analysis, and check the two share an e-class. A domain-conditional rewrite
    bridges them only where its preconditions hold over `box`; a wider box can only fail
    to prove an equivalence, never assert a false one."""
    from egglog.bindings import EGraph

    try:
        term = fpcore_to_math(body)
    except ValueError:
        return False
    rules = (paths.ROOT / "rules.egglog").read_text()
    content = (paths.BENCHMARKS / f"{benchmark}.egglog").read_text()
    seeds = _seeds(box) if box else ""
    source = (rules + content + f"\n(let __candidate__ {term})\n" + seeds
              + f"\n(run {runs})\n(check (= start __candidate__))")
    try:
        with _quiet_stderr():
            egraph = EGraph()
            egraph.run_program(*egraph.parse_program(source))
        return True
    except Exception:
        return False
