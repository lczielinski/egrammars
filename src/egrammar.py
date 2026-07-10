"""Check whether a candidate FPCore program is equivalent to a benchmark's reference
over an input box, via egglog congruence with the interval analysis seeded from the box."""

from __future__ import annotations

import contextlib
import os
import sys

import paths

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
    return (paths.EGGLOG / f"{benchmark}.egglog").read_text().splitlines()[0].removeprefix(";; ")


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


_RULES = (paths.ROOT / "rules.egglog").read_text()
_RULES_UNIFIED = _RULES.replace(" :ruleset expand", "")  # all rules in the default ruleset


def _setup(benchmark, box, term, rules: str = _RULES) -> str:
    content = (paths.EGGLOG / f"{benchmark}.egglog").read_text()
    seeds = _seeds(box) if box else ""
    return rules + content + f"\n(let __candidate__ {term})\n" + seeds


def _proved(eg) -> bool:
    try:
        eg.run_program(*eg.parse_program("(check (= start __candidate__))"))
        return True
    except Exception:
        return False


def _nodes(eg) -> int:
    return sum(n for _, n in eg.run_program(*eg.parse_program("(print-size)"))[0].sizes)


# Two proof passes (see `equivalent`), sound from either: `_run_aggressive` fires every
# rule at once (fast, proves anything that doesn't blow up); `_run_throttled` fires the
# combinatorial `expand` rules one step at a time, renormalizing in between, to rescue the
# proofs that would otherwise blow the graph up. (egg's per-rule backoff scheduler by hand.)
_CHEAP = "(run-schedule (repeat 4 (run)))"   # bounded cheap normalization (saturate can hang)
_EXPAND = "(run expand 1)"                   # one throttled expansion step


def _run_aggressive(eg, max_iters: int, node_cap: int) -> bool:
    """All rules at once, one iteration at a time. Rejects early on saturation or blowup."""
    for _ in range(max_iters + 1):
        if _proved(eg):
            return True
        if _nodes(eg) > node_cap:
            return False
        if not eg.run_program(*eg.parse_program("(run 1)"))[0].report.updated:
            return _proved(eg)               # saturated
    return _proved(eg)


def _run_throttled(eg, max_iters: int, node_cap: int) -> bool:
    """`expand` one step per round, renormalizing in between so the graph can't run away."""
    for _ in range(max_iters + 1):
        eg.run_program(*eg.parse_program(_CHEAP))
        if _proved(eg):
            return True
        if _nodes(eg) > node_cap:
            return False
        updated = eg.run_program(*eg.parse_program(_EXPAND))[0].report.updated
        if _proved(eg):
            return True
        if _nodes(eg) > node_cap:
            return False
        if not updated:                      # saturated
            return False
    return _proved(eg)


def _pass_worker(setup: str, throttled: bool, max_iters: int, node_cap: int, q) -> None:
    """Run one pass on a fresh e-graph in a child process (killed if it runs past budget)."""
    try:
        with _quiet_stderr():
            from egglog.bindings import EGraph
            eg = EGraph()
            eg.run_program(*eg.parse_program(setup))
            run = _run_throttled if throttled else _run_aggressive
            q.put(run(eg, max_iters, node_cap))
    except Exception:
        q.put(False)


def _pass_in_budget(setup, throttled, max_iters, node_cap, budget) -> bool:
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_pass_worker, args=(setup, throttled, max_iters, node_cap, q))
    p.start()
    p.join(budget)
    if p.is_alive():                 # over budget -> give up on this pass
        p.terminate()
        p.join()
        return False
    try:
        return q.get_nowait()
    except Exception:
        return False


def equivalent(benchmark: str, box: dict[str, tuple[float, float]] | None,
               body, timeout: float = 10.0, max_iters: int = 1000,
               node_cap: int = 100_000) -> bool:
    """True if branch-free FPCore AST `body` is provably equal to the reference over `box`:
    add it to the reference's e-graph, saturate with the box seeded into the interval
    analysis, and check the two share an e-class. A domain-conditional rewrite bridges them
    only where its preconditions hold over `box`; a wider box can only fail to prove an
    equivalence, never assert a false one.

    Runs the aggressive pass then the throttled pass (see above), each as a child process
    the parent kills at its slice of the `timeout` (seconds) budget."""
    try:
        term = fpcore_to_math(body)
    except ValueError:
        return False
    aggressive_setup = _setup(benchmark, box, term, _RULES_UNIFIED)
    throttled_setup = _setup(benchmark, box, term, _RULES)
    if _pass_in_budget(aggressive_setup, False, max_iters, node_cap, timeout * 0.5):
        return True
    return _pass_in_budget(throttled_setup, True, max_iters, node_cap, timeout * 0.5)
