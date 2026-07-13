"""Prove a candidate FPCore program equivalent to a benchmark's reference over a box,
via egglog saturation with the interval analysis seeded from the box."""

from __future__ import annotations

import contextlib
import os
import sys

import benchmarks
import paths

CONSTRUCTOR = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div", "sqrt": "Sqrt"}

RULES = (paths.ROOT / "rules.egglog").read_text()
RULES_UNIFIED = RULES.replace(" :ruleset expand", "")  # all rules in the default ruleset


@contextlib.contextmanager
def quiet_stderr():
    """Drop egglog's Rust-side stderr noise; errors still raise."""
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def seeds(box: dict[str, tuple[float, float]]) -> str:
    """Interval-analysis seeds for the box."""
    return "".join(f"(set (lo {v}) {lo}) (set (hi {v}) {hi})\n" for v, (lo, hi) in box.items())


def fpcore_to_math(node) -> str:
    """Branch-free FPCore AST -> egglog `Math` term. Raises ValueError on a non-integer
    literal or unsupported form (`Num` is i64)."""
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


def _setup(benchmark, box, term, rules: str) -> str:
    return (rules + benchmarks.read_source(benchmark)
            + f"\n(let __candidate__ {term})\n" + (seeds(box) if box else ""))


def _proved(eg) -> bool:
    try:
        eg.run_program(*eg.parse_program("(check (= start __candidate__))"))
        return True
    except Exception:
        return False


def node_count(eg) -> int:
    return sum(n for _, n in eg.run_program(*eg.parse_program("(print-size)"))[0].sizes)


_CHEAP = "(run-schedule (repeat 4 (run)))"   # bounded cheap normalization (saturate can hang)
_EXPAND = "(run expand 1)"                   # one throttled expansion step


def _run_aggressive(eg, max_iters: int, node_cap: int) -> bool:
    """All rules at once, one iteration at a time; bail on saturation or blowup."""
    for _ in range(max_iters + 1):
        if _proved(eg):
            return True
        if node_count(eg) > node_cap:
            return False
        if not eg.run_program(*eg.parse_program("(run 1)"))[0].report.updated:
            return _proved(eg)  # saturated
    return _proved(eg)


def _run_throttled(eg, max_iters: int, node_cap: int) -> bool:
    """One `expand` step per round, renormalizing in between to bound growth."""
    for _ in range(max_iters + 1):
        eg.run_program(*eg.parse_program(_CHEAP))
        if _proved(eg):
            return True
        if node_count(eg) > node_cap:
            return False
        updated = eg.run_program(*eg.parse_program(_EXPAND))[0].report.updated
        if _proved(eg):
            return True
        if node_count(eg) > node_cap or not updated:
            return False
    return _proved(eg)


def _pass_worker(setup: str, throttled: bool, max_iters: int, node_cap: int, q) -> None:
    try:
        with quiet_stderr():
            from egglog.bindings import EGraph
            eg = EGraph()
            eg.run_program(*eg.parse_program(setup))
            run = _run_throttled if throttled else _run_aggressive
            q.put(run(eg, max_iters, node_cap))
    except Exception:
        q.put(False)


def _pass_in_budget(setup, throttled, max_iters, node_cap, budget) -> bool:
    """One pass in a child process, killed if it outlives `budget` seconds. Spawn, not
    fork: a child forked after the parent has run egglog inherits poisoned Rust state
    and hangs."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_pass_worker, args=(setup, throttled, max_iters, node_cap, q))
    p.start()
    p.join(budget)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    try:
        return q.get(timeout=1.0)  # child may exit before its queue feeder flushes
    except Exception:
        return False


def equivalent(benchmark: str, box: dict[str, tuple[float, float]] | None,
               body, timeout: float = 10.0, max_iters: int = 1000,
               node_cap: int = 100_000) -> bool:
    """True if branch-free FPCore AST `body` provably equals the reference over `box`:
    seed the interval analysis from the box, saturate, and check the two share an
    e-class. Domain-conditional rewrites fire only where their preconditions hold over
    `box`, so a wider box can only fail to prove, never assert a false equivalence.
    Runs the aggressive pass then the throttled pass, each at half the `timeout`."""
    try:
        term = fpcore_to_math(body)
    except ValueError:
        return False
    if _pass_in_budget(_setup(benchmark, box, term, RULES_UNIFIED),
                       False, max_iters, node_cap, timeout * 0.5):
        return True
    return _pass_in_budget(_setup(benchmark, box, term, RULES),
                           True, max_iters, node_cap, timeout * 0.5)
