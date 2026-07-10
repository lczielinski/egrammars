"""Classify a candidate the e-graph failed to prove, by sampling the box and comparing to
the reference pointwise:

- "equal"        : matches everywhere (within tolerance) => likely a MISSING e-graph RULE.
- "different"    : disagrees somewhere => a genuinely NON-EQUIVALENT program.
- "indeterminate": no finite sample point in both => can't tell.

Heuristic (sampling can't prove equivalence), but a loose tolerance separates the two
cleanly: equal forms differ only by rounding (~1e-16), a wrong program by O(1)."""

from __future__ import annotations

import math
import random

import regions


def _eval(node, env: dict):
    if isinstance(node, str):
        return env[node] if node in env else float(node)
    head = node[0]
    if head == "if":
        (op, lhs, rhs), then, els = node[1], node[2], node[3]
        a, b = _eval(lhs, env), _eval(rhs, env)
        hit = {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
        return _eval(then if hit else els, env)
    if head == "sqrt":
        return math.sqrt(_eval(node[1], env))
    if head == "-" and len(node) == 2:
        return -_eval(node[1], env)
    a, b = _eval(node[1], env), _eval(node[2], env)
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[head]


def classify(reference: str, candidate: str, box: dict | None,
             n: int = 400, tol: float = 1e-6) -> dict:
    """Sample the box and compare candidate vs reference pointwise."""
    ref_body = regions.body_of(reference)
    cand_body = regions.body_of(candidate)
    variables = regions.variables_of(reference)
    fbox = regions.float_box(box) or {v: (-100.0, 100.0) for v in variables}
    rng = random.Random(0)  # deterministic

    worst = 0.0
    valid = 0
    for _ in range(n):
        env = {v: rng.uniform(*fbox[v]) for v in variables}
        try:
            a, b = _eval(ref_body, env), _eval(cand_body, env)
        except (ValueError, ZeroDivisionError, OverflowError, KeyError):
            continue  # singularity / domain error at this point — skip
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        valid += 1
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        worst = max(worst, rel)

    if valid == 0:
        verdict = "indeterminate"
    elif worst <= tol:
        verdict = "equal"
    else:
        verdict = "different"
    return {"verdict": verdict, "max_rel_diff": worst, "samples": valid,
            "hint": {"equal": "likely a missing e-graph rule",
                     "different": "model produced a non-equivalent program",
                     "indeterminate": "no finite sample point in the box"}[verdict]}
