"""Numerically classify a candidate the e-graph *failed* to prove equivalent, to tell
the two failure modes apart:

- verdict "equal"       : candidate matches the reference at every sampled point (within
                          tolerance) => it is almost certainly algebraically equivalent
                          and the e-graph is MISSING A RULE.
- verdict "different"   : candidate disagrees with the reference somewhere => the model
                          produced a genuinely NON-EQUIVALENT program.
- verdict "indeterminate": no sample point evaluated to a finite value in both (e.g. the
                          whole box is a singularity) => can't tell.

This is a heuristic (sampling can't prove equivalence), but a loose tolerance cleanly
separates the two: algebraically-equal forms differ only by rounding (~1e-16), while a
wrong program differs by O(1)."""

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


def _float_box(box: dict | None) -> dict | None:
    if not box:
        return None
    return {v: tuple(map(float, s.strip("[]").split(","))) for v, s in box.items()}


def classify(reference: str, candidate: str, box: dict | None,
             n: int = 400, tol: float = 1e-6) -> dict:
    """Sample the box and compare candidate vs reference pointwise."""
    ref_body = regions.body_of(reference)
    cand_body = regions.body_of(candidate)
    variables = regions.variables_of(reference)
    fbox = _float_box(box) or {v: (-100.0, 100.0) for v in variables}
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
