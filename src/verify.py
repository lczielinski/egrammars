"""Verify candidates: prove each branch equivalent over its region (prove.py); classify
unproven candidates by sampling the box:

- "equal"        : matches the reference everywhere sampled => likely a MISSING RULE.
- "different"    : disagrees somewhere => genuinely non-equivalent.
- "indeterminate": no finite sample point => can't tell.

The classification is heuristic, but a loose tolerance separates the cases cleanly:
equal forms differ only by rounding (~1e-16), a wrong program by O(1)."""

from __future__ import annotations

import math
import random

import prove
import regions


def validate(benchmark, box, program, timeout) -> bool:
    """True if every branch equals the reference over the region its guards select.
    `if` is total, so per-branch validity implies whole-program validity."""
    for conds, leaf in regions.split_branches(regions.body_of(program)):
        region = regions.narrow_box(box, conds) if box else None
        if box and region is None:
            continue  # unreachable branch
        if not prove.equivalent(benchmark, region, leaf, timeout=timeout):
            return False
    return True


def evaluate_candidates(benchmark, reference, box, candidates, timeout):
    """(proven_programs, attempts); an unproven candidate gets a `numeric`
    classification (missing rule vs. genuinely non-equivalent)."""
    programs, attempts = [], []
    for prog in candidates:
        proven = validate(benchmark, box, prog, timeout=timeout)
        rec = {"program": prog, "proven_equivalent": proven}
        if proven:
            programs.append(prog)
        else:
            rec["numeric"] = classify(reference, prog, box)
        attempts.append(rec)
    return programs, attempts


def attempt_line(rec: dict) -> str:
    if rec["proven_equivalent"]:
        return f"  proven      {rec['program']}"
    tag = {"equal": "MISSING-RULE?", "different": "not-equivalent",
           "indeterminate": "indeterminate"}[rec["numeric"]["verdict"]]
    return f"  {tag:<13} {rec['program']}"


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
    ref_body = regions.body_of(reference)
    cand_body = regions.body_of(candidate)
    variables = regions.variables_of(reference)
    box = box or {v: (-100.0, 100.0) for v in variables}
    rng = random.Random(0)  # deterministic

    worst = 0.0
    valid = 0
    for _ in range(n):
        env = {v: rng.uniform(*box[v]) for v in variables}
        try:
            a, b = _eval(ref_body, env), _eval(cand_body, env)
        except (ValueError, ZeroDivisionError, OverflowError, KeyError):
            continue  # singularity / domain error at this point
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
