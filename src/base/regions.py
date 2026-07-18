"""Interval boxes (var -> (lo, hi)) and FPCore `if`-tree guards."""

import re

NEGATE = {"<": ">=", ">": "<=", "<=": ">", ">=": "<"}


def tokenize(s: str):
    return re.findall(r'\(|\)|"[^"]*"|[^\s()]+', s)


def parse(toks):
    t = toks.pop(0)
    if t != "(":
        return t
    node = []
    while toks[0] != ")":
        node.append(parse(toks))
    toks.pop(0)
    return node


def parse_program(s: str):
    return parse(tokenize(s))


def variables_of(program: str) -> list[str]:
    ast = parse_program(program)
    return ast[1] if isinstance(ast, list) and ast[:1] == ["FPCore"] else []


def body_of(program):
    ast = parse_program(program) if isinstance(program, str) else program
    return ast[2] if isinstance(ast, list) and ast[:1] == ["FPCore"] else ast


def split_branches(node):
    """Yield (guards, leaf) per leaf; a non-simple condition contributes no guard
    (sound: fewer guards only widens the region)."""
    if isinstance(node, list) and node and node[0] == "if":
        _, cond, then, els = node
        simple = (isinstance(cond, list) and len(cond) == 3 and cond[0] in NEGATE
                  and isinstance(cond[1], str) and isinstance(cond[2], str))
        op, lhs, rhs = cond if simple else (None, None, None)
        for conds, expr in split_branches(then):
            yield ([(op, lhs, rhs)] if simple else []) + conds, expr
        for conds, expr in split_branches(els):
            yield ([(NEGATE[op], lhs, rhs)] if simple else []) + conds, expr
    else:
        yield [], node


def _as_float(tok):
    try:
        return float(tok)
    except (ValueError, TypeError):
        return None


def narrow_box(box: dict, conds) -> dict | None:
    """`box` restricted to where all `conds` hold, or None if empty. A comparison
    that doesn't pin a variable to a number is ignored (sound superset)."""
    iv = {v: list(b) for v, b in box.items()}
    for op, lhs, rhs in conds:
        hi_side = op in ("<", "<=")
        if lhs in iv and (n := _as_float(rhs)) is not None:
            i = 1 if hi_side else 0
            iv[lhs][i] = (min if hi_side else max)(iv[lhs][i], n)
        elif rhs in iv and (n := _as_float(lhs)) is not None:
            i = 0 if hi_side else 1
            iv[rhs][i] = (max if hi_side else min)(iv[rhs][i], n)
    if any(lo > hi for lo, hi in iv.values()):
        return None
    return {v: (lo, hi) for v, (lo, hi) in iv.items()}
