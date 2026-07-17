"""E-grammar: compile an e-graph of programs equivalent to a benchmark into a GBNF
grammar (xgrammar's EBNF dialect).

`build(benchmark, box)` saturates egglog with the interval analysis seeded from the
box, strips identity/cyclic spellings, and emits a grammar whose language is the
programs provably equivalent to the reference over that box. `rules` returns just
the e-class productions for splicing into decoding.py's head/arm grammars.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

from base import benchmarks, paths

SATURATION_RUNS = 6
SATURATION_BUDGET = 20.0  # seconds; skip remaining rounds past this
NODE_CAP = 100_000
START = "__start__"

GRAMMAR_RULES = (paths.ROOT / "rules.egglog").read_text()


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


def interval_seeds(box: dict[str, tuple[float, float]]) -> str:
    """Interval-analysis seeds for the box."""
    return "".join(f"(set (lo {v}) {lo}) (set (hi {v}) {hi})\n" for v, (lo, hi) in box.items())


def node_count(eg) -> int:
    return sum(n for _, n in eg.run_program(*eg.parse_program("(print-size)"))[0].sizes)


@dataclass(frozen=True, order=True)
class ENode:
    op: str
    children: tuple[str, ...]  # e-class ids


EClassMapping = dict[str, set[ENode]]


def saturate(benchmark_source: str, runs: int = SATURATION_RUNS, seeds: str = "",
             node_cap: int = NODE_CAP, budget: float = SATURATION_BUDGET):
    """Saturate with all rules fired together, one step at a time, to a fixpoint,
    `runs`, `node_cap`, or the time `budget` (late rounds on division-heavy
    benchmarks can cost minutes for negligible grammar gain -- and arm grammars are
    compiled mid-decode). Extraction wants breadth; stopping early only shrinks the
    language, never breaks soundness."""
    from egglog.bindings import EGraph

    source = GRAMMAR_RULES + benchmark_source + seeds
    source += f"\n(relation {START} (Math))\n({START} start)"  # mark the root e-class
    egraph = EGraph()
    t0 = time.monotonic()
    with quiet_stderr():
        egraph.run_program(*egraph.parse_program(source))
        for _ in range(runs):
            if time.monotonic() - t0 > budget or node_count(egraph) > node_cap:
                break
            r0 = time.monotonic()
            if not egraph.run_program(*egraph.parse_program("(run 1)"))[0].report.updated:
                break  # saturated
            # round cost grows ~5-10x per round: if this one used a big slice of the
            # budget, the next would overshoot it many times over -- stop now
            if time.monotonic() - r0 > budget / 4:
                break
    return egraph


CONSTRUCTORS = frozenset({"Num", "Var", "Add", "Sub", "Neg", "Sqrt", "Mul", "Div"})


def extract(egraph) -> tuple[str, EClassMapping]:
    """The root e-class and eclass -> set of e-nodes (analysis nodes dropped)."""
    nodes = json.loads(egraph.serialize([]).to_json())["nodes"]

    keep: set[str] = set()
    for node in nodes.values():
        if node["op"].strip('"') in CONSTRUCTORS or node["op"] == START:
            keep.add(node["eclass"])
            keep.update(nodes[c]["eclass"] for c in node["children"])

    root = None
    eclasses: defaultdict[str, set[ENode]] = defaultdict(set)
    for node in nodes.values():
        op = node["op"].strip('"')  # string leaves are serialized with quotes
        if node["eclass"] not in keep:
            continue
        children = tuple(nodes[child]["eclass"] for child in node["children"])
        if op == START:
            root = children[0]
        else:
            eclasses[node["eclass"]].add(ENode(op, children))
    assert root is not None, "benchmark must define a `start` expression"
    return root, dict(eclasses)


def _zero_one_classes(eclasses: EClassMapping) -> tuple[set[str], set[str]]:
    """E-classes that provably denote 0 and 1, by fixpoint over the operators."""

    def is_literal(eclass: str, value: str) -> bool:
        return any(
            enode.op == "Num"
            and any(
                leaf.op == value and not leaf.children
                for leaf in eclasses.get(enode.children[0], ())
            )
            for enode in eclasses.get(eclass, ())
            if enode.children
        )

    zero = {eclass for eclass in eclasses if is_literal(eclass, "0")}
    one = {eclass for eclass in eclasses if is_literal(eclass, "1")}

    def denotes_zero(enode: ENode) -> bool:
        match enode.op, enode.children:
            case ("Neg" | "Sqrt", (a,)):
                return a in zero
            case ("Mul", (a, b)):
                return a in zero or b in zero
            case ("Add", (a, b)):
                return a in zero and b in zero
            case ("Sub", (a, b)):
                return a == b or (a in zero and b in zero)
            case ("Div", (a, _)):
                return a in zero
        return False

    def denotes_one(enode: ENode) -> bool:
        match enode.op, enode.children:
            case ("Sqrt", (a,)):
                return a in one
            case ("Mul", (a, b)):
                return a in one and b in one
            case ("Div", (a, b)):
                return a == b or (a in one and b in one)
        return False

    changed = True
    while changed:
        changed = False
        for eclass, enodes in eclasses.items():
            if eclass not in zero and any(denotes_zero(e) for e in enodes):
                zero.add(eclass)
                changed = True
            if eclass not in one and any(denotes_one(e) for e in enodes):
                one.add(eclass)
                changed = True
    return zero, one


def strip_identity_enodes(root: str, eclasses: EClassMapping) -> tuple[str, EClassMapping]:
    """Alias identity-only e-classes into the operand they equal, drop identity
    spellings (x*1, x+0, ...) where a real spelling remains, and prune non-minimal
    cyclic spellings."""
    zero, one = _zero_one_classes(eclasses)

    parent = {eclass: eclass for eclass in eclasses}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def reduces_to(enode: ENode) -> str | None:
        """The operand this enode equals, or None if not an identity."""
        match enode.op, enode.children:
            case ("Mul", (a, b)):
                return a if a in zero else b if b in zero or a in one else (
                    a if b in one else None
                )
            case ("Add", (a, b)):
                return b if a in zero else a if b in zero else None
            case ("Sub", (a, b)):
                return a if b in zero else None
            case ("Div", (a, b)):
                return a if b in one or a in zero else None

    def resolve(enodes: set[ENode]) -> set[ENode]:
        return {ENode(e.op, tuple(find(c) for c in e.children)) for e in enodes}

    changed = True
    while changed:
        changed = False
        merged: defaultdict[str, set[ENode]] = defaultdict(set)
        for eclass, enodes in eclasses.items():
            merged[find(eclass)] |= resolve(enodes)
        zero, one = {find(z) for z in zero}, {find(o) for o in one}
        for rep, enodes in merged.items():
            targets = {reduces_to(e) for e in enodes}
            if None in targets:  # has a genuine (non-identity) spelling
                continue
            targets = {find(t) for t in targets} - {rep}
            if targets:
                parent[rep] = min(targets)
                changed = True

    cleaned: defaultdict[str, set[ENode]] = defaultdict(set)
    for eclass, enodes in eclasses.items():
        cleaned[find(eclass)] |= resolve(enodes)
    root, zero, one = find(root), {find(z) for z in zero}, {find(o) for o in one}

    def is_padding(enode: ENode) -> bool:
        match enode.op, enode.children:
            case ("Mul", (a, b)):
                return a in one or b in one or a in zero or b in zero
            case ("Add", (a, b)):
                return a in zero or b in zero
            case ("Sub", (a, b)):
                return a == b or a in zero or b in zero
            case ("Div", (a, b)):
                return a == b or b in one or a in zero
        return False

    stripped: EClassMapping = {}
    for eclass, enodes in cleaned.items():
        stripped[eclass] = {e for e in enodes if not is_padding(e)} or enodes

    # Acyclic spelling selection: order classes by (min completion size, BFS depth
    # from root DESC, id) -- the root sorts last among ties, so it can reference
    # everything -- and keep a spelling only if every child strictly precedes its
    # class. The kept grammar is a DAG (no unbounded recursion, hence no monster
    # programs) while deep-but-terminating spellings like conjugates survive. Every
    # class keeps its minimal spelling: that spelling's children are strictly
    # smaller, so they always precede it.
    size = _min_sizes(stripped)
    depth = {root: 0}
    frontier = [root]
    while frontier:
        nxt = []
        for c in frontier:
            for e in stripped[c]:
                for ch in e.children:
                    if ch in stripped and ch not in depth:
                        depth[ch] = depth[c] + 1
                        nxt.append(ch)
        frontier = nxt
    order = {c: i for i, c in enumerate(
        sorted(stripped, key=lambda c: (size[c], -depth.get(c, 1 << 30), c)))}
    return root, {
        c: {e for e in enodes if all(order[ch] < order[c] for ch in e.children
                                     if ch in order)} or
           {min(enodes, key=lambda e: (1 + sum(size[ch] for ch in e.children), e))}
        for c, enodes in stripped.items()
    }


def _min_sizes(eclasses: EClassMapping) -> dict[str, float]:
    """Size of the smallest program per e-class (fixpoint; inf on pure cycles)."""
    size = {c: float("inf") for c in eclasses}
    changed = True
    while changed:
        changed = False
        for c, enodes in eclasses.items():
            best = min(1 + sum(size[ch] for ch in e.children) for e in enodes)
            if best < size[c]:
                size[c], changed = best, True
    return size


SPELLING = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/", "Neg": "-", "Sqrt": "sqrt"}


def reachable(root: str, eclasses: EClassMapping) -> list[str]:
    """E-classes reachable from root, in BFS order."""
    order, queue = [], [root]
    seen = {root}
    while queue:
        eclass = queue.pop(0)
        order.append(eclass)
        for enode in sorted(eclasses[eclass]):
            if enode.op in ("Var", "Num"):
                continue
            for child in enode.children:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
    return order


def intersect(root: str, eclasses: EClassMapping) -> str:
    """GBNF grammar: one nonterminal per e-class, one production per e-node."""
    order = reachable(root, eclasses)
    name = {eclass: f"e{i}" for i, eclass in enumerate(order)}

    def leaf(eclass: str) -> str:
        (terminal,) = {enode.op for enode in eclasses[eclass]}
        return terminal

    # e.g. Add(e1, e2) -> "(+ " e1 " " e2 ")"
    def production(enode: ENode) -> str:
        if enode.op in ("Var", "Num"):
            return f'"{leaf(enode.children[0])}"'
        spelled = SPELLING[enode.op]
        inner = ' " " '.join(name[child] for child in enode.children)
        return f'"({spelled} " {inner} ")"'

    variables = sorted(
        {leaf(e.children[0]) for ec in order for e in eclasses[ec] if e.op == "Var"}
    )
    lines = [f'root ::= "(FPCore ({" ".join(variables)}) " {name[root]} ")"']
    for eclass in order:
        productions = sorted({production(enode) for enode in eclasses[eclass]})
        lines.append(f"{name[eclass]} ::= {' | '.join(productions)}")
    return "\n".join(lines) + "\n"


def _build(benchmark: str, box, runs: int) -> str:
    seeds = interval_seeds(box) if box else ""
    root, eclasses = extract(saturate(benchmarks.read_source(benchmark), runs, seeds))
    root, eclasses = strip_identity_enodes(root, eclasses)
    return intersect(root, eclasses)


def _build_worker(benchmark, box, runs, q) -> None:
    try:
        q.put(_build(benchmark, box, runs))
    except Exception:
        q.put(None)


def build(benchmark: str, box: dict[str, tuple[float, float]] | None = None,
          runs: int = SATURATION_RUNS) -> str:
    """Grammar of programs equivalent to the reference over `box`; `box=None` seeds
    no intervals (whole-domain equivalences only). Compiled in a budget-killed
    subprocess -- a single saturation round can cost minutes (or panic egglog) on
    division-heavy benchmarks -- retrying with fewer rounds until one fits."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    for r in dict.fromkeys((runs, runs - 2, 3, 2, 1)):
        if not 1 <= r <= runs:
            continue
        q = ctx.Queue()
        p = ctx.Process(target=_build_worker, args=(benchmark, box, r, q))
        p.start()
        try:
            # get() before join(): a grammar bigger than the pipe buffer would
            # block the child's queue feeder until the parent reads it
            g = q.get(timeout=2 * SATURATION_BUDGET)
        except Exception:
            g = None
        if p.is_alive():
            p.terminate()
        p.join()
        if g is not None:
            return g
    raise RuntimeError(f"grammar compilation failed for {benchmark!r}")


def rules(benchmark: str, box: dict[str, tuple[float, float]] | None = None,
          runs: int = SATURATION_RUNS) -> str:
    """The e-class productions (root `e0`) without the FPCore-wrapper `start` rule."""
    return "\n".join(build(benchmark, box, runs).strip().split("\n")[1:])
