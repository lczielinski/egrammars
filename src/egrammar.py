"""E-grammar: compile an e-graph of programs equivalent to a benchmark into a lark
grammar.

`build(benchmark, box)` saturates egglog with the interval analysis seeded from the
box, strips identity/cyclic spellings, and emits a grammar whose language is the
programs provably equivalent to the reference over that box. `rules` returns just
the e-class productions for splicing into decoding.py's head/arm grammars.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

import benchmarks
import prove

SATURATION_RUNS = 6
NODE_CAP = 100_000
MAX_SPELLINGS = 8
START = "__start__"

# All rules unified, minus the block rules.egglog marks grammar-excluded (recursive
# rules that are safe under the prover's throttling but flood a full saturation).
GRAMMAR_RULES = re.sub(r";; BEGIN grammar-excluded.*?;; END grammar-excluded\n", "",
                       prove.RULES_UNIFIED, flags=re.S)


@dataclass(frozen=True, order=True)
class ENode:
    op: str
    children: tuple[str, ...]  # e-class ids


EClassMapping = dict[str, set[ENode]]


def saturate(benchmark_source: str, runs: int = SATURATION_RUNS, seeds: str = "",
             node_cap: int = NODE_CAP):
    """Saturate with all rules fired together, one step at a time, to a fixpoint,
    `runs`, or `node_cap`. Extraction wants breadth (every equivalent spelling),
    unlike prove's throttled proving schedule."""
    from egglog.bindings import EGraph

    source = GRAMMAR_RULES + benchmark_source + seeds
    source += f"\n(relation {START} (Math))\n({START} start)"  # mark the root e-class
    egraph = EGraph()
    with prove.quiet_stderr():
        egraph.run_program(*egraph.parse_program(source))
        for _ in range(runs):
            if prove.node_count(egraph) > node_cap:
                break
            if not egraph.run_program(*egraph.parse_program("(run 1)"))[0].report.updated:
                break  # saturated
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

    # Keep an e-node only if it is a shortest spelling or doesn't recurse into its own cycle.
    min_depth = {eclass: float("inf") for eclass in stripped}

    def depth(enode: ENode) -> float:
        return 1 + max((min_depth[c] for c in enode.children), default=0)

    changed = True
    while changed:
        changed = False
        for eclass, enodes in stripped.items():
            if (best := min(depth(e) for e in enodes)) < min_depth[eclass]:
                min_depth[eclass], changed = best, True

    scc = _strongly_connected_components(stripped)
    return root, {
        eclass: {
            e
            for e in enodes
            if depth(e) <= min_depth[eclass]
            or all(scc[c] != scc[eclass] for c in e.children)
        }
        for eclass, enodes in stripped.items()
    }


def _strongly_connected_components(eclasses: EClassMapping) -> dict[str, str]:
    """Tarjan's SCC: eclass -> a shared id per cycle."""
    order: dict[str, int] = {}
    low: dict[str, int] = {}
    scc: dict[str, str] = {}
    stack: list[str] = []
    on_stack: set[str] = set()

    def connect(node: str) -> None:
        order[node] = low[node] = len(order)
        stack.append(node)
        on_stack.add(node)
        for child in {c for e in eclasses[node] for c in e.children}:
            if child not in order:
                connect(child)
                low[node] = min(low[node], low[child])
            elif child in on_stack:
                low[node] = min(low[node], order[child])
        if low[node] == order[node]:  # root of an SCC
            while True:
                member = stack.pop()
                on_stack.discard(member)
                scc[member] = node
                if member == node:
                    break

    for eclass in eclasses:
        if eclass not in order:
            connect(eclass)
    return scc


def limit_spellings(eclasses: EClassMapping, k: int = MAX_SPELLINGS) -> EClassMapping:
    """Keep the k smallest-completion spellings per e-class. Program size multiplies
    across classes, so uncapped spelling counts make the language monster-heavy and
    the sampler wanders; the cap keeps rewrite diversity but bounds the depth."""
    size = {c: float("inf") for c in eclasses}

    def nsize(e: ENode) -> float:
        return 1 + sum(size[c] for c in e.children)

    changed = True
    while changed:
        changed = False
        for c, enodes in eclasses.items():
            if (best := min(nsize(e) for e in enodes)) < size[c]:
                size[c], changed = best, True

    return {c: set(sorted(enodes, key=lambda e: (nsize(e), e))[:k])
            for c, enodes in eclasses.items()}


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
    """Lark grammar: one nonterminal per e-class, one production per e-node."""
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
    lines = [f'start: "(FPCore ({" ".join(variables)}) " {name[root]} ")"']
    for eclass in order:
        productions = sorted({production(enode) for enode in eclasses[eclass]})
        lines.append(f"{name[eclass]}: {' | '.join(productions)}")
    return "\n".join(lines) + "\n"


def build(benchmark: str, box: dict[str, tuple[float, float]] | None = None,
          runs: int = SATURATION_RUNS) -> str:
    """Grammar of programs equivalent to the reference over `box`; `box=None` seeds
    no intervals (whole-domain equivalences only)."""
    seeds = prove.seeds(box) if box else ""
    root, eclasses = extract(saturate(benchmarks.read_source(benchmark), runs, seeds))
    root, eclasses = strip_identity_enodes(root, eclasses)
    return intersect(root, limit_spellings(eclasses))


def rules(benchmark: str, box: dict[str, tuple[float, float]] | None = None,
          runs: int = SATURATION_RUNS) -> str:
    """The e-class productions (root `e0`) without the FPCore-wrapper `start` rule."""
    return "\n".join(build(benchmark, box, runs).strip().split("\n")[1:])
