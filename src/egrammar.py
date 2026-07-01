"""Compile an e-graph of equivalent programs into a context-free grammar.

Given a benchmark (a reference program plus egglog rewrite rules), this:
  1. builds the e-graph
  2. removes identity padding and non-minimal cyclic spellings
  3. intersects it with the simple FPCore syntax grammar

The resulting grammar's language is a cleaned subset of FPCore programs equivalent
to the reference (under the rules, up to the saturation cap)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import paths

SATURATION_RUNS = 6
START = "__start__"


@dataclass(frozen=True, order=True)
class ENode:
    op: str
    children: tuple[str, ...]  # e-class ids


EClassMapping = dict[str, set[ENode]]


def saturate(benchmark_source: str, runs: int = SATURATION_RUNS) -> "EGraph":
    from egglog.bindings import EGraph

    source = (paths.ROOT / "rules.egglog").read_text()
    source += benchmark_source
    source += f"\n(run {runs})"
    # Mark `start` so we can find its e-class after saturation.
    source += f"\n(relation {START} (Math))\n({START} start)"
    egraph = EGraph(record=True)
    egraph.run_program(*egraph.parse_program(source))
    return egraph


def extract(egraph: EGraph) -> tuple[str, EClassMapping]:
    """Read the e-graph back out: the root e-class and eclass -> set of e-nodes."""
    nodes = json.loads(egraph.serialize([]).to_json())["nodes"]
    root = None
    eclasses: defaultdict[str, set[ENode]] = defaultdict(set)
    for node in nodes.values():
        op = node["op"].strip('"')  # string leaves are serialized with quotes
        children = tuple(nodes[child]["eclass"] for child in node["children"])
        if op == START:
            root = children[0]
        else:
            eclasses[node["eclass"]].add(ENode(op, children))
    assert root is not None, "benchmark must define a `start` expression"
    return root, dict(eclasses)


def _zero_one_classes(eclasses: EClassMapping) -> tuple[set[str], set[str]]:
    """The e-classes that provably denote 0 and 1, by fixpoint over the operators."""

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


def strip_identity_enodes(
    root: str, eclasses: EClassMapping
) -> tuple[str, EClassMapping]:
    zero, one = _zero_one_classes(eclasses)

    # Alias identity-only classes into the operand they equal.
    parent = {eclass: eclass for eclass in eclasses}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def reduces_to(enode: ENode) -> str | None:
        """The operand this enode equals, or None if it is not an identity.
        Note `(- 0 b)` is `-b`, not an identity, so it is excluded."""
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

    # Drop identity *spellings* where a real spelling remains.
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

    # Prune non-minimal cyclic spellings: keep an e-node only if it is a shortest
    # spelling, or does not recurse into its own cycle.
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
    """Tarjan's SCC: eclass -> a shared id for the classes in each cycle. The
    e-graph is wide but shallow, so plain recursion stays well under the limit."""
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


# FPCore spelling of each operator — the "simple grammar":
# expr -> "(+ " expr " " expr ")" | ... | VARIABLE | INTEGER.
SPELLING = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/",
    "Neg": "-", "Sqrt": "sqrt",
}


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


def _branching_header(variables: list[str], root_name: str) -> list[str]:
    cmps = ("<", ">", "<=", ">=")
    cond = " | ".join(f'"({op} " operand " " operand ")"' for op in cmps)
    operand = " | ".join([f'"{v}"' for v in variables] + ["NUMBER"])
    return [
        f'start: "(FPCore ({" ".join(variables)}) " body ")"',
        f'body: {root_name} | "(if " cond " " body " " body ")"',
        f"cond: {cond}",
        f"operand: {operand}",
        r'NUMBER: /-?[0-9]+(\.[0-9]+)?/',
    ]


def intersect(root: str, eclasses: EClassMapping, branching: bool = False) -> str:
    """The FPCore syntax grammar restricted to the e-graph, as a lark grammar:
    one nonterminal per e-class, one production per e-node."""
    order = reachable(root, eclasses)
    name = {eclass: f"e{i}" for i, eclass in enumerate(order)}

    def leaf(eclass: str) -> str:
        (terminal,) = {enode.op for enode in eclasses[eclass]}
        return terminal

    # e.g. Add(e1, e2) --> "(+ " e1 " " e2 ")"
    def production(enode: ENode) -> str:
        if enode.op in ("Var", "Num"):
            return f'"{leaf(enode.children[0])}"'
        spelled = SPELLING[enode.op]
        inner = ' " " '.join(name[child] for child in enode.children)
        return f'"({spelled} " {inner} ")"'

    variables = sorted(
        {leaf(e.children[0]) for ec in order for e in eclasses[ec] if e.op == "Var"}
    )
    if branching:
        lines = _branching_header(variables, name[root])
    else:
        lines = [f'start: "(FPCore ({" ".join(variables)}) " {name[root]} ")"']
    for eclass in order:
        productions = sorted({production(enode) for enode in eclasses[eclass]})
        lines.append(f"{name[eclass]}: {' | '.join(productions)}")
    return "\n".join(lines) + "\n"


def read_reference(benchmark: str) -> str:
    content = (paths.BENCHMARKS / f"{benchmark}.egglog").read_text()
    return content.splitlines()[0].removeprefix(";; ")


def build(benchmark: str, runs: int = SATURATION_RUNS,
          branching: bool = False) -> tuple[str, str]:
    content = (paths.BENCHMARKS / f"{benchmark}.egglog").read_text()
    reference = read_reference(benchmark)

    root, eclasses = extract(saturate(content, runs))
    root, eclasses = strip_identity_enodes(root, eclasses)
    grammar = intersect(root, eclasses, branching)
    return reference, grammar


def write_grammar(benchmark: str, grammar: str) -> Path:
    paths.LARK.mkdir(exist_ok=True)
    grammar_path = paths.LARK / f"{benchmark}.lark"
    grammar_path.write_text(grammar)
    return grammar_path
