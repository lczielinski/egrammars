"""Compile an e-graph of programs equivalent to a benchmark into a lark grammar.

build_region(benchmark, box) saturates egglog (rules.egglog, with the interval
analysis seeded from the box), strips identity/cyclic spellings, and intersects the
result into a grammar whose language is the equivalent programs over that box."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import paths

SATURATION_RUNS = 6
START = "__start__"


@contextlib.contextmanager
def _quiet_stderr():
    """Drop egglog's Rust-side stderr (the `$`-prefix lint); errors still raise."""
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


@dataclass(frozen=True, order=True)
class ENode:
    op: str
    children: tuple[str, ...]  # e-class ids


EClassMapping = dict[str, set[ENode]]


def saturate(benchmark_source: str, runs: int = SATURATION_RUNS,
             seeds: str = "") -> "EGraph":
    from egglog.bindings import EGraph

    source = (paths.ROOT / "rules.egglog").read_text()  # rules + interval analysis
    source += benchmark_source
    source += seeds             # (set (lo x) ..) (set (hi x) ..) from the region box
    source += f"\n(run {runs})"
    # Mark `start` so we can find its e-class after saturation.
    source += f"\n(relation {START} (Math))\n({START} start)"
    egraph = EGraph(record=True)
    with _quiet_stderr():
        egraph.run_program(*egraph.parse_program(source))
    return egraph


CONSTRUCTORS = frozenset(
    {"Num", "Var", "Add", "Sub", "Neg", "Sqrt", "Mul", "Div"})


def extract(egraph: EGraph) -> tuple[str, EClassMapping]:
    """The root e-class and eclass -> set of e-nodes."""
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
            continue                 # analysis (f64/lo/hi) node -- skip
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
        """The operand this enode equals, or None if it is not an identity."""
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
    """Tarjan's SCC: eclass -> a shared id for the classes in each cycle."""
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


# FPCore spelling of each operator.
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


def intersect(root: str, eclasses: EClassMapping) -> str:
    """Lark grammar of the e-graph: one nonterminal per e-class, one production per
    e-node. Its language is exactly the programs equivalent to the reference over the
    box the e-graph was saturated with -- used by skeleton mode to fill an arm."""
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
    lines = [f'start: "(FPCore ({" ".join(variables)}) " {name[root]} ")"']
    for eclass in order:
        productions = sorted({production(enode) for enode in eclasses[eclass]})
        lines.append(f"{name[eclass]}: {' | '.join(productions)}")
    return "\n".join(lines) + "\n"


def read_reference(benchmark: str) -> str:
    content = (paths.BENCHMARKS / f"{benchmark}.egglog").read_text()
    return content.splitlines()[0].removeprefix(";; ")


def _seeds(box: dict[str, tuple[float, float]]) -> str:
    return "".join(f"(set (lo {v}) {lo})(set (hi {v}) {hi})\n"
                   for v, (lo, hi) in box.items())


def build_region(benchmark: str, box: dict[str, tuple[float, float]] | None = None,
                 runs: int = SATURATION_RUNS) -> tuple[str, str]:
    """Grammar of programs equivalent to the reference over `box` (var -> (lo, hi))."""
    content = (paths.BENCHMARKS / f"{benchmark}.egglog").read_text()
    reference = read_reference(benchmark)

    seeds = _seeds(box) if box else ""
    root, eclasses = extract(saturate(content, runs, seeds))
    root, eclasses = strip_identity_enodes(root, eclasses)
    return reference, intersect(root, eclasses)


# FPCore head -> Math constructor, for the equivalence checker.
CONSTRUCTOR = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div", "sqrt": "Sqrt"}


def fpcore_to_math(node) -> str:
    """Render a branch-free FPCore AST (from `regions.parse`) as an egglog `Math` term.
    Raises ValueError on a non-integer literal or unsupported form (`Num` is i64)."""
    if isinstance(node, str):
        try:
            value = float(node)
        except ValueError:
            return f'(Var "{node}")'  # not a number -> a variable
        if value != int(value):
            raise ValueError(f"non-integer literal {node!r}")
        return f"(Num {int(value)})"
    if not node:
        raise ValueError("empty node")
    head, operands = node[0], node[1:]
    if head == "-" and len(operands) == 1:
        return f"(Neg {fpcore_to_math(operands[0])})"
    ctor = CONSTRUCTOR.get(head)
    arity = 1 if head == "sqrt" else 2
    if ctor is None or len(operands) != arity:
        raise ValueError(f"unsupported form {node!r}")
    return f"({ctor} " + " ".join(fpcore_to_math(o) for o in operands) + ")"


def equivalent(benchmark: str, box: dict[str, tuple[float, float]] | None,
               body, runs: int = SATURATION_RUNS) -> bool:
    """True if the branch-free FPCore AST `body` is provably equal to the benchmark
    reference over `box` (var -> (lo, hi)) under the rewrite rules + interval analysis.

    Adds `body` to the reference's e-graph, saturates with the box seeded into the
    interval analysis, and checks the two land in the same e-class -- so a
    domain-conditional rewrite can only bridge them where its preconditions are proven
    over `box`. A wider box (e.g. an ignored var-vs-var guard) is conservative: it can
    only fail to prove an equivalence, never assert a false one."""
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
    egraph = EGraph()
    try:
        with _quiet_stderr():
            egraph.run_program(*egraph.parse_program(source))
        return True
    except Exception:
        return False
