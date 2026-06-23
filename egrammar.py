"""Compile an e-graph of equivalent programs into a context-free grammar.

Given a benchmark (a reference program plus egglog rewrite rules), this:
  1. builds the e-graph
  2. removes identity padding and non-minimal cyclic spellings
  3. intersects it with the simple FPCore syntax grammar

The resulting grammar's language is a cleaned subset of FPCore programs equivalent
to the reference (under the rules, up to the saturation cap).

Usage:
    uv run egrammar.py quadratic            # writes out/quadratic.lark + .txt
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from egglog.bindings import EGraph

HERE = Path(__file__).resolve().parent
SATURATION_RUNS = 6
START = "__start__"


# --- step 1: build the e-graph ----------------------------------------------


@dataclass(frozen=True, order=True)
class ENode:
    op: str
    children: tuple[str, ...]  # e-class ids


EClassMapping = dict[str, set[ENode]]


def saturate(benchmark_source: str) -> EGraph:
    source = (HERE / "rules.egglog").read_text()
    source += benchmark_source
    source += f"\n(run {SATURATION_RUNS})"
    # Mark the benchmark's `start` expression so we can find its e-class later.
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


# --- step 2: clean the e-graph ----------------------------------------------


def strip_identity_enodes(eclasses: EClassMapping) -> EClassMapping:
    """Remove identity-padding enodes and non-minimal cyclic spellings."""

    def has_num(eclass: str, literal: str) -> bool:
        return any(
            enode.op == "Num"
            and any(
                leaf.op == literal and not leaf.children
                for leaf in eclasses.get(enode.children[0], ())
            )
            for enode in eclasses.get(eclass, ())
            if enode.children
        )

    zeroish = {eclass for eclass in eclasses if has_num(eclass, "0")}
    oneish = {eclass for eclass in eclasses if has_num(eclass, "1")}

    def denotes_zero(enode: ENode) -> bool:
        match enode.op, enode.children:
            case ("Neg" | "Sqrt", (a,)):
                return a in zeroish
            case ("Mul", (a, b)):
                return a in zeroish or b in zeroish
            case ("Add", (a, b)):
                return a in zeroish and b in zeroish
            case ("Sub", (a, b)):
                return a == b or (a in zeroish and b in zeroish)
            case ("Div", (a, _)):
                return a in zeroish
        return False

    def denotes_one(enode: ENode) -> bool:
        match enode.op, enode.children:
            case ("Sqrt", (a,)):
                return a in oneish
            case ("Mul", (a, b)):
                return a in oneish and b in oneish
            case ("Div", (a, b)):
                return a == b or (a in oneish and b in oneish)
        return False

    changed = True
    while changed:
        changed = False
        for eclass, enodes in eclasses.items():
            if eclass not in zeroish and any(denotes_zero(e) for e in enodes):
                zeroish.add(eclass)
                changed = True
            if eclass not in oneish and any(denotes_one(e) for e in enodes):
                oneish.add(eclass)
                changed = True

    def is_identity(enode: ENode) -> bool:
        if len(enode.children) != 2:
            return False
        a, b = enode.children
        match enode.op:
            case "Mul":
                return a in oneish or b in oneish or a in zeroish or b in zeroish
            case "Add":
                return a in zeroish or b in zeroish
            case "Sub":
                return a == b or a in zeroish or b in zeroish
            case "Div":
                return a == b or b in oneish or a in zeroish
        return False

    stripped: EClassMapping = {}
    for eclass, enodes in eclasses.items():
        non_identity = {enode for enode in enodes if not is_identity(enode)}
        stripped[eclass] = non_identity or set(enodes)

    min_depth = {eclass: float("inf") for eclass in stripped}

    def enode_depth(enode: ENode) -> float:
        return 1 + max(
            (min_depth.get(child, float("inf")) for child in enode.children),
            default=0,
        )

    changed = True
    while changed:
        changed = False
        for eclass, enodes in stripped.items():
            best = min((enode_depth(e) for e in enodes), default=float("inf"))
            if best < min_depth[eclass]:
                min_depth[eclass] = best
                changed = True

    graph: dict[str, set[str]] = {eclass: set() for eclass in stripped}
    for eclass, enodes in stripped.items():
        for enode in enodes:
            graph[eclass].update(enode.children)
            for child in enode.children:
                graph.setdefault(child, set())

    scc_of: dict[str, int] = {}
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()

    def connect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for child in graph[node]:
            if child not in indices:
                connect(child)
                lowlinks[node] = min(lowlinks[node], lowlinks[child])
            elif child in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[child])

        if lowlinks[node] == indices[node]:
            while True:
                member = stack.pop()
                on_stack.remove(member)
                scc_of[member] = lowlinks[node]
                if member == node:
                    break

    for eclass in graph:
        if eclass not in indices:
            connect(eclass)

    return {
        eclass: {
            enode
            for enode in enodes
            if enode_depth(enode) <= min_depth[eclass]
            or all(scc_of.get(child) != scc_of[eclass] for child in enode.children)
        }
        for eclass, enodes in stripped.items()
    }


# --- step 3: the simple grammar ----------------------------------------------

# Concrete FPCore spelling of each operator the rules know about. This *is* the
# simple grammar: expr -> "(+ " expr " " expr ")" | ... | VARIABLE | INTEGER,
# wrapped as "(FPCore (" args ") " expr ")".
SPELLING = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/",  # binary
    "Neg": "-", "Sqrt": "sqrt",                      # unary
}


# --- step 4: intersection -----------------------------------------------------


def reachable(root: str, eclasses: EClassMapping) -> list[str]:
    """E-classes reachable from the root, in BFS order (= grammar rule order).

    Var/Num children are string/number leaves that get inlined into their
    production, so they are not visited (they would be useless rules).
    """
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
    """The FPCore syntax grammar restricted to the e-graph, as a lark grammar.

    The syntax grammar's one `expr` nonterminal splits into one nonterminal per
    e-class; each e-node becomes the production spelling it. Whitespace is fixed
    to one canonical style (single spaces) so the grammar is plain string
    literals — same language up to formatting, much simpler.
    """
    order = reachable(root, eclasses)
    name = {eclass: f"e{i}" for i, eclass in enumerate(order)}

    def leaf(eclass: str) -> str:
        """The variable name or integer literal stored in a Var/Num child class."""
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
    lines = [f'start: "(FPCore ({" ".join(variables)}) " {name[root]} ")" "\\n"?']
    for eclass in order:
        productions = sorted({production(enode) for enode in eclasses[eclass]})
        lines.append(f"{name[eclass]}: {' | '.join(productions)}")
    return "\n".join(lines) + "\n"


# --- entry point ---------------------------------------------------------------


def build(benchmark: str) -> tuple[str, str]:
    """Compile a benchmark into (reference, grammar): the original program text and
    a lark grammar of cleaned equivalent programs."""
    content = (HERE / "benchmarks" / f"{benchmark}.egglog").read_text()
    reference = content.splitlines()[0].removeprefix(";; ")

    root, eclasses = extract(saturate(content))
    eclasses = strip_identity_enodes(eclasses)
    grammar = intersect(root, eclasses)
    return reference, grammar


def make_prompt(reference: str) -> str:
    """The user prompt handed to the sampler alongside the grammar."""
    return (
        (HERE / "prompt_header.md").read_text()
        + f"\n\nThe original program is:\n{reference}\n\n"
        "Produce one complete FPCore program that is algebraically equivalent to "
        "the original but evaluates with different floating-point behavior."
    )


def write_artifacts(
    benchmark: str, grammar: str, prompt: str, out_dir: Path
) -> tuple[Path, Path]:
    """Write the grammar (.lark) and prompt (.txt), returning their paths."""
    out_dir.mkdir(exist_ok=True)
    grammar_path = out_dir / f"{benchmark}.lark"
    prompt_path = out_dir / f"{benchmark}.txt"
    grammar_path.write_text(grammar)
    prompt_path.write_text(prompt)
    return grammar_path, prompt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="benchmark name, e.g. quadratic")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()

    reference, grammar = build(args.benchmark)
    prompt = make_prompt(reference)
    grammar_path, prompt_path = write_artifacts(
        args.benchmark, grammar, prompt, args.out
    )

    print(f"reference: {reference}")
    print(f"grammar:   {grammar_path} ({grammar.count(chr(10))} rules)")
    print(f"prompt:    {prompt_path}")


if __name__ == "__main__":
    main()
