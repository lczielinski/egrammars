"""Compile an e-graph of equivalent programs into a context-free grammar.

Given a benchmark (a reference program plus egglog rewrite rules), this:
  1. builds the e-graph
  2. intersects it with the simple FPCore syntax grammar

The resulting grammar's language is exactly the set of FPCore programs equivalent
to the reference (under the rules, up to the saturation cap)

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


def saturate(benchmark_source: str) -> EGraph:
    source = (HERE / "rules.egglog").read_text()
    source += benchmark_source
    source += f"\n(run {SATURATION_RUNS})"
    # Mark the benchmark's `start` expression so we can find its e-class later.
    source += f"\n(relation {START} (Math))\n({START} start)"
    egraph = EGraph(record=True)
    egraph.run_program(*egraph.parse_program(source))
    return egraph


def extract(egraph: EGraph) -> tuple[str, dict[str, set[ENode]]]:
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


# --- step 2: the simple grammar ----------------------------------------------

# Concrete FPCore spelling of each operator the rules know about. This *is* the
# simple grammar: expr -> "(+ " expr " " expr ")" | ... | VARIABLE | INTEGER,
# wrapped as "(FPCore (" args ") " expr ")".
SPELLING = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/",  # binary
    "Neg": "-", "Sqrt": "sqrt",                      # unary
}


# --- step 3: intersection -----------------------------------------------------


def reachable(root: str, eclasses: dict[str, set[ENode]]) -> list[str]:
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


def intersect(root: str, eclasses: dict[str, set[ENode]]) -> str:
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
    the lark grammar whose language is exactly the programs equivalent to it."""
    content = (HERE / "benchmarks" / f"{benchmark}.egglog").read_text()
    reference = content.splitlines()[0].removeprefix(";; ")

    root, eclasses = extract(saturate(content))
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
