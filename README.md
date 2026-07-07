# egrammars

Synthesize numerically-accurate branch-on-the-input programs by grammar-constrained
decoding over an e-graph of equivalent forms
([casa](https://github.com/large-loris-models/casa) + egglog).

## Pipeline

Per "Combining E-Graphs with Abstract Interpretation". [run.py](src/run.py) generates
in one of two `--mode`s; both then check with FPTaylor.

**`check` (default) — generate freely, verify after.** The model writes a whole
program in one pass under a light, static *syntax* grammar (well-formed FPCore, allowed
ops/vars, threshold or var-vs-var conditions — [run.py](src/run.py) `syntax_grammar`),
branching however it likes with an **arbitrary** threshold. Validity is checked
afterwards: [regions.py](src/regions.py) splits the `if`-tree, each branch's guards
narrow the box, and `egrammar.equivalent` ([egrammar.py](src/egrammar.py)) adds the arm
to the reference's e-graph and proves them equal *over that sub-box* — the interval
analysis ([rules.egglog](rules.egglog)), seeded from the narrowed box, bridges them only
where it proves the domain-conditional preconditions. A program with any non-equivalent
branch is dropped. Per-branch validity implies whole-program validity because `if` is
total. Semantics live in the checker; the grammar only enforces syntax.

**`skeleton` — a few skeletons, options per branch, assembled.** In one ASAP call the
model (constrained to a skeleton grammar) proposes `--skeletons` distinct `if`-trees
with `?` arm holes. For each hole, its guards narrow the box, egglog builds a grammar
*sound over that sub-box*, and one ASAP call draws `--arms` distinct fills from it
(region grammars cached across skeletons). Each skeleton is then assembled as the
cross-product of its holes' options, capped at `--max-combos` programs. Sound by
construction — no post-hoc check needed. Drawing skeletons and arms as batched ASAP
calls (rather than one `n_samples=1` call each) is what gives it dedup + reweighting
instead of degrading to GCD.

**Check** ([fptaylor_check.py](src/fptaylor_check.py)): FPTaylor bounds each branch over
the sub-interval where it applies (same `if`-tree split as above).

## Usage

```bash
uv run src/run.py sqrtminus --model openai/gpt-oss-120b   # check mode (default)
uv run src/run.py sqrtminus --mode skeleton               # constrained skeleton + holes
uv run src/fptaylor_check.py quadratic          # also runs automatically after run.py
```

`run.py` writes `equivalents/<name>-NNN.json`, `fptaylor_check.py` writes
`fptaylor/<name>-NNN.json`. Flags: `--mode --samples` (check) `--skeletons --arms
--max-combos` (skeleton) `--max-attempts --temperature --model --effort --saturation`.

## Requirements

- casa (torch, transformers, llguidance, xgrammar) + egglog.
- FPTaylor: the `fptaylor` binary on PATH, `$FPTAYLOR_BASE` set, and a per-benchmark box
  in `INTERVALS` ([fptaylor_check.py](src/fptaylor_check.py)) — which also seeds the
  interval analysis.
