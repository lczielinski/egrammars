# egrammars

Synthesize numerically-accurate branch-on-the-input programs by grammar-constrained
decoding over an e-graph of equivalent forms
([casa](https://github.com/large-loris-models/casa) + egglog).

## Pipeline

1. **Generate** ([run.py](src/run.py)): the model reasons once, then draws
   `--samples` distinct programs in one ASAP call under a light static *syntax* grammar, 
   branching however it likes with an **arbitrary** threshold. 
2. **Verify** ([egrammar.py](src/egrammar.py)): [regions.py](src/regions.py) splits the
   `if`-tree; each branch's guards narrow the box, and `egrammar.equivalent` adds the arm
   to the reference's e-graph and proves them equal *over that sub-box* — the interval
   analysis ([rules.egglog](rules.egglog)), seeded from the narrowed box, bridges them
   only where it proves the domain-conditional preconditions. A program with any
   non-equivalent branch is dropped; `if` is total, so per-branch validity implies
   whole-program validity. Semantics live in the checker; the grammar only enforces
   syntax.
3. **Bound** ([fptaylor_check.py](src/fptaylor_check.py)): FPTaylor bounds each branch
   over the sub-interval where it applies.

## Usage

```bash
uv run src/run.py sqrtminus --model openai/gpt-oss-120b
uv run src/fptaylor_check.py quadratic          # also runs automatically after run.py
```

`run.py` writes `equivalents/<name>-NNN.json`, `fptaylor_check.py` writes
`fptaylor/<name>-NNN.json`. Flags: `--samples --max-attempts --temperature --model
--effort --saturation`.

## Requirements

- casa (torch, transformers, llguidance, xgrammar) + egglog.
- FPTaylor: the `fptaylor` binary on PATH, `$FPTAYLOR_BASE` set, and a per-benchmark box
  in `INTERVALS` ([fptaylor_check.py](src/fptaylor_check.py)) — which also seeds the
  interval analysis.
