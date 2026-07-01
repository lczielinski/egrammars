# egrammars

Compile an e-graph of equivalent programs into a context-free grammar, then sample from
it with grammar-constrained decoding via [casa](https://github.com/large-loris-models/casa).

## Pipeline

1. **Build & clean the e-graph** ([egrammar.py](src/egrammar.py)): run egglog on the
   benchmark plus the rewrite rules ([rules.egglog](rules.egglog)), then strip identity
   padding and non-minimal cyclic spellings.
2. **Intersect into a grammar**: one nonterminal per e-class, one production per e-node,
   spelled in FPCore syntax — its language is the cleaned set of equivalent programs.
3. **Sample** ([run.py](src/run.py)): the model reasons about where the original loses
   accuracy (cancellation, overflow), then a grammar-constrained pass writes one program
   that branches with `if` into the form accurate in each region.
4. **Check** ([fptaylor_check.py](src/fptaylor_check.py)): bound each program's rounding
   error with FPTaylor, splitting `if` branches over the sub-interval where each applies.

## Usage

```bash
uv run src/run.py quadratic --samples 50 --model openai/gpt-oss-120b
uv run src/fptaylor_check.py quadratic                    # runs automatically after run.py
```

`run.py` compiles the grammar on demand (cached in `lark/`) and writes
`equivalents/<benchmark>-NNN.json`; `fptaylor_check.py` writes
`fptaylor/<benchmark>-NNN.json`. Flags: `--samples`, `--max-attempts`, `--saturation`,
`--model`, `--effort` (see the [run.py](src/run.py) docstring).

## Requirements

- casa pulls in the sampling runtime (torch, transformers, llguidance, xgrammar).
- egglog is needed only to compile a grammar not already cached in `lark/`.
- FPTaylor needs the `fptaylor` binary on PATH and `$FPTAYLOR_BASE` set; a per-benchmark
  interval box in `INTERVALS` ([fptaylor_check.py](src/fptaylor_check.py)).
