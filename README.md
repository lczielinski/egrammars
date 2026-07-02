# egrammars

Synthesize numerically-accurate branch-on-the-input programs by grammar-constrained
decoding over an e-graph of equivalent forms
([casa](https://github.com/large-loris-models/casa) + egglog).

## Pipeline

Per "Combining E-Graphs with Abstract Interpretation", with each branch's grammar built
on the fly:

1. **Generate** ([run.py](src/run.py)): the model emits the program in shared-context
   segments. A head grammar lets it output a whole-box form, or open `(FPCore (v) (if
   (op v <threshold>)` with an **arbitrary** numeric threshold; once the threshold is
   known, each arm is generated under a grammar built on the fly ([egrammar.py](src/egrammar.py))
   over the box narrowed by that condition. Segment boundaries are token-aligned, so
   there's no cross-grammar token straddle.
2. **Sound arms** ([rules.egglog](rules.egglog)): the interval analysis, seeded from the
   arm's narrowed box, re-enables the domain-conditional rewrites only where it proves
   their preconditions — so a form like `sqrt(x²+1)+x` (valid only for one sign of `x`)
   can't appear in the wrong arm.
3. **Check** ([fptaylor_check.py](src/fptaylor_check.py)): FPTaylor bounds each branch
   ([regions.py](src/regions.py) splits the `if`-tree) over the sub-interval where it
   applies.

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
