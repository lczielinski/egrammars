# egrammars

Synthesize numerically-accurate branch-on-the-input programs by grammar-constrained
decoding over an e-graph of equivalent forms
([casa](https://github.com/large-loris-models/casa) + egglog).

## Pipeline

Two phases (per "Combining E-Graphs with Abstract Interpretation"), so each branch is
equivalent to the original over the region it governs:

1. **Partition** ([run.py](src/run.py)): the model, constrained to a skeleton grammar
   ([regions.py](src/regions.py)), emits an `if`-tree over the input range with `?` arm
   holes — splitting only where the accurate form must change.
2. **Fill** ([egrammar.py](src/egrammar.py)): each hole's guards narrow the box; egglog
   builds a grammar sound over that sub-box. The interval analysis in
   [rules.egglog](rules.egglog), seeded from the box, re-enables the domain-conditional
   rewrites only where it proves their preconditions (so e.g. `sqrt(x²+1)+x`, valid only
   for one sign of `x`, can't appear). A constrained pass fills each arm.
3. **Check** ([fptaylor_check.py](src/fptaylor_check.py)): FPTaylor bounds each branch
   over its sub-interval.

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
