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
uv run src/run.py x_by_xy --model openai/gpt-oss-120b
uv run src/fptaylor_check.py x_by_xy            # also runs automatically after run.py
```

`run.py` writes `equivalents/<name>-NNN.json`, `fptaylor_check.py` writes
`fptaylor/<name>-NNN.json`. Flags: `--samples --max-attempts --temperature --model
--effort --saturation`.

## FPBench (comparing against Herbie)

FPCore is the common ground with [Herbie](https://herbie.uwplse.org): both tools consume
it. The benchmarks in `benchmarks/*.egglog` are the [FPBench](https://fpbench.org) suite
(vendored raw under `benchmarks/fpbench/`) filtered to the cores expressible in *this*
tool's subset — only `+ - * / sqrt`, integer literals, and a branch-free reference
(`let`/`let*` inlined). 57 of the 130 cores survived; the rest need transcendentals,
loops, arrays, or non-integer constants the `Num i64` e-graph can't represent. Each
core's input box (in `benchmarks/fpbench_intervals.json`, loaded into `INTERVALS`) was
read from its `:pre`, with a wide default where `:pre` left a variable unbounded;
`benchmarks/fpbench_manifest.json` records the provenance. Run one like any other
benchmark (`uv run src/run.py kepler0`), or the whole suite with
[run_suite.py](src/run_suite.py).

Note the metric mismatch: this tool reports *sound worst-case* FPTaylor bounds over the
box, whereas Herbie reports *average-case* bits/ULP error over sampled points — a
like-for-like comparison means bounding Herbie's output with the same harness.

## Requirements

- casa (torch, transformers, llguidance, xgrammar) + egglog.
- FPTaylor: the `fptaylor` binary on PATH with its opam environment active
  (`eval $(opam env)`; otherwise its native libs `dllnums.so` / `interval.cmi` fail to
  load and every bound comes back unbounded), and a per-benchmark box in `INTERVALS`
  ([fptaylor_check.py](src/fptaylor_check.py)) — which also seeds the interval analysis.
