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
uv run src/run.py x_by_xy          # one benchmark (default model: openai/gpt-oss-120b)
uv run src/run.py                  # every benchmark, model loaded once
uv run src/run.py --summary-only   # re-print the results table
uv run src/fptaylor_check.py x_by_xy   # bound a run's programs (also runs after run.py)
```

`run.py` writes `equivalents/<name>-NNN.json` (proven programs plus an `attempts` log of
every candidate the model tried and why each was rejected), `fptaylor_check.py` writes
`fptaylor/<name>-NNN.json`. Flags: `--samples --max-attempts --temperature --model
--effort --saturation --time-budget --shard`.

## FPBench (comparing against Herbie)

FPCore is the common ground with [Herbie](https://herbie.uwplse.org): both tools consume
it. The benchmarks in `benchmarks/*.egglog` are the [FPBench](https://fpbench.org) suite
(vendored raw under `benchmarks/fpbench/`) filtered to the cores expressible in *this*
tool's subset — only `+ - * / sqrt`, integer literals, and a branch-free reference
(`let`/`let*` inlined). 57 of the 130 cores survived that filter, and trivially-accurate
ones (single ops, plain sums, already-optimal forms like `1/(x+1)`, and `sqrt(x^2+y^2)`
whose only fix is unexpressible scaling) were then dropped, leaving 44. The rest of the
130 need transcendentals, loops, arrays, or non-integer constants the `Num i64` e-graph
can't represent. Each
core's input box (in `benchmarks/fpbench_intervals.json`, loaded into `INTERVALS`) was
read from its `:pre`, with a wide default where `:pre` left a variable unbounded;
`benchmarks/fpbench_manifest.json` records the provenance. Run one benchmark
(`uv run src/run.py kepler0`), all of them (`uv run src/run.py`), or one shard per GPU
across the suite ([scripts/run_gpus.sh](scripts/run_gpus.sh)).

Note the metric mismatch: this tool reports *sound worst-case* FPTaylor bounds over the
box, whereas Herbie reports *average-case* bits/ULP error over sampled points — a
like-for-like comparison means bounding Herbie's output with the same harness.

## Requirements

- casa (torch, transformers, llguidance, xgrammar) + egglog.
- FPTaylor: the `fptaylor` binary on PATH with its opam environment active
  (`eval $(opam env)`; otherwise its native libs `dllnums.so` / `interval.cmi` fail to
  load and every bound comes back unbounded), and a per-benchmark box in `INTERVALS`
  ([fptaylor_check.py](src/fptaylor_check.py)) — which also seeds the interval analysis.
