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

## Benchmark suite (FPBench + Herbie)

FPCore is the common ground with [Herbie](https://herbie.uwplse.org) and
[FPBench](https://fpbench.org): all three tools consume it. The suite lives under
`benchmarks/`:

```
benchmarks/
  egglog/          one <name>.egglog per benchmark: the reference term the tool rewrites
  intervals.json   per-benchmark input box {var: "[lo,hi]"}, loaded into INTERVALS
  sources/
    fpbench/       raw FPBench dump the original 33 were filtered from
    herbie/        vendored Herbie bench/ tree (HERBIE_COMMIT.txt pins the revision)
```

Every benchmark is filtered to *this* tool's subset — only `+ - * / sqrt`, integer
literals, and a branch-free reference (`let`/`let*` inlined, variadic operators folded
left). The `Num i64` e-graph rules out transcendentals, loops, arrays, and non-integer
constants.

- **33 FPBench cores** (`delta`, `kepler*`, `rigidbody*`, `nmse_*`, …): the FPBench suite
  filtered to the subset, with trivially-accurate and duplicate cores dropped. Their
  boxes were read from each core's `:pre` (hand-refined; a wide default where `:pre` left
  a variable unbounded).
- **14 Herbie cores** imported by [scripts/import_fpcore.py](scripts/import_fpcore.py)
  from the vendored `sources/herbie/` tree. Of Herbie's 730 cores, 312 are
  subset-expressible; after dropping the 200+ auto-extracted `haskell.fpcore` entries
  (no `:pre`), cores already covered by the FPBench 33, and — crucially — cores where no
  subset-expressible rewrite lowers the *worst-case* bound this tool measures (their real
  fix is average-case, or needs overflow-scaling / `fma` the subset can't express), 14
  genuinely-improvable programs remain: catastrophic cancellation (`kahan_p9`,
  `conte_x_minus_sqrt`, `expand_square`, `complex_square_real`, `som_setup_w`),
  near-pole / difference-of-ratios (`conte_near_pole`, `asymptote_c`), summation order
  (`martel_p6`), factoring (`fastmath_dist4`), and more. Each was verified to admit a
  lower-error equivalent under the FPTaylor harness before inclusion. The `SPEC` list in
  `import_fpcore.py` records each import's source core, per-variable box provenance
  (`pre` / `pre-hi` / `pre-lo` from the source `:pre`, or `cur` curated), and a note.

To (re)generate or extend the Herbie set, edit the `SPEC` list in `import_fpcore.py` and
run `uv run scripts/import_fpcore.py` (idempotent; `--list` prints every
subset-expressible source core). Run one benchmark (`uv run src/run.py kepler0`), all of
them (`uv run src/run.py`), or one shard per GPU ([scripts/run_gpus.sh](scripts/run_gpus.sh)).

Note the metric mismatch: this tool reports *sound worst-case* FPTaylor bounds over the
box, whereas Herbie reports *average-case* bits/ULP error over sampled points — a
like-for-like comparison means bounding Herbie's output with the same harness.

## Requirements

- casa (torch, transformers, llguidance, xgrammar) + egglog.
- FPTaylor: the `fptaylor` binary on PATH with its opam environment active
  (`eval $(opam env)`; otherwise its native libs `dllnums.so` / `interval.cmi` fail to
  load and every bound comes back unbounded), and a per-benchmark box in `INTERVALS`
  ([fptaylor_check.py](src/fptaylor_check.py)) — which also seeds the interval analysis.
