# egrammars

Synthesize numerically-accurate FPCore rewrites — optionally branching on the input —
with LLM sampling ([casa](https://github.com/large-loris-models/casa)) checked or
constrained by an e-graph (egglog).

## Two decoding modes

**`--decoding light` (default) — generate freely, verify after.** The model reasons
once, then samples `--samples` distinct programs under a light *syntax* grammar
([decoding.py](src/decoding.py)): any well-formed program, branching on any
variable/threshold. Each candidate is then verified ([verify.py](src/verify.py)):
its `if`-tree is split, each branch's guards narrow the input box, and
[prove.py](src/prove.py) proves the branch equal to the reference over that sub-box
(egglog saturation with the interval analysis in [rules.egglog](rules.egglog) seeded
from the box — domain-conditional rewrites fire only where sound). Any non-equivalent
branch drops the program; `if` is total, so per-branch validity implies whole-program
validity. Unproven candidates are classified by sampling: numerically equal (a
missing e-graph rule) vs. genuinely non-equivalent.

**`--decoding egraph` — correct by construction.** The grammar itself is the set of
programs provably equivalent to the reference: [egrammar.py](src/egrammar.py)
saturates the e-graph over the box and compiles it to a CFG (one nonterminal per
e-class). Branching needs the grammar to depend on the model's own output: the head
grammar offers either a whole-box equivalent form or the opening
`(if (op var NUMBER)`; when the model emits a conditional, `DynamicRegionRecognizer`
([decoding.py](src/decoding.py)) reads the threshold, narrows the box by the guard,
rebuilds the e-grammar over each arm's sub-box, and swaps it in mid-decode — to the
model it is one uninterrupted generation.

Both modes then bound each surviving program's worst-case double rounding error with
FPTaylor ([fptaylor_check.py](src/fptaylor_check.py)), per branch over its sub-box.

## Layout

```
src/
  run.py            CLI, per-benchmark pipeline, GPU sharding
  generate.py       prompt, reasoning handoff, ASAP sampling
  decoding.py       syntax/head/arm grammars, mid-decode grammar swapping
  egrammar.py       e-graph -> lark grammar compiler
  prove.py          egglog equivalence prover
  verify.py         per-branch proving + numeric classification of failures
  fptaylor_check.py FPTaylor error bounds (runs automatically after generation)
  summary.py        per-run results table
  benchmarks.py     reference terms and interval boxes
  regions.py        box arithmetic and if-tree splitting
  paths.py          repo layout and per-run result directories
```

## Usage

```bash
uv run src/run.py x_by_xy                     # one benchmark
uv run src/run.py all                         # whole suite, one shard per GPU
uv run src/run.py all --decoding egraph       # constrained decoding
uv run src/run.py --summary-only              # re-print the latest run's table
```

Every invocation writes one self-contained directory,
`results/<timestamp>-<decoding>/` (name it with `--run NAME`):

```
results/2026-07-13-142530-light/
  equivalents/<benchmark>.json   proven programs + every attempt and why rejected
  fptaylor/<benchmark>.json      error bounds (written right after each benchmark)
  summary.md                     aggregate table
  log/gpu<i>.log                 per-shard logs when sharded
```

`all` shards across every visible GPU, all shards writing into the same run
directory; the parent summarizes when they finish.
`--summary-only [--run NAME]` re-summarizes the latest (or named) run. Other flags:
`--samples --temperature --model --saturation --time-budget`.

## Benchmarks (FPBench + Herbie)

`benchmarks/egglog/<name>.egglog` holds each reference term;
`benchmarks/intervals.json` the input box. All are filtered to this tool's subset:
`+ - * / sqrt`, integer literals, branch-free reference (the `Num i64` e-graph rules
out transcendentals and non-integer constants).

- **33 FPBench cores**, boxes from each core's `:pre`.
- **14 Herbie cores**: of 312 subset-expressible cores, those remaining after
  dropping duplicates, cores without a `:pre`, and cores where no subset-expressible
  rewrite lowers the *worst-case* bound this tool measures. Note Herbie itself
  reports average-case error over sampled points, so its numbers aren't directly
  comparable to the FPTaylor bounds here.

## Requirements

- casa (torch, transformers, llguidance) + egglog.
- FPTaylor: the `fptaylor` binary on PATH with its opam env active
  (`eval $(opam env)`), else every bound comes back unbounded.
