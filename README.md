# egrammar

Compile an e-graph of equivalent programs into a context-free grammar, then sample
from it with grammar-constrained sampling via the
[casa](https://github.com/large-loris-models/casa) library.

## How it works

1. **Build the e-graph** ([egrammar.py](src/egrammar.py)): run egglog on the benchmark's
   reference program plus the rewrite rules ([rules.egglog](rules.egglog)), saturating for 6 rounds. The root e-class then holds every
   recognized rewrite of the program.
2. **Clean the e-graph** (`strip_identity_enodes`): the rules keep identity-elimination
   rewrites (`(* 1 x) → x`, …) because they canonicalize intermediates during
   saturation, helping downstream rules match and discover more equivalences — but
   egglog's symmetric merges then leave padded spellings (`(* x 1)`, `(/ a a)`, …)
   in each e-class. This pass removes them:
   - find e-classes that provably **denote 0 / 1** (a fixpoint, so it catches derived
     identities like `(/ a a)`, not just literal `(Num 1)`);
   - **alias** any class whose every spelling is padding to the value it denotes (so
     `(* x (/ a a))` collapses into `x`);
   - **strip** padding spellings from classes that still have a real one;
   - **prune** non-minimal cyclic spellings (the `4ac`-as-deep-nesting reshuffles)
     with a Tarjan-SCC + min-depth pass.
3. **Intersect with the simple grammar** ([egrammar.py](src/egrammar.py)): the intersection has **one
   nonterminal per e-class** and **one production per e-node**, spelled in FPCore
   syntax with fixed whitespace:

   ```lark
   start: "(FPCore (a b c) " e0 ")" "\n"?
   e0: "(/ " e1 " " e2 ")" | "(* " e3 " " e4 ")" | ...
   e7: "b"
   ```

## Usage

### Compile the grammar

```bash
uv run src/egrammar.py quadratic        # writes lark/quadratic.lark
```

### Compile *and* sample in one step

[run_cars.py](src/run_cars.py) compiles the grammar (reusing `lark/<benchmark>.lark` if
present), then drives one of casa's grammar-constrained samplers over a language
model to harvest a *variety* of distinct programs, printing each program live as it
is accepted or rejected. Since the grammar's language is exactly the programs
provably equivalent to the reference, every accepted sample is equivalent by
construction — a deduplicated set of alternative spellings of the same computation.

```bash
uv run src/run_cars.py quadratic                       # 20 programs, CARS sampler
uv run src/run_cars.py quadratic --samples 50 --sampler ars
uv run src/run_cars.py quadratic --sampler mcmc-restart --steps 20
```

It samples with `Qwen/Qwen2.5-14B-Instruct` and writes a numbered run file
`equivalents/<benchmark>-NNN.json` (each run gets the next number, so repeated runs
don't overwrite each other). `--sampler` selects either a casa rejection-family
sampler — `cars` (default), `ars`, `rsft`, `rs` (tuned with `--max-attempts`) — or an
MCMC variant — `mcmc-uniform`, `mcmc-priority`, `mcmc-restart` (tuned with `--steps`,
which runs that many MCMC steps per chain and keeps each chain's final program). The
full option list is in the [run_cars.py](src/run_cars.py) module docstring.

### Bound the rounding error

[gappa_check.py](src/gappa_check.py) bounds the floating-point rounding error of each
harvested program with [Gappa](https://gappa.gitlabpages.inria.fr/) (requires the
`gappa` binary on PATH; no Python deps beyond the stdlib):

```bash
uv run src/gappa_check.py quadratic            # analyzes the latest equivalents run
uv run src/gappa_check.py quadratic --run 2    # a specific run
uv run src/gappa_check.py sqrtminus --subdiv 64 # wide box; subdivide to bound it
```

It reads `equivalents/<benchmark>-NNN.json` and writes the matching
`gappa/<benchmark>-NNN.json`: per program, the exact real-valued enclosure plus
certified worst-case absolute and relative rounding error in IEEE-754 double. The
interval box is per-benchmark (`INTERVALS` in the script) and deliberately narrow,
because Gappa's interval arithmetic loses variable correlations on a wide box (it
can't prove a cancelling denominator/value is nonzero). For a wider box, `--subdiv N`
makes Gappa bisect each variable into `N` pieces to recover those correlations, at
`N^(#vars)` cost; bounds it still can't prove are reported as `n/a`.

Sampling is delegated to the [casa](https://github.com/large-loris-models/casa)
library, which implements CARS (Constrained Adaptive Rejection Sampling; see the
[paper](https://arxiv.org/pdf/2506.05754)) alongside the simpler ARS/RSFT/RS
rejection samplers. casa pulls in the sampling runtime (torch, transformers,
llguidance, xgrammar, accelerate); egglog is needed only to compile a grammar that
is not already cached in `lark/`.

## Caveats

- **Saturation cap**: like chopchop, equivalence is "reachable within 6 egglog runs",
  not full algebraic equivalence (`SATURATION_RUNS`).
- **No identity padding**: `strip_identity_enodes` removes the trivial respellings
  (`(* x 1)`, `(+ x 0)`, `(- x x)`, `(/ a a)`, …) egglog's merges leave behind, so the
  grammar won't pass off `(* 1 original)` as new. It works semantically (the
  denotes-0/1 fixpoint), catching derived identities, and aliases fully-padding
  classes rather than keeping one as a fallback. The `(* 4 x) → (* 2 (* 2 x))` rule is
  also dropped from [rules.egglog](rules.egglog) — ×2 is exact, so it only gave an
  FP-identical respelling.
- **Cyclic reshuffles are pruned**: the SCC + min-depth pass drops non-minimal cyclic
  spellings like `4ac` as `(* (* (* (* c a) (/ 1 a)) a) 4)`, matching chopchop's checker.
- **Fixed argument list and spacing**: the grammar pins the argument list (sorted) and
  one whitespace style, so the model has no formatting freedom — keeps productions
  plain string literals.

## Grammar sizes (rules ≈ e-classes reachable from the root)

| benchmark | rules | benchmark | rules |
|-----------|------:|-----------|------:|
| lerp      |    15 | distance  |   545 |
| power     |    18 | quadratic |   292 |
| sqrtminus |    23 | variance  | 11409 |
| subfrac   |    27 | gravity   |    63 |

All validate under `llguidance.LLMatcher.validate_grammar`.
