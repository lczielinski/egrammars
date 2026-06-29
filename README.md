# egrammars

Compile an e-graph of equivalent programs into a context-free grammar, then sample
from it with grammar-constrained sampling via the [casa](https://github.com/large-loris-models/casa) library.

## How it works

1. **Build the e-graph** ([egrammar.py](src/egrammar.py)): run egglog on the benchmark's
   reference program plus the rewrite rules ([rules.egglog](rules.egglog)), saturating for a
   configurable number of rounds (default 6; `--saturation`). The root e-class then holds
   every recognized rewrite of the program. The grammar grows roughly exponentially in the
   round count, so symmetry-heavy expressions may need a lower `--saturation` to stay
   tractable for constrained decoding.
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

[run.py](src/run.py) compiles the grammar then drives one of casa's grammar-constrained samplers over a language
model to harvest a *variety* of distinct programs, printing each program live as it
is accepted or rejected. 

```bash
uv run src/run.py quadratic                            # 20 programs, asap sampler
uv run src/run.py quadratic --samples 50 --sampler ars
uv run src/run.py quadratic --sampler mcmc-restart --steps 20
uv run src/run.py quadratic --model openai/gpt-oss-120b --reason
```

It writes a numbered run file `equivalents/<benchmark>-NNN.json`. Key flags:

- `--model ID` — any HuggingFace causal LM (default `Qwen/Qwen2.5-14B-Instruct`); larger
  or reasoning models such as `openai/gpt-oss-120b` work too.
- `--sampler` — a casa rejection-family sampler `asap` (default), `cars`, `ars`, `rsft`,
  `rs`, `gcd` (tuned with `--max-attempts`), or an MCMC variant `mcmc-uniform`,
  `mcmc-priority`, `mcmc-restart` (tuned with `--steps`, kept per chain).
- `--reason` — two-phase: let the model reason *unconstrained* about which rewrites
  improve accuracy, then fold that reasoning into the prompt before grammar-constrained
  sampling. Useful with a reasoning model, but only helps when the stabilizing rewrite is
  actually reachable in the grammar.
- `--saturation N` — rounds when compiling a grammar that isn't cached yet.

The full option list is in the [run.py](src/run.py) module docstring.

### Bound the rounding error

[fptaylor_check.py](src/fptaylor_check.py) bounds the floating-point rounding error of
each harvested program with [FPTaylor](https://github.com/soarlab/FPTaylor) (requires the
`fptaylor` binary on PATH and `$FPTAYLOR_BASE` set; no Python deps beyond the stdlib).
It runs automatically at the end of `run.py`, or standalone:

```bash
uv run src/fptaylor_check.py quadratic         # analyzes the latest equivalents run
uv run src/fptaylor_check.py quadratic --run 2 # a specific run
```

Inputs are treated as exact doubles over a per-benchmark interval box (`INTERVALS` in the
script), every operation is rounded to IEEE-754 double, and the worst-case absolute and
relative error of each program is reported, written to `fptaylor/<benchmark>-NNN.json`
(reusing the equivalents run number). Notes:

- FPTaylor's symbolic Taylor forms handle wider boxes than interval arithmetic would, but
  every `sqrt` argument must stay `>= 0` and every denominator clear of `0` across the box
  or the bound is `+inf`.
- It often omits relative error through divisions even when the value is far from zero; in
  that case the checker derives a (looser, sound) bound as `abs_err / min|value|` from the
  reported value range, flagged `rel_err_derived`. When the range genuinely straddles zero,
  relative error is reported as `n/a`.
- A cancellation-heavy rewrite can make FPTaylor's optimizer grind, so each program has a
  timeout (`TIMEOUT` in the script); a program that exceeds it is recorded as `timeout` and
  the run continues.

Sampling is delegated to the [casa](https://github.com/large-loris-models/casa)
library, which implements CARS (Constrained Adaptive Rejection Sampling; see the
[paper](https://arxiv.org/pdf/2506.05754)) and its mask-every-step variant `asap`,
alongside the simpler `ars`/`rsft`/`rs`/`gcd` samplers and MCMC. casa pulls in the
sampling runtime (torch, transformers, llguidance, xgrammar, accelerate); egglog is
needed only to compile a grammar that is not already cached in `lark/`.

## Grammar sizes (rules ≈ e-classes reachable from the root)

| benchmark | rules | benchmark | rules |
|-----------|------:|-----------|------:|
| lerp      |    15 | distance  |   545 |
| power     |    18 | quadratic |   292 |
| sqrtminus |    23 | variance  | 11409 |
| subfrac   |    27 | gravity   |    63 |

Sizes are for the cached grammars and depend on both `--saturation` and the rule set;
adding rules or rounds grows them (e.g. `heron` is 744 rules at saturation 4 but 44.5k at
6). All validate under `llguidance.LLMatcher.validate_grammar`.
