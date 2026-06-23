# egrammar

Compile an e-graph of equivalent programs into a context-free grammar, then sample
from it with grammar-constrained sampling ([CARS](cars.py), vendored here).

The grammar-flavored version of [chopchop](../chopchop)'s e-graph case study:
rather than checking realizability token by token during decoding, we compile the
*whole* constraint into a grammar up front and let the sampler enforce it.

## How it works

1. **Build the e-graph** ([egrammar.py](egrammar.py)): run egglog on the benchmark's
   reference program plus the rewrite rules ([rules.egglog](rules.egglog), chopchop's
   `let.egglog`), saturating for 6 rounds. The root e-class then holds every
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
3. **Intersect with the simple grammar** ([egrammar.py](egrammar.py)): the FPCore
   syntax grammar's single `expr` allows every operator everywhere; the e-graph allows
   only some e-class's spellings at each position. The intersection has **one
   nonterminal per e-class** and **one production per e-node**, spelled in FPCore
   syntax with fixed whitespace:

   ```lark
   start: "(FPCore (a b c) " e0 ")" "\n"?
   e0: "(/ " e1 " " e2 ")" | "(* " e3 " " e4 ")" | ...
   e7: "b"
   ```

The grammar's language is the cleaned set of root-e-class spellings — programs
provably equivalent to the reference, minus the trivial respellings. The e-graph is
cyclic (`a` = `(* (/ 1 2) (* 2 a))` = ...); a cycle just becomes a recursive rule.

## Usage

### Compile the grammar

```bash
uv run egrammar.py quadratic        # writes out/quadratic.lark + out/quadratic.txt
```

(`.lark` is llguidance's lark dialect, loaded via `llguidance.grammar_from`; `.txt`
is the prompt.)

### Compile *and* sample in one step

[run_cars.py](run_cars.py) compiles the grammar (reusing `out/<benchmark>.lark` if
present), then drives the CARS sampler over a language model to harvest a *variety*
of distinct programs. Since the grammar's language is exactly the programs provably
equivalent to the reference, every accepted sample is equivalent by construction — a
deduplicated set of alternative spellings of the same computation.

```bash
uv run --extra cars run_cars.py quadratic                  # 20 programs
uv run --extra cars run_cars.py quadratic --samples 50 --steps 500
```

It samples with `Qwen/Qwen2.5-14B-Instruct` and writes `out/<benchmark>.equivalents.json`.

The sampler is [cars.py](cars.py) — a self-contained port of CARS (Constrained
Adaptive Rejection Sampling), trimmed to the single "cars" style (learn level 3,
constrained first token; see the [paper](https://arxiv.org/pdf/2506.05754)). The
`cars` extra (in `pyproject.toml`) pulls in its deps — torch, transformers,
llguidance, xgrammar; the base install stays egglog-only, and egglog is needed only
on a cache miss.

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
