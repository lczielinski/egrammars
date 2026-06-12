# egrammar

Compile an e-graph of equivalent programs into a context-free grammar, then sample
from it with grammar-constrained sampling ([CARS](cars.py), vendored here).

This is the grammar-flavored version of [chopchop](../chopchop)'s e-graph case
study: instead of checking realizability token by token during decoding, we
compile the *whole* constraint into a grammar up front and let the sampler
enforce it.

## How it works

1. **Build the e-graph** ([egrammar.py](egrammar.py), step 1): run egglog on the
   benchmark's reference program plus the rewrite rules ([rules.egglog](rules.egglog),
   chopchop's `let.egglog`), saturating for 6 rounds. The root e-class now
   contains every recognized-equivalent rewrite of the program.
2. **Intersect with the simple grammar** (steps 2–3): the FPCore syntax grammar
   has a single `expr` nonterminal that allows every operator everywhere; the
   e-graph only allows, at each position, the spellings of some e-class. Their
   intersection is the grammar with **one nonterminal per e-class** and **one
   production per e-node**, each production spelled in FPCore concrete syntax
   with canonical whitespace:

   ```lark
   start: "(FPCore (a b c) " e0 ")" "\n"?
   e0: "(/ " e1 " " e2 ")" | "(* " e3 " " e4 ")" | ...
   e7: "b"
   ```

The grammar's language is exactly the FPCore spellings of the root e-class —
i.e. the programs provably equivalent to the reference. The e-graph is cyclic
(`a` = `(* (/ 1 2) (* 2 a))` = ...); that's fine, a cycle just becomes a
recursive grammar rule.

## Usage

### Compile the grammar

```bash
uv run egrammar.py quadratic        # writes out/quadratic.lark + out/quadratic.txt
```

(The `.lark` output is in llguidance's lark dialect, loaded via
`llguidance.grammar_from`; the `.txt` is the prompt.)

### Compile *and* sample in one step

[run_cars.py](run_cars.py) closes the loop: it compiles the grammar (reusing
`out/<benchmark>.lark` if present) and then drives the CARS sampler over a language
model to harvest a *variety* of distinct programs from the grammar's language.
Because the grammar's language is exactly the programs provably equivalent to the
reference, every accepted sample is equivalent by construction — the output is a
deduplicated set of alternative spellings of the same computation.

```bash
uv run --extra cars run_cars.py quadratic                  # 20 programs
uv run --extra cars run_cars.py quadratic --samples 50 --steps 500
```

It samples with `Qwen/Qwen2.5-14B-Instruct` and writes
`out/<benchmark>.equivalents.json` (the deduplicated programs).

The sampler itself lives in [cars.py](cars.py) — a self-contained port of CARS
(Constrained Adaptive Rejection Sampling), trimmed to the single "cars" style
(learn level 3, constrained first token; see the
[paper](https://arxiv.org/pdf/2506.05754)). The `cars` extra (declared in
`pyproject.toml`) pulls in its runtime deps — torch, transformers, llguidance,
xgrammar — so `uv` manages them for you; the base install stays egglog-only, and
egglog is only needed on a cache miss (when the `.lark` file does not yet exist).

## Caveats

- **Saturation cap**: like chopchop, equivalence is "reachable within 6 egglog
  runs", not full algebraic equivalence (`SATURATION_RUNS` in egrammar.py).
- **Identity padding is kept**: the rules merge `x` with `(* 1 x)`, `(+ 0 x)`,
  etc., so every e-class contains identity-padded spellings of itself, and the
  grammar accepts them — a sampler can pass off `(* 1 original)` as a "new"
  program. These are genuinely equivalent, just trivial; filter them downstream
  if it matters.
- **Looser than chopchop's checker**: chopchop additionally breaks e-graph
  cycles (`strip_identity_enodes`'s SCC pass), which removes reshuffle
  spellings like `4ac` as `(* (* (* (* c a) (/ 1 a)) a) 4)`. We keep them —
  they are genuinely equivalent (verified with sympy), but it means this
  grammar accepts some programs chopchop's checker would reject, and a sampler
  can wander into deeply padded spellings. Port that pass if it matters.
- **Fixed argument list and spacing**: the grammar pins the FPCore argument
  list (sorted variable order) and one canonical whitespace style, so the model
  has no formatting freedom — intentional, it keeps the grammar plain string
  literals.

## Grammar sizes (rules ≈ e-classes reachable from the root)

| benchmark | rules | benchmark | rules |
|-----------|------:|-----------|------:|
| lerp      |    15 | distance  |   545 |
| power     |    18 | quadratic |   664 |
| sqrtminus |    22 | variance  | 11409 |
| subfrac   |    27 | gravity   |    63 |

All validate under `llguidance.LLMatcher.validate_grammar`.
