"""Sample equivalent programs from an egrammar-compiled grammar with casa.

Options:
    benchmark            benchmark name, e.g. quadratic (positional, required)
    --samples N          distinct programs to collect / chains to run (default 20)
    --sampler NAME       which casa sampler (default cars); one of:
                           cars   rejection + first-token constraint + learning
                           ars    rejection + learning (no first-token constraint)
                           rsft   rejection + first-token constraint
                           rs     plain rejection sampling
                           mcmc-uniform    MCMC, resample a uniformly-random position
                           mcmc-priority   MCMC, resample a high-entropy position
                           mcmc-restart    MCMC, always resample from the start
    --max-attempts N     rejection samplers: cap on attempts per sample (default 200)
    --steps N            mcmc samplers: MCMC steps per chain (default 10)
    --temperature T      sampling temperature applied to the model (default 1.0);
                         T<1 sharpens, T>1 flattens the grammar-constrained model

Examples:
    uv run src/run.py quadratic --sampler mcmc-restart --steps 20
"""

import argparse
import json

import paths

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REJECTION = ("rs", "ars", "rsft", "cars")
MCMC_VARIANTS = ("uniform", "priority", "restart")
SAMPLERS = REJECTION + tuple(f"mcmc-{v}" for v in MCMC_VARIANTS)


def ensure_artifacts(benchmark: str) -> tuple[str, str, str]:
    import egrammar

    grammar_path = paths.LARK / f"{benchmark}.lark"
    if grammar_path.exists():
        reference, grammar = egrammar.read_reference(benchmark), grammar_path.read_text()
    else:
        print(f"Compiling grammar for {benchmark!r} (no cached grammar)")
        reference, grammar = egrammar.build(benchmark)
        egrammar.write_grammar(benchmark, grammar)
    return grammar, egrammar.make_prompt(reference), reference


def distinct(results) -> list[str]:
    seen: set[str] = set()
    programs: list[str] = []
    for r in results:
        text = r.text.strip()
        if text not in seen:
            seen.add(text)
            programs.append(text)
    return programs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("benchmark", help="benchmark name, e.g. quadratic")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sampler", choices=SAMPLERS, default="cars")
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    grammar_str, prompt, reference = ensure_artifacts(args.benchmark)

    import casa

    budget = (f"<= {args.max_attempts} attempts/sample" if args.sampler in REJECTION
              else f"{args.steps} MCMC steps/chain")
    print(f"benchmark: {args.benchmark}")
    print(f"grammar:   {paths.LARK / f'{args.benchmark}.lark'} "
          f"({grammar_str.count(chr(10))} rules)")
    print(f"model:     {MODEL_ID}")
    print(f"sampler:   {args.sampler}")
    print(f"temp:      {args.temperature}")
    print(f"target {args.samples} programs, {budget}\n")

    llm = casa.LLM.from_pretrained(MODEL_ID)
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)

    # casa prints each program live (verbose=True) as it is accepted/rejected.
    if args.sampler in REJECTION:
        sampler = getattr(casa, args.sampler.upper())(
            llm, grammar, verbose=True, temperature=args.temperature
        )
        results = sampler.sample(
            prompt, n_samples=args.samples, max_attempts=args.max_attempts
        )
    else:  # mcmc-<variant>
        sampler = casa.MCMC(
            llm, grammar, variant=args.sampler.split("-", 1)[1], verbose=True,
            temperature=args.temperature,
        )
        results = sampler.sample(
            prompt, n_samples=args.samples, n_steps=args.steps, return_steps=False
        )
    programs = distinct(results)

    print(f"\n{'=' * 70}")
    print(f"original program:     {reference}")
    print(f"distinct equivalents: {len(programs)}")
    print(f"{'=' * 70}")
    for i, program in enumerate(programs):
        print(f"{i:3d}  {program}")

    _, summary = paths.next_path(paths.EQUIVALENTS, args.benchmark)
    summary.write_text(json.dumps(
        {"benchmark": args.benchmark, "reference": reference,
         "model": MODEL_ID, "sampler": args.sampler, "programs": programs},
        indent=2,
    ))
    print(f"\nwrote {len(programs)} distinct equivalent programs to {summary}")


if __name__ == "__main__":
    main()
