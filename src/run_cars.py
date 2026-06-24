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
    --max-new-tokens N   generation length cap (default 1024)

Examples:
    uv run src/run_cars.py quadratic --sampler mcmc-restart --steps 20
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


def stream_rejection(sampler, prompt, n_samples, max_attempts) -> list[str]:
    """Run a casa rejection sampler attempt-by-attempt, printing each program as it
    is accepted ([k/n]) or rejected ([reject]). Returns the distinct accepted ones."""
    from casa.utils.oracle_logits_processor import OracleLogitsProcessor

    proc = OracleLogitsProcessor(
        tokenizer=sampler.llm.tokenizer,
        grammar_constraint=sampler.grammar.recognizer,
        device=sampler.llm.device,
        learn_level=sampler.learn_level,
        constrain_first=sampler.constrain_first,
    )
    prompt_ids = sampler._encode_prompt(prompt)
    programs: list[str] = []
    seen: set[str] = set()

    for _ in range(n_samples):
        for _ in range(max_attempts):
            try:
                text = sampler._generate_one(prompt_ids, proc).text.strip()
            except ValueError:
                rejected = sampler.llm.tokenizer.decode(
                    proc.generated_tokens, skip_special_tokens=True
                ).strip()
                print(f"[reject] {rejected}", flush=True)
                continue
            if text not in seen:
                seen.add(text)
                programs.append(text)
            print(f"[{len(programs)}/{n_samples}] {text}", flush=True)
            break
    return programs


def stream_mcmc(sampler, prompt, n_samples, n_steps) -> list[str]:
    """Run casa's MCMC sampler and print every proposal with its accept/reject
    decision, then keep each chain's final program. (casa computes a whole chain in
    one call, so a chain's steps print together once it finishes.)"""
    chains = sampler.sample(
        prompt, n_samples=n_samples, n_steps=n_steps, return_steps=True
    )
    programs: list[str] = []
    seen: set[str] = set()
    for c, steps in enumerate(chains):
        if not steps:
            continue
        for s, step in enumerate(steps):
            tag = "accept" if step.accepted else "reject"
            print(f"[chain {c} step {s}] {tag} p={step.acceptance_prob:.2f}: "
                  f"{step.proposal.text.strip()}", flush=True)
        last = steps[-1]
        final = (last.proposal if last.accepted else last.current).text.strip()
        if final not in seen:
            seen.add(final)
            programs.append(final)
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
    parser.add_argument("--max-new-tokens", type=int, default=1024)
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
    print(f"target {args.samples} programs, {budget}\n")

    llm = casa.LLM.from_pretrained(MODEL_ID)
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)

    if args.sampler in REJECTION:
        sampler = getattr(casa, args.sampler.upper())(
            llm, grammar, max_new_tokens=args.max_new_tokens
        )
        programs = stream_rejection(sampler, prompt, args.samples, args.max_attempts)
    else:  # mcmc-<variant>
        sampler = casa.MCMC(
            llm, grammar, variant=args.sampler.split("-", 1)[1],
            max_new_tokens=args.max_new_tokens,
        )
        programs = stream_mcmc(sampler, prompt, args.samples, args.steps)

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
