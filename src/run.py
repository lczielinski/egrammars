"""Sample equivalent programs from an egrammar-compiled grammar with casa.

Options:
    benchmark            benchmark name, e.g. quadratic (positional, required)
    --samples N          distinct programs to collect / chains to run (default 20)
    --sampler NAME       which casa sampler (default asap); one of:
                           cars   rejection + first-token constraint + learning
                           asap   cars + grammar mask every step (no rejects)
                           gcd    grammar-constrained decoding (masked, no learning)
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
    --model ID           HuggingFace model id to load (default Qwen2.5-14B-Instruct)
    --reason             two-phase: let the model reason (unconstrained) about
                         which rewrites improve accuracy, then fold that reasoning
                         into the prompt before grammar-constrained sampling. Lets
                         a reasoning model (e.g. gpt-oss) actually think first.
    --reason-tokens N    token budget for the reasoning phase (default 1024)
    --saturation N       rewrite-rule iterations when compiling a grammar (default
                         6; only used if not cached). Lower it for symmetry-heavy
                         expressions whose grammar explodes (e.g. heron: 4)

Examples:
    uv run src/run.py quadratic --sampler mcmc-restart --steps 20
    uv run src/run.py quadratic --model openai/gpt-oss-120b --reason
"""

import argparse
import json

import paths

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REJECTION = ("rs", "ars", "rsft", "cars", "asap", "gcd")
MCMC_VARIANTS = ("uniform", "priority", "restart")
SAMPLERS = REJECTION + tuple(f"mcmc-{v}" for v in MCMC_VARIANTS)


def ensure_artifacts(benchmark: str, saturation: int) -> tuple[str, str, str]:
    import egrammar

    grammar_path = paths.LARK / f"{benchmark}.lark"
    if grammar_path.exists():
        reference, grammar = egrammar.read_reference(benchmark), grammar_path.read_text()
    else:
        print(f"Compiling grammar for {benchmark!r} (no cached grammar, "
              f"saturation={saturation})")
        reference, grammar = egrammar.build(benchmark, saturation)
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


def reasoning_prompt(reference: str) -> str:
    """Phase-1 prompt: same header/reference, but ask the model to analyze where
    this expression loses accuracy instead of emitting a program."""
    return (
        (paths.ROOT / "prompt_header.md").read_text()
        + f"\n\nThe original program is:\n{reference}\n\n"
        "Think step by step about THIS expression's floating-point behavior: where "
        "does it lose accuracy (catastrophic cancellation, division by a near-zero "
        "or near-equal quantity, growth of intermediate magnitudes), and which of "
        "the rewrites above would most improve its worst-case rounding error? "
        "Reason in prose; do NOT write any FPCore program yet."
    )


def generate_reasoning(llm, reference: str, max_new_tokens: int,
                       temperature: float) -> str:
    """Run the model unconstrained on the reasoning prompt; return its text."""
    import torch

    text = llm.format_prompt(reasoning_prompt(reference))
    ids = llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False
    ).to(llm.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=llm.tokenizer.pad_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    with torch.no_grad():
        out = llm.model.generate(ids, **gen_kwargs)
    return llm.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def prompt_with_reasoning(base_prompt: str, reasoning: str) -> str:
    """Fold phase-1 reasoning into the phase-2 (grammar-constrained) prompt."""
    return (f"{base_prompt}\n\nYour analysis of this expression:\n{reasoning}\n\n"
            "Using that analysis, produce the program.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("benchmark", help="benchmark name, e.g. quadratic")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sampler", choices=SAMPLERS, default="asap")
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--reason", action="store_true")
    parser.add_argument("--reason-tokens", type=int, default=1024)
    parser.add_argument("--saturation", type=int, default=6,
                        help="rewrite-rule iterations when compiling a grammar "
                             "(only used if not already cached; lower it for "
                             "symmetry-heavy expressions like heron)")
    args = parser.parse_args()

    grammar_str, prompt, reference = ensure_artifacts(args.benchmark, args.saturation)

    import casa

    budget = (f"<= {args.max_attempts} attempts/sample" if args.sampler in REJECTION
              else f"{args.steps} MCMC steps/chain")
    print(f"benchmark: {args.benchmark}")
    print(f"grammar:   {paths.LARK / f'{args.benchmark}.lark'} "
          f"({grammar_str.count(chr(10))} rules)")
    print(f"model:     {args.model}")
    print(f"sampler:   {args.sampler}")
    print(f"temp:      {args.temperature}")
    print(f"reason:    {args.reason}")
    print(f"target {args.samples} programs, {budget}\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)

    reasoning = None
    if args.reason:
        print(f"{'=' * 70}\nreasoning phase (unconstrained)\n{'=' * 70}")
        reasoning = generate_reasoning(llm, reference, args.reason_tokens,
                                       args.temperature)
        print(f"{reasoning}\n{'=' * 70}\n")
        prompt = prompt_with_reasoning(prompt, reasoning)

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

    n, summary = paths.next_path(paths.EQUIVALENTS, args.benchmark)
    summary.write_text(json.dumps(
        {"benchmark": args.benchmark, "reference": reference,
         "model": args.model, "sampler": args.sampler,
         "reasoning": reasoning, "programs": programs},
        indent=2,
    ))
    print(f"\nwrote {len(programs)} distinct equivalent programs to {summary}")

    if programs:
        import fptaylor_check
        print(f"\n{'=' * 70}\nfptaylor rounding-error analysis\n{'=' * 70}")
        try:
            fptaylor_check.check(args.benchmark, run=n)
        except KeyError:
            print(f"skipping fptaylor: no interval box configured for "
                  f"{args.benchmark!r} (add one to INTERVALS in fptaylor_check.py)")
        except FileNotFoundError as e:
            # Either the equivalents file (shouldn't happen) or the fptaylor binary.
            msg = "fptaylor binary not found on PATH" if "fptaylor" in str(e).lower() \
                else str(e)
            print(f"skipping fptaylor: {msg}")


if __name__ == "__main__":
    main()
