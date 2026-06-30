"""Sample equivalent programs from an egrammar-compiled grammar with casa.

Options:
    benchmark            benchmark name, e.g. quadratic (positional, required)
    --samples N          distinct programs to collect (default 20)
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
    --reason-tokens N    cap on the reasoning phase (default 4096). A harmony model
                         (gpt-oss) on a rejection sampler reasons automatically: it
                         thinks in its analysis channel, then the program is sampled
                         grammar-constrained in its own final channel. Generation
                         stops the moment the final channel opens.
    --saturation N       rewrite-rule iterations when compiling a grammar (default
                         6; only used if not cached). Lower it for symmetry-heavy
                         expressions whose grammar explodes

Examples:
    uv run src/run.py quadratic --sampler mcmc-restart --steps 20
    uv run src/run.py quadratic --model openai/gpt-oss-120b
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
    return (
        (paths.ROOT / "prompt_header.md").read_text()
        + f"\n\nThe original program is:\n{reference}\n\n"
        "Think step by step about THIS expression's floating-point behavior: where "
        "does it lose accuracy (catastrophic cancellation, division by a near-zero "
        "or near-equal quantity, growth of intermediate magnitudes), and which of "
        "the rewrites above would most improve its worst-case rounding error? "
        "Reason in prose; do NOT write any FPCore program yet."
    )


def channel_ids(tokenizer):
    """The (<|channel|>, <|message|>) token ids if the tokenizer has harmony
    channels (gpt-oss), else None."""
    ch = tokenizer.convert_tokens_to_ids("<|channel|>")
    msg = tokenizer.convert_tokens_to_ids("<|message|>")
    if None in (ch, msg) or tokenizer.unk_token_id in (ch, msg):
        return None
    return ch, msg


def at_final_header(seq, ch_id, msg_id, decode) -> bool:
    """True if token list `seq` ends exactly at a final channel header
    (...<|channel|>final<|message|>). `decode` maps token ids to text."""
    if not seq or seq[-1] != msg_id:
        return False
    try:
        ch = len(seq) - 1 - seq[::-1].index(ch_id)
    except ValueError:
        return False
    return decode(seq[ch + 1:-1]).strip() == "final"


def think_then_handoff(llm, prompt, max_new_tokens, temperature, effort="high"):
    """Reason in the analysis channel, halting the instant the final channel
    opens. Returns (prompt_ids, analysis_text, opened_final) for the constrained
    pass to continue from."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

    ch_id, msg_id = channel_ids(llm.tokenizer)
    try:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, reasoning_effort=effort,
        )
    except TypeError:
        text = llm.format_prompt(prompt)
    ids = llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False
    ).to(llm.device)
    start = ids.shape[1]

    class StopAtFinal(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return at_final_header(input_ids[0].tolist(), ch_id, msg_id,
                                   llm.tokenizer.decode)

    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                      attention_mask=torch.ones_like(ids),
                      pad_token_id=llm.tokenizer.pad_token_id,
                      stopping_criteria=StoppingCriteriaList([StopAtFinal()]),
                      streamer=TextStreamer(llm.tokenizer, skip_prompt=True,
                                            skip_special_tokens=True))
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    with torch.no_grad():
        out = llm.model.generate(ids, **gen_kwargs)

    opened_final = out[0, -1].item() == msg_id
    analysis = llm.tokenizer.decode(
        out[0][start:], skip_special_tokens=True
    ).strip()
    return out, analysis, opened_final


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
    parser.add_argument("--reason-tokens", type=int, default=4096)
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
    print(f"target {args.samples} programs, {budget}\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)

    # A harmony model (gpt-oss) on a rejection sampler reasons in its analysis
    # channel, then the program is sampled grammar-constrained as a continuation
    # in its own final channel.
    reasoning, prompt_ids = None, None
    if args.sampler in REJECTION and channel_ids(llm.tokenizer):
        print(f"{'=' * 70}\nreasoning phase\n{'=' * 70}")
        prompt_ids, reasoning, opened = think_then_handoff(
            llm, prompt, args.reason_tokens, args.temperature)
        print(f"\n{'=' * 70}\n")
        if not opened:
            print(f"warning: model did not open its final channel within "
                  f"--reason-tokens={args.reason_tokens}; raise it. Sampling "
                  f"without reasoning.\n")
            prompt_ids = None

    # casa prints each program live (verbose=True) as it is accepted/rejected.
    if args.sampler in REJECTION:
        sampler = getattr(casa, args.sampler.upper())(
            llm, grammar, verbose=True, temperature=args.temperature
        )
        results = sampler.sample(
            prompt if prompt_ids is None else None, n_samples=args.samples,
            max_attempts=args.max_attempts, prompt_ids=prompt_ids,
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
