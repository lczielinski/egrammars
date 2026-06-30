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
    --saturation N       rewrite-rule iterations when compiling a grammar (default
                         6; only used if not cached). Lower it for symmetry-heavy
                         expressions whose grammar explodes

A harmony model (gpt-oss) on a rejection sampler reasons automatically: it first
proposes a menu of distinct rewrite ideas, then samples each idea under its own
grammar-constrained prompt so the programs spread across ideas.

Examples:
    uv run src/run.py quadratic --sampler mcmc-restart --steps 20
    uv run src/run.py quadratic --model openai/gpt-oss-120b
"""

import argparse
import json
import os
import re

import paths

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REJECTION = ("rs", "ars", "rsft", "cars", "asap", "gcd")
MCMC_VARIANTS = ("uniform", "priority", "restart")
SAMPLERS = REJECTION + tuple(f"mcmc-{v}" for v in MCMC_VARIANTS)


def make_prompt(reference: str) -> str:
    return (
        (paths.ROOT / "prompt_header.md").read_text()
        + f"\n\nThe original program is:\n{reference}\n\n"
        "Produce one complete FPCore program that is algebraically equivalent to "
        "the original but evaluates with different floating-point behavior."
    )


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
    return grammar, make_prompt(reference), reference


def distinct(results) -> list[str]:
    seen: set[str] = set()
    programs: list[str] = []
    for r in results:
        text = r.text.strip()
        if text not in seen:
            seen.add(text)
            programs.append(text)
    return programs


def free_cuda() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def channel_ids(tokenizer):
    """The (<|channel|>, <|message|>) token ids if the tokenizer has harmony
    channels (gpt-oss), else None."""
    ch = tokenizer.convert_tokens_to_ids("<|channel|>")
    msg = tokenizer.convert_tokens_to_ids("<|message|>")
    if None in (ch, msg) or tokenizer.unk_token_id in (ch, msg):
        return None
    return ch, msg


def final_channel(decoded: str) -> str:
    """The text the model put in its harmony `final` channel."""
    if "final<|message|>" in decoded:
        decoded = decoded.split("final<|message|>", 1)[1]
    for end in ("<|return|>", "<|end|>", "<|endoftext|>"):
        decoded = decoded.split(end, 1)[0]
    return decoded.strip()


def parse_ideas(text: str) -> list[str]:
    """Pull rewrite-idea phrases from a numbered or bulleted menu."""
    ideas = []
    for line in text.splitlines():
        m = re.match(r"\s*(?:\d+[.)]|[-*])\s+(.+)", line)
        if m and m.group(1).strip():
            ideas.append(m.group(1).strip())
    return ideas


def ideas_prompt(reference: str) -> str:
    return (
        "The following expression is evaluated in IEEE-754 double precision:\n\n"
        f"    {reference}\n\n"
        "It is built from + - * / and sqrt over its variables. Reason about where "
        "it loses floating-point accuracy -- catastrophic cancellation, division "
        "by a near-zero quantity, growth of intermediate magnitudes. Then list "
        "several distinct algebraic rewrites that keep the same exact value but "
        "round differently and improve accuracy: one per line, numbered, each "
        "naming the rewrite and the input regime where it helps. List ideas only; "
        "do not write a program."
    )


def propose_ideas(llm, reference, temperature, effort="high"):
    import torch
    from transformers import TextStreamer

    full = ideas_prompt(reference)
    try:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": full}],
            tokenize=False, add_generation_prompt=True, reasoning_effort=effort,
        )
    except TypeError:
        text = llm.format_prompt(full)
    ids = llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False
    ).to(llm.device)

    max_ctx = getattr(llm.model.config, "max_position_embeddings", None) or 8192
    gen_kwargs = dict(max_new_tokens=max(256, max_ctx - ids.shape[1] - 64),
                      attention_mask=torch.ones_like(ids),
                      pad_token_id=llm.tokenizer.pad_token_id,
                      streamer=TextStreamer(llm.tokenizer, skip_prompt=True,
                                            skip_special_tokens=True))
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    with torch.no_grad():
        out = llm.model.generate(ids, **gen_kwargs)

    decoded = llm.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    return parse_ideas(final_channel(decoded))


def condition_on(idea: str) -> str:
    return (f"\n\nApply this specific rewrite to the original program, and no "
            f"other:\n{idea}\nOutput the single program.")


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

    # A harmony model (gpt-oss) reasons once into a menu of distinct rewrite
    # ideas; each idea then conditions its own batch of constrained samples, so
    # the programs spread across ideas instead of orbiting a single one.
    ideas = []
    if args.sampler in REJECTION and channel_ids(llm.tokenizer):
        print(f"{'=' * 70}\nreasoning phase (proposing rewrite ideas)\n{'=' * 70}")
        ideas = propose_ideas(llm, reference, args.temperature)
        print(f"\n{'=' * 70}\nproposed {len(ideas)} rewrite idea(s)\n{'=' * 70}")
        free_cuda()

    # casa prints each program live (verbose=True) as it is accepted/rejected.
    if args.sampler in REJECTION:
        sampler = getattr(casa, args.sampler.upper())(
            llm, grammar, verbose=True, temperature=args.temperature
        )
        if ideas:
            results = []
            per = max(1, -(-args.samples // len(ideas)))  # ceil
            for i, idea in enumerate(ideas):
                print(f"\n[idea {i + 1}/{len(ideas)}] {idea}")
                results += sampler.sample(prompt + condition_on(idea),
                                          n_samples=per,
                                          max_attempts=args.max_attempts)
                free_cuda()
                if len(distinct(results)) >= args.samples:
                    break
        else:
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
         "ideas": ideas, "programs": programs},
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
