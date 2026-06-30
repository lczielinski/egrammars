"""Sample equivalent programs from an egrammar-compiled grammar with casa.

Sampling uses casa's asap sampler.

Options:
    benchmark            benchmark name, e.g. quadratic (positional, required)
    --samples N          distinct programs to collect (default 20)
    --max-attempts N     cap on attempts per sample (default 200)
    --temperature T      sampling temperature applied to the model (default 1.0);
                         T<1 sharpens, T>1 flattens the grammar-constrained model
    --model ID           HuggingFace model id to load (default Qwen2.5-14B-Instruct)
    --effort LEVEL       gpt-oss reasoning effort for the reasoning phase:
                         low, medium, or high (default medium)
    --branching          produce one program that branches on the input with `if`,
                         each arm an e-graph equivalent. The model reasons briefly,
                         then a constrained pass writes the branching program. Uses
                         a separate <name>-branching.lark grammar; fptaylor is
                         skipped (it can't analyze `if`).
    --saturation N       rewrite-rule iterations when compiling a grammar (default
                         6; only used if not cached). Lower it for symmetry-heavy
                         expressions whose grammar explodes

Without --branching, a harmony model (gpt-oss) reasons automatically: it proposes a
menu of distinct rewrite ideas, then samples each idea under its own grammar-
constrained prompt so the programs spread across ideas.

Examples:
    uv run src/run.py quadratic --samples 50
    uv run src/run.py sqrtshift --model openai/gpt-oss-120b --branching --saturation 4
"""

import argparse
import json
import os
import re
import warnings

import paths

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Transitive dep `kernels` (via transformers) warns about a future API change.
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def make_prompt(reference: str) -> str:
    return (
        (paths.ROOT / "prompt_header.md").read_text()
        + f"\n\nThe original program is:\n{reference}\n\n"
        "Produce one complete FPCore program that is algebraically equivalent to "
        "the original but evaluates with different floating-point behavior."
    )


# Appended to the prompt in --branching mode: permits an `if` over the inputs so
# each region can use the rewrite that is accurate there.
BRANCHING_GOAL = (
    "\n\nThis expression loses accuracy on some inputs. In ADDITION to those five "
    "operators you may use (if cond a b), where cond is a comparison -- (< a b), "
    "(> a b), (<= a b), (>= a b) over the variables and numeric thresholds. Produce "
    "one program that branches on the input so each region uses an algebraically-"
    "equivalent form that is accurate there; every branch must equal the original "
    "in exact arithmetic."
)


def ensure_artifacts(benchmark: str, saturation: int,
                     branching: bool = False) -> tuple[str, str, str]:
    import egrammar

    suffix = "-branching" if branching else ""
    grammar_path = paths.LARK / f"{benchmark}{suffix}.lark"
    if grammar_path.exists():
        reference, grammar = egrammar.read_reference(benchmark), grammar_path.read_text()
    else:
        print(f"Compiling grammar for {benchmark!r} (no cached grammar, "
              f"saturation={saturation}, branching={branching})")
        reference, grammar = egrammar.build(benchmark, saturation, branching)
        egrammar.write_grammar(benchmark, grammar, branching)
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
        "It is written with only these five operations -- addition, subtraction, "
        "multiplication, division, and square root -- over its variables. List "
        "several distinct algebraic rewrites that keep the same exact value but "
        "round more accurately (e.g. avoiding catastrophic cancellation, division "
        "by a near-zero quantity, or large intermediate magnitudes), one per line, "
        "numbered, each naming the rewrite and the input regime where it helps. "
        "Every rewrite must stay within those same five operations and the original "
        "variables (integer constants are fine); introduce no other functions such "
        "as exp, log, abs, pow, fma, or min/max. Output only the numbered list -- "
        "no preamble or explanation, and no program."
    )


def propose_ideas(llm, reference, temperature, effort="medium"):
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


def think_then_handoff(llm, prompt, temperature, effort="medium"):
    """Reason in the analysis channel, halting the instant the final channel opens.
    Returns (prompt_ids, analysis_text, opened_final) so a grammar-constrained pass
    can continue in the same final channel, attending to the reasoning."""
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

    max_ctx = getattr(llm.model.config, "max_position_embeddings", None) or 8192
    gen_kwargs = dict(max_new_tokens=max(256, max_ctx - start - 64),
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
    analysis = llm.tokenizer.decode(out[0][start:], skip_special_tokens=True).strip()
    return out, analysis, opened_final


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("benchmark", help="benchmark name, e.g. quadratic")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="medium",
                        help="gpt-oss reasoning effort for the reasoning phase")
    parser.add_argument("--branching", action="store_true",
                        help="let the program branch on the input with `if`; the "
                             "model reasons briefly, then samples one branching "
                             "program whose every arm is an e-graph equivalent")
    parser.add_argument("--saturation", type=int, default=6,
                        help="rewrite-rule iterations when compiling a grammar "
                             "(only used if not already cached; lower it for "
                             "symmetry-heavy expressions like variance)")
    args = parser.parse_args()

    grammar_str, prompt, reference = ensure_artifacts(
        args.benchmark, args.saturation, args.branching)

    import casa

    suffix = "-branching" if args.branching else ""
    print(f"benchmark: {args.benchmark}")
    print(f"grammar:   {paths.LARK / f'{args.benchmark}{suffix}.lark'} "
          f"({grammar_str.count(chr(10))} rules)")
    print(f"model:     {args.model}")
    print(f"temp:      {args.temperature}")
    print(f"target {args.samples} programs, <= {args.max_attempts} attempts/sample\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)

    # Reasoning. --branching: the model reasons briefly, then a constrained pass
    # writes one branching program (each arm an e-graph equivalent) in its own
    # final channel. Otherwise a harmony model proposes a menu of rewrite ideas,
    # each conditioning its own batch of constrained samples.
    ideas, prompt_ids, bprompt = [], None, prompt + BRANCHING_GOAL
    if args.branching and channel_ids(llm.tokenizer):
        print(f"{'=' * 70}\nreasoning phase (branching)\n{'=' * 70}")
        prompt_ids, _, opened = think_then_handoff(
            llm, bprompt, args.temperature, args.effort)
        print(f"\n{'=' * 70}\n")
        free_cuda()
        if not opened:
            print("warning: model never opened its final channel; sampling "
                  "without reasoning.\n")
            prompt_ids = None
    elif not args.branching and channel_ids(llm.tokenizer):
        print(f"{'=' * 70}\nreasoning phase (proposing rewrite ideas)\n{'=' * 70}")
        ideas = propose_ideas(llm, reference, args.temperature, args.effort)
        print(f"\n{'=' * 70}\nproposed {len(ideas)} rewrite idea(s)\n{'=' * 70}")
        free_cuda()

    # casa prints each program live (verbose=True) as it is accepted/rejected.
    sampler = casa.ASAP(llm, grammar, verbose=True, temperature=args.temperature)
    if args.branching:
        results = sampler.sample(
            bprompt if prompt_ids is None else None, n_samples=args.samples,
            max_attempts=args.max_attempts, prompt_ids=prompt_ids,
        )
    elif ideas:
        results = []
        per = max(1, -(-args.samples // len(ideas)))  # ceil
        for i, idea in enumerate(ideas):
            print(f"\n[idea {i + 1}/{len(ideas)}] {idea}")
            results += sampler.sample(prompt + condition_on(idea),
                                      n_samples=per, max_attempts=args.max_attempts)
            free_cuda()
            if len(distinct(results)) >= args.samples:
                break
    else:
        results = sampler.sample(
            prompt, n_samples=args.samples, max_attempts=args.max_attempts
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
         "model": args.model, "ideas": ideas, "programs": programs},
        indent=2,
    ))
    print(f"\nwrote {len(programs)} distinct equivalent programs to {summary}")

    if args.branching:
        print("\nskipping fptaylor: branching programs (if) are not supported by "
              "the checker yet")
    elif programs:
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
