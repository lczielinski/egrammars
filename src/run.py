"""Sample equivalent programs from an egrammar-compiled grammar with casa.

Sampling uses casa's asap sampler. The model produces one program that branches on
the input with `if`: it reasons about where the original loses accuracy -- input
regions where a + or - cancels (large condition number) or an intermediate overflows
-- then a grammar-constrained pass writes a single program that branches into the
algebraically-equivalent form that is accurate in each region. fptaylor then bounds
each branch's rounding error over the sub-interval where that branch applies.

Options:
    benchmark            benchmark name, e.g. quadratic (positional, required)
    --samples N          distinct programs to collect (default 20)
    --max-attempts N     cap on attempts per sample (default 200)
    --temperature T      sampling temperature applied to the model (default 1.0);
                         T<1 sharpens, T>1 flattens the grammar-constrained model
    --model ID           HuggingFace model id to load (default Qwen2.5-14B-Instruct)
    --effort LEVEL       gpt-oss reasoning effort for the reasoning phase:
                         low, medium, or high (default medium)
    --saturation N       rewrite-rule iterations when compiling a grammar (default
                         6; only used if not cached). Lower it for symmetry-heavy
                         expressions whose grammar explodes

Examples:
    uv run src/run.py quadratic --samples 50
    uv run src/run.py sqrtshift --model openai/gpt-oss-120b --saturation 4
"""

import argparse
import json
import os
import warnings

import paths

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def make_prompt(reference: str) -> str:
    header = (paths.ROOT / "prompt_header.md").read_text()
    return f"{header}\n\nThe original program is:\n{reference}"


def ensure_artifacts(benchmark: str, saturation: int) -> tuple[str, str, str]:
    import egrammar

    grammar_path = paths.LARK / f"{benchmark}-branching.lark"
    if grammar_path.exists():
        reference, grammar = egrammar.read_reference(benchmark), grammar_path.read_text()
    else:
        print(f"Compiling grammar for {benchmark!r} "
              f"(no cached grammar, saturation={saturation})")
        reference, grammar = egrammar.build(benchmark, saturation, branching=True)
        egrammar.write_grammar(benchmark, grammar, branching=True)
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
    parser.add_argument("--saturation", type=int, default=6,
                        help="rewrite-rule iterations when compiling a grammar")
    args = parser.parse_args()

    grammar_str, prompt, reference = ensure_artifacts(args.benchmark, args.saturation)

    import casa

    print(f"benchmark: {args.benchmark}")
    print(f"grammar:   {paths.LARK / f'{args.benchmark}-branching.lark'} "
          f"({grammar_str.count(chr(10))} rules)")
    print(f"model:     {args.model}")
    print(f"temp:      {args.temperature}")
    print(f"target {args.samples} programs, <= {args.max_attempts} attempts/sample\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)

    # On a harmony model (gpt-oss) the model reasons first
    prompt_ids = None
    if channel_ids(llm.tokenizer):
        print(f"{'=' * 70}\nreasoning phase\n{'=' * 70}")
        prompt_ids, _, opened = think_then_handoff(
            llm, prompt, args.temperature, args.effort)
        print(f"\n{'=' * 70}\n")
        free_cuda()
        if not opened:
            print("warning: model never opened its final channel; sampling "
                  "without reasoning.\n")
            prompt_ids = None

    sampler = casa.ASAP(llm, grammar, verbose=True, temperature=args.temperature)
    results = sampler.sample(
        prompt if prompt_ids is None else None, n_samples=args.samples,
        max_attempts=args.max_attempts, prompt_ids=prompt_ids,
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
         "model": args.model, "programs": programs},
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
