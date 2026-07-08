"""Sample numerically-accurate branching programs, then verify each is equivalent.

The model writes a whole program in one pass under a light FPCore syntax grammar,
branching freely. Validity is checked afterwards: each branch's guards narrow the input
box and egglog proves the arm equivalent to the reference over that sub-box
(egrammar.equivalent); a program with any non-equivalent branch is dropped. fptaylor
then bounds each branch over the sub-interval where it applies.

    uv run src/run.py sqrtminus --model openai/gpt-oss-120b
"""

import argparse
import json
import os
import warnings

import egrammar
import paths
import regions

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def program_prompt(reference: str, box: dict | None) -> str:
    preamble = (paths.ROOT / "prompt_header.md").read_text()
    ranges = ""
    if box:
        spans = ", ".join(f"{v} in {iv}" for v, iv in box.items())
        ranges = f"\nThe program is only evaluated on inputs in these ranges: {spans}."
    return (
        preamble + f"\n\nThe original program is:\n{reference}\n" + ranges
        + "\n\nOutput ONE FPCore program that is algebraically equivalent to the "
        "original and numerically accurate across the range. You may branch with "
        "`(if cond ...)` on a variable and a numeric threshold when different regions "
        "need different forms; otherwise output a single form. Output only the "
        "single-line program."
    )


def float_box(box: dict | None) -> dict | None:
    if not box:
        return None
    return {v: tuple(map(float, iv.strip("[]").split(","))) for v, iv in box.items()}


def free_cuda() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def channel_ids(tokenizer):
    """(<|channel|>, <|message|>) ids for a harmony model (gpt-oss), else None."""
    ch = tokenizer.convert_tokens_to_ids("<|channel|>")
    msg = tokenizer.convert_tokens_to_ids("<|message|>")
    if None in (ch, msg) or tokenizer.unk_token_id in (ch, msg):
        return None
    return ch, msg


def at_final_header(seq, ch_id, msg_id, decode) -> bool:
    """True if `seq` ends at a `...<|channel|>final<|message|>` header."""
    if not seq or seq[-1] != msg_id:
        return False
    try:
        ch = len(seq) - 1 - seq[::-1].index(ch_id)
    except ValueError:
        return False
    return decode(seq[ch + 1:-1]).strip() == "final"


def _encode(llm, prompt, effort):
    try:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    except TypeError:
        text = llm.format_prompt(prompt)
    return llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False).to(llm.device)


def think_then_handoff(llm, prompt, temperature, effort):
    """Reason in the analysis channel, stopping when the final channel opens; return the
    token ids so generation can continue there, or None if it never opened."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

    ch_id, msg_id = channel_ids(llm.tokenizer)
    ids = _encode(llm, prompt, effort)
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
    return out if out[0, -1].item() == msg_id else None


def initial_ids(llm, prompt, args):
    """Reasoned context to generate from: reasoning + open final channel on a harmony
    model, else just the encoded prompt."""
    if channel_ids(llm.tokenizer):
        ids = think_then_handoff(llm, prompt, args.temperature, args.effort)
        free_cuda()
        if ids is not None:
            return ids
    return _encode(llm, prompt, args.effort)


def syntax_grammar(variables):
    """Light static FPCore syntax grammar over `variables`: well-formed programs,
    branching freely on a variable-vs-threshold condition. Constrains syntax only."""
    vs = " ".join(variables)
    var_alts = " | ".join(f'"{v}"' for v in variables)
    cond = " | ".join(f'"({op} " var " " NUMBER ")"' for op in ("<", ">", "<=", ">="))
    return "\n".join([
        f'start: "(FPCore ({vs}) " e ")"',
        f'e: NUMBER | {var_alts}',
        '  | "(+ " e " " e ")" | "(- " e " " e ")" | "(- " e ")"',
        '  | "(* " e " " e ")" | "(/ " e " " e ")" | "(sqrt " e ")"',
        '  | "(if " cond " " e " " e ")"',
        f"cond: {cond}",
        f"var: {var_alts}",
        r'NUMBER: /-?[0-9]+(\.[0-9]+)?/',
    ]) + "\n"


def asap_samples(llm, grammar_str, args, base_ids):
    """Up to `args.samples` distinct grammar-valid programs from ONE ASAP call: the
    oracle trie persists, so each is masked out and never re-proposed (dedup) and the
    proposal is reweighted toward the true grammar-aligned distribution."""
    import casa
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)
    res = casa.ASAP(llm, grammar, verbose=True, temperature=args.temperature).sample(
        prompt_ids=base_ids, n_samples=args.samples, max_attempts=args.max_attempts)
    free_cuda()
    return [r.text.strip() for r in res]


def validate(benchmark, box, program, runs) -> bool:
    """True if every branch is equivalent to the reference over the region its guards
    select. `if` is total, so per-branch validity implies whole-program validity."""
    for conds, leaf in regions.split_branches(regions.body_of(program)):
        region = regions.narrow_box(box, conds) if box else None
        if box and region is None:
            continue  # unreachable branch
        if not egrammar.equivalent(benchmark, float_box(region), leaf, runs):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--saturation", type=int, default=6)
    args = parser.parse_args()

    import casa
    import fptaylor_check

    reference = egrammar.read_reference(args.benchmark)
    variables = regions.variables_of(reference)
    box = fptaylor_check.INTERVALS.get(args.benchmark)
    print(f"benchmark:   {args.benchmark}\nreference:   {reference}\n"
          f"model:       {args.model}\ntemperature: {args.temperature}\n"
          f"effort:      {args.effort}\nsamples:     {args.samples}\n"
          f"saturation:  {args.saturation}\nbox:         {box or '(none)'}\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)

    base_ids = initial_ids(llm, program_prompt(reference, box), args)
    programs = []
    for prog in asap_samples(llm, syntax_grammar(variables), args, base_ids):
        if validate(args.benchmark, box, prog, args.saturation):
            programs.append(prog)
        else:
            print(f"rejected (not equivalent over the box): {prog}")

    print(f"\noriginal: {reference}\ndistinct programs: {len(programs)}")
    for i, program in enumerate(programs):
        print(f"{i:3d}  {program}")

    n, summary = paths.next_path(paths.EQUIVALENTS, args.benchmark)
    summary.write_text(json.dumps(
        {"benchmark": args.benchmark, "reference": reference,
         "config": {"model": args.model, "temperature": args.temperature,
                    "effort": args.effort, "samples": args.samples,
                    "max_attempts": args.max_attempts, "saturation": args.saturation,
                    "box": box},
         "programs": programs}, indent=2))
    print(f"\nwrote {len(programs)} programs to {summary}")

    if programs:
        try:
            fptaylor_check.check(args.benchmark, run=n)
        except KeyError:
            print(f"skipping fptaylor: no interval box for {args.benchmark!r}")
        except FileNotFoundError as e:
            msg = "fptaylor binary not on PATH" if "fptaylor" in str(e).lower() else str(e)
            print(f"skipping fptaylor: {msg}")


if __name__ == "__main__":
    main()
