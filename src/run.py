"""Sample accurate branching programs, building each branch's grammar on the fly.

The model generates in shared-context segments (one growing token sequence, so it
sees everything it has written):
  - a HEAD grammar lets it emit either a complete no-branch program, or the opening
    `(FPCore (v) (if (op v <threshold>)` with an ARBITRARY numeric threshold;
  - once the threshold is known, each arm's box is narrowed by the condition and a
    region grammar is built ON THE FLY over that sub-box (sound by construction), and
    generation continues under it.
fptaylor then bounds each branch over the sub-interval where it applies.

    uv run src/run.py sqrtminus --model openai/gpt-oss-120b
"""

import argparse
import json
import os
import re
import warnings

import egrammar
import paths
import regions

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
COND = re.compile(r"\(if \((<=|>=|<|>) (\w+) (-?\d+(?:\.\d+)?)\)")


def _preamble() -> str:
    return (paths.ROOT / "prompt_header.md").read_text()


def _range_line(box: dict | None) -> str:
    if not box:
        return ""
    ranges = ", ".join(f"{v} in {iv}" for v, iv in box.items())
    return f"\nThe program is only evaluated on inputs in these ranges: {ranges}."


def program_prompt(reference: str, box: dict | None) -> str:
    return (
        _preamble() + f"\n\nThe original program is:\n{reference}\n" + _range_line(box)
        + "\n\nOutput ONE FPCore program that is algebraically equivalent to the "
        "original and numerically accurate across the range. You may branch with "
        "`(if cond ...)` on a variable and a numeric threshold when different regions "
        "need different forms; otherwise output a single form. Output only the "
        "single-line program."
    )


def dedup(programs) -> list[str]:
    seen, out = set(), []
    for p in programs:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def float_box(box: dict | None) -> dict | None:
    if not box:
        return None
    return {v: tuple(map(float, iv.strip("[]").split(","))) for v, iv in box.items()}


def free_cuda() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def channel_ids(tokenizer):
    """(<|channel|>, <|message|>) token ids for a harmony model (gpt-oss), else None."""
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
    """Prompt token ids, applying the chat template (with gpt-oss reasoning effort)."""
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
    """The context to generate the program from: reasoning + open final channel on a
    harmony model, else just the encoded prompt."""
    if channel_ids(llm.tokenizer):
        ids = think_then_handoff(llm, prompt, args.temperature, args.effort)
        free_cuda()
        if ids is not None:
            return ids
    return _encode(llm, prompt, args.effort)


def sample_segment(llm, grammar_str, ids, args):
    """One grammar-constrained segment continued from `ids`; returns the SamplingResult
    (with .text and .token_ids) or None."""
    import casa
    grammar = casa.Grammar.from_string(grammar_str, llm.tokenizer)
    res = casa.ASAP(llm, grammar, verbose=True, temperature=args.temperature).sample(
        prompt_ids=ids, n_samples=1, max_attempts=args.max_attempts)
    free_cuda()
    return res[0] if res else None


def extend(ids, token_ids):
    import torch
    return torch.cat([ids, torch.tensor([token_ids], device=ids.device)], dim=1)


def head_grammar(benchmark, box, variables, runs):
    """Emit either a complete no-branch program, or the opening
    `(FPCore (v) (if (op var <NUMBER>)` -- an arbitrary threshold, stopping at the `)`."""
    vs = " ".join(variables)
    cmps = ("<", ">", "<=", ">=")
    cond = " | ".join(f'"({op} " operand " " NUMBER ")"' for op in cmps)
    return "\n".join([
        f'start: "(FPCore ({vs}) " e0 ")" | "(FPCore ({vs}) (if " cond',
        egrammar.region_rules(benchmark, box, runs),
        f"cond: {cond}",
        "operand: " + " | ".join(f'"{v}"' for v in variables),
        r'NUMBER: /-?[0-9]+(\.[0-9]+)?/',
    ]) + "\n"


def arm_grammar(benchmark, box, runs, closes):
    """A leaf over `box`, with a leading space and `closes` trailing `)` (0 for the
    then-arm; 2 for the else-arm, to close the `if` and the `FPCore`)."""
    suffix = "".join(' ")"' for _ in range(closes))
    return f'start: " " e0{suffix}\n{egrammar.region_rules(benchmark, box, runs)}\n'


def generate_program(llm, benchmark, reference, variables, box, args):
    ids = initial_ids(llm, program_prompt(reference, box), args)
    if box is None:  # no interval box configured -> no branching, plain equivalents
        r = sample_segment(llm, egrammar.build_region(benchmark, None, args.saturation)[1],
                            ids, args)
        return r.text.strip() if r else None

    fbox = float_box(box)
    head = sample_segment(llm, head_grammar(benchmark, fbox, variables, args.saturation),
                          ids, args)
    if head is None:
        return None
    text = head.text.strip()
    m = COND.search(text)
    if m is None:  # complete no-branch program
        return text
    op, var, thr = m.group(1), m.group(2), m.group(3)
    then_box = regions.narrow_box(box, [(op, var, thr)]) or box
    else_box = regions.narrow_box(box, [(regions.NEGATE[op], var, thr)]) or box

    ids = extend(ids, head.token_ids)
    then = sample_segment(llm, arm_grammar(benchmark, float_box(then_box), args.saturation, 0),
                          ids, args)
    if then is None:
        return None
    ids = extend(ids, then.token_ids)
    els = sample_segment(llm, arm_grammar(benchmark, float_box(else_box), args.saturation, 2),
                         ids, args)
    if els is None:
        return None
    return text + then.text + els.text


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
    variables = regions.parse(regions.tokenize(reference))[1]  # (FPCore (vars) ...)
    box = fptaylor_check.INTERVALS.get(args.benchmark)
    print(f"benchmark: {args.benchmark}\nreference: {reference}\n"
          f"model:     {args.model}\nbox:       {box or '(none)'}\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)

    programs = []
    for i in range(args.samples):
        print(f"{'=' * 70}\nsample {i + 1}/{args.samples}\n{'=' * 70}")
        prog = generate_program(llm, args.benchmark, reference, variables, box, args)
        if prog:
            programs.append(prog)

    programs = dedup(programs)
    print(f"\noriginal: {reference}\ndistinct programs: {len(programs)}")
    for i, program in enumerate(programs):
        print(f"{i:3d}  {program}")

    n, summary = paths.next_path(paths.EQUIVALENTS, args.benchmark)
    summary.write_text(json.dumps(
        {"benchmark": args.benchmark, "reference": reference,
         "model": args.model, "programs": programs}, indent=2))
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
