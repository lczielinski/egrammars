"""Sample accurate branching programs region-by-region (casa + egglog).

Phase 1: the model, constrained to a skeleton grammar, emits an `if`-tree over the
input range with `?` arm holes. Phase 2/3: each hole's guards narrow the box, egglog
builds a grammar sound over that sub-box, and a constrained pass fills the arm.
fptaylor then bounds each branch over its sub-interval.

    uv run src/run.py sqrtminus --model openai/gpt-oss-120b
"""

import argparse
import json
import os
import warnings

import paths
import regions

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def _preamble() -> str:
    return (paths.ROOT / "prompt_header.md").read_text()


def _range_line(box: dict | None) -> str:
    if not box:
        return ""
    ranges = ", ".join(f"{v} in {iv}" for v, iv in box.items())
    return f"\nThe program is only evaluated on inputs in these ranges: {ranges}."


def partition_prompt(reference: str, box: dict | None) -> str:
    return (
        _preamble() + f"\n\nThe original program is:\n{reference}\n" + _range_line(box)
        + "\n\nDecide how to split this input range into pieces that each need a "
        "different accurate form. Split ONLY where the accurate form must change -- a "
        "sign flip, or a cancellation/overflow that one form avoids and another does "
        "not. If a single form is accurate across the whole range, do NOT split. "
        "Output ONLY an FPCore `if`-skeleton whose arms are `?` placeholders -- e.g. "
        "(FPCore (x) (if (> x 0) ? ?)) to split at x=0, or (FPCore (x) ?) for no "
        "split. Use `?` for every arm; write no real subexpression."
    )


def arm_prompt(reference: str, box: dict | None) -> str:
    return (
        _preamble() + f"\n\nThe original program is:\n{reference}\n" + _range_line(box)
        + "\n\nOutput ONE FPCore program that is algebraically equivalent to the "
        "original and numerically accurate across this range. Output only the "
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


def think_then_handoff(llm, prompt, temperature, effort):
    """Reason in the analysis channel, stopping when the final channel opens; return
    the token ids so a constrained pass can continue there, or None if it never did."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

    ch_id, msg_id = channel_ids(llm.tokenizer)
    try:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    except TypeError:
        text = llm.format_prompt(prompt)
    ids = llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False).to(llm.device)
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


def constrained_sample(llm, grammar, prompt, args, reason: bool) -> str | None:
    """One grammar-constrained program; on a harmony model, `reason` first thinks
    unconstrained, then continues under the grammar in the same final channel."""
    import casa
    prompt_ids = None
    if reason and channel_ids(llm.tokenizer):
        prompt_ids = think_then_handoff(llm, prompt, args.temperature, args.effort)
        free_cuda()
    sampler = casa.ASAP(llm, grammar, verbose=True, temperature=args.temperature)
    results = sampler.sample(
        prompt if prompt_ids is None else None, n_samples=1,
        max_attempts=args.max_attempts, prompt_ids=prompt_ids)
    free_cuda()
    return results[0].text.strip() if results else None


def build_branching_program(llm, benchmark, reference, variables, box, skel_grammar,
                            args, grammar_cache) -> str | None:
    """Propose a partition, fill each region's arm from a region-sound grammar, assemble."""
    import casa
    import egrammar

    skeleton = constrained_sample(llm, skel_grammar, partition_prompt(reference, box),
                                  args, reason=True)
    if skeleton is None:
        return None
    arm_bodies = []
    for conds in regions.leaf_paths(skeleton):
        rbox = regions.narrow_box(box, conds) if box else None
        if box and rbox is None:  # region empty in the box; arm never executes
            arm_bodies.append(regions.strip_wrapper(reference, variables))
            continue
        key = None if rbox is None else tuple(sorted(rbox.items()))
        if key not in grammar_cache:
            _, g = egrammar.build_region(benchmark, float_box(rbox), args.saturation)
            grammar_cache[key] = casa.Grammar.from_string(g, llm.tokenizer)
        arm = constrained_sample(llm, grammar_cache[key], arm_prompt(reference, rbox),
                                 args, reason=False)
        if arm is None:
            return None
        arm_bodies.append(regions.strip_wrapper(arm, variables))
    return regions.assemble(skeleton, arm_bodies)


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
    import egrammar
    import fptaylor_check

    reference = egrammar.read_reference(args.benchmark)
    variables = regions.variables_of(reference)
    box = fptaylor_check.INTERVALS.get(args.benchmark)
    print(f"benchmark: {args.benchmark}\nreference: {reference}\n"
          f"model:     {args.model}\nbox:       {box or '(none)'}\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)
    skel_grammar = casa.Grammar.from_string(
        regions.skeleton_grammar(variables), llm.tokenizer)

    grammar_cache, programs = {}, []
    for i in range(args.samples):
        print(f"{'=' * 70}\nsample {i + 1}/{args.samples}\n{'=' * 70}")
        prog = build_branching_program(llm, args.benchmark, reference, variables, box,
                                       skel_grammar, args, grammar_cache)
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
