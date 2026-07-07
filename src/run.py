"""Sample accurate branching programs, in one of two modes (--mode).

check (default):   the model writes a WHOLE program in one pass under a light,
    static SYNTAX grammar (well-formed FPCore, allowed ops/vars, threshold or
    var-vs-var conditions), free to branch however it likes. Validity is checked
    AFTER the fact: each branch's guards narrow the box and egglog proves the arm
    equivalent to the reference over that sub-box (egrammar.equivalent). Programs
    with any non-equivalent branch are dropped. Semantics live in the checker;
    the grammar only enforces syntax.

skeleton:          the two-phase constrained approach. The model, constrained to a
    skeleton grammar, emits an `if`-tree with `?` arm holes; then each hole's guards
    narrow the box, egglog builds a grammar sound over that sub-box, and a
    constrained pass fills the arm. Sound by construction, but multiple generations.

Either way, fptaylor then bounds each branch over the sub-interval where it applies.

    uv run src/run.py sqrtminus --model openai/gpt-oss-120b
    uv run src/run.py sqrtminus --mode skeleton
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
    """The context to generate from: reasoning + open final channel on a harmony model,
    else just the encoded prompt."""
    if channel_ids(llm.tokenizer):
        ids = think_then_handoff(llm, prompt, args.temperature, args.effort)
        free_cuda()
        if ids is not None:
            return ids
    return _encode(llm, prompt, args.effort)


def make_grammar(llm, grammar_str):
    import casa
    return casa.Grammar.from_string(grammar_str, llm.tokenizer)


def sample(llm, grammar, args, *, prompt_ids=None, prompt=None):
    """One grammar-constrained program (its stripped text) continued from `prompt_ids`
    or generated from `prompt`, or None if the sampler produced nothing."""
    import casa
    res = casa.ASAP(llm, grammar, verbose=True, temperature=args.temperature).sample(
        prompt, prompt_ids=prompt_ids, n_samples=1, max_attempts=args.max_attempts)
    free_cuda()
    return res[0].text.strip() if res else None


# --- check mode: free generation under a light syntax grammar, checked after ---------

def syntax_grammar(variables):
    """A light, static FPCore syntax grammar: well-formed programs over the allowed ops
    and `variables`, branching freely, with threshold or var-vs-var conditions. It
    constrains SYNTAX only -- equivalence is checked afterwards by egrammar.equivalent."""
    vs = " ".join(variables)
    var_alts = " | ".join(f'"{v}"' for v in variables)
    cmps = ("<", ">", "<=", ">=")
    cond = " | ".join(f'"({op} " operand " " operand ")"' for op in cmps)
    return "\n".join([
        f'start: "(FPCore ({vs}) " e ")"',
        f'e: NUMBER | {var_alts}',
        '  | "(+ " e " " e ")" | "(- " e " " e ")" | "(- " e ")"',
        '  | "(* " e " " e ")" | "(/ " e " " e ")" | "(sqrt " e ")"',
        '  | "(if " cond " " e " " e ")"',
        f"cond: {cond}",
        f"operand: {var_alts} | NUMBER",
        r'NUMBER: /-?[0-9]+(\.[0-9]+)?/',
    ]) + "\n"


def validate(benchmark, box, program, runs) -> bool:
    """True if every branch of `program` is equivalent to the reference over the region
    its guards select. Per-branch validity implies whole-program validity: `if` is
    total, so each input lands in exactly one branch."""
    for conds, leaf in regions.split_branches(regions.body_of(program)):
        region = regions.narrow_box(box, conds) if box else None
        if box and region is None:
            continue  # branch unreachable in the box -- nothing to prove
        if not egrammar.equivalent(benchmark, float_box(region), leaf, runs):
            return False
    return True


# --- skeleton mode: constrained skeleton, then fill each arm from a region grammar ----

def constrained_sample(llm, grammar, prompt, args, reason: bool):
    """One grammar-constrained program; on a harmony model, `reason` first thinks
    unconstrained, then continues under the grammar in the same final channel."""
    prompt_ids = None
    if reason and channel_ids(llm.tokenizer):
        prompt_ids = think_then_handoff(llm, prompt, args.temperature, args.effort)
        free_cuda()
    return sample(llm, grammar, args, prompt_ids=prompt_ids,
                  prompt=(prompt if prompt_ids is None else None))


def build_skeleton_program(llm, benchmark, reference, variables, box, skel_grammar,
                           args, grammar_cache):
    """Propose a partition, fill each region's arm from a region-sound grammar, assemble."""
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
            grammar_cache[key] = make_grammar(llm, g)
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
    parser.add_argument("--mode", choices=["check", "skeleton"], default="check")
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
    print(f"benchmark:   {args.benchmark}\nreference:   {reference}\nmode:        {args.mode}\n"
          f"model:       {args.model}\ntemperature: {args.temperature}\n"
          f"effort:      {args.effort}\nsamples:     {args.samples}\n"
          f"saturation:  {args.saturation}\nbox:         {box or '(none)'}\n")

    load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
    llm = casa.LLM.from_pretrained(args.model, **load_kwargs)

    programs = []
    if args.mode == "check":
        # Reason once, then draw all programs from that shared context in ONE ASAP call
        # so its trie persists across samples: each grammar-valid program it yields is
        # masked out and never proposed again (dedup during generation), and the
        # proposal is reweighted toward the true grammar-aligned distribution. Calling
        # it once per sample would reset that state and degrade to plain GCD.
        base_ids = initial_ids(llm, program_prompt(reference, box), args)
        syntax = make_grammar(llm, syntax_grammar(variables))
        sampler = casa.ASAP(llm, syntax, verbose=True, temperature=args.temperature)
        candidates = sampler.sample(prompt_ids=base_ids, n_samples=args.samples,
                                    max_attempts=args.max_attempts)
        for r in candidates:
            prog = r.text.strip()
            if validate(args.benchmark, box, prog, args.saturation):
                programs.append(prog)
            else:
                print(f"rejected (not equivalent over the box): {prog}")
    else:  # skeleton
        skel_grammar = make_grammar(llm, regions.skeleton_grammar(variables))
        grammar_cache = {}
        for i in range(args.samples):
            print(f"{'=' * 70}\nsample {i + 1}/{args.samples}\n{'=' * 70}")
            prog = build_skeleton_program(llm, args.benchmark, reference, variables, box,
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
         "config": {"mode": args.mode, "model": args.model,
                    "temperature": args.temperature, "effort": args.effort,
                    "samples": args.samples, "max_attempts": args.max_attempts,
                    "saturation": args.saturation, "box": box},
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
