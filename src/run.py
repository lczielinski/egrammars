"""Sample accurate branching programs, in one of two modes (--mode).

check (default):   the model writes a WHOLE program in one pass under a light,
    static SYNTAX grammar (well-formed FPCore, allowed ops/vars, threshold or
    var-vs-var conditions), free to branch however it likes. Validity is checked
    AFTER the fact: each branch's guards narrow the box and egglog proves the arm
    equivalent to the reference over that sub-box (egrammar.equivalent). Programs
    with any non-equivalent branch are dropped. Semantics live in the checker;
    the grammar only enforces syntax.

skeleton:          the constrained two-phase approach, batched through ASAP. In one
    call the model proposes --skeletons distinct `if`-trees with `?` arm holes; then for
    each hole the guards narrow the box, egglog builds a grammar sound over that sub-box,
    and (after reasoning once about the sub-box) one ASAP call draws --arms distinct
    fills from it (region grammars cached). Each skeleton is assembled as the
    cross-product of its holes' options (capped at
    --max-combos). Sound by construction. Batching the draws keeps ASAP's dedup +
    reweighting instead of degrading to per-call GCD.

Either way, fptaylor then bounds each branch over the sub-interval where it applies.

    uv run src/run.py sqrtminus --model openai/gpt-oss-120b
    uv run src/run.py sqrtminus --mode skeleton
"""

import argparse
import itertools
import json
import math
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


def asap_samples(llm, grammar, args, n_samples, *, prompt_ids=None, prompt=None):
    """Up to `n_samples` DISTINCT grammar-valid programs (stripped text) from ONE ASAP
    call, continued from `prompt_ids` or generated from `prompt`. Drawing them in a
    single call is what makes it ASAP rather than GCD: the oracle trie persists, so each
    program it yields is masked out and never re-proposed (dedup), and the proposal is
    reweighted toward the true grammar-aligned distribution."""
    import casa
    res = casa.ASAP(llm, grammar, verbose=True, temperature=args.temperature).sample(
        prompt, prompt_ids=prompt_ids, n_samples=n_samples, max_attempts=args.max_attempts)
    free_cuda()
    return [r.text.strip() for r in res]


def reasoned_samples(llm, grammar, prompt, args, n_samples):
    """Reason once about `prompt` (on a harmony model, in the analysis channel), then
    draw `n_samples` distinct grammar-valid programs from that shared reasoned context
    in one ASAP call."""
    prompt_ids = None
    if channel_ids(llm.tokenizer):
        prompt_ids = think_then_handoff(llm, prompt, args.temperature, args.effort)
        free_cuda()
    return asap_samples(llm, grammar, args, n_samples,
                        prompt=(prompt if prompt_ids is None else None),
                        prompt_ids=prompt_ids)


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


# --- skeleton mode: a few skeletons, each hole filled from a region grammar, assembled -

def arm_options(llm, benchmark, reference, variables, rbox, args):
    """`args.arms` DISTINCT region-sound arm bodies over `rbox`: reason once about this
    sub-box, then draw them in one ASAP call from a grammar whose language is exactly the
    forms equivalent to the reference and sound over the sub-box (no post-hoc check
    needed). Called once per distinct region and cached, so the reasoning is shared."""
    _, grammar_str = egrammar.build_region(benchmark, float_box(rbox), args.saturation)
    grammar = make_grammar(llm, grammar_str)
    bodies = reasoned_samples(llm, grammar, arm_prompt(reference, rbox), args, args.arms)
    return [regions.strip_wrapper(b, variables) for b in bodies]


def build_skeleton_programs(llm, benchmark, reference, variables, box, skel_grammar,
                            args, arm_cache):
    """Draw `args.skeletons` distinct `if`-skeletons in one ASAP call (reasoning once
    about the partition first); fill each hole with `args.arms` distinct region-sound
    options (reasoning once per region, then one ASAP call, cached); assemble the
    per-skeleton cross-product of options, capped at `args.max_combos` per skeleton."""
    skeletons = reasoned_samples(llm, skel_grammar, partition_prompt(reference, box),
                                 args, args.skeletons)

    programs = []
    for skeleton in skeletons:
        hole_options, reachable = [], True
        for conds in regions.leaf_paths(skeleton):
            rbox = regions.narrow_box(box, conds) if box else None
            if box and rbox is None:  # region empty in the box; arm never executes
                hole_options.append([regions.strip_wrapper(reference, variables)])
                continue
            key = None if rbox is None else tuple(sorted(rbox.items()))
            if key not in arm_cache:
                arm_cache[key] = arm_options(llm, benchmark, reference, variables, rbox, args)
            if not arm_cache[key]:  # region grammar produced nothing
                reachable = False
                break
            hole_options.append(arm_cache[key])
        if not reachable:
            continue
        total = math.prod(len(opts) for opts in hole_options)
        if total > args.max_combos:
            print(f"[skeleton] {skeleton}: {total} combinations, capping to "
                  f"{args.max_combos}")
        for combo in itertools.islice(itertools.product(*hole_options), args.max_combos):
            programs.append(regions.assemble(skeleton, list(combo)))
    return programs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark")
    parser.add_argument("--mode", choices=["check", "skeleton"], default="check")
    parser.add_argument("--samples", type=int, default=20,
                        help="check mode: distinct programs to draw")
    parser.add_argument("--skeletons", type=int, default=3,
                        help="skeleton mode: distinct if-skeletons to draw")
    parser.add_argument("--arms", type=int, default=3,
                        help="skeleton mode: options to draw per skeleton hole")
    parser.add_argument("--max-combos", type=int, default=12,
                        help="skeleton mode: cap on assembled programs per skeleton")
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
    budget = (f"samples:     {args.samples}" if args.mode == "check" else
              f"skeletons:   {args.skeletons}\narms:        {args.arms}\n"
              f"max_combos:  {args.max_combos}")
    print(f"benchmark:   {args.benchmark}\nreference:   {reference}\nmode:        {args.mode}\n"
          f"model:       {args.model}\ntemperature: {args.temperature}\n"
          f"effort:      {args.effort}\n{budget}\n"
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
        programs = build_skeleton_programs(llm, args.benchmark, reference, variables, box,
                                           skel_grammar, args, arm_cache={})

    programs = dedup(programs)
    print(f"\noriginal: {reference}\ndistinct programs: {len(programs)}")
    for i, program in enumerate(programs):
        print(f"{i:3d}  {program}")

    n, summary = paths.next_path(paths.EQUIVALENTS, args.benchmark)
    summary.write_text(json.dumps(
        {"benchmark": args.benchmark, "reference": reference,
         "config": {"mode": args.mode, "model": args.model,
                    "temperature": args.temperature, "effort": args.effort,
                    "samples": args.samples, "skeletons": args.skeletons,
                    "arms": args.arms, "max_combos": args.max_combos,
                    "max_attempts": args.max_attempts,
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
