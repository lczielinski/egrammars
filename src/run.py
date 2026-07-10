"""Generate numerically-accurate FPCore rewrites and verify each is equivalent.

The model reasons once, then samples programs under a light FPCore syntax grammar,
branching freely. Each candidate is verified afterwards: every branch's guards narrow the
input box and egglog proves the arm equivalent to the reference over that sub-box
(egrammar.equivalent); a program with any non-equivalent branch is dropped. FPTaylor then
bounds the survivors. Every candidate the model tried is recorded in the `attempts` field
with why it was rejected (missing rule vs. genuinely non-equivalent).

    uv run src/run.py x_by_xy              # one benchmark
    uv run src/run.py                      # every benchmark, model loaded once
    uv run src/run.py --shard 0/8          # one shard per GPU (see scripts/run_gpus.sh)
    uv run src/run.py --summary-only       # re-print the results table, no generation
"""

import argparse
import json
import os
import warnings

import egrammar
import fptaylor_check
import paths
import regions

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")

MODEL_ID = "openai/gpt-oss-120b"


def program_prompt(reference: str, box: dict | None) -> str:
    preamble = (paths.ROOT / "prompt_header.md").read_text()
    ranges = ""
    if box:
        spans = ", ".join(f"{v} in {iv}" for v, iv in box.items())
        ranges = f"\nThe program is only evaluated on inputs in these ranges: {spans}."
    return (
        preamble + f"\n\nThe original program is:\n{reference}\n" + ranges
        + "\n\nGoal: ONE FPCore program, algebraically equivalent to the original, that "
        "is accurate across the whole input range.\n"
        "1. Using the condition numbers, find where inside the range the program loses "
        "accuracy -- a `+` or `-` whose operands nearly cancel, a `/` by a near-zero "
        "value, or an intermediate that overflows or underflows.\n"
        "2. If a single algebraically-equivalent form is accurate everywhere in the "
        "range, output just that form with NO `if`. Prefer this: only branch when "
        "different parts of the range genuinely need different forms. A needless `if` is "
        "worse than one clean form.\n"
        "3. When you do branch, split on a variable and a numeric threshold and give "
        "each fragile region a rewrite that is well-conditioned and in-range there, so "
        "every input takes the form accurate for it. The arms must be genuinely "
        "different forms; never repeat the same form in both. Every branch must equal "
        "the original in exact arithmetic; only the rounding may differ.\n\n"
        "Output only the single-line program, then immediately stop."
    )


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


def validate(benchmark, box, program, timeout) -> bool:
    """True if every branch is equivalent to the reference over the region its guards
    select. `if` is total, so per-branch validity implies whole-program validity. Each
    branch check saturates until proven or its `timeout` budget."""
    for conds, leaf in regions.split_branches(regions.body_of(program)):
        region = regions.narrow_box(box, conds) if box else None
        if box and region is None:
            continue  # unreachable branch
        if not egrammar.equivalent(benchmark, regions.float_box(region), leaf,
                                   timeout=timeout):
            return False
    return True


def evaluate_candidates(benchmark, reference, box, candidates, timeout):
    """Returns (proven_programs, attempts). Every candidate is recorded; an unproven one
    gets a `numeric` classification (missing rule vs. genuinely non-equivalent)."""
    import numcheck
    programs, attempts = [], []
    for prog in candidates:
        proven = validate(benchmark, box, prog, timeout=timeout)
        rec = {"program": prog, "proven_equivalent": proven}
        if proven:
            programs.append(prog)
        else:
            rec["numeric"] = numcheck.classify(reference, prog, box)
        attempts.append(rec)
    return programs, attempts


def attempt_line(rec: dict) -> str:
    if rec["proven_equivalent"]:
        return f"  proven      {rec['program']}"
    num = rec["numeric"]
    tag = {"equal": "MISSING-RULE?", "different": "not-equivalent",
           "indeterminate": "indeterminate"}[num["verdict"]]
    return f"  {tag:<13} {rec['program']}"


def all_benchmarks() -> list[str]:
    return sorted(p.stem for p in paths.EGGLOG.glob("*.egglog"))


def shard(benchmarks: list[str], spec: str | None) -> list[str]:
    """`spec` = "I/N": keep every N-th benchmark from I (round-robin, so slow cores
    spread evenly across the N GPU shards). None -> all."""
    if not spec:
        return benchmarks
    i, n = (int(x) for x in spec.split("/"))
    return benchmarks[i::n]


def run_benchmark(llm, benchmark: str, args) -> None:
    """Generate, verify, bound, and write equivalents/<benchmark>-NNN.json for one
    benchmark, against an already-loaded model."""
    reference = egrammar.read_reference(benchmark)
    box = fptaylor_check.INTERVALS.get(benchmark)
    print(f"\n{'=' * 70}\n{benchmark}   box: {box or '(none)'}\n{reference}\n{'=' * 70}")

    base_ids = initial_ids(llm, program_prompt(reference, box), args)
    candidates = asap_samples(llm, syntax_grammar(regions.variables_of(reference)),
                              args, base_ids)
    programs, attempts = evaluate_candidates(
        benchmark, reference, box, candidates, timeout=args.time_budget)

    missing = sum(a.get("numeric", {}).get("verdict") == "equal" for a in attempts)
    print(f"{len(attempts)} candidates: {len(programs)} proven, {missing} missing-rule?, "
          f"{len(attempts) - len(programs) - missing} non-equivalent")
    for rec in attempts:
        print(attempt_line(rec))

    n, summary = paths.next_path(paths.EQUIVALENTS, benchmark)
    summary.write_text(json.dumps(
        {"benchmark": benchmark, "reference": reference,
         "config": {"model": args.model, "temperature": args.temperature,
                    "effort": args.effort, "samples": args.samples,
                    "max_attempts": args.max_attempts,
                    "time_budget": args.time_budget, "box": box},
         "programs": programs, "attempts": attempts}, indent=2))
    print(f"wrote {summary}")
    if programs:
        try:
            fptaylor_check.check(benchmark, run=n)
        except (KeyError, FileNotFoundError) as e:
            print(f"skipping fptaylor: {e}")


def _metric(r):
    """Comparable error for a result: relative ulps if defined, else absolute error."""
    if r and r.get("rel_err_ulps") is not None:
        return ("rel_ulp", r["rel_err_ulps"])
    if r and r.get("abs_err") is not None:
        return ("abs", r["abs_err"])
    return (None, None)


def _benchmark_row(b):
    """Per-benchmark stats from its latest equivalents (+ fptaylor) run, or None."""
    _, esrc = paths.latest(paths.EQUIVALENTS, b)
    if esrc is None:
        return None
    attempts = json.loads(esrc.read_text()).get("attempts", [])
    v = lambda pred: sum(1 for a in attempts if pred(a))
    verd = lambda a: a.get("numeric", {}).get("verdict")
    unproven = lambda a: not a["proven_equivalent"]
    valid = v(lambda a: a["proven_equivalent"])
    row = {"benchmark": b, "candidates": len(attempts), "valid": valid,
           "missing_rule": v(lambda a: unproven(a) and verd(a) == "equal"),
           "non_equiv": v(lambda a: unproven(a) and verd(a) == "different"),
           "indeterminate": v(lambda a: unproven(a) and verd(a) == "indeterminate"),
           "best_ulp": None, "verdict": "unmeasurable" if valid else "no-valid"}
    _, fsrc = paths.latest(paths.FPTAYLOR, b)
    if fsrc is not None:
        fd = json.loads(fsrc.read_text())
        results = fd.get("results", [])
        row["best_ulp"] = next((r["rel_err_ulps"] for r in results
                                if r.get("rel_err_ulps") is not None), None)
        if row["valid"]:
            best_k, best_m = None, None
            for r in results:
                k, m = _metric(r)
                if m is not None and (best_m is None or m < best_m):
                    best_k, best_m = k, m
            rk, rm = _metric(fd.get("reference_result"))
            if rm is None or best_m is None or rk != best_k:
                row["verdict"] = "unmeasurable"
            elif best_m < rm * 0.99:
                row["verdict"] = "improved"
            elif best_m > rm * 1.01:
                row["verdict"] = "worse"
            else:
                row["verdict"] = "no-change"
    return row


def summarize(benchmarks: list[str]) -> None:
    """Aggregate the latest runs into console output and summary.md: fraction of candidates
    valid / missing-rule / non-equivalent, and of benchmarks improved over the reference."""
    rows = [r for r in (_benchmark_row(b) for b in benchmarks) if r]
    if not rows:
        print("no results found (run some benchmarks first)")
        return

    tot = sum(r["candidates"] for r in rows) or 1
    prog = {k: sum(r[k] for r in rows) for k in ("valid", "missing_rule", "non_equiv", "indeterminate")}
    nb = len(rows)
    with_valid = sum(r["valid"] > 0 for r in rows)
    with_missing = sum(r["missing_rule"] > 0 for r in rows)
    verd = {k: sum(r["verdict"] == k for r in rows)
            for k in ("improved", "no-change", "worse", "unmeasurable", "no-valid")}

    def pct(n, d):
        return f"{100 * n / d:.1f}%"

    lines = [
        "# Benchmark run summary", "",
        f"Benchmarks with results: **{nb}**   |   candidates evaluated: **{tot}**", "",
        "## Programs (all candidates the model produced)",
        f"- valid (proven equivalent): **{prog['valid']} ({pct(prog['valid'], tot)})**",
        f"- invalid — missing e-graph rule (numerically equal, unproven): {prog['missing_rule']} ({pct(prog['missing_rule'], tot)})",
        f"- invalid — non-equivalent (model error): {prog['non_equiv']} ({pct(prog['non_equiv'], tot)})",
        f"- indeterminate (no finite sample point): {prog['indeterminate']} ({pct(prog['indeterminate'], tot)})",
        "",
        "## Benchmarks",
        f"- produced >=1 valid rewrite: **{with_valid}/{nb} ({pct(with_valid, nb)})**",
        f"- accuracy improved over reference: **{verd['improved']}/{nb} ({pct(verd['improved'], nb)})**",
        f"- valid rewrite but no accuracy gain: {verd['no-change']}/{nb}",
        f"- best valid rewrite was worse than reference: {verd['worse']}/{nb}",
        f"- unmeasurable (box straddles zero / singularity): {verd['unmeasurable']}/{nb}",
        f"- no valid rewrite found: {verd['no-valid']}/{nb}",
        f"- had a missing-rule candidate: {with_missing}/{nb} ({pct(with_missing, nb)})",
        "",
        "## Per-benchmark",
        "| benchmark | candidates | valid | best rel (ulp) | vs reference |",
        "|---|--:|--:|--:|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["verdict"] != "improved", r["benchmark"])):
        ulp = f"{r['best_ulp']:.1f}" if r["best_ulp"] is not None else "-"
        lines.append(f"| {r['benchmark']} | {r['candidates']} | {r['valid']} | {ulp} | {r['verdict']} |")

    text = "\n".join(lines) + "\n"
    out = paths.ROOT / "summary.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", nargs="?", help="one benchmark; omit to run all")
    p.add_argument("--shard", metavar="I/N", help="run only benchmarks[I::N] (one per GPU)")
    p.add_argument("--summary-only", action="store_true",
                   help="skip generation; just re-print the results table")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--max-attempts", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--time-budget", type=float, default=10.0, metavar="SECONDS",
                   help="per-check budget: saturate each branch until proven or this many "
                        "seconds elapse")
    args = p.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else shard(all_benchmarks(), args.shard)

    if not args.summary_only:
        import casa
        load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
        llm = casa.LLM.from_pretrained(args.model, **load_kwargs)
        for b in benchmarks:
            try:
                run_benchmark(llm, b, args)
            except Exception as e:  # one bad benchmark shouldn't sink a whole suite run
                print(f"!! {b} failed: {e!r}")

    if len(benchmarks) > 1 or args.summary_only:
        summarize(benchmarks)


if __name__ == "__main__":
    main()
