"""Model-facing side: prompt, reasoning handoff, and constrained sampling."""

from base import paths

MODEL_ID = "openai/gpt-oss-120b"


def program_prompt(reference: str, box: dict | None) -> str:
    preamble = (paths.ROOT / "prompt_header.md").read_text()
    ranges = ""
    if box:
        spans = ", ".join(f"{v} in [{lo}, {hi}]" for v, (lo, hi) in box.items())
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
        "the original in exact arithmetic; only the rounding may differ.\n"
        "4. Keep the program as SMALL as accuracy allows: rewrite only the fragile "
        "subterms and copy every already-accurate part of the original unchanged. "
        "Every operation is a rounding step, so re-associations and expansions that "
        "don't fix a fragile spot make the result worse, not better. Between two "
        "equally accurate forms, always output the smaller one.\n\n"
        "Output only the single-line program, then immediately stop."
    )


def load_model(model_id: str):
    import casa
    kwargs = {"dtype": "auto"} if "gpt-oss" in model_id.lower() else {}
    return casa.LLM.from_pretrained(model_id, **kwargs)


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


def _encode(llm, prompt):
    try:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
    except (TypeError, ValueError):
        text = llm.format_prompt(prompt)
    return llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False).to(llm.device)


def think_then_handoff(llm, prompt, temperature):
    """Reason in the analysis channel until the final channel opens; return the token
    ids so generation can continue there, or None if it never opened."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

    ch_id, msg_id = channel_ids(llm.tokenizer)
    ids = _encode(llm, prompt)
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


def initial_ids(llm, prompt, temperature):
    """Context to sample from: reasoning + open final channel on a harmony model,
    else just the encoded prompt."""
    if channel_ids(llm.tokenizer):
        ids = think_then_handoff(llm, prompt, temperature)
        free_cuda()
        if ids is not None:
            return ids
    return _encode(llm, prompt)


def sample_programs(llm, grammar, n_samples, temperature, base_ids):
    """Up to `n_samples` distinct grammar-valid programs from ONE ASAP call: the
    oracle trie persists, so each sample is masked out and never re-proposed."""
    import casa
    res = casa.ASAP(llm, grammar, verbose=True, temperature=temperature).sample(
        prompt_ids=base_ids, n_samples=n_samples)
    free_cuda()
    return [r.text.strip() for r in res]
