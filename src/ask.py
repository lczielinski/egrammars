"""Prompt a model unconstrained and stream its reasoning + answer live.

Edit PROMPT below, then run. Used to watch how a reasoning model (gpt-oss) thinks
-- here, reasoning from per-operation condition numbers toward a program that
branches on the input.

Usage:
    uv run src/ask.py
    uv run src/ask.py --model openai/gpt-oss-120b --effort high
"""

import argparse

import torch
from transformers import TextStreamer

import casa

CONDITION_RULES = """\
Condition numbers (how much each operation amplifies relative input error):
  x + y :  G_x = |x/(x+y)|,  G_y = |y/(x+y)|     -- blows up when x is close to -y
  x - y :  G_x = |x/(x-y)|,  G_y = |y/(x-y)|     -- blows up when x is close to y
  x * y :  G_x = G_y = 1                          -- always well-conditioned
  x / y :  G_x = G_y = 1                          -- always well-conditioned
A large condition number means small rounding errors in the inputs are magnified
in the result. So + and - lose accuracy exactly when their operands nearly cancel;
* and / never amplify relative error."""

# The expression to analyze -- swap which EXPR is active:
EXPR = "(-b + sqrt(b*b - 4*a*c)) / (2*a)"  # quadratic root: two cancellation sites
                                           # (-b + sqrt when b > 0, and b*b - 4*a*c
                                           # when b*b ~= 4*a*c); branch on sign of b
# EXPR = "sqrt(x + 1) - sqrt(x)"           # near-equal sqrts; cancels for large x

PROMPT = f"""\
{CONDITION_RULES}

The expression below is evaluated in IEEE-754 double precision:

    {EXPR}

Goal: a single program that stays numerically accurate across the WHOLE input
domain by using a different algebraically-equivalent form in each region.

Steps:
1. Using the condition numbers, find every input region where some operation loses
   accuracy -- a + or - whose operands nearly cancel, or a / by a near-zero value.
2. For each region, give a rewrite whose operations are all well-conditioned there,
   so it has low rounding error throughout that region. Different regions may need
   different rewrites.
3. Combine them into ONE program that branches on the inputs (comparisons or sign
   tests) so every input takes the form that is accurate for it. Cover the entire
   domain, including the region where the original form is already accurate.

Rules: every branch must equal the original exactly in real arithmetic (only the
rounding may differ); use only +, -, *, /, sqrt and the original variables. Output
the single program as an S-expression."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"],
                    help="gpt-oss reasoning effort (default high)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="cap on generated tokens (default: fill the context "
                         "window, so reasoning isn't cut off before the answer)")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    llm = casa.LLM.from_pretrained(args.model, dtype="auto")

    # gpt-oss takes reasoning_effort in its chat template; other models ignore it.
    try:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False, add_generation_prompt=True, reasoning_effort=args.effort,
        )
    except TypeError:
        text = llm.format_prompt(PROMPT)

    ids = llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False
    ).to(llm.device)

    max_ctx = getattr(llm.model.config, "max_position_embeddings", None) or 8192
    max_new = args.max_new_tokens or max(256, max_ctx - ids.shape[1] - 64)
    gen_kwargs = dict(max_new_tokens=max_new,
                      attention_mask=torch.ones_like(ids),
                      pad_token_id=llm.tokenizer.pad_token_id,
                      streamer=TextStreamer(llm.tokenizer, skip_prompt=True,
                                            skip_special_tokens=True))
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)

    print("=" * 70)
    print(f"PROMPT\n{PROMPT}")
    print("=" * 70)
    with torch.no_grad():
        llm.model.generate(ids, **gen_kwargs)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
