"""Prompt a model unconstrained and print its full reasoning + answer.

Edit PROMPT below, then run. Lets you eyeball how a reasoning model (gpt-oss)
thinks before wiring it into the pipeline.

Usage:
    uv run src/ask.py
    uv run src/ask.py --model openai/gpt-oss-120b --effort high
"""

import argparse

import torch

import casa

PROMPT = """\
Why does sqrt(x*x + 1) - x lose floating-point accuracy for large x, and how
would you rewrite it to avoid the cancellation?
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"],
                    help="gpt-oss reasoning effort (default high)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
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

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      attention_mask=torch.ones_like(ids),
                      pad_token_id=llm.tokenizer.pad_token_id)
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)

    with torch.no_grad():
        out = llm.model.generate(ids, **gen_kwargs)
    new = out[0][ids.shape[1]:]

    print("=" * 70)
    print(f"PROMPT\n{PROMPT}")
    print("=" * 70)
    # skip_special_tokens=False so the analysis (reasoning) and final channels
    # both show; gpt-oss hides its chain-of-thought behind <|channel|> markers.
    print(llm.tokenizer.decode(new, skip_special_tokens=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
