"""Prompt a model unconstrained and print its full reasoning + answer.

Lets you eyeball how well a reasoning model (e.g. gpt-oss) thinks about a
benchmark's floating-point instability, before wiring it into the pipeline.

Usage:
    uv run src/ask.py "Why is sqrt(x*x+1)-x inaccurate for large x?"
    uv run src/ask.py --benchmark sqrtminus --effort high
    uv run src/ask.py --benchmark quadratic --model openai/gpt-oss-120b
"""

import argparse

import torch

import casa
import egrammar
import run


def build_prompt(args) -> str:
    if args.benchmark:
        # The same instability-reasoning prompt the --reason flow uses.
        return run.reasoning_prompt(egrammar.read_reference(args.benchmark))
    if args.prompt:
        return args.prompt
    raise SystemExit("give a prompt or --benchmark")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=None)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--benchmark", default=None,
                    help="use this benchmark's instability-reasoning prompt")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"],
                    help="gpt-oss reasoning effort (default high)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    prompt = build_prompt(args)
    llm = casa.LLM.from_pretrained(args.model, dtype="auto")

    # gpt-oss takes reasoning_effort in its chat template; other models ignore it.
    try:
        text = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
            reasoning_effort=args.effort,
        )
    except TypeError:
        text = llm.format_prompt(prompt)

    ids = llm.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False
    ).to(llm.device)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                      pad_token_id=llm.tokenizer.pad_token_id)
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)

    with torch.no_grad():
        out = llm.model.generate(ids, **gen_kwargs)
    new = out[0][ids.shape[1]:]

    print("=" * 70)
    print(f"PROMPT\n{prompt}")
    print("=" * 70)
    # skip_special_tokens=False so the analysis (reasoning) and final channels
    # both show; gpt-oss hides its chain-of-thought behind <|channel|> markers.
    print(llm.tokenizer.decode(new, skip_special_tokens=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
