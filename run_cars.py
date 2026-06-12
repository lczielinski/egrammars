"""Run CARS on an egrammar-compiled grammar to harvest equivalent programs.

Compiles a benchmark's equivalence grammar (reusing out/<benchmark>.lark if it
already exists) and samples a variety of distinct programs from it with the
vendored CARS sampler (cars.py). Because the grammar's language is exactly the
programs provably equivalent to the reference, every accepted sample is equivalent
by construction — the output is a deduplicated set of alternative spellings of the
same computation.

Usage:
    uv run --extra cars run_cars.py quadratic
    uv run --extra cars run_cars.py quadratic --samples 50 --steps 500

The `cars` extra (see pyproject.toml) pulls in the runtime deps (torch,
transformers, llguidance, xgrammar, ...). egglog (the base dependency) is only used
to compile a grammar that is not already cached in out/.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"


def ensure_artifacts(benchmark: str, out_dir: Path) -> tuple[Path, Path]:
    """Return paths to the grammar (.lark) and prompt (.txt), compiling if needed."""
    grammar_path = out_dir / f"{benchmark}.lark"
    prompt_path = out_dir / f"{benchmark}.txt"
    if not (grammar_path.exists() and prompt_path.exists()):
        print(f"Compiling grammar for {benchmark!r} (no cached artifacts in {out_dir})")
        import egrammar  # local; pulls in egglog, only needed on a cache miss

        reference, grammar = egrammar.build(benchmark)
        egrammar.write_artifacts(
            benchmark, grammar, egrammar.make_prompt(reference), out_dir
        )
    return grammar_path, prompt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("benchmark", help="benchmark name, e.g. quadratic")
    parser.add_argument("--samples", type=int, default=20,
                        help="number of accepted programs to collect (default 20)")
    parser.add_argument("--steps", type=int, default=200,
                        help="cap on generation attempts; sampling stops here even "
                             "if fewer programs were accepted (default 200)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.3,
                        help="sampling temperature; >1 flattens the distribution "
                             "for more diverse programs (default 1.3)")
    parser.add_argument("--dtype", default="bfloat16",
                        help="torch dtype for the model (default bfloat16)")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    grammar_path, prompt_path = ensure_artifacts(args.benchmark, out_dir)
    grammar = grammar_path.read_text()
    prompt = prompt_path.read_text()

    import torch
    from cars import ConstrainedModel, sample_programs

    dtype = getattr(torch, args.dtype)
    print(f"benchmark: {args.benchmark}")
    print(f"grammar:   {grammar_path} ({grammar.count(chr(10))} rules)")
    print(f"model:     {MODEL_ID} ({dtype}, T={args.temperature})")
    print(f"target {args.samples} programs, <= {args.steps} attempts\n")

    model = ConstrainedModel(MODEL_ID, grammar, dtype=dtype)
    programs = sample_programs(
        model, prompt, args.samples, args.steps, args.max_new_tokens,
        args.temperature,
    )

    parts = prompt.split("The original program is:\n", 1)
    reference = parts[1].split("\n\n", 1)[0].strip() if len(parts) > 1 else "?"

    print(f"\n{'=' * 70}")
    print(f"original program:     {reference}")
    print(f"distinct equivalents: {len(programs)}")
    print(f"{'=' * 70}")
    for i, program in enumerate(programs):
        print(f"{i:3d}  {program}")

    summary = out_dir / f"{args.benchmark}.equivalents.json"
    summary.write_text(json.dumps(
        {"benchmark": args.benchmark, "reference": reference,
         "model": MODEL_ID, "programs": programs},
        indent=2,
    ))
    print(f"\nwrote {len(programs)} distinct equivalent programs to {summary}")


if __name__ == "__main__":
    main()
