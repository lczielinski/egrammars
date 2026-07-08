"""Run the full generate -> verify -> bound pipeline over EVERY benchmark with the model
loaded once, then print a comparison-ready summary (best relative-error bound per
benchmark). This is `run.py` in a loop over `benchmarks/*.egglog`, sharing one loaded
model; to run a single benchmark use `run.py` directly.

    uv run src/run_suite.py                 # generate + bound every benchmark (1 GPU)
    uv run src/run_suite.py --shard 0/8     # only benchmarks[0::8] — one shard per GPU
    uv run src/run_suite.py --summary-only  # just re-print the table from existing runs

To use all GPUs, run one shard per GPU concurrently (reasoning dominates, so this is a
near-linear speedup) — see scripts/run_gpus.sh, which pins CUDA_VISIBLE_DEVICES per shard.

FPTaylor must be on PATH with its opam environment active (`eval $(opam env)`), else its
native libs (dllnums.so, interval.cmi) fail to load and every bound comes back unbounded.
"""

import argparse
import json

import paths
import regions
import run as run_module  # reuse the exact generation/verification path


def all_benchmarks() -> list[str]:
    return sorted(p.stem for p in paths.BENCHMARKS.glob("*.egglog"))


def shard(benchmarks: list[str], spec: str | None) -> list[str]:
    """`spec` = "I/N": keep every N-th benchmark starting at I (round-robin, so the
    slowest cores spread evenly across shards). None => all."""
    if not spec:
        return benchmarks
    i, n = (int(x) for x in spec.split("/"))
    return benchmarks[i::n]


def generate_one(llm, benchmark: str, args) -> int | None:
    """Mirror run.main() for a single benchmark against an already-loaded model; return
    the run number written, or None if nothing survived verification."""
    import egrammar
    import fptaylor_check

    reference = egrammar.read_reference(benchmark)
    variables = regions.variables_of(reference)
    box = fptaylor_check.INTERVALS.get(benchmark)
    print(f"\n{'='*70}\n{benchmark}\nreference: {reference}\nbox: {box or '(none)'}\n{'='*70}")

    base_ids = run_module.initial_ids(
        llm, run_module.program_prompt(reference, box), args)
    candidates = run_module.asap_samples(
        llm, run_module.syntax_grammar(variables), args, base_ids)
    programs, attempts = run_module.evaluate_candidates(
        benchmark, reference, box, candidates, args.saturation)

    missing = sum(a.get("numeric", {}).get("verdict") == "equal" for a in attempts)
    print(f"{len(attempts)} candidates: {len(programs)} proven, {missing} missing-rule?, "
          f"{len(attempts) - len(programs) - missing} non-equivalent")
    n, summary = paths.next_path(paths.EQUIVALENTS, benchmark)
    summary.write_text(json.dumps(
        {"benchmark": benchmark, "reference": reference,
         "config": {"model": args.model, "temperature": args.temperature,
                    "effort": args.effort, "samples": args.samples,
                    "max_attempts": args.max_attempts, "saturation": args.saturation,
                    "box": box},
         "programs": programs, "attempts": attempts}, indent=2))
    if not programs:
        return None
    try:
        fptaylor_check.check(benchmark, run=n)
    except (KeyError, FileNotFoundError) as e:
        print(f"skipping fptaylor: {e}")
    return n


def summarize(benchmarks: list[str]) -> None:
    """One row per benchmark: best (lowest) relative-error bound found, in ulps."""
    print(f"\n{'='*70}\nSUMMARY  (best relative-error bound per benchmark)\n{'='*70}")
    print(f"{'benchmark':<32}{'#eqv':>6}{'best rel (ulp)':>16}  best program")
    rows = []
    for b in benchmarks:
        _, src = paths.latest(paths.FPTAYLOR, b)
        if src is None:
            continue
        data = json.loads(src.read_text())
        results = data.get("results", [])
        best = next((r for r in results if r.get("rel_err_ulps") is not None), None)
        ulp = f"{best['rel_err_ulps']:.1f}" if best else "unbounded"
        prog = best["program"] if best else (results[0]["program"] if results else "")
        rows.append((b, len(results), ulp, prog))
    for b, n, ulp, prog in rows:
        print(f"{b:<32}{n:>6}{ulp:>16}  {prog[:60]}")
    print(f"\n{len(rows)} benchmarks with results")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", metavar="I/N",
                    help="run only benchmarks[I::N] — one shard per GPU (see scripts/run_gpus.sh)")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip generation; just re-print the table from existing runs")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--model", default=run_module.MODEL_ID)
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--saturation", type=int, default=6)
    args = ap.parse_args()

    benchmarks = shard(all_benchmarks(), args.shard)
    tag = f" (shard {args.shard})" if args.shard else ""
    print(f"{len(benchmarks)} benchmarks{tag}: {', '.join(benchmarks)}")

    if not args.summary_only:
        import casa
        load_kwargs = {"dtype": "auto"} if "gpt-oss" in args.model.lower() else {}
        llm = casa.LLM.from_pretrained(args.model, **load_kwargs)
        for b in benchmarks:
            try:
                generate_one(llm, b, args)
            except Exception as e:  # one bad benchmark shouldn't sink the whole suite
                print(f"!! {b} failed: {e!r}")

    # A shard summarizes only its own slice; run --summary-only afterward for the full table.
    summarize(benchmarks)


if __name__ == "__main__":
    main()
