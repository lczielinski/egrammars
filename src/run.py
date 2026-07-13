"""Generate numerically-accurate FPCore rewrites and verify each is equivalent.

Two decoding modes feed the same verification:

  --decoding light  (default): sample programs under a light FPCore syntax grammar;
    the model branches freely and each candidate is checked against the e-graph after.
  --decoding egraph: grammar-constrained decoding where the grammar IS the set of
    programs provably equivalent to the reference. The model either emits a whole-box
    form or opens `(if (op var NUMBER)`; once the threshold is known, each arm's
    grammar is rebuilt on the fly over the guard-narrowed box, so every program is
    equivalent by construction. To the model it is one generation.

Each invocation writes a fresh results/<run>/ directory holding equivalents/,
fptaylor/, and summary.md. `all` self-shards across every visible GPU.

    uv run src/run.py x_by_xy              # one benchmark
    uv run src/run.py all                  # whole suite, one shard per GPU
    uv run src/run.py all --decoding egraph
    uv run src/run.py --summary-only       # re-print the latest run's table
"""

import argparse
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime

import benchmarks
import decoding
import egrammar
import fptaylor_check
import generate
import paths
import summary
import verify

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")


def run_benchmark(llm, benchmark: str, run, args) -> None:
    """Generate, verify, bound, and write <run>/equivalents/<benchmark>.json."""
    reference = benchmarks.read_reference(benchmark)
    box = benchmarks.INTERVALS.get(benchmark)
    print(f"\n{'=' * 70}\n{benchmark}   box: {box or '(none)'}\n{reference}\n{'=' * 70}")

    base_ids = generate.initial_ids(llm, generate.program_prompt(reference, box),
                                    args.temperature)
    grammar = decoding.build_grammar(llm, benchmark, reference, box,
                                     args.decoding, args.saturation)
    candidates = generate.sample_programs(llm, grammar, args.samples,
                                          args.temperature, base_ids)
    programs, attempts = verify.evaluate_candidates(
        benchmark, reference, box, candidates, timeout=args.time_budget)

    missing = sum(a.get("numeric", {}).get("verdict") == "equal" for a in attempts)
    print(f"{len(attempts)} candidates: {len(programs)} proven, {missing} missing-rule?, "
          f"{len(attempts) - len(programs) - missing} non-equivalent")
    for rec in attempts:
        print(verify.attempt_line(rec))

    out = paths.equivalents_path(run, benchmark)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"benchmark": benchmark, "reference": reference,
         "config": {"model": args.model, "temperature": args.temperature,
                    "samples": args.samples, "decoding": args.decoding,
                    "run": run.name, "time_budget": args.time_budget, "box": box},
         "programs": programs, "attempts": attempts}, indent=2))
    print(f"wrote {out}")
    if programs:
        try:
            fptaylor_check.check(benchmark, run)
        except (KeyError, FileNotFoundError) as e:
            print(f"skipping fptaylor: {e}")


def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def run_shards(n: int, run, args) -> bool:
    """The whole suite as N subprocesses, one per GPU, all writing into `run`."""
    log_dir = run / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, __file__, "all", "--run", run.name,
            "--samples", str(args.samples), "--temperature", str(args.temperature),
            "--model", args.model, "--decoding", args.decoding,
            "--saturation", str(args.saturation), "--time-budget", str(args.time_budget)]
    print(f"launching {n} shards over {n} GPUs; logs in {log_dir}")
    procs = []
    for i in range(n):
        log = open(log_dir / f"gpu{i}.log", "w")
        procs.append((i, subprocess.Popen(
            base + ["--shard", f"{i}/{n}"],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": str(i)},
            stdout=log, stderr=subprocess.STDOUT), log))
    ok = True
    for i, p, log in procs:
        code = p.wait()
        log.close()
        print(f"shard {i} {'done' if code == 0 else f'FAILED (see {log_dir}/gpu{i}.log)'}")
        ok &= code == 0
    return ok


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", nargs="?", help="benchmark name, or `all` for the suite")
    p.add_argument("--summary-only", action="store_true",
                   help="no generation; summarize --run (default: the latest run)")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--model", default=generate.MODEL_ID)
    p.add_argument("--decoding", choices=["light", "egraph"], default="light",
                   help="light: static syntax grammar, verify after; egraph: decode "
                        "under the e-graph of equivalent programs, rebuilt per arm "
                        "when an `if` is emitted")
    p.add_argument("--run", default=None, metavar="NAME",
                   help="results/<NAME> (default: <timestamp>-<decoding>)")
    p.add_argument("--saturation", type=int, default=egrammar.SATURATION_RUNS,
                   metavar="RUNS", help="egraph decoding: saturation rounds when "
                        "compiling the e-grammar")
    p.add_argument("--time-budget", type=float, default=10.0, metavar="SECONDS",
                   help="per-check budget: saturate each branch until proven or "
                        "this many seconds elapse")
    p.add_argument("--shard", metavar="I/N", help=argparse.SUPPRESS)  # internal
    args = p.parse_args()

    if args.summary_only:
        run = paths.run_dir(args.run) if args.run else paths.latest_run()
        if run is None:
            p.error(f"no runs in {paths.RESULTS}")
        return summary.summarize(run)
    if not args.benchmark:
        p.error("give a benchmark name or `all`")

    name = args.run or f"{datetime.now():%Y-%m-%d-%H%M%S}-{args.decoding}"
    run = paths.run_dir(name)

    names = benchmarks.suite() if args.benchmark == "all" else [args.benchmark]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        names = names[i::n]  # round-robin spreads slow cores evenly
    elif args.benchmark == "all" and (n := gpu_count()) > 1:
        ok = run_shards(n, run, args)
        summary.summarize(run)
        sys.exit(0 if ok else 1)

    llm = generate.load_model(args.model)
    for b in names:
        try:
            run_benchmark(llm, b, run, args)
        except Exception as e:  # one bad benchmark shouldn't sink a suite run
            print(f"!! {b} failed: {e!r}")

    if not args.shard:  # shard children leave the summary to the parent
        summary.summarize(run)


if __name__ == "__main__":
    main()
