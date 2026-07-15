"""Generate numerically-accurate FPCore rewrites by grammar-constrained decoding,
where the grammar IS the set of programs provably equivalent to the reference. The
model reasons once, then either emits a whole-box form or opens `(if (op var
NUMBER)`; once the threshold is known, each arm's grammar is rebuilt on the fly over
the guard-narrowed box, so every program is equivalent by construction. Alongside
the samples, the classic min-cost extraction from the same e-grammar is recorded as
a model-free baseline.

Each invocation writes a fresh results/<run>/ directory holding equivalents/,
fptaylor/, and summary.md. `all` self-shards across every visible GPU.

    uv run src/run.py x_by_xy              # one benchmark
    uv run src/run.py all                  # whole suite, one shard per GPU
    uv run src/run.py --summary-only       # re-print the latest run's table
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime

from analysis import fptaylor_check, summary
from base import benchmarks, paths, regions
from synth import decoding, egrammar, generate

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning, module="kernels")


def run_benchmark(llm, benchmark: str, run, args, tag: str = "") -> None:
    """Generate, bound, and write <run>/equivalents/<benchmark>.json."""
    reference = benchmarks.read_reference(benchmark)
    box = benchmarks.INTERVALS.get(benchmark)
    print(f"\n{'=' * 70}\n{benchmark}{f'  [{tag}]' if tag else ''}   "
          f"box: {box or '(none)'}\n{reference}\n{'=' * 70}", flush=True)

    base_ids = generate.initial_ids(llm, generate.program_prompt(reference, box),
                                    args.temperature)
    grammar = decoding.build_grammar(llm, benchmark, box, args.saturation)
    programs = generate.sample_programs(llm, grammar, args.samples,
                                        args.temperature, base_ids)
    print(f"{len(programs)} programs (equivalent by construction)")
    for p in programs:
        print(f"  [cost {regions.cost(p):3d}] {p}")

    # model-free baseline: classic min-cost extraction from the same e-grammar
    try:
        extraction = egrammar.min_program(egrammar.build(benchmark, box, args.saturation))
        print(f"e-graph extraction [cost {regions.cost(extraction):3d}] {extraction}")
    except Exception as e:
        extraction = None
        print(f"no extraction baseline: {e!r}")

    out = paths.equivalents_path(run, benchmark)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"benchmark": benchmark, "reference": reference,
         "config": {"model": args.model, "temperature": args.temperature,
                    "samples": args.samples, "saturation": args.saturation,
                    "run": run.name, "box": box},
         "programs": programs, "extraction": extraction}, indent=2))
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
    """The whole suite as N subprocesses, one per GPU, all writing into `run`.
    Progress = completed benchmarks, counted off the run directory."""
    log_dir = run / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, __file__, "all", "--run", run.name,
            "--samples", str(args.samples), "--temperature", str(args.temperature),
            "--model", args.model, "--saturation", str(args.saturation)]
    print(f"launching {n} shards over {n} GPUs; tail -f {log_dir}/gpu0.log to watch one")
    procs = []
    for i in range(n):
        log = open(log_dir / f"gpu{i}.log", "w")
        procs.append((i, subprocess.Popen(
            base + ["--shard", f"{i}/{n}"],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": str(i)},
            stdout=log, stderr=subprocess.STDOUT), log))

    total, start, done = len(benchmarks.suite()), time.time(), -1
    while any(p.poll() is None for _, p, _ in procs):
        if (d := len(paths.benchmarks_in(run))) != done:
            done = d
            print(f"  {done}/{total} benchmarks done  ({time.time() - start:.0f}s)",
                  flush=True)
        time.sleep(10)

    ok = True
    for i, p, log in procs:
        log.close()
        if p.returncode != 0:
            print(f"shard {i} FAILED (see {log_dir}/gpu{i}.log)")
            ok = False
    print(f"{len(paths.benchmarks_in(run))}/{total} benchmarks done "
          f"({time.time() - start:.0f}s)")
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
    p.add_argument("--run", default=None, metavar="NAME",
                   help="results/<NAME> (default: <timestamp>)")
    p.add_argument("--saturation", type=int, default=egrammar.SATURATION_RUNS,
                   metavar="RUNS", help="saturation rounds when compiling the e-grammar")
    p.add_argument("--shard", metavar="I/N", help=argparse.SUPPRESS)  # internal
    args = p.parse_args()

    if args.summary_only:
        run = paths.run_dir(args.run) if args.run else paths.latest_run()
        if run is None:
            p.error(f"no runs in {paths.RESULTS}")
        return summary.summarize(run)
    if not args.benchmark:
        p.error("give a benchmark name or `all`")

    name = args.run or f"{datetime.now():%Y-%m-%d-%H%M%S}"
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
    for k, b in enumerate(names, 1):
        try:
            run_benchmark(llm, b, run, args, tag=f"{k}/{len(names)}")
        except Exception as e:  # one bad benchmark shouldn't sink a suite run
            print(f"!! {b} failed: {e!r}")

    if not args.shard:  # shard children leave the summary to the parent
        summary.summarize(run)


if __name__ == "__main__":
    main()
