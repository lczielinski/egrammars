"""Repository layout and per-run result directories (results/<run>/)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS = ROOT / "benchmarks"
EGGLOG = BENCHMARKS / "egglog"          # per-benchmark reference terms (<name>.egglog)
INTERVALS_FILE = BENCHMARKS / "intervals.json"
RESULTS = ROOT / "results"


def run_dir(name: str) -> Path:
    return RESULTS / name


def equivalents_path(run: Path, benchmark: str) -> Path:
    return run / "equivalents" / f"{benchmark}.json"


def fptaylor_path(run: Path, benchmark: str) -> Path:
    return run / "fptaylor" / f"{benchmark}.json"


def benchmarks_in(run: Path) -> list[str]:
    d = run / "equivalents"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def latest_run() -> Path | None:
    runs = [p for p in RESULTS.iterdir() if p.is_dir()] if RESULTS.exists() else []
    return max(runs, key=lambda p: p.stat().st_mtime, default=None)
