"""Repository paths and numbered per-run output files (<name>-NNN.json)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"
EGGLOG = BENCHMARKS / "egglog"          # per-benchmark reference terms (<name>.egglog)
INTERVALS_FILE = BENCHMARKS / "intervals.json"
EQUIVALENTS = ROOT / "equivalents"
FPTAYLOR = ROOT / "fptaylor"


def _run_numbers(folder: Path, benchmark: str) -> list[int]:
    if not folder.exists():
        return []
    nums = []
    for p in folder.glob(f"{benchmark}-*.json"):
        if m := re.fullmatch(rf"{re.escape(benchmark)}-(\d+)", p.stem):
            nums.append(int(m.group(1)))
    return sorted(nums)


def path_for(folder: Path, benchmark: str, n: int) -> Path:
    return folder / f"{benchmark}-{n:03d}.json"


def next_path(folder: Path, benchmark: str) -> tuple[int, Path]:
    runs = _run_numbers(folder, benchmark)
    n = (runs[-1] + 1) if runs else 1
    folder.mkdir(parents=True, exist_ok=True)
    return n, path_for(folder, benchmark, n)


def latest(folder: Path, benchmark: str) -> tuple[int | None, Path | None]:
    runs = _run_numbers(folder, benchmark)
    if not runs:
        return None, None
    return runs[-1], path_for(folder, benchmark, runs[-1])


def benchmarks_in(folder: Path) -> list[str]:
    """Distinct benchmark names that have a `<name>-NNN.json` file in `folder`."""
    if not folder.exists():
        return []
    names = set()
    for p in folder.glob("*-*.json"):
        if m := re.fullmatch(r"(.+)-(\d+)", p.stem):
            names.add(m.group(1))
    return sorted(names)
