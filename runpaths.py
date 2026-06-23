"""Numbered per-run output paths, so repeated runs don't overwrite each other.

Layout under out/:
    out/equivalents/<benchmark>-NNN.json   (written by run_cars.py)
    out/gappa/<benchmark>-NNN.json         (written by gappa_check.py)

A gappa file reuses the run number of the equivalents file it analyzed, so the
two stay linked. NNN is zero-padded to 3 digits.
"""

from __future__ import annotations

import re
from pathlib import Path


def _run_numbers(folder: Path, benchmark: str) -> list[int]:
    if not folder.exists():
        return []
    nums = []
    for p in folder.glob(f"{benchmark}-*.json"):
        m = re.fullmatch(rf"{re.escape(benchmark)}-(\d+)", p.stem)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def path_for(folder: Path, benchmark: str, n: int) -> Path:
    return folder / f"{benchmark}-{n:03d}.json"


def next_path(folder: Path, benchmark: str) -> tuple[int, Path]:
    """The next unused run number and its path (creates `folder`)."""
    runs = _run_numbers(folder, benchmark)
    n = (runs[-1] + 1) if runs else 1
    folder.mkdir(parents=True, exist_ok=True)
    return n, path_for(folder, benchmark, n)


def latest(folder: Path, benchmark: str) -> tuple[int | None, Path | None]:
    """The highest existing run number and its path, or (None, None)."""
    runs = _run_numbers(folder, benchmark)
    if not runs:
        return None, None
    return runs[-1], path_for(folder, benchmark, runs[-1])
