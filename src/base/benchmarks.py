"""Benchmark data: reference terms and interval boxes."""

import json

from base import paths


def suite() -> list[str]:
    return sorted(p.stem for p in paths.EGGLOG.glob("*.egglog"))


def read_reference(benchmark: str) -> str:
    """The FPCore reference on the benchmark file's first comment line."""
    return (paths.EGGLOG / f"{benchmark}.egglog").read_text().splitlines()[0].removeprefix(";; ")


def read_source(benchmark: str) -> str:
    """The benchmark's egglog source (reference term as `start`)."""
    return (paths.EGGLOG / f"{benchmark}.egglog").read_text()


def _load_intervals() -> dict:
    """intervals.json ({var: "[lo,hi]"} strings) -> {benchmark: {var: (lo, hi)}}."""
    if not paths.INTERVALS_FILE.exists():
        return {}
    raw = json.loads(paths.INTERVALS_FILE.read_text())
    return {b: {v: tuple(float(x) for x in s.strip("[]").split(","))
                for v, s in box.items()}
            for b, box in raw.items()}


INTERVALS = _load_intervals()  # benchmark -> input box
