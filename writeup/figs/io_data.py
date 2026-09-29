"""Shared CSV access for the figure scripts (data from extract.py)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SIZES = ["100k", "500k", "1M", "3M", "6M", "24M"]
N_NONEMB = {"100k": 65_728, "500k": 492_160, "1M": 984_192, "3M": 3_016_960,
            "6M": 6_033_664, "24M": 24_253_184}
FINAL = 17.723                                    # self-play round 2816, the paper's ICL point
MATCH = [1.6169, 3.2275, 6.4487, 12.8912, 17.723]
UP_PROGRAMS = {"100k": 1024, "500k": 2048, "1M": 1024, "3M": 2048, "6M": 1024}
ARM = {"sp": "Self-play", "dclm": "DCLM text", "up": "Universal prior", "sp2dclm": "Self-play → DCLM"}


def rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def near(t, target, tol=0.02):
    return abs(float(t) - target) <= tol * target


def seeds(table, arm, size, tokens, **kw) -> dict:
    """{seed: value} for one (arm, size, token count, extra column filters)."""
    return {r["seed"]: float(r["value"]) for r in table
            if r["arm"] == arm and r["size"] == size and near(r["tokens"], tokens)
            and all(r[k] == str(v) for k, v in kw.items())}


def up_tokens(size):
    return UP_PROGRAMS[size] * 4096 * 2817 / 1e9


def curve(table, arm, size, suite, task, tokens=FINAL):
    """m -> list of per-seed values."""
    out = defaultdict(list)
    for r in table:
        if (r["arm"] == arm and r["size"] == size and r["suite"] == suite and r["task"] == task
                and near(r["tokens"], tokens)):
            out[int(r["m"])].append(float(r["value"]))
    return dict(sorted(out.items()))


def as_runs(by_m: dict):
    """(ms, runs[n_seeds, n_m]) using only m values every seed has."""
    n = min(len(v) for v in by_m.values())
    ms = [m for m, v in by_m.items() if len(v) == n]
    return np.array(ms, float), np.array([by_m[m] for m in ms]).T
