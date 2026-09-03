"""training/dataset.py

Loading, the synthetic-data guard, and the leave-one-caster-out splitter.

THE GUARD is the most important thing in this file.  Every entry point that can
produce a number for the report goes through `load_dataset`, and `load_dataset`
will not touch data/raw unless there are at least MIN_REAL_CSV_FILES real
captures in it.  Synthetic data is reachable only by explicitly passing
allow_synthetic=True, and when it is, every returned record is flagged so the
callers can stamp "SYNTHETIC -- NOT A RESULT" on their output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import wand_config as C
from preprocess import KinematicFeatures, preprocess


@dataclass
class Capture:
    tensor: np.ndarray            # (64, 8) float32
    kin: KinematicFeatures
    label: int
    caster: str
    path: Path
    synthetic: bool


def _read_csv(path: Path) -> tuple[np.ndarray, int, str, bool]:
    label, caster, synthetic = -1, "unknown", False
    rows: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "SYNTHETIC" in line.upper() or "synthetic=1" in line:
                    synthetic = True
                if "GESTURE_START" in line:
                    for tok in line.split():
                        if tok.startswith("label="):
                            label = int(tok.split("=")[1])
                        elif tok.startswith("caster="):
                            caster = tok.split("=")[1]
                continue
            if line.startswith("idx"):
                continue
            parts = line.split(",")
            # idx, qw, qx, qy, qz, lax, lay, laz, gx, gy, gz, dt_us, dup[, synthetic]
            if len(parts) < 13:
                continue
            rows.append([int(float(p)) for p in parts[1:11]])
    return np.asarray(rows, dtype=np.int16), label, caster, synthetic


def _count_real_csv(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob("*.csv"):
        with open(p) as f:
            head = f.read(400)
        if "SYNTHETIC" not in head.upper() and "synthetic=1" not in head:
            n += 1
    return n


def load_dataset(root: str | Path = C.REAL_DATA_DIR,
                 allow_synthetic: bool = False) -> list[Capture]:
    """Load every capture under `root`.

    Raises unless either (a) root holds at least MIN_REAL_CSV_FILES genuine
    captures, or (b) allow_synthetic is explicitly True.
    """
    root = Path(root)
    n_real = _count_real_csv(root)

    if not allow_synthetic and n_real < C.MIN_REAL_CSV_FILES:
        raise RuntimeError(
            f"No real dataset present -- refusing to run on synthetic data.\n"
            f"  {root} holds {n_real} genuine captures, need >= {C.MIN_REAL_CSV_FILES}.\n"
            f"  Collect data with training/collect.py, or pass --allow-synthetic\n"
            f"  if you are only smoke-testing the pipeline (results will be\n"
            f"  stamped SYNTHETIC and must never enter the report).")

    out: list[Capture] = []
    for path in sorted(root.rglob("*.csv")):
        raw, label, caster, synthetic = _read_csv(path)
        if label < 0 or len(raw) < C.MIN_GESTURE_SAMPLES:
            continue
        if len(raw) > C.MAX_GESTURE_SAMPLES + C.PREROLL_SAMPLES + C.POSTROLL_SAMPLES:
            raw = raw[: C.MAX_GESTURE_SAMPLES + C.PREROLL_SAMPLES + C.POSTROLL_SAMPLES]
        tensor, kin = preprocess(raw)
        out.append(Capture(tensor, kin, label, caster, path, synthetic))

    if not out:
        raise RuntimeError(f"{root} contained no usable captures")
    return out


def to_arrays(caps: list[Capture]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.stack([c.tensor for c in caps]).astype(np.float32)      # (n, 64, 8)
    y = np.array([c.label for c in caps], dtype=np.int32)
    g = np.array([c.caster for c in caps])
    return X, y, g


def loco_folds(groups: np.ndarray):
    """Leave-One-Caster-Out splits.

    This is THE validation protocol for this project, and it is not the same as
    a random split.  A random split puts reps from the same person on both sides
    of the line, so the model can memorise that person's idiosyncratic wrist
    signature and still look excellent.  The judges are strangers.  LOCO tests
    the only question that matters -- does this work on a hand it has never seen
    -- and it will read several points lower than a random split.  Report the
    lower number; it is the one you can defend when a judge picks up the wand.
    """
    for caster in sorted(set(groups)):
        test = groups == caster
        yield caster, ~test, test
