"""training/parity_check.py  --  task P8, the most valuable test in this repo.

Asserts that training/preprocess.py and firmware/wand_infer/preprocess.cpp
produce the same tensor from the same bytes, to within 1e-3 per element.

WHY THIS IS THE MOST VALUABLE TEST
    Every other bug in this project announces itself. A wiring fault gives no
    I2C device. A bad build flag fails to compile. A bad model gives obviously
    wrong classes. A preprocessing divergence gives none of that: both chains run
    clean, the training accuracy is excellent, the firmware is fast and stable,
    and the wand classifies at chance. There is nothing to grep for and no error
    to read. Teams lose whole days to it and usually conclude the model is
    "overfitted" when in fact it is being fed a distribution it has never seen.

    A single scale constant, a filter seeded from zero on one side and from the
    accelerometer on the other, an off-by-one in the resample -- any of these is
    enough. This test finds all of them in about two seconds.

RUN IT BEFORE YOU HAVE HARDWARE. preprocess.cpp includes only stdint and math,
so parity_host.cpp compiles on a laptop and the check works on any CSV --
synthetic included, because this test is about the CODE, not the data. Running
it on synthetic captures is a completely valid result and should be in the report.

    python parity_check.py --data ../data/synthetic --n 25
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import wand_config as C
from dataset import _read_csv
from preprocess import gate_kinematics, preprocess

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware" / "wand_infer"


def build_host_binary() -> Path:
    exe = FW / "parity_host"
    cmd = ["g++", "-O2", "-std=c++17",
           f"-I{ROOT/'shared'}", f"-I{ROOT/'firmware'/'common'}", "-o", str(exe),
           str(FW / "parity_host.cpp"), str(FW / "preprocess.cpp"),
           str(FW / "reject.cpp")]
    subprocess.run(cmd, check=True)
    return exe


# Mirror of enum RejectReason in firmware/wand_infer/reject.h. Only the Stage 1
# values can be produced by gateKinematics(); the rest are listed so the indices
# line up with the enum and a future insertion in the middle is obvious here.
GATE_CODE = {
    None: 0, "TOO_SHORT": 1, "TOO_LONG": 2,
    "TOO_STILL": 3, "TOO_SHALLOW": 4, "IMPACT": 5,
}


def run_host(exe: Path, csv: Path) -> tuple[np.ndarray, list[float], int]:
    r = subprocess.run([str(exe), str(csv)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"parity_host failed on {csv}: {r.stderr.strip()}")
    kin: list[float] = []
    gate = -1
    rows = []
    for line in r.stdout.splitlines():
        if line.startswith("# kin"):
            kin = [float(v) for v in line.split()[2:]]
        elif line.startswith("# gate"):
            gate = int(line.split()[2])
        elif line.strip():
            rows.append([float(v) for v in line.split(",")])
    return np.asarray(rows, dtype=np.float64), kin, gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--n", type=int, default=25, help="captures to check")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    exe = build_host_binary()
    files = sorted(Path(args.data).rglob("*.csv"))[: args.n]
    if not files:
        sys.exit(f"no CSVs under {args.data}")

    worst = 0.0
    worst_file = ""
    worst_kin = 0.0
    checked = 0
    gate_mismatches: list[str] = []

    for csv in files:
        raw, label, caster, _ = _read_csv(csv)
        if len(raw) < C.MIN_GESTURE_SAMPLES:
            continue
        py_t, py_kin = preprocess(raw)
        cpp_t, cpp_kin, cpp_gate = run_host(exe, csv)

        if cpp_t.shape != py_t.shape:
            sys.exit(f"SHAPE MISMATCH {csv}: python {py_t.shape} vs c++ {cpp_t.shape}")

        d = float(np.abs(cpp_t - py_t.astype(np.float64)).max())
        dk = float(np.abs(np.asarray(cpp_kin) - np.asarray(py_kin.as_row())).max())
        if d > worst:
            worst, worst_file = d, csv.name
        worst_kin = max(worst_kin, dk)

        # Stage 1 DECISION parity, not just feature parity. A gate that agrees on
        # every number and disagrees on the verdict is still a wand that behaves
        # differently from the model that was validated.
        py_gate = GATE_CODE[gate_kinematics(py_kin)]
        if py_gate != cpp_gate:
            gate_mismatches.append(f"{csv.name}: python={py_gate} c++={cpp_gate}")
        checked += 1

    print(f"checked {checked} captures")
    print(f"  worst tensor element difference : {worst:.3e}   ({worst_file})")
    print(f"  worst kinematic difference      : {worst_kin:.3e}")
    print(f"  Stage 1 gate verdict mismatches : {len(gate_mismatches)}")
    print(f"  tolerance                       : {args.tol:.1e}")

    if gate_mismatches:
        for m in gate_mismatches[:5]:
            print(f"    {m}")
        sys.exit("PARITY FAILED -- gateKinematics() disagrees with gate_kinematics().")
    if worst > args.tol or worst_kin > args.tol:
        sys.exit("PARITY FAILED -- the Python and C++ chains disagree. "
                 "Fix this before training anything.")
    print("PARITY OK -- preprocess.py/.cpp and the Stage 1 gate all agree.")


if __name__ == "__main__":
    main()
