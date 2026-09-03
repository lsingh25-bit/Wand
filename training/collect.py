"""training/collect.py

Drives the logger firmware over serial and writes real captures to data/raw/.
This is the ONLY program allowed to write into data/raw.

    python collect.py --port COM5 --caster ankit --gesture LUMOS --reps 15

Design notes that matter for data quality:

  * The dt_us column is validated on every capture, not sampled. A capture whose
    inter-sample interval leaves 10000 +/- 200 us is rejected and re-prompted,
    because timing corruption produces data that looks entirely valid, trains
    without complaint, and quietly degrades the model. The usual cause is an LED
    or OLED refresh firing during capture, which is why the logger must not touch
    either while capturing.
  * The duplicate-frame rate is validated too. The BNO055's fusion runs at its own
    100 Hz, unlocked from ours, so a few repeated frames per capture are normal and
    harmless. A HIGH rate is not: it means the I2C bus is hanging, or the chip has
    dropped out of fusion mode, and the capture is then partly sample-and-hold.
  * The gyro calibration status is recorded in every capture's header. The fusion's
    continuously-estimated gyro bias is what replaced the software bias removal the
    MPU6050 build needed; a capture taken below status 3 is quietly wrong, and
    recording it means such captures can be found and discarded later rather than
    silently poisoning the dataset.
  * The script randomises the prompted SPEED and GRIP for each rep. Left to
    themselves people converge on one comfortable way of performing a gesture
    within about five reps, and a dataset of one style per caster is exactly the
    overfitting the invariance test is designed to expose.
  * Captures are named with the caster, so leave-one-caster-out can group them.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import serial  # pyserial

import wand_config as C

SPEEDS = ["SLOW", "NORMAL", "FAST"]
GRIPS = ["standard grip", "rolled wrist", "loose / fingertip"]
POSES = ["standing", "seated", "arm extended"]


def read_capture(ser: serial.Serial, timeout_s: float = 30.0):
    rows, meta, started = [], {}, False
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode("utf-8", "replace").strip()
        if not line:
            continue
        if line.startswith("# GESTURE_START"):
            started = True
            rows = []
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
            continue
        if line.startswith("# GESTURE_END"):
            return rows, meta
        if not started or line.startswith("#") or line.startswith("idx"):
            continue
        rows.append(line)
    return None, {}


def validate(rows: list[str]) -> tuple[bool, str]:
    if len(rows) < C.MIN_GESTURE_SAMPLES:
        return False, f"only {len(rows)} samples (< {C.MIN_GESTURE_SAMPLES})"
    if len(rows) > C.MAX_GESTURE_SAMPLES + C.PREROLL_SAMPLES + C.POSTROLL_SAMPLES:
        return False, f"{len(rows)} samples, longer than the 2 s window"

    bad = dups = 0
    for r in rows[1:]:
        parts = r.split(",")
        # idx, qw, qx, qy, qz, lax, lay, laz, gx, gy, gz, dt_us, dup
        if len(parts) < 13:
            return False, f"malformed row: {len(parts)} columns, expected >= 13"
        if abs(int(parts[11]) - C.DT_US) > C.DT_TOL_US:
            bad += 1
        if int(parts[12]):
            dups += 1

    if bad:
        return False, (f"{bad}/{len(rows)} samples outside {C.DT_US}+/-{C.DT_TOL_US} us "
                       f"-- something is refreshing the LED or OLED during capture")

    frac = dups / max(len(rows) - 1, 1)
    if frac > C.MAX_DUP_FRACTION:
        return False, (f"{frac*100:.0f}% duplicate fusion frames (limit "
                       f"{C.MAX_DUP_FRACTION*100:.0f}%) -- the I2C bus is stalling or "
                       f"the BNO055 has dropped out of fusion mode")

    # A quaternion that never moves means the chip is not fusing: either it fell
    # back to CONFIG mode, or it is in a non-fusion mode where the QUA registers
    # read zero. Either way the orientation channels would be constant, and a
    # model trained on that has four dead inputs and nobody would notice.
    q = np.array([[int(v) for v in r.split(",")[1:5]] for r in rows], dtype=float)
    if np.abs(q - q[0]).max() < 1.0:
        return False, "quaternion never changes -- the BNO055 is not in fusion mode"
    if np.linalg.norm(q, axis=1).max() < 0.5 * C.BNO_LSB_PER_QUAT:
        return False, "quaternion norm is near zero -- no valid fusion output"

    return True, f"ok ({dups} duplicate frames)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--caster", required=True)
    ap.add_argument("--gesture", required=True, choices=[n.replace(" ", "_") for n in C.GESTURE_NAMES])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default=C.REAL_DATA_DIR)
    args = ap.parse_args()

    label = [n.replace(" ", "_") for n in C.GESTURE_NAMES].index(args.gesture)
    outdir = Path(args.out) / args.caster / args.gesture
    outdir.mkdir(parents=True, exist_ok=True)

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.write(f"SET label={label} caster={args.caster}\n".encode())

    saved = 0
    rng = random.Random()
    while saved < args.reps:
        # [[prompts]]
        if label == int(C.WandGestureClass.NOISE):
            prompt = rng.choice([
                "walk three steps holding the wand normally",
                "scratch your head with the wand in hand",
                "talk with your hands for two seconds",
                "put the wand down on the table",
                "start a spell and abandon it halfway",
                "pass the wand to your other hand",
            ])
            print(f"\n[{saved+1}/{args.reps}] NOISE -- {prompt}")
        else:
            print(f"\n[{saved+1}/{args.reps}] {args.gesture}  "
                  f"speed={rng.choice(SPEEDS)}  grip={rng.choice(GRIPS)}  "
                  f"pose={rng.choice(POSES)}")
        print("    hold the trigger, perform, release ...", end="", flush=True)
        # [[/prompts]]

        rows, meta = read_capture(ser)
        if rows is None:
            print(" timeout, retrying")
            continue
        ok, why = validate(rows)
        if not ok:
            print(f" REJECTED: {why}")
            continue

        path = outdir / f"{args.caster}_{args.gesture}_{saved}_{int(time.time())}.csv"
        with open(path, "w") as f:
            f.write(f"# GESTURE_START label={label} caster={args.caster} "
                    f"rep={saved} nsamples={len(rows)} "
                    f"calib={meta.get('calib', '?')} mode={meta.get('mode', '?')}\n")
            f.write(",".join(C.CSV_COLUMNS) + "\n")
            f.write("\n".join(rows) + "\n")
            f.write("# GESTURE_END\n")
        saved += 1
        print(f" saved {len(rows)} samples -> {path.name}")

    print(f"\ndone: {saved} captures in {outdir}")


if __name__ == "__main__":
    sys.exit(main())
