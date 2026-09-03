"""training/audit_corpus.py

Physical plausibility audit of a gesture corpus -- synthetic OR real.

WHAT THIS IS FOR
    synth_bootstrap.py derives its accelerometer signal from the wand's
    geometry rather than picking amplitudes by hand.  That is only an
    improvement if the numbers that come out are the numbers a real wand
    produces, and "it looks about right in a plot" is not a check.  This script
    reduces a corpus to the handful of quantities that CAN be checked against
    physics and against a stopwatch, and states a pass or a warning against a
    band for each one.

    Every band below is a property of a person waving a 50 cm stick, not of this
    project's model, so the same script is the acceptance test for the REAL
    corpus once it exists.  Run it on data/raw the day the first fifty captures
    land: if the real numbers fall outside these bands, the bands are what was
    wrong, and the synthetic corpus that was tuned to them has to be regenerated
    before it is used to size anything.

WHAT IT DELIBERATELY DOES NOT REPORT
    Accuracy, separability, or anything a classifier could be scored on.  This
    is an instrumentation check.  Numbers from a synthetic corpus stay
    meaningless as results no matter how physical they are.

    python training/audit_corpus.py data/synthetic
    python training/audit_corpus.py data/raw
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import wand_config as C
from dataset import _read_csv, load_dataset
from preprocess import (gate_kinematics, quat_conj, quat_rotate,
                        sanitise_quaternions, to_physical)

# -----------------------------------------------------------------------------
# Plausibility bands.
#
# Sources, in order of how much weight to put on them:
#   * arithmetic from the geometry in shared/wand_config.h,
#   * the constants the Stage 1 gate and the tensor normalisation already commit
#     to (a corpus that violates those makes the shipped firmware wrong),
#   * the range of human wrist and forearm motion in the biomechanics
#     literature, which is where the angular-rate ceiling comes from.
#
# These are BANDS, not targets. A class landing outside one is a question to
# answer, not necessarily a bug -- but it must be answered before the corpus is
# used to size the arena, set the divisors, or sweep a threshold.
# -----------------------------------------------------------------------------
BANDS = {
    # name:            (low, high, unit, why)
    "peak_w": (1.5, 14.0, "rad/s",
               "below 1.5 nothing was cast; above 14 (800 dps) exceeds "
               "sustained human wrist rate and would also exceed NORM_GYRO"),
    "peak_a": (0.15, 5.90, "g",
               "the +/-4 g rail is on the RAW signal and the fusion subtracts "
               "gravity afterwards, so per-axis linear acceleration reaches "
               "5 g and its magnitude a little under 6"),
    "tip_speed": (0.3, 8.0, "m/s",
                  "wand tip speed about the butt; a hard cast is 3-6 m/s, "
                  "above 8 the wand is being thrown"),
    "duration": (0.45, 2.30, "s",
                 "REJ_MIN_DURATION_S .. REJ_MAX_DURATION_S, whole buffer"),
}

# How often a class is ALLOWED to saturate the +/-4 g range. Some saturation on
# the hardest casts is real and expected; a class that saturates most of the
# time has lost its amplitude information and the generator is overdriving it.
CLIP_WARN_FRACTION = 0.35


def reconstruct_gravity(quat: np.ndarray) -> np.ndarray:
    """Body-frame gravity, in g, recovered from the orientation quaternion.

    The logger stores the fusion's gravity-free linear acceleration, so the raw
    accelerometer signal is not in the file -- but it can be rebuilt, because
    the same file carries the orientation the fusion used to remove gravity:

        g_body = R^T * (0, 0, 1) = rotate(conj(q), world_up)

    That makes the +/-4 g saturation rate measurable from a CSV alone, which
    matters because saturation is invisible in the linear channel: a clipped
    sample looks like an ordinary large number there, not like a rail.
    """
    up = np.zeros_like(quat[:, 1:4])
    up[:, 2] = 1.0
    return quat_rotate(quat_conj(quat), up)


def audit(root: Path, allow_synthetic: bool) -> int:
    caps = load_dataset(root, allow_synthetic=allow_synthetic)
    synthetic = any(c.synthetic for c in caps)
    stamp = "SYNTHETIC -- NOT A RESULT | " if synthetic else ""

    per: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    gate_reasons: dict[int, Counter] = defaultdict(Counter)

    for cap in caps:
        # Re-read through the pipeline's own parser rather than a second one --
        # an audit that reads the file differently from the trainer can only
        # ever audit the reader.
        raw, _, _, _ = _read_csv(cap.path)
        quat, acc, gyro = to_physical(raw)
        quat = sanitise_quaternions(quat)

        amag = np.linalg.norm(acc, axis=1)
        wmag = np.linalg.norm(gyro, axis=1)
        gb = reconstruct_gravity(quat)
        raw_a = acc + gb                       # what the accelerometer saw, in g
        clipped = np.any(np.abs(raw_a) >= C.BNO_FUSION_ACC_RANGE_G - 1e-3, axis=1)

        d = per[cap.label]
        d["peak_w"].append(float(wmag.max()))
        d["peak_a"].append(float(amag.max()))
        d["tip_speed"].append(float(wmag.max()) * C.WAND_LENGTH_M)
        d["duration"].append(cap.kin.duration_s)
        d["path_rad"].append(cap.kin.path_rad)
        d["spike_run"].append(float(cap.kin.spike_run))
        d["clip_frac"].append(float(clipped.mean()))
        d["rms_w"].append(float(np.sqrt((wmag ** 2).mean())))

        gate_reasons[cap.label][gate_kinematics(cap.kin) or "PASS"] += 1

    # ---------------------------------------------------------------- report
    print(f"\n{stamp}corpus audit: {root}  ({len(caps)} captures)")
    print(f"geometry: wand {C.WAND_LENGTH_M*100:.0f} cm, BNO055 "
          f"{C.SENSOR_FROM_TIP_M*100:.0f} cm below the tip, +X towards the tip")
    print(f"lever arm to the sensor: {C.PIVOT_WRIST_M + C.SENSOR_FROM_BUTT_M:.2f} m "
          f"(wrist) .. {C.PIVOT_SHOULDER_M + C.SENSOR_FROM_BUTT_M:.2f} m (shoulder)")

    print("\n--- kinematics, median [5th, 95th percentile] ---")
    hdr = f"{'class':<20}{'n':>4} {'peak |w| rad/s':>22} {'peak |a| g':>20} " \
          f"{'tip m/s':>16} {'dur s':>14} {'path rad':>12} {'spike':>7} {'clip%':>7}"
    print(hdr)
    print("-" * len(hdr))

    def pct(v):
        a = np.asarray(v)
        return np.percentile(a, 50), np.percentile(a, 5), np.percentile(a, 95)

    warnings: list[str] = []
    for label in range(C.NUM_CLASSES):
        d = per.get(label)
        if not d:
            continue
        name = C.GESTURE_NAMES[label]
        cells = []
        for key in ("peak_w", "peak_a", "tip_speed", "duration"):
            m, lo, hi = pct(d[key])
            cells.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
            blo, bhi, unit, why = BANDS[key]
            if m < blo or m > bhi:
                warnings.append(f"{name}: median {key} = {m:.2f} {unit} is outside "
                                f"[{blo}, {bhi}] -- {why}")
        pr_m, _, _ = pct(d["path_rad"])
        sr_m, _, _ = pct(d["spike_run"])
        cf = float(np.mean(d["clip_frac"]))
        if cf > CLIP_WARN_FRACTION:
            warnings.append(f"{name}: saturates the +/-4 g range on "
                            f"{100*cf:.0f}% of samples -- amplitude information "
                            f"is being destroyed, reduce the commanded excursions")
        print(f"{name:<20}{len(d['peak_w']):>4} {cells[0]:>22} {cells[1]:>20} "
              f"{cells[2]:>16} {cells[3]:>14} {pr_m:>12.2f} {sr_m:>7.0f} {100*cf:>6.1f}%")

    # ---- the one relationship that only holds if the physics is real --------
    # A wand rotating about a pivot 45-100 cm away MUST show acceleration that
    # tracks angular rate. If peak |a| and peak |w| are uncorrelated across a
    # spell class, the accelerometer channel is not coming from the geometry --
    # which is exactly the defect this revision exists to remove, so it is
    # checked rather than assumed.
    print("\n--- lever-arm coupling: corr(peak |w|, peak |a|) within each class ---")
    print("    a rotation-driven class should be strongly positive; STUPEFY is")
    print("    hand-driven by construction and is the one allowed to be weak.")
    for label in range(C.NUM_CLASSES):
        d = per.get(label)
        if not d or len(d["peak_w"]) < 8:
            continue
        rho = float(np.corrcoef(d["peak_w"], d["peak_a"])[0, 1])
        flag = ""
        if label not in (int(C.WandGestureClass.STUPEFY),
                         int(C.WandGestureClass.NOISE)) and rho < 0.30:
            flag = "   <-- WEAK: acceleration is not tracking rotation"
            warnings.append(f"{C.GESTURE_NAMES[label]}: peak |a| barely correlates "
                            f"with peak |w| (rho = {rho:.2f}); the lever-arm term "
                            f"should dominate this class")
        print(f"    {C.GESTURE_NAMES[label]:<20} rho = {rho:+.2f}{flag}")

    # ---- Stage 1 gate ------------------------------------------------------
    print("\n--- Stage 1 kinematic gate (what the firmware would do) ---")
    for label in range(C.NUM_CLASSES):
        cnt = gate_reasons.get(label)
        if not cnt:
            continue
        tot = sum(cnt.values())
        parts = ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in cnt.most_common())
        name = C.GESTURE_NAMES[label]
        print(f"    {name:<20} {parts}")
        if label != int(C.WandGestureClass.NOISE):
            passed = cnt.get("PASS", 0) / tot
            if passed < 0.97:
                warnings.append(f"{name}: Stage 1 rejects {100*(1-passed):.0f}% of "
                                f"valid casts. A gate that eats real spells is worse "
                                f"than no gate -- re-sweep it in calibrate.py")

    # ---- tensor headroom ---------------------------------------------------
    X = np.stack([c.tensor for c in caps])
    print("\n--- tensor occupancy per channel group (fraction of range used) ---")
    print("    a group hugging 1.00 is clipping; one below ~0.30 is wasting int8 codes")
    for lo, hi, nm in ((C.CH_ACC, C.CH_ACC + 3, "accel (anchored)"),
                       (C.CH_GYRO, C.CH_GYRO + 3, "gyro (anchored)"),
                       (C.CH_QUAT, C.CH_QUAT + 4, "quaternion"),
                       (C.CH_AMAG, C.CH_AMAG + 1, "|a|"),
                       (C.CH_WMAG, C.CH_WMAG + 1, "|w|")):
        v = np.abs(X[:, :, lo:hi])
        print(f"    {nm:<20} p50 {np.percentile(v, 50):.3f}   "
              f"p99 {np.percentile(v, 99):.3f}   max {v.max():.3f}   "
              f"at-rail {100*np.mean(v >= 0.999):.2f}%")

    # ---- verdict -----------------------------------------------------------
    print()
    if warnings:
        print(f"{len(warnings)} PLAUSIBILITY WARNING(S):")
        for w in warnings:
            print(f"  ! {w}")
        return 1
    print("all classes fall inside their physical plausibility bands.")
    if synthetic:
        print("\nThis says the generator is producing signals of the right SHAPE and")
        print("SCALE for the wand that was built. It says nothing whatsoever about")
        print("whether the classifier works. Only real captures can say that.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=C.SYNTHETIC_DATA_DIR)
    args = ap.parse_args()
    root = Path(args.root)
    # Auditing is a measurement, not a result, so synthetic input is allowed --
    # but it is always announced, on every line of the output.
    sys.exit(audit(root, allow_synthetic=True))


if __name__ == "__main__":
    main()
