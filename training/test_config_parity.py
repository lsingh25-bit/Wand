"""training/test_config_parity.py

Asserts wand_config.py still equals shared/wand_config.h (and wand_thresholds.h).

The header is the source of truth; this parses it and compares. A constant that
drifts between the two is one of the two silent-failure classes in this project
(the other being the preprocessing divergence that parity_check.py covers).

    pytest training/test_config_parity.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import wand_config as C

ROOT = Path(__file__).resolve().parents[1]
HEADERS = [ROOT / "shared" / "wand_config.h", ROOT / "shared" / "wand_thresholds.h"]

CONSTEXPR = re.compile(
    r"constexpr\s+(?:int|float|uint8_t|uint16_t|uint32_t)\s+(\w+)\s*=\s*"
    r"(0[xX][0-9a-fA-F]+|[-\d.eE]+)f?\s*[;*]")


def header_constants() -> dict[str, float]:
    out: dict[str, float] = {}
    for h in HEADERS:
        assert h.exists(), f"missing {h}"
        for name, val in CONSTEXPR.findall(h.read_text()):
            out[name] = float(int(val, 16)) if val.lower().startswith("0x") else float(val)
    return out


# Constants whose C++ definition is an expression rather than a literal, or which
# are intentionally Python-only.
SKIP = {"TENSOR_LEN", "LINACC_LSB_PER_G", "TFLM_ARENA_BYTES",
        "CAL_RECALL_AT_OP", "CAL_MISFIRE_AT_OP"}


def test_headers_exist():
    for h in HEADERS:
        assert h.exists()


def test_shared_constants_match():
    hdr = header_constants()
    checked = 0
    mismatches = []
    for name, hval in hdr.items():
        if name in SKIP or not hasattr(C, name):
            continue
        pval = float(getattr(C, name))
        if abs(pval - hval) > 1e-9:
            mismatches.append(f"{name}: header={hval} python={pval}")
        checked += 1
    assert not mismatches, "config drift:\n  " + "\n  ".join(mismatches)
    assert checked > 25, f"only {checked} constants cross-checked; the regex is not matching"


@pytest.mark.parametrize("name,expected", [
    ("DT_S", 1.0 / 100),
    ("TENSOR_LEN", 64 * 12),
    ("LINACC_LSB_PER_G", 100.0 * 9.80665),
    ("TFLM_ARENA_BYTES", 28 * 1024),
])
def test_derived_constants(name, expected):
    assert abs(float(getattr(C, name)) - expected) < 1e-9


def test_channel_layout_is_self_consistent():
    """The 12 channels must tile [0, N_CHANNELS) exactly once, with no gap and no
    overlap. A silently overlapping group would make one channel overwrite
    another and the tensor would still have the right shape."""
    assert len(C.CHANNEL_NAMES) == C.N_CHANNELS
    assert C.CH_ACC == 0
    assert C.CH_GYRO == C.CH_ACC + 3
    assert C.CH_QUAT == C.CH_GYRO + 3
    assert C.CH_AMAG == C.CH_QUAT + 4
    assert C.CH_WMAG == C.CH_AMAG + 1
    assert C.CH_WMAG + 1 == C.N_CHANNELS


def test_accel_normalisation_matches_the_reportable_range():
    """NORM_ACC_LIN must equal what the sensor can physically report -- no more
    (wasted int8 codes) and no less (clipped casts).

    R3 got this backwards and it cost 2.4% of all samples to clipping. The rail
    is on the RAW accelerometer, and the fusion SUBTRACTS gravity afterwards:

        lin = raw - g,   raw in [-4, +4] g,   g in [-1, +1] g
        =>  lin in [-5, +5] g

    so the reportable range is the rail PLUS one gravity, not minus it. The
    measured per-axis maximum over the bootstrap corpus is exactly 5.00 g."""
    reportable = C.BNO_FUSION_ACC_RANGE_G + 1.0
    assert abs(C.NORM_ACC_LIN - reportable) < 1e-9
    assert abs(C.NORM_AMAG - reportable) < 1e-9
    # The spike level must sit inside the range with room above it, or a window
    # could reach the level only by railing and the width test would be reading
    # the rail rather than the motion.
    assert C.REJ_SPIKE_G < reportable - 1.0


def test_enum_matches_header():
    src = (ROOT / "shared" / "wand_config.h").read_text()
    body = src.split("enum WandGestureClass")[1].split("}")[0]
    pairs = re.findall(r"(GESTURE_\w+)\s*=\s*(-?\d+)", body)
    hdr = {k: int(v) for k, v in pairs}

    for member in C.WandGestureClass:
        key = f"GESTURE_{member.name}"
        assert key in hdr, f"{key} missing from the header enum"
        assert hdr[key] == int(member), f"{key}: header={hdr[key]} python={int(member)}"

    assert hdr["GESTURE_NONE"] == -1
    assert C.NUM_CLASSES == len(C.GESTURE_NAMES) == 6
    assert C.NUM_SPELL_CLASSES == C.NUM_CLASSES - 1


def test_noise_is_last_class():
    """NOISE must stay at the END of the enum. If it is ever inserted in the
    middle, every gesture recorded before that change is silently relabelled."""
    assert int(C.WandGestureClass.NOISE) == C.NUM_CLASSES - 1


def test_thresholds_flagged_uncalibrated_until_real_data():
    """A demo build must not ship placeholder thresholds. This test is expected
    to FAIL until calibrate.py has run against data/raw -- that failure is the
    reminder, not a bug."""
    src = (ROOT / "shared" / "wand_thresholds.h").read_text()
    calibrated = "WAND_THRESHOLDS_CALIBRATED 1" in src
    if not calibrated:
        pytest.skip("thresholds not yet calibrated on real data "
                    "(run training/calibrate.py --data data/raw before the demo)")
    assert C.THRESHOLDS_CALIBRATED
