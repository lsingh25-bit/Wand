"""training/wand_config.py

Python mirror of shared/wand_config.h.  Every value here MUST equal the value in
the header; training/test_config_parity.py parses the header and asserts it.

Nothing in the training pipeline may re-type one of these numbers.

REVISION R4 -- BNO055 in IMU fusion mode, 12-channel anchored tensor, and the
physical wand geometry promoted into the shared config because the bootstrap
generator derives the accelerometer signal from it.
"""
from __future__ import annotations

from enum import IntEnum
from pathlib import Path

# Anchored to the repository root, not left as a bare relative string, because a
# bare "data/raw" resolves against the CALLER's cwd -- and every one of these
# scripts gets invoked both ways in practice ("python train_cnn.py" from inside
# training/, and "python training/train_cnn.py" from the repo root). The two
# invocations used to silently write to two different directories: the second
# form was fine, the first quietly created a phantom training/firmware/wand_infer/
# and training/data/ next to nothing, and the ONE real model_data.cc at the repo
# root was simply never touched -- with no error, because argparse defaults look
# perfectly valid either way. Anchoring here makes the default correct regardless
# of where the script is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ============================ SENSOR =========================================
I2C_ADDR_BNO = 0x28
BNO_REG_PAGE_ID = 0x07
BNO_CHIP_ID_VALUE = 0xA0
BNO_MODE_IMU = 0x08

BNO_LSB_PER_MS2 = 100.0
BNO_LSB_PER_DPS = 16.0
BNO_LSB_PER_QUAT = 16384.0
G_MS2 = 9.80665
DEG2RAD = 0.017453292519943295

LINACC_LSB_PER_G = BNO_LSB_PER_MS2 * G_MS2   # 980.665

BNO_FUSION_ACC_RANGE_G = 4.0
BNO_MIN_GYRO_CALIB = 3

# ======================= PHYSICAL WAND GEOMETRY ==============================
# Measured on the built wand.  See the header for the stack-up diagram and for
# why the synthetic generator derives acceleration from these numbers rather
# than inventing it.  Body axes: +X along the tube towards the tip, +Z out of
# the BNO055's PCB face, +Y completing the right-handed set (caster's left).
WAND_LENGTH_M = 0.500
SENSOR_FROM_TIP_M = 0.130
SENSOR_FROM_BUTT_M = 0.370        # = WAND_LENGTH_M - SENSOR_FROM_TIP_M
SENSOR_RADIAL_OFF_M = 0.012

PIVOT_WRIST_M = 0.080
PIVOT_ELBOW_M = 0.330
PIVOT_SHOULDER_M = 0.620

# ============================ SAMPLING =======================================
FS_HZ = 100
DT_S = 0.01
DT_US = 10_000
DT_TOL_US = 200
MAX_DUP_FRACTION = 0.12

# ============================ CAPTURE WINDOW =================================
RING_LEN = 320
PREROLL_SAMPLES = 20
POSTROLL_SAMPLES = 10
MIN_GESTURE_SAMPLES = 35
MAX_GESTURE_SAMPLES = 200

# ============================ FEATURE TENSOR =================================
N_RESAMPLE = 64
N_CHANNELS = 12
TENSOR_LEN = N_RESAMPLE * N_CHANNELS

CH_ACC = 0
CH_GYRO = 3
CH_QUAT = 6
CH_AMAG = 10
CH_WMAG = 11

QUAT_MIN_NORM = 0.50

# R4: 5.0 = the +/-4 g fusion rail on the RAW signal plus the 1 g of gravity the
# fusion subtracts back out.  See shared/wand_config.h for the derivation and for
# why NORM_GYRO was measured, considered, and deliberately left at 12.0.
NORM_ACC_LIN = 5.0
NORM_GYRO = 12.0
NORM_AMAG = 5.0
NORM_WMAG = 12.0

CHANNEL_NAMES = (
    "ax_anch", "ay_anch", "az_anch",
    "wx_anch", "wy_anch", "wz_anch",
    "qw_rel", "qx_rel", "qy_rel", "qz_rel",
    "amag", "wmag",
)

# ============================ CLASSES ========================================


class WandGestureClass(IntEnum):
    LUMOS = 0
    ALOHOMORA = 1
    EXPELLIARMUS = 2
    WINGARDIUM_LEVIOSA = 3
    STUPEFY = 4
    NOISE = 5


NUM_SPELL_CLASSES = 5
NUM_CLASSES = 6

GESTURE_NAMES = [
    "LUMOS",
    "ALOHOMORA",
    "EXPELLIARMUS",
    "WINGARDIUM LEVIOSA",
    "STUPEFY",
    "NOISE",
]

# ==================== REJECTION CASCADE THRESHOLDS ===========================
REJ_MIN_DURATION_S = 0.35  # R5: real-data sweep, see shared/wand_config.h comment
REJ_MAX_DURATION_S = 2.30
# R4: loosened from 2.20 / 1.00 / 1.80 after the joint sweep. See the header --
# the old setting was strictly dominated, and the gate's real job is only to
# reject a window in which the caster did not move.
REJ_MIN_PEAK_W = 0.45   # R5: real-data sweep, see shared/wand_config.h comment
REJ_MIN_PATH_RAD = 0.10
REJ_MIN_PEAK_A = 1.00

REJ_SPIKE_G = 2.00
REJ_SPIKE_MIN_RUN = 8  # R5: real-data sweep, see shared/wand_config.h comment

REJ_PERSIST_WINDOWS = 3
REJ_PERSIST_AGREE = 2
REJ_REFRACTORY_MS = 500

SEG_ENTER_W = 2.80
SEG_EXIT_W = 0.90
SEG_EXIT_HOLD = 12

# --- calibrated, mirrored from shared/wand_thresholds.h (see that file) -------
# These three are MEASURED from the pooled LOCO probability distribution, not
# chosen. calibrate.py rewrites both this block and the header together.
THRESHOLDS_CALIBRATED = False
REJ_TAU_CONF = 0.300
REJ_TAU_MARGIN = 0.100
REJ_TAU_NOISE = 0.350
# ------------------------------------------------------------------------------

# ============================ MODEL ==========================================
TFLM_ARENA_BYTES = 28 * 1024

# ==================== DATASET GUARD ==========================================
# The single tripwire that stops a synthetic number reaching the report.
MIN_REAL_CSV_FILES = 50
REAL_DATA_DIR = str(_REPO_ROOT / "data" / "raw")
SYNTHETIC_DATA_DIR = str(_REPO_ROOT / "data" / "synthetic")

# CSV column order emitted by wand_logger and consumed by everything downstream.
CSV_COLUMNS = ("idx", "qw", "qx", "qy", "qz",
               "lax", "lay", "laz", "gx", "gy", "gz", "dt_us", "dup")
