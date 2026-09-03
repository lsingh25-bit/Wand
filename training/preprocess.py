"""training/preprocess.py

THE reference implementation of the signal chain.  firmware/wand_infer/preprocess.cpp
is a line-for-line port of this file, and training/parity_check.py asserts the two
agree to 1e-3 on captured data.

If you change anything in this file you MUST change preprocess.cpp in the same
commit, or the parity test will fail -- which is exactly what it is for.

================================ REVISION R3 ==================================
The chain got SHORTER when the sensor changed to a BNO055 running IMU fusion.

Deleted, because the sensor's own Cortex-M0 already does all three, better:
    * gyro bias removal      -> the fusion estimates and removes bias continuously
    * complementary filter   -> the fusion is a proper Kalman-class estimator
    * gravity projection     -> the LIA registers are already gravity-free

What replaced them is not more filtering but a change of FRAME.  The chip gives
us an orientation quaternion alongside the data, so we can express the whole
gesture in the frame the wand occupied at the instant the caster started moving.

Chain now:
    raw int16  ->  physical units (g, rad/s, unit quaternion)
               ->  quaternion sanitising and hemisphere continuity
               ->  q_rel[i] = conj(q[0]) * q[i]        (anchor to gesture start)
               ->  rotate linear accel and angular rate into the anchored frame
               ->  assemble 12 channels
               ->  resample to a fixed 64 steps
               ->  clip and divide by frozen constants
    = float32 tensor of shape (64, 12)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import wand_config as C


# -----------------------------------------------------------------------------
# Quaternion helpers.  Hamilton convention, (w, x, y, z), unit norm.
# Every one of these has a byte-for-byte twin in preprocess.cpp.
# -----------------------------------------------------------------------------
def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate.  For a unit quaternion this is also the inverse."""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b.  Broadcasts over leading axes."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q, without forming a rotation matrix.

        t  = 2 * (q_vec x v)
        v' = v + q_w * t + q_vec x t

    Fifteen multiplies and twelve adds, versus building a 3x3 matrix and
    multiplying.  More importantly it is a short, closed-form expression with no
    branches, which is what makes the C++ port trivially checkable against this.
    """
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def sanitise_quaternions(q: np.ndarray) -> np.ndarray:
    """Normalise, replace degenerate quaternions with the identity, and enforce
    hemisphere continuity along the time axis.

    Three separate hazards, all of which produce a valid-looking tensor:

    1.  A quaternion read while the BNO055 is still in CONFIG mode, or during the
        I2C hiccup that clock stretching can cause, comes back as all zeros.
        Normalising that is a division by zero.  We substitute the identity, and
        because the identity means "no rotation relative to the reference" it is
        the safest possible fallback -- the sample simply contributes no rotation.

    2.  Quaternions double-cover rotations: q and -q are the same orientation.
        The BNO055 is free to hand back either, and it does flip sign.  Without
        the continuity pass, channels 6..9 contain sign discontinuities that look
        like instantaneous 360-degree rotations, and no convolution can learn
        through that.

    3.  Continuity is enforced HERE, before the anchoring, because quaternion
        multiplication is bilinear -- a sign fixed now stays fixed in q_rel.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    norm = np.linalg.norm(q, axis=1)

    bad = norm < C.QUAT_MIN_NORM
    q[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    norm[bad] = 1.0
    q /= norm[:, None]

    # Hemisphere continuity: keep every sample on the same side as the one before.
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


# -----------------------------------------------------------------------------
# Step 1 -- raw counts to physical units
# -----------------------------------------------------------------------------
def to_physical(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """raw: (N, 10) int16 columns qw qx qy qz lax lay laz gx gy gz.

    Returns (quaternion, linear acceleration in g, angular rate in rad/s).

    The divisors are fixed by the BNO055 datasheet and by the UNIT_SEL register,
    which the firmware writes explicitly at boot.  This is a real improvement over
    the MPU6050, where the scale depended on a range register that could be
    changed without anyone updating the constant -- here there is no range
    register to get wrong, because the fusion algorithm owns it.
    """
    raw = np.asarray(raw, dtype=np.float64)
    quat = raw[:, 0:4] / C.BNO_LSB_PER_QUAT
    acc = raw[:, 4:7] / C.LINACC_LSB_PER_G                 # m/s^2 counts -> g
    gyro = (raw[:, 7:10] / C.BNO_LSB_PER_DPS) * C.DEG2RAD   # dps counts -> rad/s
    return quat, acc, gyro


# -----------------------------------------------------------------------------
# Step 2 -- anchor the gesture to its own starting orientation
# -----------------------------------------------------------------------------
def anchor(quat: np.ndarray) -> np.ndarray:
    """q_rel[i] = conj(q[0]) * q[i] -- orientation relative to gesture start.

    q[0] is the first PRE-ROLL sample, 200 ms before the trigger went down, when
    the caster is still holding position.  That is the best available description
    of "how this person was holding the wand before they started", which is
    exactly the reference we want to remove.

    q_rel[0] is the identity by construction, so channel 6 always starts at 1.0
    and channels 7..9 always start at 0.0.  That fixed anchor is genuinely useful
    to the network: every gesture begins at the same point in the representation,
    so the convolutions only ever have to model the DEPARTURE from it.
    """
    return quat_mul(quat_conj(quat[0])[None, :], quat)


# -----------------------------------------------------------------------------
# Step 3 -- resampling
# -----------------------------------------------------------------------------
def resample_fixed(x: np.ndarray, n_out: int = C.N_RESAMPLE) -> np.ndarray:
    """Linear resample (N, C) -> (n_out, C) on a normalised index axis.

    Index-space rather than time-space interpolation, because the logger already
    guarantees a uniform 10 ms grid -- it rejects any capture whose dt_us leaves
    the +/- 200 us tolerance.  Index space is exactly reproducible in C++ with two
    multiplies and an add, with no floating-point time accumulation to diverge.

    This step is what gives speed invariance: a 0.6 s Lumos and a 1.4 s Lumos both
    become 64 steps, so the network sees the SHAPE of the trajectory and is blind
    to how fast the judge happened to wave.  Duration is deliberately NOT fed to
    the network -- it is checked separately, as a hard range, in the Stage 1 gate.

    The quaternion channels are interpolated linearly rather than by slerp, and
    are deliberately NOT renormalised afterwards.  Two reasons: between adjacent
    100 Hz samples the rotation is under a degree, where lerp and slerp differ by
    less than 1e-5; and a renormalisation step is one more place for the Python
    and C++ chains to disagree, for a correction smaller than the sensor noise.
    """
    n_in = len(x)
    if n_in == n_out:
        return x.astype(np.float64).copy()
    out = np.zeros((n_out, x.shape[1]), dtype=np.float64)
    step = (n_in - 1) / (n_out - 1)
    for k in range(n_out):
        pos = k * step
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, n_in - 1)
        f = pos - i0
        out[k] = x[i0] * (1.0 - f) + x[i1] * f
    return out


def normalise(x: np.ndarray) -> np.ndarray:
    """Divide each channel group by its frozen constant and clip to [-1, 1].

    Deliberately NOT a per-sample z-score.  Two reasons:

    1.  A per-sample z-score would normalise away amplitude, and amplitude is a
        real discriminative feature here -- Stupefy is a hard ballistic thrust and
        Wingardium Leviosa is a gentle swish.  Z-scoring makes them look alike.
    2.  A per-sample statistic means the MCU must buffer the whole gesture, compute
        mean and variance, then walk it again.  Fixed divisors are one multiply per
        sample and can run streaming.

    The quaternion channels need no divisor at all -- a unit quaternion already
    lives in [-1, 1] on every component.  That is a small but real bonus of
    carrying orientation as a quaternion rather than as Euler angles or a rotation
    vector: it is natively in the range int8 quantisation wants.
    """
    out = np.empty_like(x)
    out[:, C.CH_ACC:C.CH_ACC + 3] = np.clip(x[:, C.CH_ACC:C.CH_ACC + 3] / C.NORM_ACC_LIN, -1.0, 1.0)
    out[:, C.CH_GYRO:C.CH_GYRO + 3] = np.clip(x[:, C.CH_GYRO:C.CH_GYRO + 3] / C.NORM_GYRO, -1.0, 1.0)
    out[:, C.CH_QUAT:C.CH_QUAT + 4] = np.clip(x[:, C.CH_QUAT:C.CH_QUAT + 4], -1.0, 1.0)
    out[:, C.CH_AMAG] = np.clip(x[:, C.CH_AMAG] / C.NORM_AMAG, 0.0, 1.0)
    out[:, C.CH_WMAG] = np.clip(x[:, C.CH_WMAG] / C.NORM_WMAG, 0.0, 1.0)
    return out


# -----------------------------------------------------------------------------
# Stage 1 gate features -- computed BEFORE resampling, where duration still means
# something in seconds and a spike is still a number of samples.
# -----------------------------------------------------------------------------
@dataclass
class KinematicFeatures:
    duration_s: float
    peak_w: float       # rad/s
    peak_a: float       # g, linear (gravity already removed by the sensor)
    path_rad: float     # integral of |w| dt -- total angle swept
    spike_run: int      # longest run of samples with |a| >= REJ_SPIKE_G

    def as_row(self) -> list[float]:
        return [self.duration_s, self.peak_w, self.peak_a, self.path_rad, float(self.spike_run)]


def kinematic_features(acc: np.ndarray, gyro: np.ndarray,
                       dt: float = C.DT_S) -> KinematicFeatures:
    n = len(gyro)
    wmag = np.linalg.norm(gyro, axis=1)
    amag = np.linalg.norm(acc, axis=1)

    run = best = 0
    for v in amag:
        run = run + 1 if v >= C.REJ_SPIKE_G else 0
        best = max(best, run)

    return KinematicFeatures(
        duration_s=n * dt,
        peak_w=float(wmag.max()) if n else 0.0,
        peak_a=float(amag.max()) if n else 0.0,
        path_rad=float(wmag.sum() * dt),
        spike_run=best,
    )


# -----------------------------------------------------------------------------
# The whole chain
# -----------------------------------------------------------------------------
def preprocess(raw: np.ndarray) -> tuple[np.ndarray, KinematicFeatures]:
    """raw: (N, 10) int16 -> ((64, 12) float32 tensor, KinematicFeatures).

    This is the ONLY function the training scripts call, and its C++ twin is the
    only function wand_infer.cpp calls.  Keeping the entry point singular is what
    makes the parity test meaningful.
    """
    quat, acc, gyro = to_physical(raw)
    quat = sanitise_quaternions(quat)
    q_rel = anchor(quat)

    # Rotate translation and rotation into the anchored frame.  This is the step
    # that decouples them: channels 0..2 become pure translation in a frame that
    # does not spin with the wand, instead of translation smeared through the
    # wand's own rotation the way a body-frame representation leaves it.
    acc_anch = quat_rotate(q_rel, acc)
    gyro_anch = quat_rotate(q_rel, gyro)

    kin = kinematic_features(acc, gyro)

    amag = np.linalg.norm(acc, axis=1)[:, None]     # rotation-invariant, so the
    wmag = np.linalg.norm(gyro, axis=1)[:, None]    # anchoring does not change these
    chans = np.concatenate([acc_anch, gyro_anch, q_rel, amag, wmag], axis=1)   # (N, 12)

    tensor = normalise(resample_fixed(chans))
    return tensor.astype(np.float32), kin


# -----------------------------------------------------------------------------
# Python mirror of firmware/wand_infer/reject.cpp::gateKinematics.
# Kept here so that reported accuracy is the accuracy of the SHIPPED cascade,
# not of a bare argmax the device never performs.
# -----------------------------------------------------------------------------
def gate_kinematics(k: KinematicFeatures) -> str | None:
    """Return None if the window is plausible, else the rejection reason."""
    if k.duration_s < C.REJ_MIN_DURATION_S:
        return "TOO_SHORT"
    if k.duration_s > C.REJ_MAX_DURATION_S:
        return "TOO_LONG"

    rotational = (k.peak_w >= C.REJ_MIN_PEAK_W) and (k.path_rad >= C.REJ_MIN_PATH_RAD)
    ballistic = k.peak_a >= C.REJ_MIN_PEAK_A
    if not rotational and not ballistic:
        return "TOO_STILL" if k.peak_w < C.REJ_MIN_PEAK_W else "TOO_SHALLOW"

    # Impact by spike WIDTH, not magnitude.  The fusion locks the accelerometer to
    # +/-4 g, so a table knock and a hard Stupefy both saturate near 3 g and
    # magnitude cannot separate them.  Duration can: a collision reaches the spike
    # level for one or two samples, a thrust holds it for tens.
    if k.peak_a >= C.REJ_SPIKE_G and k.spike_run < C.REJ_SPIKE_MIN_RUN:
        return "IMPACT"
    return None
