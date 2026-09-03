"""training/augment.py

Augmentation is this project's main defence against "The Wand Chooses the Wizard".

With five casters you have five hand-writing styles.  The judges will be a sixth,
seventh and eighth.  You cannot collect your way out of that in the time left, so
the job of augmentation is to manufacture the *between-caster* variation you
cannot record, and force the network to learn the parts of a gesture that are
common to everyone.

Every transform below operates on the already-preprocessed (64, 12) tensor, and
each is physically exact rather than a generic "add noise" -- which matters,
because an augmentation that produces impossible signals teaches the network to
separate real from impossible instead of Lumos from Alohomora.

Channel layout (R3):
    0..2   linear acceleration, anchored frame
    3..5   angular rate,        anchored frame
    6..9   relative quaternion  (w, x, y, z)
    10     |linear acceleration|
    11     |angular rate|

THE INVARIANT.  Every tensor the real chain produces satisfies q_rel[0] = identity
-- channel 6 starts at 1.0, channels 7..9 at 0.0.  The network will come to rely
on that fixed starting point, so every augmented tensor must satisfy it too.  Most
of the transforms below preserve it by construction; `reanchor` at the end of
`augment_one` guarantees it for the ones that do not.
"""
from __future__ import annotations

import numpy as np

import wand_config as C

Q0, Q1 = C.CH_QUAT, C.CH_QUAT + 4
A0, A1 = C.CH_ACC, C.CH_ACC + 3
W0, W1 = C.CH_GYRO, C.CH_GYRO + 3


# -----------------------------------------------------------------------------
# quaternion helpers (mirrors of preprocess.py's, kept local so augmentation
# cannot accidentally depend on the reference chain and hide a bug in it)
# -----------------------------------------------------------------------------
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qv = q[..., 0:1], q[..., 1:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = angle * 0.5
    return np.concatenate([[np.cos(h)], np.sin(h) * axis])


# -----------------------------------------------------------------------------
def grip_rotation(x: np.ndarray, rng: np.random.Generator,
                  roll_deg: float = 25.0, off_deg: float = 10.0) -> np.ndarray:
    """Rotate the sensor's mounting frame -- i.e. simulate a different grip.

    This is the highest-value augmentation for the invariance test.  Two people
    holding the same wand differ mostly by rotation about its long axis (how far
    the wrist is rolled), and secondarily by small pitch and yaw offsets.

    Under a constant grip change G, the anchored representation transforms as:

        a'    = G a          (and likewise for the angular rate)
        q_rel' = G q_rel G^-1   -- a CONJUGATION, not a left-multiply

    The conjugation is the part worth being careful about.  q_rel is a rotation
    expressed in the anchored frame, and rotating the frame itself is a change of
    basis, which for rotations is conjugation.  Left-multiplying instead -- the
    obvious-looking mistake -- would compose an extra rotation into the gesture
    and produce trajectories no wrist could perform.

    Note also that anchoring does NOT make this augmentation redundant.  Anchoring
    removes which way the caster was FACING; it does not remove how they were
    holding the wand, because the anchor rotates with the grip.  The two are
    complementary, and the report says so explicitly rather than overclaiming.
    """
    g = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]),
                              np.deg2rad(rng.uniform(-roll_deg, roll_deg)))
    g = _qmul(g, _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]),
                                       np.deg2rad(rng.uniform(-off_deg, off_deg))))
    g = _qmul(g, _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]),
                                       np.deg2rad(rng.uniform(-off_deg, off_deg))))
    gc = _qconj(g)

    out = x.copy()
    n = x.shape[0]
    gb = np.broadcast_to(g, (n, 4))
    out[:, A0:A1] = _qrot(gb, x[:, A0:A1])
    out[:, W0:W1] = _qrot(gb, x[:, W0:W1])
    out[:, Q0:Q1] = _qmul(_qmul(gb, x[:, Q0:Q1]), np.broadcast_to(gc, (n, 4)))
    return out


def nonuniform_time_warp(x: np.ndarray, rng: np.random.Generator,
                         n_knots: int = 4, sigma: float = 0.18) -> np.ndarray:
    """Warp time non-uniformly along the 64-step axis.

    Uniform speed change is already removed by the fixed-length resample, so
    scaling the whole gesture would be a no-op.  What is NOT removed, and what
    genuinely varies between people, is *rhythm*: one caster lingers at the top of
    the Lumos loop and snaps the descent, another does the reverse.

    Endpoints are pinned, which is what keeps q_rel[0] at the identity.
    """
    n = x.shape[0]
    knots = np.linspace(0, n - 1, n_knots)
    offs = rng.normal(0.0, sigma * n / n_knots, n_knots)
    offs[0] = offs[-1] = 0.0
    warp = np.interp(np.arange(n), knots, offs)
    src = np.clip(np.arange(n) + warp, 0, n - 1)

    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(src, np.arange(n), x[:, c])
    return out


def magnitude_scale(x: np.ndarray, rng: np.random.Generator,
                    lo: float = 0.82, hi: float = 1.20) -> np.ndarray:
    """Scale how forcefully and how widely the gesture was performed.

    Accelerometer and gyroscope get *independent* factors: a strong caster is not
    necessarily a wide-swinging one, and coupling them would teach a correlation
    that does not exist.

    The subtlety introduced in R3: scaling the angular rate without also scaling
    the ORIENTATION would make the tensor internally contradictory -- channels
    3..5 would describe a rotation that channels 6..9 do not perform, and the
    network would happily learn to detect that contradiction instead of learning
    the gesture.  So the relative quaternion's rotation angle is scaled by the
    same factor: theta -> k*theta about the same axis.
    """
    ka, kg = rng.uniform(lo, hi), rng.uniform(lo, hi)
    out = x.copy()
    out[:, A0:A1] *= ka
    out[:, W0:W1] *= kg
    out[:, C.CH_AMAG] *= ka
    out[:, C.CH_WMAG] *= kg

    q = x[:, Q0:Q1]
    qw = np.clip(q[:, 0], -1.0, 1.0)
    qv = q[:, 1:4]
    vn = np.linalg.norm(qv, axis=1)
    theta = 2.0 * np.arctan2(vn, qw)
    axis = qv / (vn[:, None] + 1e-12)
    h = (theta * kg) * 0.5
    out[:, Q0] = np.cos(h)
    out[:, Q0 + 1:Q1] = np.sin(h)[:, None] * axis
    return out


def time_shift(x: np.ndarray, rng: np.random.Generator, max_shift: int = 5) -> np.ndarray:
    """Shift the gesture within its window, edge-padded.

    Models trigger-timing variance: some casters press the button and then start
    moving, others are already moving.  The 20-sample pre-roll absorbs most of
    this, but not all of it, and a network that keys off the exact start index
    will fail the moment a judge's thumb is a beat late.

    A negative shift moves the anchor off sample 0; `reanchor` puts it back.
    """
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return x
    out = np.empty_like(x)
    if s > 0:
        out[:s] = x[0]
        out[s:] = x[:-s]
    else:
        out[s:] = x[-1]
        out[:s] = x[-s:]
    return out


def sensor_noise(x: np.ndarray, rng: np.random.Generator,
                 acc_sigma: float = 0.006, gyro_sigma: float = 0.004,
                 quat_sigma: float = 0.004, bias_sigma: float = 0.006) -> np.ndarray:
    """Additive noise on every channel group, plus a residual rate offset.

    The quaternion term models the fusion's own orientation estimation error,
    which is small but not zero and grows while gyro calibration is imperfect.
    Training with it present is what makes the network tolerant of a sensor that
    has only just reached calibration status 3 -- which, in a demo queue, it will
    have.  Sigmas are in normalised units.
    """
    out = x.copy()
    out[:, A0:A1] += rng.normal(0, acc_sigma, (x.shape[0], 3))
    out[:, W0:W1] += rng.normal(0, gyro_sigma, (x.shape[0], 3))
    out[:, W0:W1] += rng.normal(0, bias_sigma, (1, 3))       # constant over the window
    out[:, Q0 + 1:Q1] += rng.normal(0, quat_sigma, (x.shape[0], 3))
    q = out[:, Q0:Q1]
    out[:, Q0:Q1] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    return out


def reanchor(x: np.ndarray) -> np.ndarray:
    """Restore q_rel[0] = identity, rotating the vector channels to match.

    Applied once at the end of the pipeline, so that whatever the individual
    transforms did, the result satisfies the same invariant the real signal chain
    produces.  Skipping this would let the network see augmented tensors whose
    first sample is not the identity and real tensors whose first sample always
    is -- a difference it could learn instead of the gesture.

        q'   = conj(q[0]) * q
        a'   = rotate(conj(q[0]), a)
    """
    out = x.copy()
    q = out[:, Q0:Q1]
    q0 = q[0] / (np.linalg.norm(q[0]) + 1e-12)
    q0c = _qconj(q0)
    n = x.shape[0]
    q0cb = np.broadcast_to(q0c, (n, 4))
    out[:, Q0:Q1] = _qmul(q0cb, q)
    out[:, A0:A1] = _qrot(q0cb, out[:, A0:A1])
    out[:, W0:W1] = _qrot(q0cb, out[:, W0:W1])
    return out


PIPELINE = (grip_rotation, nonuniform_time_warp, magnitude_scale, time_shift, sensor_noise)


def augment_one(x: np.ndarray, rng: np.random.Generator, p: float = 0.75) -> np.ndarray:
    out = x
    for fn in PIPELINE:
        if rng.random() < p:
            out = fn(out, rng)
    out = reanchor(out)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def augment_dataset(X: np.ndarray, y: np.ndarray, factor: int = 8,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Expand a training split by `factor`.  The originals are kept as-is at
    index 0 of each block so the clean signal is always in the training set."""
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(factor - 1):
        Xs.append(np.stack([augment_one(x, rng) for x in X]))
        ys.append(y)
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)
