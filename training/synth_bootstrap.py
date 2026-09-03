"""training/synth_bootstrap.py

QUARANTINED physics-based gesture generator.

================================ REVISION R4 ==================================
The generator now DERIVES the accelerometer signal from the wand's measured
geometry instead of inventing it.

WHY THAT CHANGE MATTERS MORE THAN IT SOUNDS
    Up to R3 each spell was two independently hand-tuned programmes: an angular
    rate curve and, separately, an acceleration curve.  Nothing tied them
    together.  That is not a small liberty -- it is the wrong physics, and it is
    wrong in the direction that flatters the model.  A BNO055 bolted 37 cm from
    the caster's wrist is not measuring the caster's hand.  It is measuring the
    hand PLUS a lever arm, and for every gesture in this vocabulary the lever
    term is the larger of the two by a factor of three to ten.  A generator that
    treats acceleration as a free parameter hands the network a channel that
    carries independent information, when on the real wand that channel is very
    nearly a deterministic function of the gyro channel.  A model trained on the
    former learns a decision rule that does not exist in the latter.

    So: the caster's motion is specified as the two things a human actually
    commands -- how the wand ROTATES and where the HAND goes -- and everything
    the accelerometer sees follows from rigid-body kinematics.

THE GEOMETRY (measured on the built wand; see shared/wand_config.h)
    Total length 50 cm.  BNO055 at 13 cm below the tip, so 37 cm up from the
    butt.  Sensor +X runs along the tube towards the tip.  The tube is gripped
    over the electronics at the butt end, and the wand rotates about a joint
    BEHIND that grip -- wrist, elbow or shoulder depending on the gesture.

THE RELATION
    For a point rigidly attached to a rotating body,

        a_sensor = a_pivot  +  alpha x r  +  omega x (omega x r)
                   \______/    \_______/     \______________/
                    hand         Euler         centripetal
                   travels      (angular      (always inward,
                                accel)         always present)

    all three terms expressed in the sensor's own frame, with r the fixed vector
    from the pivot to the sensor.  The BNO055's linear-acceleration registers
    report exactly this quantity -- the fusion has already removed gravity -- so
    the generator can emit it directly.

    With r = 0.45 m (wrist pivot) the centripetal term alone is 0.45*omega^2:
    at the 6 rad/s a brisk flick reaches, that is 16 m/s^2, or 1.7 g, from
    rotation alone with the hand held perfectly still.  This is why the wand
    saturates its +/-4 g fusion-locked range on sharp casts, and it is why the
    impact test had to be redesigned around spike WIDTH.  None of that is
    visible to a generator that picks acceleration amplitudes by hand.

MOTION PRIMITIVES
    Both commanded quantities use the minimum-jerk profile (Flash & Hogan 1985),
    which is the standard model of human point-to-point limb movement:

        s(u) = 10u^3 - 15u^4 + 6u^5,    u = (t - t0) / duration

    Its first derivative is zero at both ends (the limb starts and stops at
    rest) and so is its SECOND, which matters here: a profile with non-zero
    endpoint acceleration would inject a step into alpha and therefore a
    discontinuity into the accelerometer channel that no real arm produces and
    that the 100 Hz sampling would alias.  Rotations are specified as an
    EXCURSION IN RADIANS over a window, which integrates to exactly that angle;
    hand motions as a DISPLACEMENT IN METRES.  Both are quantities that can be
    checked against a video of someone casting, which hand-picked amplitudes in
    counts cannot.

WHAT IT IS NOT
    It is not a dataset.  Numbers produced from it are meaningless as accuracy
    figures.  An earlier iteration of this project reported 100.0% leave-one-
    caster-out accuracy with 0.00% standard deviation from exactly this kind of
    generator; that number was fiction and nearly reached a report.  Better
    physics makes the corpus a better ENGINEERING TARGET -- the tensor ranges,
    the clipping rate, the gate pass rates and the arena size are now the ones
    the real wand will produce -- and does not make it evidence.

THE QUARANTINE (three independent locks, unchanged)
    1.  Everything is written under data/synthetic/, never data/raw/.
    2.  Every file gets a `# SYNTHETIC` first line and every row carries a
        synthetic=1 column, so a stray file cannot be mistaken for a capture.
    3.  train_cnn.py / calibrate.py refuse to report metrics unless
        --allow-synthetic is passed, and when it is passed every printed number
        and every figure title is prefixed "SYNTHETIC -- NOT A RESULT".

BNO055 BEHAVIOURS REPRODUCED ON PURPOSE
    Each has a code path downstream that must survive it:
      * +/-4 g clipping, applied to the RAW signal (linear + gravity) before
        gravity is subtracted, which is where the chip applies it;
      * quaternion sign flips -- q and -q are the same rotation and the chip
        hands back either, exercising the hemisphere-continuity pass;
      * all-zero quaternion frames, as an I2C hiccup produces;
      * duplicate frames, from the beat between the chip's 100 Hz fusion clock
        and the ESP32's 100 Hz sampling clock.

    python training/synth_bootstrap.py            # regenerate the corpus
    python training/audit_corpus.py data/synthetic  # check it against physics
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import wand_config as C

RNG_GLOBAL = np.random.default_rng(20260901)

# Body-frame axis indices, so no programme below indexes a bare 0/1/2.
#
#   AX_ROLL   about +X, the tube's own axis   -- spinning the wand
#   AX_PITCH  about +Y                        -- POSITIVE takes the tip DOWN
#   AX_YAW    about +Z                        -- POSITIVE swings the tip to the
#                                                caster's left
# The pitch sign is the one that catches people: +Y points to the caster's left
# and +Z is up through the wand's back, so by the right-hand rule a positive
# rotation about +Y carries +X (the tip) towards -Z (down).  Every "tip up" move
# below therefore carries a NEGATIVE pitch excursion, and it is written that way
# rather than with a flipped axis so the frame stays the one the firmware uses.
AX_ROLL, AX_PITCH, AX_YAW = 0, 1, 2


# =============================================================================
# motion primitives
# =============================================================================
def minjerk_s(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk position profile, 0 -> 1 over u in [0, 1] (Flash & Hogan).

    Clamped outside the interval: before the move the limb is at the start, and
    after it, at the end.  s'(0) = s'(1) = 0 and s''(0) = s''(1) = 0, so both
    the velocity and the acceleration go to zero at each end -- which is what
    keeps the derived accelerometer trace free of the step discontinuities a
    raised-cosine or a trapezoid would leave.
    """
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)


def minjerk_rate(u: np.ndarray) -> np.ndarray:
    """d/du of minjerk_s: 30 u^2 (1-u)^2, zero outside [0, 1].

    Unit area over the interval, so multiplying by (excursion / duration) gives
    a rate profile that integrates to EXACTLY the requested excursion.  Peak
    value is 1.875, at u = 0.5.
    """
    inside = (u >= 0.0) & (u <= 1.0)
    uc = np.clip(u, 0.0, 1.0)
    return np.where(inside, 30.0 * uc ** 2 * (1.0 - uc) ** 2, 0.0)


def d2dt2(x: np.ndarray, dt: float) -> np.ndarray:
    """Second time derivative of an (N, 3) path by the three-point stencil.

    Edges are replicated rather than one-sided-differenced.  Every path this is
    applied to is at rest at both ends by construction, so the replicated value
    is the correct one (zero) and a one-sided formula would only add noise.
    """
    a = np.zeros_like(x)
    a[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt * dt)
    if len(x) > 2:
        a[0], a[-1] = a[1], a[-2]
    return a


def ddt(x: np.ndarray, dt: float) -> np.ndarray:
    """First time derivative by central differences, edges replicated."""
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    if len(x) > 2:
        d[0], d[-1] = d[1], d[-2]
    return d


class Move:
    """Accumulates one gesture as commanded rotation + commanded hand path.

    The caller writes what a person does -- "swing the tip up through 50 degrees
    over 280 ms", "carry the hand 30 cm forward" -- and never touches an
    acceleration.  Acceleration is not an input to this class; it is an output
    of synth_capture(), computed from these two curves and the geometry.
    """

    def __init__(self, t: np.ndarray, dt: float):
        self.t = t
        self.dt = dt
        self.w = np.zeros((len(t), 3))     # body angular rate, rad/s
        self.p = np.zeros((len(t), 3))     # hand path in the ANCHOR frame, m
        self.impulse = np.zeros((len(t), 3))   # collision term, m/s^2, anchor frame

    # -- rotation -------------------------------------------------------------
    def turn(self, axis: int, radians: float, t0: float, dur: float) -> "Move":
        """Rotate `radians` about a body axis over [t0, t0+dur], min-jerk."""
        u = (self.t - t0) / dur
        self.w[:, axis] += (radians / dur) * minjerk_rate(u)
        return self

    def cone(self, amp: float, turns: float, t0: float, dur: float,
             phase: float = 0.0, roll: float = 0.0) -> "Move":
        """Sustained conical sweep: pitch and yaw in quadrature, plus roll.

        The signature of Expelliarmus.  Enveloped by a min-jerk ramp in and out
        so the sweep starts and stops smoothly rather than switching on.
        """
        u = (self.t - t0) / dur
        env = np.where((u >= 0.0) & (u <= 1.0),
                       minjerk_s(np.clip(u * 2.0, 0, 1))
                       * minjerk_s(np.clip((1.0 - u) * 2.0, 0, 1)), 0.0)
        ph = 2.0 * np.pi * turns * np.clip(u, 0.0, 1.0) + phase
        self.w[:, AX_PITCH] += amp * env * np.cos(ph)
        self.w[:, AX_YAW] += amp * env * np.sin(ph)
        self.w[:, AX_ROLL] += roll * env
        return self

    def wobble(self, axis: int, amp: float, freq: float, phase: float) -> "Move":
        """Continuous low-amplitude oscillation -- fidgeting, gait, scratching."""
        self.w[:, axis] += amp * np.sin(2.0 * np.pi * freq * self.t + phase)
        return self

    # -- hand path ------------------------------------------------------------
    def reach(self, vec, t0: float, dur: float) -> "Move":
        """Carry the hand by `vec` metres over [t0, t0+dur] and hold it there."""
        u = (self.t - t0) / dur
        self.p += np.asarray(vec, float)[None, :] * minjerk_s(u)[:, None]
        return self

    def circle(self, radius: float, ax_a: int, ax_b: int, t0: float, dur: float,
               turns: float = 1.0, phase: float = 0.0) -> "Move":
        """Trace a circle of `radius` in the (ax_a, ax_b) plane of the anchor frame.

        This is how a person actually draws a loop in the air with a stick: the
        hand travels round a circle while the wand tilts through a much smaller
        angle.  Modelling a loop as a 2*pi rotation of the wand instead would
        demand ~20 rad/s of angular rate, which is not a gesture anyone performs
        twice in a demo.  The angular sweep is applied separately with turn().

        The phase ramps through min-jerk, so the hand accelerates into the loop
        and decelerates out of it instead of stepping onto the circle at speed.
        """
        u = (self.t - t0) / dur
        ph = 2.0 * np.pi * turns * minjerk_s(u) + phase
        seg = np.zeros((len(self.t), 3))
        seg[:, ax_a] = radius * (np.cos(ph) - np.cos(phase))
        seg[:, ax_b] = radius * (np.sin(ph) - np.sin(phase))
        self.p += seg
        return self

    def bob(self, axis: int, amp: float, freq: float, phase: float) -> "Move":
        """Continuous hand oscillation -- walking gait, arm sway."""
        self.p[:, axis] += amp * np.sin(2.0 * np.pi * freq * self.t + phase)
        return self

    # -- collisions -----------------------------------------------------------
    def impact(self, vec, t0: float, dur: float = 0.022) -> "Move":
        """A collision: a half-sine acceleration impulse `dur` seconds wide.

        Modelled as an impulse added directly to the pivot acceleration rather
        than as a hand path, because that is what it is -- a contact force, not
        a commanded movement.  The width is the point.  At 100 Hz a 22 ms
        contact puts two, at most three, samples above the spike level, whereas
        a Stupefy thrust holds the same level for six to ten.  That gap is the
        entire basis of the Stage 1 impact test, so the generator has to produce
        it honestly: if a real knock rings for longer than this, the test's
        REJ_SPIKE_MIN_RUN needs re-sweeping against real captures, not this file
        widening until the numbers look good.
        """
        u = (self.t - t0) / dur
        shape = np.where((u >= 0.0) & (u <= 1.0), np.sin(np.pi * np.clip(u, 0, 1)), 0.0)
        self.impulse += np.asarray(vec, float)[None, :] * shape[:, None]
        return self


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


# =============================================================================
# orientation integration
# =============================================================================
def rot_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate body angular rates into per-sample rotation matrices (N,3,3).

    Second-order exponential map per step, re-orthonormalised every 16 steps so
    numerical drift cannot make the gravity projection nonsensical over 2 s.
    R[i] maps BODY vectors to ANCHOR-frame vectors, with R[0] = I.
    """
    n = len(omega)
    R = np.zeros((n, 3, 3))
    cur = np.eye(3)
    for i in range(n):
        wx, wy, wz = omega[i] * dt
        K = np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])
        cur = cur @ (np.eye(3) + K + 0.5 * K @ K)
        if i % 16 == 15:
            u, _, vt = np.linalg.svd(cur)
            cur = u @ vt
        R[i] = cur
    return R


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices -> (N,4) unit quaternions, Shepperd's method.

    Branching on the largest diagonal term avoids the numerical blow-up the
    naive trace formula suffers near a 180-degree rotation, which a Lumos loop
    passes straight through.
    """
    n = len(R)
    q = np.zeros((n, 4))
    for i in range(n):
        m = R[i]
        t = m[0, 0] + m[1, 1] + m[2, 2]
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
                    (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                    (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                    0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        q[i] /= np.linalg.norm(q[i])
    return q


def euler_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Z-Y-X intrinsic rotation, used only for the caster's holding pose."""
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                              np.sin(ry), np.cos(rz), np.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# =============================================================================
# spell programmes
#
# Every number below is either an ANGLE IN RADIANS the wand turns through, a
# DISTANCE IN METRES the hand travels, or a FRACTION OF THE GESTURE DURATION.
# There is not a single acceleration constant in this section, and there must
# never be one: the moment somebody adds one, the accelerometer channel stops
# being a consequence of the gesture and goes back to being a free parameter.
#
# Each returns (Move, pivot_distance_m).  The pivot distance is measured
# BACKWARDS from the butt of the wand along its axis, so the lever arm to the
# sensor is pivot + SENSOR_FROM_BUTT_M.
# =============================================================================
def _lumos(t, T, g, m: Move):
    """Upward loop.  The hand carries the wand round a vertical circle while the
    tip pitches steadily up through about 125 degrees.  Elbow-driven, so the
    lever arm is long and the centripetal term is the dominant one -- Lumos and
    Alohomora are the pair the problem statement warns about, and what separates
    them here is the SIGN of the loop and the absence of a terminal spike."""
    m.turn(AX_PITCH, -g(1.05), 0.10 * T, 0.48 * T)
    m.turn(AX_PITCH, -g(1.15), 0.50 * T, 0.44 * T)
    m.turn(AX_ROLL, g(0.30), 0.20 * T, 0.60 * T)
    m.circle(g(0.115), 0, 2, 0.10 * T, 0.82 * T, turns=1.0, phase=-np.pi / 2)
    return m, C.PIVOT_ELBOW_M


def _alohomora(t, T, g, m: Move):
    """Backward loop, then a sharp flick.  The loop runs opposite to Lumos in
    both the hand path and the pitch sign; the flick is a short high-rate yaw
    burst in the last fifth.  Because the flick is wrist-driven at the end of an
    elbow-driven loop the effective pivot sits between the two."""
    m.turn(AX_PITCH, g(1.85), 0.08 * T, 0.60 * T)
    m.turn(AX_ROLL, -g(0.32), 0.10 * T, 0.55 * T)
    m.turn(AX_YAW, g(1.00), 0.74 * T, 0.22 * T)          # the flick
    m.circle(g(0.100), 0, 2, 0.08 * T, 0.64 * T, turns=-1.0, phase=np.pi / 2)
    m.reach([0.0, g(0.06), 0.0], 0.74 * T, 0.22 * T)
    return m, 0.5 * (C.PIVOT_WRIST_M + C.PIVOT_ELBOW_M)


def _expelliarmus(t, T, g, m: Move):
    """Spiral swirl.  Sustained simultaneous pitch and yaw in quadrature with a
    steady roll -- the tip traces a cone.  Longest of the five and the only one
    whose energy is spread across the whole window rather than concentrated in a
    flick, which is what the path_rad arm of the Stage 1 gate keys on."""
    m.cone(amp=g(3.60), turns=1.6, t0=0.05 * T, dur=0.90 * T, roll=g(0.9))
    m.circle(g(0.065), 1, 2, 0.05 * T, 0.90 * T, turns=1.6)
    return m, C.PIVOT_ELBOW_M


def _wingardium(t, T, g, m: Move):
    """Swish and flick.  A broad lateral yaw sweep driven from the elbow, then a
    crisp upward pitch from the wrist.  The two sub-moves are ordered, and that
    ordering is the reason the classifier flattens the temporal axis instead of
    global-average-pooling it."""
    m.turn(AX_YAW, g(1.55), 0.05 * T, 0.50 * T)          # swish
    m.reach([0.0, g(0.20), 0.0], 0.05 * T, 0.50 * T)
    m.turn(AX_PITCH, -g(0.95), 0.60 * T, 0.30 * T)       # flick
    m.reach([0.0, 0.0, g(0.06)], 0.60 * T, 0.30 * T)
    return m, 0.35 * C.PIVOT_WRIST_M + 0.65 * C.PIVOT_ELBOW_M


def _stupefy(t, T, g, m: Move):
    """Ballistic forward thrust: the hand drives 40 cm along the wand's axis and
    is pulled back, with almost no rotation.  Its near-zero gyro RMS against
    every other class is the fastest sanity check that labels did not get
    shuffled, and it is the class that forced the Stage 1 energy test to be a
    disjunction -- a single rotation floor rejects Stupefy by construction."""
    m.reach([g(0.40), 0.0, 0.0], 0.05 * T, 0.42 * T)
    m.reach([-g(0.40), 0.0, 0.0], 0.52 * T, 0.44 * T)
    m.turn(AX_PITCH, -g(0.30), 0.10 * T, 0.40 * T)
    return m, C.PIVOT_WRIST_M + 0.04


SPELLS = {
    C.WandGestureClass.LUMOS:              (_lumos,        (0.80, 1.40)),
    C.WandGestureClass.ALOHOMORA:          (_alohomora,    (0.85, 1.45)),
    C.WandGestureClass.EXPELLIARMUS:       (_expelliarmus, (1.05, 1.65)),
    C.WandGestureClass.WINGARDIUM_LEVIOSA: (_wingardium,   (0.75, 1.30)),
    C.WandGestureClass.STUPEFY:            (_stupefy,      (0.55, 0.90)),
}


# =============================================================================
# the NOISE class -- six sub-behaviours, matching the problem statement's own list
#
# These go through the identical physics.  That is deliberate and it is the part
# most likely to be got wrong: if the noise class were generated by a simpler
# path than the spells, the network could separate the two on an artefact of the
# generator rather than on the motion, and would then fail on the first real
# false positive it met.
# =============================================================================
def _noise_walk(t, T, rng, m: Move):
    """Walking with the wand in hand: 2 Hz gait bob, 1 Hz arm sway, shoulder pivot."""
    ph = rng.uniform(0, 6.3, 4)
    m.bob(2, 0.030, 1.9, ph[0]).bob(1, 0.045, 0.95, ph[1]).bob(0, 0.020, 0.95, ph[2])
    m.wobble(AX_PITCH, 0.55, 0.95, ph[1]).wobble(AX_YAW, 0.40, 0.95, ph[3])
    m.wobble(AX_ROLL, 0.25, 1.9, ph[0])
    return m, C.PIVOT_SHOULDER_M


def _noise_scratch(t, T, rng, m: Move):
    """Scratching an itch: small, fast, wrist-only, almost no travel."""
    f = rng.uniform(3.6, 6.2)
    ph = rng.uniform(0, 6.3, 3)
    m.wobble(AX_ROLL, 1.35, f, ph[0]).wobble(AX_YAW, 0.85, f, ph[1] + 1.1)
    m.wobble(AX_PITCH, 0.50, f * 0.5, ph[2])
    m.bob(1, 0.012, f, ph[0]).bob(2, 0.008, f, ph[1])
    return m, C.PIVOT_WRIST_M


def _noise_gesticulate(t, T, rng, m: Move):
    """Talking with your hands: several broad, slow, unstructured moves.

    The hardest false positive to reject, because it genuinely overlaps a spell
    in energy.  Nothing here is shaped -- excursions, timings and directions are
    all drawn independently -- which is precisely the property that separates it
    from a spell and the one the network has to learn."""
    for _ in range(rng.integers(3, 6)):
        ax = int(rng.integers(3))
        m.turn(ax, rng.uniform(-1.15, 1.15), rng.uniform(0, 0.75) * T,
               rng.uniform(0.22, 0.55) * T)
    for _ in range(rng.integers(2, 4)):
        v = rng.normal(0, 0.075, 3)
        m.reach(v, rng.uniform(0, 0.7) * T, rng.uniform(0.25, 0.6) * T)
    return m, C.PIVOT_ELBOW_M


def _noise_setdown(t, T, rng, m: Move):
    """Placing the wand on a table: a slow tip-down tilt, a 30 cm descent, then
    contact.  The contact is the impulse the spike test exists to catch."""
    m.turn(AX_PITCH, 0.85, 0.05 * T, 0.65 * T)
    m.reach([0.0, 0.0, -0.30], 0.05 * T, 0.70 * T)
    m.impact([0.0, 0.0, rng.uniform(45.0, 95.0)], 0.76 * T, rng.uniform(0.018, 0.028))
    return m, C.PIVOT_ELBOW_M


def _noise_knock(t, T, rng, m: Move):
    """Rapping the wand against a table or a chair, once or twice.  The purest
    test of the impact rule: large peak acceleration, essentially zero width."""
    n_hits = int(rng.integers(1, 3))
    for k in range(n_hits):
        t0 = (0.35 + 0.28 * k) * T
        m.turn(AX_PITCH, 0.45, t0 - 0.16 * T, 0.16 * T)
        m.reach([0.0, 0.0, -0.10], t0 - 0.16 * T, 0.16 * T)
        v = rng.normal(0, 25.0, 3)
        v[2] = rng.uniform(60.0, 130.0)
        m.impact(v, t0, rng.uniform(0.012, 0.024))
        m.reach([0.0, 0.0, 0.10], t0 + 0.02, 0.14 * T)
    return m, C.PIVOT_WRIST_M


def _noise_still(t, T, rng, m: Move):
    """The trigger pressed with nothing behind it -- a fumbled button, a caster
    steadying themselves before starting, the wand held out while someone talks.

    Physiological hand tremor only: 8-12 Hz at a fraction of a degree, plus a
    slow postural drift.  This is the sub-behaviour the Stage 1 ENERGY floor
    exists for, and until it was in the corpus that floor had no job to do --
    every other noise kind is energetic enough to clear any floor low enough to
    let a gentle cast through, which is why the floor should stay low and leave
    the discriminating to the impact test and the trained NOISE class."""
    ph = rng.uniform(0, 6.3, 3)
    f = rng.uniform(8.0, 11.0)
    for ax in range(3):
        m.wobble(ax, rng.uniform(0.03, 0.10), f * rng.uniform(0.9, 1.1), ph[ax])
        m.turn(ax, rng.normal(0, 0.06), rng.uniform(0, 0.6) * T, 0.4 * T)
    m.reach(rng.normal(0, 0.006, 3), 0.1 * T, 0.7 * T)
    return m, C.PIVOT_ELBOW_M


def _noise_drop(t, T, rng, m: Move):
    """Dropping the wand.

    NOT a free-fall detector test -- there is no free-fall signature to detect
    in fusion mode.  The BNO's gravity estimate coasts through the fall, so its
    linear-acceleration registers report about 1 g downward for the whole
    descent rather than the ~0 g a bare accelerometer would show.  That is
    modelled here literally: the pivot acceleration during the fall is -g in the
    anchor frame's vertical, and the trailing impact is a narrow impulse."""
    t_rel, t_hit = 0.22 * T, 0.62 * T
    fall = (t >= t_rel) & (t < t_hit)
    m.impulse[fall, 2] += -C.G_MS2
    m.wobble(AX_ROLL, 2.2, 0.7, rng.uniform(0, 6.3))
    m.wobble(AX_PITCH, 1.5, 0.5, rng.uniform(0, 6.3))
    v = rng.normal(0, 30.0, 3)
    v[2] = rng.uniform(80.0, 160.0)
    m.impact(v, t_hit, rng.uniform(0.010, 0.020))
    return m, C.PIVOT_SHOULDER_M


NOISE_KINDS = [_noise_walk, _noise_scratch, _noise_gesticulate,
               _noise_setdown, _noise_knock, _noise_still, _noise_drop]
NOISE_WEIGHTS = np.array([0.18, 0.14, 0.25, 0.12, 0.12, 0.12, 0.07])


# =============================================================================
# generation
# =============================================================================
def synth_capture(label: int, caster_seed: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Return an (N, 10) int16 array in exactly the layout the logger emits:
    qw qx qy qz | lax lay laz | gx gy gz."""
    cr = np.random.default_rng(caster_seed)

    # Per-caster style.  These are the axes along which real casters differ, and
    # the leave-one-caster-out split is only meaningful because they exist.
    gain = cr.uniform(0.82, 1.22)        # how vigorously this person casts
    dur_bias = cr.uniform(0.86, 1.16)    # how fast
    hold = cr.uniform(-0.40, 0.40, 3)    # the pose the wand is held in
    pivot_scale = cr.uniform(0.80, 1.25)  # arm length / wrist-vs-elbow habit
    gyro_bias = cr.normal(0.0, 0.020, 3)  # residual fusion bias, rad/s

    def g(x):
        """Per-rep excursion scaling: caster style times within-caster variation."""
        return x * gain * rng.uniform(0.90, 1.10)

    pre, post = C.PREROLL_SAMPLES, C.POSTROLL_SAMPLES

    if label == int(C.WandGestureClass.NOISE):
        kind = NOISE_KINDS[rng.choice(len(NOISE_KINDS), p=NOISE_WEIGHTS)]
        T = float(rng.uniform(0.65, 1.95))
        n_g = int(round(T * C.FS_HZ))
        t = np.arange(n_g) * C.DT_S
        mv, pivot = kind(t, T, rng, Move(t, C.DT_S))
        mv.w *= gain
    else:
        fn, (lo, hi) = SPELLS[C.WandGestureClass(label)]
        T = float(np.clip(rng.uniform(lo, hi) * dur_bias, 0.45, 1.95))
        n_g = int(round(T * C.FS_HZ))
        t = np.arange(n_g) * C.DT_S
        mv, pivot = fn(t, T, g, Move(t, C.DT_S))

    # --- pre-roll and post-roll ---------------------------------------------
    # The rate pads with zeros (the caster is still), but the hand PATH pads by
    # HOLDING its endpoint value -- padding a position with zeros would inject a
    # step, and the second derivative of a step is an impulse that would appear
    # as a phantom impact at the seam of every single capture.
    w = np.vstack([np.zeros((pre, 3)), mv.w, np.zeros((post, 3))])
    p = np.vstack([np.repeat(mv.p[:1], pre, axis=0), mv.p,
                   np.repeat(mv.p[-1:], post, axis=0)])
    imp = np.vstack([np.zeros((pre, 3)), mv.impulse, np.zeros((post, 3))])
    n = len(w)

    # --- rigid-body kinematics ----------------------------------------------
    # r is the fixed vector from the pivot to the sensor, in the BODY frame:
    # along the tube by (pivot + 37 cm), plus the 12 mm the PCB stands off the
    # tube's centreline.  The radial offset is small but it is what makes a pure
    # roll about the wand's own axis visible at all -- with r exactly on the
    # axis, roll would produce no acceleration whatsoever.
    lever = pivot * pivot_scale + C.SENSOR_FROM_BUTT_M
    r = np.array([lever, 0.0, C.SENSOR_RADIAL_OFF_M])

    alpha = ddt(w, C.DT_S)                                   # rad/s^2, body frame
    a_lever = cross(alpha, r) + cross(w, cross(w, r))        # m/s^2, body frame

    R = rot_from_omega(w, C.DT_S)                            # body -> anchor
    Rhold = euler_matrix(*hold)                              # anchor -> world
    R = np.einsum('ij,njk->nik', Rhold, R)                   # body -> world

    a_pivot_anchor = d2dt2(p, C.DT_S) + imp                  # m/s^2, anchor frame
    a_pivot_world = np.einsum('ij,nj->ni', Rhold, a_pivot_anchor)

    # a_sensor = a_pivot + alpha x r + omega x (omega x r), in the body frame.
    lin_ms2 = np.einsum('nji,nj->ni', R, a_pivot_world) + a_lever
    lin_body = lin_ms2 / C.G_MS2                             # -> g

    # --- gravity and the +/-4 g clip ----------------------------------------
    # Gravity is +1 g "up" in the world frame.  The raw accelerometer measures
    # imparted acceleration PLUS the gravity reaction in its own frame; the
    # fusion subtracts its gravity estimate and reports the remainder as LIA.
    # The clipping happens on the RAW signal, before that subtraction, and
    # reproducing it in the right place matters: clipping the linear channel
    # directly would be wrong by up to 1 g and would understate how often a hard
    # cast actually saturates.
    g_world = np.array([0.0, 0.0, 1.0])
    g_body = np.einsum('nji,j->ni', R, g_world)
    raw_body = np.clip(lin_body + g_body,
                       -C.BNO_FUSION_ACC_RANGE_G, C.BNO_FUSION_ACC_RANGE_G)
    lin_body = raw_body - g_body

    # --- sensor imperfection -------------------------------------------------
    # Smaller than a bare MPU6050's, because the fusion has already suppressed
    # most of the white noise; what is left is mostly its own estimation error.
    lin_body += rng.normal(0.0, 0.008, lin_body.shape)
    w = w + gyro_bias * 0.25 + rng.normal(0.0, 0.004, w.shape)

    quat = quat_from_matrix(R)
    dq = rng.normal(0.0, 0.0015, (n, 3))
    quat[:, 1:4] += dq
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # --- BNO055 behaviours the downstream code must survive ------------------
    quat[rng.random(n) < 0.04] *= -1.0        # q and -q are the same rotation
    quat[rng.random(n) < 0.004] = 0.0         # I2C hiccup -> all-zero frame

    raw = np.empty((n, 10), dtype=np.int16)
    raw[:, 0:4] = np.clip(quat * C.BNO_LSB_PER_QUAT, -32768, 32767).astype(np.int16)
    raw[:, 4:7] = np.clip(lin_body * C.LINACC_LSB_PER_G, -32768, 32767).astype(np.int16)
    raw[:, 7:10] = np.clip((w / C.DEG2RAD) * C.BNO_LSB_PER_DPS,
                           -32768, 32767).astype(np.int16)

    for i in range(1, n):                     # fusion-clock beat -> duplicates
        if rng.random() < 0.02:
            raw[i] = raw[i - 1]
    return raw


def write_csv(path: Path, raw: np.ndarray, label: int, caster: str, rep: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# SYNTHETIC -- generated by synth_bootstrap.py -- NOT A REAL CAPTURE\n")
        f.write(f"# GESTURE_START label={label} caster={caster} rep={rep} "
                f"nsamples={len(raw)} synthetic=1\n")
        f.write("idx,qw,qx,qy,qz,lax,lay,laz,gx,gy,gz,dt_us,dup,synthetic\n")
        prev = None
        for i, row in enumerate(raw):
            dup = 1 if (prev is not None and np.array_equal(row, prev)) else 0
            prev = row
            f.write(f"{i}," + ",".join(str(int(v)) for v in row) +
                    f",{C.DT_US},{dup},1\n")
        f.write("# GESTURE_END\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casters", type=int, default=6)
    ap.add_argument("--reps", type=int, default=18, help="reps per spell per caster")
    ap.add_argument("--noise-reps", type=int, default=45, help="NOISE reps per caster")
    ap.add_argument("--out", default=C.SYNTHETIC_DATA_DIR)
    args = ap.parse_args()

    out = Path(args.out)
    # LOCK 1: refuse to write anywhere near the real dataset.
    if "raw" in out.parts or os.path.abspath(out).endswith(os.path.normpath(C.REAL_DATA_DIR)):
        raise RuntimeError(f"refusing to write synthetic data into {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "DO_NOT_TRAIN_FOR_REPORT.txt").write_text(
        "Every file under this directory is machine-generated.\n"
        "It exists to prove the pipeline runs. No number derived from it may\n"
        "appear in the technical report, the PPT, or any submitted result.\n")

    total = 0
    for c in range(args.casters):
        caster = f"synth{c:02d}"
        for label in range(C.NUM_CLASSES):
            reps = args.noise_reps if label == int(C.WandGestureClass.NOISE) else args.reps
            for rep in range(reps):
                rng = np.random.default_rng(RNG_GLOBAL.integers(2**31))
                raw = synth_capture(label, caster_seed=1000 + c, rng=rng)
                name = C.GESTURE_NAMES[label].replace(" ", "_")
                write_csv(out / caster / name / f"{caster}_{name}_{rep}.csv",
                          raw, label, caster, rep)
                total += 1
    print(f"wrote {total} SYNTHETIC captures to {out}/  "
          f"({args.casters} casters x {C.NUM_SPELL_CLASSES} spells x {args.reps} "
          f"+ {args.noise_reps} noise)")
    print(f"geometry: wand {C.WAND_LENGTH_M * 100:.0f} cm, sensor "
          f"{C.SENSOR_FROM_TIP_M * 100:.0f} cm below the tip "
          f"({C.SENSOR_FROM_BUTT_M * 100:.0f} cm from the butt), +X towards the tip")
    print("now run:  python training/audit_corpus.py data/synthetic")


if __name__ == "__main__":
    main()
