// firmware/wand_infer/preprocess.cpp
// -----------------------------------------------------------------------------
// LINE-FOR-LINE PORT of training/preprocess.py.  These two files are a matched
// pair: change one without the other and training/parity_check.py fails, which
// is the entire reason that test exists.
//
// The failure this guards against has no visible symptom.  If the Python chain
// and this chain disagree by even a scale factor, the model trains happily to
// 92% in Colab, the firmware compiles, the device runs at full speed -- and
// classifies at chance, because it is being fed a distribution the network has
// never seen.  There is no error message.  You find it on stage.
//
// R3: the chain got SHORTER.  Gyro bias removal, the complementary filter and
// gravity projection are gone -- the BNO055's own Cortex-M0 does all three, and
// does them better.  What is left is a change of FRAME, which is cheap: no
// transcendental functions at all on the hot path except the four sqrtf calls
// for the magnitudes.  That is roughly 2.5 ms off the latency budget, and it also
// deletes the exact step (complementary-filter initialisation) that the parity
// negative control proved was the easiest thing in the project to get wrong.
//
// Rules that keep the two in step:
//   * every constant comes from wand_config.h, none is retyped here
//   * Python uses float64 and this uses float32.  The 1e-3 parity tolerance is
//     set wide enough to absorb that and tight enough to catch any real
//     algorithmic divergence.
//   * the loops are written in the same order as the Python, deliberately, even
//     where a different order would be marginally faster on Xtensa.
// -----------------------------------------------------------------------------
#include "preprocess.h"

#include <math.h>

// Scratch buffers. Static, not stack: the FreeRTOS task stack is 8 KB and these
// are ~20 KB. Static also means no allocation on the hot path, so the latency
// figure quoted in the report has no malloc variance in it.
static constexpr int MAXN = MAX_GESTURE_SAMPLES + PREROLL_SAMPLES + POSTROLL_SAMPLES;  // 230

static float s_quat[MAXN][4];    // sanitised, hemisphere-continuous
static float s_acc[MAXN][3];     // g, gravity already removed by the sensor
static float s_gyro[MAXN][3];    // rad/s, bias already removed by the fusion
static float s_chan[MAXN][N_CHANNELS];

// ---------------------------------------------------------------------------
// quaternion helpers -- twins of training/preprocess.py
// ---------------------------------------------------------------------------
static inline void qMul(const float* a, const float* b, float* out) {
  const float aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const float bw = b[0], bx = b[1], by = b[2], bz = b[3];
  out[0] = aw * bw - ax * bx - ay * by - az * bz;
  out[1] = aw * bx + ax * bw + ay * bz - az * by;
  out[2] = aw * by - ax * bz + ay * bw + az * bx;
  out[3] = aw * bz + ax * by - ay * bx + az * bw;
}

// Rotate v by q:  t = 2 * (q_vec x v);  v' = v + q_w * t + q_vec x t
// Fifteen multiplies, no branches, no matrix. Same closed form as the Python.
static inline void qRot(const float* q, const float* v, float* out) {
  const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  const float tx = 2.0f * (qy * v[2] - qz * v[1]);
  const float ty = 2.0f * (qz * v[0] - qx * v[2]);
  const float tz = 2.0f * (qx * v[1] - qy * v[0]);
  out[0] = v[0] + qw * tx + (qy * tz - qz * ty);
  out[1] = v[1] + qw * ty + (qz * tx - qx * tz);
  out[2] = v[2] + qw * tz + (qx * ty - qy * tx);
}

bool wandPreprocess(const RawSample* raw, int n, float* out, KinFeatures* kin) {
  if (n < MIN_GESTURE_SAMPLES || n > MAXN) return false;

  // --- Step 1: raw counts -> physical units --------------------------------
  // The divisors are fixed by the BNO055 datasheet and by the UNIT_SEL register
  // the firmware writes at boot. Unlike the MPU6050 there is no range register
  // that someone can change without updating a constant -- in fusion mode the
  // algorithm owns the range, which removes a whole class of silent error.
  for (int i = 0; i < n; ++i) {
    s_quat[i][0] = (float)raw[i].qw / BNO_LSB_PER_QUAT;
    s_quat[i][1] = (float)raw[i].qx / BNO_LSB_PER_QUAT;
    s_quat[i][2] = (float)raw[i].qy / BNO_LSB_PER_QUAT;
    s_quat[i][3] = (float)raw[i].qz / BNO_LSB_PER_QUAT;

    s_acc[i][0] = (float)raw[i].lax / LINACC_LSB_PER_G;
    s_acc[i][1] = (float)raw[i].lay / LINACC_LSB_PER_G;
    s_acc[i][2] = (float)raw[i].laz / LINACC_LSB_PER_G;

    s_gyro[i][0] = ((float)raw[i].gx / BNO_LSB_PER_DPS) * DEG2RAD;
    s_gyro[i][1] = ((float)raw[i].gy / BNO_LSB_PER_DPS) * DEG2RAD;
    s_gyro[i][2] = ((float)raw[i].gz / BNO_LSB_PER_DPS) * DEG2RAD;
  }

  // --- Step 2: sanitise quaternions ----------------------------------------
  // Three hazards, all of which otherwise produce a valid-LOOKING tensor:
  //   1. an all-zero frame (I2C hiccup, or the chip still in CONFIG mode) would
  //      be a divide-by-zero on normalisation -> substitute the identity, which
  //      contributes no rotation and is the safest possible fallback;
  //   2. q and -q are the same rotation and the chip hands back either, so
  //      without a continuity pass channels 6..9 contain sign flips that look
  //      like instantaneous 360-degree rotations;
  //   3. continuity is fixed HERE, before anchoring, because quaternion
  //      multiplication is bilinear -- a sign fixed now stays fixed in q_rel.
  for (int i = 0; i < n; ++i) {
    float nn = sqrtf(s_quat[i][0] * s_quat[i][0] + s_quat[i][1] * s_quat[i][1] +
                     s_quat[i][2] * s_quat[i][2] + s_quat[i][3] * s_quat[i][3]);
    if (nn < QUAT_MIN_NORM) {
      s_quat[i][0] = 1.0f; s_quat[i][1] = 0.0f;
      s_quat[i][2] = 0.0f; s_quat[i][3] = 0.0f;
      nn = 1.0f;
    }
    const float inv = 1.0f / nn;
    for (int c = 0; c < 4; ++c) s_quat[i][c] *= inv;
  }
  for (int i = 1; i < n; ++i) {
    float d = 0.0f;
    for (int c = 0; c < 4; ++c) d += s_quat[i][c] * s_quat[i - 1][c];
    if (d < 0.0f) for (int c = 0; c < 4; ++c) s_quat[i][c] = -s_quat[i][c];
  }

  // --- Step 3: anchor to gesture start, and rotate into that frame ---------
  // q_rel[i] = conj(q[0]) * q[i].  q[0] is the first PRE-ROLL sample, 200 ms
  // before the trigger, when the caster is still holding position -- the best
  // available description of "how this person was holding the wand before they
  // started", which is exactly the reference we want to remove.
  //
  // Rotating the vectors by q_rel is what decouples translation from rotation.
  // In the old body-frame representation a constant world-frame push appeared as
  // a time-varying signal, because the frame itself was spinning with the wand.
  {
    const float q0c[4] = {s_quat[0][0], -s_quat[0][1], -s_quat[0][2], -s_quat[0][3]};
    for (int i = 0; i < n; ++i) {
      float qr[4];
      qMul(q0c, s_quat[i], qr);

      float a[3], w[3];
      qRot(qr, s_acc[i], a);
      qRot(qr, s_gyro[i], w);

      // magnitudes are rotation-invariant, so they are taken before/after
      // interchangeably; the Python takes them from the unrotated vectors and so
      // does this, to keep the two byte-identical in intent as well as in value.
      const float am = sqrtf(s_acc[i][0] * s_acc[i][0] + s_acc[i][1] * s_acc[i][1] +
                             s_acc[i][2] * s_acc[i][2]);
      const float wm = sqrtf(s_gyro[i][0] * s_gyro[i][0] + s_gyro[i][1] * s_gyro[i][1] +
                             s_gyro[i][2] * s_gyro[i][2]);

      s_chan[i][CH_ACC + 0] = a[0];
      s_chan[i][CH_ACC + 1] = a[1];
      s_chan[i][CH_ACC + 2] = a[2];
      s_chan[i][CH_GYRO + 0] = w[0];
      s_chan[i][CH_GYRO + 1] = w[1];
      s_chan[i][CH_GYRO + 2] = w[2];
      s_chan[i][CH_QUAT + 0] = qr[0];
      s_chan[i][CH_QUAT + 1] = qr[1];
      s_chan[i][CH_QUAT + 2] = qr[2];
      s_chan[i][CH_QUAT + 3] = qr[3];
      s_chan[i][CH_AMAG] = am;
      s_chan[i][CH_WMAG] = wm;
    }
  }

  // --- Stage 1 gate features, on the unresampled signal --------------------
  {
    float peak_w = 0.f, peak_a = 0.f, path = 0.f;
    int run = 0, best = 0;
    for (int i = 0; i < n; ++i) {
      const float wm = s_chan[i][CH_WMAG];
      const float am = s_chan[i][CH_AMAG];
      if (wm > peak_w) peak_w = wm;
      if (am > peak_a) peak_a = am;
      path += wm * DT_S;
      run = (am >= REJ_SPIKE_G) ? run + 1 : 0;
      if (run > best) best = run;
    }
    kin->duration_s = (float)n * DT_S;
    kin->peak_w     = peak_w;
    kin->peak_a     = peak_a;
    kin->path_rad   = path;
    kin->spike_run  = best;
  }

  // --- Step 4: resample to a fixed 64 steps, then normalise ----------------
  // Index-space linear interpolation, identical to preprocess.py::resample_fixed.
  // The quaternion channels are lerped and deliberately NOT renormalised: between
  // adjacent 100 Hz samples the rotation is under a degree, where lerp and slerp
  // differ by under 1e-5, and a renormalisation step is one more place for the
  // two chains to disagree for a correction smaller than the sensor noise.
  {
    const float step = (float)(n - 1) / (float)(N_RESAMPLE - 1);
    for (int k = 0; k < N_RESAMPLE; ++k) {
      const float pos = (float)k * step;
      int i0 = (int)floorf(pos);
      if (i0 > n - 1) i0 = n - 1;
      int i1 = (i0 + 1 < n) ? i0 + 1 : n - 1;
      const float f = pos - (float)i0;

      float v[N_CHANNELS];
      for (int c = 0; c < N_CHANNELS; ++c)
        v[c] = s_chan[i0][c] * (1.f - f) + s_chan[i1][c] * f;

      float* o = out + k * N_CHANNELS;
      for (int c = 0; c < 3; ++c) {
        float x = v[CH_ACC + c] / NORM_ACC_LIN;
        o[CH_ACC + c] = x < -1.f ? -1.f : (x > 1.f ? 1.f : x);
      }
      for (int c = 0; c < 3; ++c) {
        float x = v[CH_GYRO + c] / NORM_GYRO;
        o[CH_GYRO + c] = x < -1.f ? -1.f : (x > 1.f ? 1.f : x);
      }
      // A unit quaternion already lives in [-1, 1] on every component, so the
      // orientation channels need no divisor at all -- a small but real bonus of
      // carrying orientation as a quaternion rather than as Euler angles.
      for (int c = 0; c < 4; ++c) {
        float x = v[CH_QUAT + c];
        o[CH_QUAT + c] = x < -1.f ? -1.f : (x > 1.f ? 1.f : x);
      }
      float a10 = v[CH_AMAG] / NORM_AMAG;  o[CH_AMAG] = a10 < 0.f ? 0.f : (a10 > 1.f ? 1.f : a10);
      float a11 = v[CH_WMAG] / NORM_WMAG;  o[CH_WMAG] = a11 < 0.f ? 0.f : (a11 > 1.f ? 1.f : a11);
    }
  }
  return true;
}
