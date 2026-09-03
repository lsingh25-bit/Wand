// firmware/wand_infer/reject.cpp
// -----------------------------------------------------------------------------
// THE FALSE-POSITIVE CASCADE -- the "Muggle Movement" defence.
//
// The scoring here is asymmetric: a missed spell scores nothing, a misfire is
// penalised. That asymmetry is the whole design brief for this file. Every stage
// below can only ever turn a fire into a no-fire; none can invent a cast.
//
// Five stages, cheapest first, because there is no reason to spend 5 ms of
// inference deciding that someone scratched their nose:
//
//   0  TRIGGER      the caster deliberately held the button           (~0 us)
//   1  KINEMATICS   duration, energy disjunction, impact spike width   (~5 us)
//   2  CLASS        the network's own trained NOISE class            (~4.6 ms)
//   3  CONFIDENCE   calibrated top-1 floor and top-1/top-2 margin      (~2 us)
//   4  PERSISTENCE  agreement across consecutive windows          (auto mode)
//   5  REFRACTORY   no second cast within 500 ms                       (~1 us)
//
// Stage 2 is the one worth defending in the report. The alternative -- train on
// four spells and reject anything with low confidence -- fails badly, because a
// softmax over four classes must sum to one: shown a shrug, the network has no
// way to say "none of these" and is forced to distribute its belief over four
// spells, frequently landing on one of them with high confidence. Giving NOISE
// its own trained output gives the network somewhere to put that belief. It is
// the difference between a threshold hack and a classifier that actually knows
// what "not a spell" looks like.
// -----------------------------------------------------------------------------
#include "reject.h"

static uint32_t s_last_fire_ms = 0;
static int8_t   s_hist[REJ_PERSIST_WINDOWS] = {GESTURE_NONE, GESTURE_NONE, GESTURE_NONE};
static uint8_t  s_hist_i = 0;

const char* rejectReasonName(RejectReason r) {
  switch (r) {
    case REJ_NONE:            return "OK";
    case REJ_TOO_SHORT:       return "TOO SHORT";
    case REJ_TOO_LONG:        return "TOO LONG";
    case REJ_TOO_STILL:       return "TOO STILL";
    case REJ_TOO_SHALLOW:     return "NO ARC";
    case REJ_IMPACT:          return "IMPACT";
    case REJ_NN_NOISE:        return "NOT A SPELL";
    case REJ_LOW_CONF:        return "UNSURE";
    case REJ_LOW_MARGIN:      return "AMBIGUOUS";
    case REJ_NOISE_PROB:      return "NOISY";
    case REJ_NOT_PERSISTENT:  return "UNSTABLE";
    case REJ_REFRACTORY:      return "TOO SOON";
    default:                  return "?";
  }
}

// ---------------------------------------------------------------------------
// Stage 1 -- kinematic plausibility.
//
// Each bound is tied to a specific behaviour from the PS's own false-positive
// list, which is deliberate: these are not generic sanity checks, they are
// answers to the named test cases.
// ---------------------------------------------------------------------------
RejectReason gateKinematics(const KinFeatures& k) {
  // No incantation is shorter than 300 ms of motion or longer than 2 s. This is
  // also where duration is enforced at all -- the network never sees it, because
  // the fixed 64-step resample makes the model deliberately speed-blind so that a
  // fast judge and a slow judge get the same answer. Bounds are on the whole
  // buffer, pre-roll and post-roll included.
  if (k.duration_s < REJ_MIN_DURATION_S) return REJ_TOO_SHORT;
  if (k.duration_s > REJ_MAX_DURATION_S) return REJ_TOO_LONG;

  // Energy, as a DISJUNCTION. A window is plausible if it is rotationally
  // energetic (the four wand-waving spells) OR ballistically energetic
  // (Stupefy, which is a thrust and barely rotates at all).
  //
  // Getting this wrong is easy and expensive. The first version of this gate
  // required rotation unconditionally and rejected every single Stupefy, on the
  // bench, before the network ever saw one -- the bonus spell was unreachable
  // and the fault looked like a model problem, not a gate problem.
  //
  // "scratching an itch" is fast but small: low peak rate AND low linear accel.
  // "random pacing" moves the wand through space but barely rotates it, so the
  // swept angle stays small AND there is no thrust. Neither passes either arm.
  {
    const bool rotational = (k.peak_w >= REJ_MIN_PEAK_W) && (k.path_rad >= REJ_MIN_PATH_RAD);
    const bool ballistic  = (k.peak_a >= REJ_MIN_PEAK_A);
    if (!rotational && !ballistic) {
      return (k.peak_w < REJ_MIN_PEAK_W) ? REJ_TOO_STILL : REJ_TOO_SHALLOW;
    }
  }

  // Impact, by spike WIDTH rather than magnitude.
  //
  // R3 change, forced by the sensor swap. With the MPU6050 at +/-8 g a 7.5 g
  // table knock was separable from a 3 g cast by magnitude alone. The BNO055's
  // fusion algorithm locks the accelerometer to +/-4 g, and gravity consumes one
  // of them, so a knock and a hard Stupefy BOTH saturate near 3 g and magnitude
  // can no longer tell them apart.
  //
  // Duration can, and always could: a collision reaches the spike level for one
  // or two samples and is gone, whereas a thrust holds it for tens. A window is
  // an impact if it reaches the spike level but never HOLDS it. This is a better
  // test than the one it replaces, and it happens to survive clipping.
  if (k.peak_a >= REJ_SPIKE_G && k.spike_run < REJ_SPIKE_MIN_RUN) return REJ_IMPACT;

  return REJ_NONE;
}

// ---------------------------------------------------------------------------
// Stages 2 and 3 -- the network's verdict, then confidence.
// ---------------------------------------------------------------------------
Decision judgeProbs(const float* probs, int n_classes) {
  Decision d{GESTURE_NONE, REJ_NONE, 0.f, 0.f, 3};

  int i1 = 0;
  for (int i = 1; i < n_classes; ++i) if (probs[i] > probs[i1]) i1 = i;
  int i2 = -1;
  for (int i = 0; i < n_classes; ++i)
    if (i != i1 && (i2 < 0 || probs[i] > probs[i2])) i2 = i;

  d.p1 = probs[i1];
  d.p2 = (i2 >= 0) ? probs[i2] : 0.f;

  // Stage 2
  if (i1 == GESTURE_NOISE) { d.reason = REJ_NN_NOISE; d.stage = 2; return d; }

  // Stage 3. Thresholds come from wand_thresholds.h and are MEASURED --
  // see that header for why hardcoding them is a mistake.
  if (probs[GESTURE_NOISE] >= REJ_TAU_NOISE)  { d.reason = REJ_NOISE_PROB;  return d; }
  if (d.p1 < REJ_TAU_CONF)                    { d.reason = REJ_LOW_CONF;    return d; }
  if ((d.p1 - d.p2) < REJ_TAU_MARGIN)         { d.reason = REJ_LOW_MARGIN;  return d; }

  d.cls = (int8_t)i1;
  d.reason = REJ_NONE;
  return d;
}

// ---------------------------------------------------------------------------
// Stages 4 and 5 -- temporal persistence and the refractory period.
// ---------------------------------------------------------------------------
Decision applyTemporal(Decision d, uint32_t now_ms) {
  // Stage 5 first: a refractory rejection is unconditional, and checking it
  // before touching the history stops a rejected repeat from polluting it.
  if (d.cls != GESTURE_NONE && (now_ms - s_last_fire_ms) < REJ_REFRACTORY_MS) {
    d.cls = GESTURE_NONE;
    d.reason = REJ_REFRACTORY;
    d.stage = 5;
    return d;
  }

#if AUTO_SEGMENT
  // Stage 4 exists only in auto-segmentation mode. With a trigger button the
  // caster has already told us where the gesture is, and demanding that three
  // consecutive windows agree would just add 2 windows of latency for nothing.
  // In auto mode the segmenter fires on every energy burst, so requiring 2 of 3
  // consecutive windows to agree removes the single-frame flukes that a sliding
  // window over continuous motion inevitably produces.
  s_hist[s_hist_i] = d.cls;
  s_hist_i = (uint8_t)((s_hist_i + 1) % REJ_PERSIST_WINDOWS);

  if (d.cls != GESTURE_NONE) {
    int agree = 0;
    for (int i = 0; i < REJ_PERSIST_WINDOWS; ++i) if (s_hist[i] == d.cls) ++agree;
    if (agree < REJ_PERSIST_AGREE) {
      d.cls = GESTURE_NONE;
      d.reason = REJ_NOT_PERSISTENT;
      d.stage = 4;
      return d;
    }
  }
#endif

  if (d.cls != GESTURE_NONE) {
    s_last_fire_ms = now_ms;
    for (int i = 0; i < REJ_PERSIST_WINDOWS; ++i) s_hist[i] = GESTURE_NONE;
  }
  return d;
}

void resetTemporal() {
  s_last_fire_ms = 0;
  s_hist_i = 0;
  for (int i = 0; i < REJ_PERSIST_WINDOWS; ++i) s_hist[i] = GESTURE_NONE;
}
