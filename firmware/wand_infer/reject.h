// firmware/wand_infer/reject.h
#ifndef WAND_REJECT_H
#define WAND_REJECT_H

#include <stdint.h>
#include "preprocess.h"
#include "wand_config.h"

enum RejectReason : uint8_t {
  REJ_NONE = 0,
  REJ_TOO_SHORT,
  REJ_TOO_LONG,
  REJ_TOO_STILL,      // peak |w| below threshold -- scratching, fidgeting
  REJ_TOO_SHALLOW,    // total swept angle too small -- pacing, arm swing
  REJ_IMPACT,         // a large acceleration that was not held -- a knock, not a cast
  REJ_NN_NOISE,       // network's own NOISE class won
  REJ_LOW_CONF,       // top-1 below the calibrated floor
  REJ_LOW_MARGIN,     // top-1 and top-2 too close (the Lumos/Alohomora case)
  REJ_NOISE_PROB,     // P(NOISE) above its veto level
  REJ_NOT_PERSISTENT, // auto-segment mode: consecutive windows disagreed
  REJ_REFRACTORY      // fired too recently
};

struct Decision {
  int8_t       cls;          // WandGestureClass, or GESTURE_NONE
  RejectReason reason;
  float        p1, p2;       // top-1 and top-2 probabilities
  uint8_t      stage;        // which cascade stage decided
};

const char* rejectReasonName(RejectReason r);

// Stage 1: kinematic plausibility, runs BEFORE the network.
RejectReason gateKinematics(const KinFeatures& k);

// Stages 2, 3: network output -> decision.
Decision judgeProbs(const float* probs, int n_classes);

// Stages 4, 5: persistence and refractory. Call once per candidate; `now_ms` is
// millis(). Mutates internal state.
Decision applyTemporal(Decision d, uint32_t now_ms);

void resetTemporal();

#endif  // WAND_REJECT_H
