// firmware/wand_infer/preprocess.h
#ifndef WAND_PREPROCESS_H
#define WAND_PREPROCESS_H

#include <stdint.h>
#include "wand_config.h"
#include "wand_types.h"   // struct RawSample -- one definition, four consumers

// Mirrors training/preprocess.py::KinematicFeatures exactly.
struct KinFeatures {
  float duration_s;
  float peak_w;      // rad/s
  float peak_a;      // g, linear
  float path_rad;    // integral |w| dt
  int   spike_run;   // longest run with |a| >= REJ_SPIKE_G
};

// raw[0..n) -> out[N_RESAMPLE * N_CHANNELS], row-major (t, c), plus kinematics.
// Returns false if n is outside [MIN_GESTURE_SAMPLES, buffer capacity].
bool wandPreprocess(const RawSample* raw, int n, float* out, KinFeatures* kin);

#endif  // WAND_PREPROCESS_H
