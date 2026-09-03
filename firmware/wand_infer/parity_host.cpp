// firmware/wand_infer/parity_host.cpp
// -----------------------------------------------------------------------------
// Host build of the firmware preprocessing chain, so parity can be tested on a
// laptop with no ESP32 and no captured data.
//
// This is the trick that unblocks the whole schedule. The Python<->C++ parity
// check is normally described as something you run once the hardware works and
// real gestures exist -- which puts the single highest-risk defect in the
// project (a silent divergence between the two chains) at the END of the
// timeline. But preprocess.cpp depends on nothing from Arduino: only stdint and
// math. So it compiles and runs here, today, against any CSV, and the divergence
// is either found or ruled out before the IMU is even wired.
//
// Build and run:
//     g++ -O2 -std=c++17 -I../../shared -o parity_host parity_host.cpp preprocess.cpp
//     ./parity_host gesture.csv
//
// Emits one line per resampled step: N_CHANNELS comma-separated floats, %.6f,
// preceded by the kinematic feature row and the Stage 1 GATE VERDICT.
// training/parity_check.py compares all three against the Python chain.
//
// The gate verdict is checked here because of a gap the negative controls found:
// a deliberate off-by-one in REJ_SPIKE_MIN_RUN passed the parity test cleanly.
// It had to -- spike_run is computed against REJ_SPIKE_G, and MIN_RUN is only
// consulted by gateKinematics(), which nothing was exercising. A test that
// covers the tensor and the features but not the DECISION made from them leaves
// the whole of Stage 1 free to diverge silently, which is exactly the failure
// class this harness exists to eliminate. reject.cpp, like preprocess.cpp,
// includes nothing from Arduino, so it links here for free.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "preprocess.h"
#include "reject.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: parity_host <capture.csv>\n");
    return 2;
  }
  std::FILE* f = std::fopen(argv[1], "r");
  if (!f) { std::perror("fopen"); return 2; }

  std::vector<RawSample> rows;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    if (line[0] == '#' || line[0] == 'i' || line[0] == '\n') continue;
    RawSample s{};
    int idx, dup = 0;
    long dt = DT_US;
    // idx,qw,qx,qy,qz,lax,lay,laz,gx,gy,gz,dt_us,dup[,synthetic]
    int got = std::sscanf(line, "%d,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%hd,%ld,%d",
                          &idx, &s.qw, &s.qx, &s.qy, &s.qz,
                          &s.lax, &s.lay, &s.laz, &s.gx, &s.gy, &s.gz, &dt, &dup);
    if (got < 11) continue;
    s.dt_us = (uint32_t)dt;
    s.dup = (uint8_t)dup;
    rows.push_back(s);
  }
  std::fclose(f);

  if (rows.empty()) { std::fprintf(stderr, "no rows parsed\n"); return 2; }

  static float tensor[TENSOR_LEN];
  KinFeatures kin{};
  if (!wandPreprocess(rows.data(), (int)rows.size(), tensor, &kin)) {
    std::fprintf(stderr, "wandPreprocess rejected n=%zu\n", rows.size());
    return 1;
  }

  std::printf("# kin %.6f %.6f %.6f %.6f %d\n",
              kin.duration_s, kin.peak_w, kin.peak_a, kin.path_rad, kin.spike_run);
  // Emitted as the raw enum value rather than its display name, because the
  // display names are for an OLED ("NO ARC") and the Python side names the
  // condition ("TOO_SHALLOW"). Comparing integers keeps the test insensitive to
  // wording and sensitive to behaviour.
  std::printf("# gate %d\n", (int)gateKinematics(kin));
  for (int k = 0; k < N_RESAMPLE; ++k) {
    for (int c = 0; c < N_CHANNELS; ++c)
      std::printf("%.6f%s", tensor[k * N_CHANNELS + c], c == N_CHANNELS - 1 ? "\n" : ",");
  }
  return 0;
}
