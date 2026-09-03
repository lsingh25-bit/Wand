// firmware/common/wand_types.h
//
// The one sample struct, shared by the BNO055 driver, the logger, the demo
// firmware and the host-side parity harness. It lives in its own header so that
// bno055_fusion.h does not have to include preprocess.h, and vice versa.
//
// R4 NOTE. An earlier draft of this file described a six-field sample in mg and
// 1/16 dps -- the BNO055 used as a plain 6-axis sensor in ACCGYRO mode. That is
// a coherent design and it is NOT this one. Running the chip in IMU fusion mode
// changes what a "sample" is: the fusion's own Cortex-M0 outputs an orientation
// quaternion and a gravity-free linear acceleration, and the entire 12-channel
// tensor, the anchoring transform and the trained model are built on those. The
// two sample layouts cannot coexist in one tree, because both want this name and
// the mismatch is a silent one -- the wrong struct still compiles, still fills
// with plausible int16s, and produces a wand that classifies at chance.
//
// If you want the non-fusion path, it is a deliberate fork of the whole
// pipeline, not a swap of this header.
#ifndef WAND_TYPES_H
#define WAND_TYPES_H

#include <stdint.h>

// One BNO055 fusion frame, exactly as the 26-byte burst read delivers it.
// Field order matches the CSV column order the logger writes and the training
// pipeline parses -- keeping those two identical is deliberate, because a
// silently transposed column pair is invisible until accuracy is inexplicable.
struct RawSample {
  int16_t qw, qx, qy, qz;   // 0x20..0x27, 16384 LSB = 1.0
  int16_t lax, lay, laz;    // 0x28..0x2D, 100 LSB = 1 m/s^2, gravity already removed
  int16_t gx, gy, gz;       // 0x14..0x19, 16 LSB = 1 dps
  uint32_t dt_us;
  uint8_t dup;              // 1 if this frame was byte-identical to the previous
};

#endif  // WAND_TYPES_H
