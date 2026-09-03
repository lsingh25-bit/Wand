// firmware/common/bno055_fusion.h
//
// BNO055 driver, IMU FUSION mode (OPR_MODE = 0x08, IMUPLUS).
//
// One driver, three consumers: the demo firmware, the logger that produces the
// training data, and the bring-up suite that decides whether the sensor is
// trustworthy at all. That sharing is the point. A bring-up tool that
// configures the chip its own way tests a sensor configuration that will never
// be flashed again, and the acceptance it grants is worthless.
//
// Not Adafruit_BNO055, for three reasons:
//   * the burst read below is ONE 26-byte I2C transaction where the library
//     issues three separate ones -- and three transactions mean the quaternion
//     and the acceleration can come from different fusion frames, a skew that
//     varies with bus timing and cannot be corrected afterwards;
//   * the mode and unit configuration is part of the frozen contract with the
//     training data, so it belongs where training/test_config_parity.py can
//     read the constants, not inside a library call;
//   * it removes a dependency whose version could differ between two
//     teammates' laptops.
#ifndef WAND_BNO055_FUSION_H
#define WAND_BNO055_FUSION_H

#include <stdint.h>

#include "wand_config.h"
#include "wand_types.h"

namespace bno {

// Named failure causes rather than a bool. Every one of these has a different
// physical fix, and at 2 a.m. the difference between "nothing is on the bus"
// and "something is on the bus but it is not a BNO055" is the difference
// between checking the wiring and checking the part.
enum InitResult : uint8_t {
  IMU_OK = 0,
  IMU_NO_DEVICE,      // nothing acknowledged at 0x28 -- wiring or power
  IMU_BAD_CHIP_ID,    // something answered, but CHIP_ID is not 0xA0
  IMU_PAGE_FAILED,    // could not force the register page to 0
  IMU_UNIT_FAILED,    // UNIT_SEL did not read back -- units would be wrong
  IMU_MODE_FAILED     // OPR_MODE did not read back as IMUPLUS
};

const char* initResultName(InitResult r);

// Full bring-up: power-on wait, page 0, CONFIGMODE, units, then IMUPLUS.
// Every configuration write is read back and verified.
InitResult begin();

// One 26-byte burst, GYR_DATA_X_LSB (0x14) through LIA_DATA_Z_MSB (0x2D).
// Fills gyro, quaternion and linear acceleration from a single fusion frame.
// Does NOT set dt_us or dup -- those belong to the sampler, which knows when it
// asked. Returns false on any bus error.
bool read(RawSample* s);

// CALIB_STAT (0x35) bits 5:4 -- the gyroscope calibration field, 0..3.
uint8_t gyroCalib();

// Bring-up assertion, not an assumption: read frames until one carries a
// unit-norm quaternion. An invalid quaternion reads as zeros, which the
// preprocessing silently replaces with the identity -- so a wand whose
// orientation channels are dead would train, run, and simply classify worse,
// with no error reported anywhere.
bool quaternionIsLive();

// Diagnostics for the bring-up suite and the logger's display.
uint8_t  lastChipId();
uint8_t  lastCalibByte();
uint32_t errorCount();
uint32_t lastReadMicros();

// Re-open the bus after a hang. The BNO055 stretches the I2C clock and a wedged
// transaction does not recover on its own.
void busRecover();

}  // namespace bno

#endif  // WAND_BNO055_FUSION_H
