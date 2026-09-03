// firmware/common/bno055_fusion.cpp
#include "bno055_fusion.h"

#include <Arduino.h>
#include <Wire.h>

namespace bno {

static uint8_t  s_chip_id = 0;
static uint8_t  s_calib = 0;
static uint32_t s_errors = 0;
static uint32_t s_last_us = 0;

const char* initResultName(InitResult r) {
  switch (r) {
    case IMU_OK:          return "OK";
    case IMU_NO_DEVICE:   return "NO DEVICE at 0x28 (wiring or power)";
    case IMU_BAD_CHIP_ID: return "WRONG CHIP_ID (not a BNO055)";
    case IMU_PAGE_FAILED: return "PAGE_ID would not take";
    case IMU_UNIT_FAILED: return "UNIT_SEL would not take (units unknown)";
    case IMU_MODE_FAILED: return "OPR_MODE would not take (not in fusion)";
    default:              return "?";
  }
}

static inline void w8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(I2C_ADDR_BNO);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static inline uint8_t r8(uint8_t reg) {
  Wire.beginTransmission(I2C_ADDR_BNO);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((int)I2C_ADDR_BNO, 1);
  return Wire.available() ? Wire.read() : 0xFF;
}

// Write, then read back and compare. A dropped configuration write is otherwise
// invisible: the sensor keeps streaming perfectly plausible data in the wrong
// mode or the wrong units, and nothing downstream reports an error. Claiming to
// guard against a dropped write without reading it back is not a guard.
static bool w8_verify(uint8_t reg, uint8_t val) {
  w8(reg, val);
  delay(2);
  return r8(reg) == val;
}

void busRecover() {
  Wire.end();
  delay(5);
  Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL, I2C_FREQ_IMU);
  Wire.setTimeOut(I2C_TIMEOUT_MS);
}

InitResult begin() {
  Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL, I2C_FREQ_IMU);
  // The BNO055 stretches the I2C clock hard. A short timeout turns that into an
  // intermittent bus hang that passes every quick bench test and dies mid-demo.
  Wire.setTimeOut(I2C_TIMEOUT_MS);

  // The chip does not answer I2C at all for ~650 ms after power-on. Polling
  // rather than a blind delay, because a bare delay hides the case where the
  // sensor is genuinely absent -- you want that to fail loudly at boot.
  const uint32_t deadline = millis() + BNO_POWERON_DELAY_MS + 800;
  bool answered = false;
  while ((int32_t)(millis() - deadline) < 0) {
    Wire.beginTransmission(I2C_ADDR_BNO);
    if (Wire.endTransmission() == 0) { answered = true; break; }
    delay(20);
  }
  if (!answered) return IMU_NO_DEVICE;

  // Give it the rest of the power-on window to start answering with a sane ID.
  for (int i = 0; i < 40; ++i) {
    s_chip_id = r8(BNO_REG_CHIP_ID);
    if (s_chip_id == BNO_CHIP_ID_VALUE) break;
    delay(20);
  }
  if (s_chip_id != BNO_CHIP_ID_VALUE) return IMU_BAD_CHIP_ID;

  // FORCE PAGE 0 FIRST. The BNO055 has two register pages, and UNIT_SEL (0x3B)
  // and OPR_MODE (0x3D) are page-0 addresses. If the chip is left on page 1 --
  // by an earlier experiment, or by a reset partway through someone else's
  // configuration sequence -- both writes below land on page-1 addresses, are
  // discarded, and the wand runs in whatever mode and units it was left in with
  // nothing anywhere reporting a problem. One register write closes it.
  if (!w8_verify(BNO_REG_PAGE_ID, 0x00)) return IMU_PAGE_FAILED;

  w8(BNO_REG_OPR_MODE, BNO_MODE_CONFIG);
  delay(BNO_MODE_SWITCH_MS);              // >= 19 ms leaving an operating mode
  w8(BNO_REG_SYS_TRIGGER, 0x00);          // internal oscillator
  w8(BNO_REG_PWR_MODE, 0x00);             // normal power
  delay(10);

  // Units written explicitly even though 0x00 is the power-on default, and
  // VERIFIED rather than assumed: a chip left in mg or rad/s by an earlier
  // experiment would silently rescale every capture by 2% (m/s^2 vs mg) or by
  // 57x (dps vs rad/s), and nothing downstream would report an error.
  if (!w8_verify(BNO_REG_UNIT_SEL, BNO_UNIT_SEL_VALUE)) return IMU_UNIT_FAILED;

  w8(BNO_REG_OPR_MODE, BNO_MODE_IMU);     // accel + gyro fusion, magnetometer OFF
  delay(BNO_MODE_SWITCH_MS);              // >= 7 ms entering an operating mode
  if ((r8(BNO_REG_OPR_MODE) & 0x0F) != BNO_MODE_IMU) return IMU_MODE_FAILED;

  s_errors = 0;
  return IMU_OK;
}

// One transaction, GYR_DATA_X_LSB (0x14) through LIA_DATA_Z_MSB (0x2D).
// 26 bytes covers gyro, Euler, quaternion and linear acceleration in a single
// burst, which guarantees every channel in a sample comes from the same fusion
// frame. The six Euler bytes are read and discarded -- they sit in the middle of
// the range so they cannot be skipped, and paying 0.5 ms for them is cheaper
// than splitting this into two transactions that could straddle a frame boundary.
bool read(RawSample* s) {
  const uint32_t t0 = micros();
  Wire.beginTransmission(I2C_ADDR_BNO);
  Wire.write(BNO_REG_GYR_DATA);
  if (Wire.endTransmission(false) != 0) { ++s_errors; return false; }
  if (Wire.requestFrom((int)I2C_ADDR_BNO, 26) != 26) { ++s_errors; return false; }

  auto rd = []() -> int16_t {
    const uint8_t lo = Wire.read();          // BNO055 is little-endian: LSB first
    return (int16_t)((uint16_t)Wire.read() << 8 | lo);
  };
  s->gx = rd(); s->gy = rd(); s->gz = rd();                  // 0x14..0x19
  (void)rd(); (void)rd(); (void)rd();                        // 0x1A..0x1F Euler
  s->qw = rd(); s->qx = rd(); s->qy = rd(); s->qz = rd();    // 0x20..0x27
  s->lax = rd(); s->lay = rd(); s->laz = rd();               // 0x28..0x2D
  s_last_us = micros() - t0;
  return true;
}

// CALIB_STAT (0x35): bits 7:6 SYS, 5:4 GYR, 3:2 ACC, 1:0 MAG.
// Only the gyroscope field matters here. The fusion's continuously-estimated
// gyro bias is what replaced the pre-roll bias subtraction the MPU6050 build
// needed, and below status 3 that estimate is not yet trustworthy -- so every
// capture taken there is quietly wrong, with nothing to indicate it.
uint8_t gyroCalib() {
  s_calib = r8(BNO_REG_CALIB_STAT);
  return (uint8_t)((s_calib >> 4) & 0x03);
}

bool quaternionIsLive() {
  RawSample s{};
  for (int attempt = 0; attempt < 20; ++attempt) {
    if (read(&s)) {
      const float n = sqrtf((float)s.qw * s.qw + (float)s.qx * s.qx +
                            (float)s.qy * s.qy + (float)s.qz * s.qz);
      if (n > 0.8f * BNO_LSB_PER_QUAT && n < 1.2f * BNO_LSB_PER_QUAT) return true;
    }
    delay(25);
  }
  return false;
}

uint8_t  lastChipId()     { return s_chip_id; }
uint8_t  lastCalibByte()  { return s_calib; }
uint32_t errorCount()     { return s_errors; }
uint32_t lastReadMicros() { return s_last_us; }

}  // namespace bno
