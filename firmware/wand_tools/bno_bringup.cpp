// firmware/wand_tools/bno_bringup.cpp
// -----------------------------------------------------------------------------
// BNO055 acceptance suite.  Build and flash this BEFORE you solder anything
// permanently and BEFORE you record a single gesture.
//
//     pio run -e bringup -t upload -t monitor
//
// It exercises the SHARED driver in firmware/common/bno055_fusion.cpp -- the same
// object file the logger and the demo firmware link. A bring-up tool that
// configures the chip its own way tests a configuration that will never be
// flashed again, and the acceptance it grants is worthless.
//
// Six tests, in the order they should be run. Each closes a specific risk that
// is otherwise discovered too late to fix:
//
//   [1] SCAN        is anything on the bus at all
//   [2] CONFIG      did the mode and unit writes actually take
//   [3] QUATERNION  is the fusion really running, or are we reading zeros
//   [4] SCALE       are the datasheet LSB constants the ones this part uses
//   [5] GEOMETRY    is the sensor mounted the way the model assumes
//   [6] STRESS      does the bus survive ten minutes at 100 Hz
//
// Test 5 is new in R4 and it is the one nobody thinks to run. The synthetic
// bootstrap corpus, the tensor divisors and the Stage 1 thresholds are all
// derived from the wand's measured geometry -- 50 cm long, sensor 13 cm below
// the tip, +X towards the tip. If the BNO055 was soldered on rotated, every one
// of those derivations is describing a different wand. There is no way to detect
// that from the data afterwards; the captures look completely normal.
// -----------------------------------------------------------------------------
#include <Arduino.h>
#include <Wire.h>

#include "bno055_fusion.h"
#include "wand_config.h"
#include "wand_types.h"

static void hr(const char* title) {
  Serial.printf("\n---- %s ----------------------------------------\n", title);
}

// ---------------------------------------------------------------------------
static void testScan() {
  hr("[1] I2C SCAN on bus 0");
  Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL, I2C_FREQ_IMU);
  Wire.setTimeOut(I2C_TIMEOUT_MS);
  delay(BNO_POWERON_DELAY_MS);

  int found = 0;
  for (uint8_t a = 1; a < 127; ++a) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("    device at 0x%02X%s\n", a,
                    a == I2C_ADDR_BNO ? "   <-- BNO055, as expected" : "");
      ++found;
    }
  }
  if (found == 0) {
    Serial.println(F("    NOTHING FOUND. This is a wiring or power fault, not a"));
    Serial.println(F("    sensor fault. Check, in this order: 3V3 present at the"));
    Serial.println(F("    BNO VIN pin with a multimeter; GND continuity; SDA on"));
    Serial.println(F("    GPIO21 and SCL on GPIO22 not swapped; 3V3 to GND NOT"));
    Serial.println(F("    shorted. Note the OLED is on bus 1 and will not appear."));
  }
}

// ---------------------------------------------------------------------------
static bool testConfig() {
  hr("[2] CONFIGURATION");
  const bno::InitResult r = bno::begin();
  Serial.printf("    begin()  -> %s\n", bno::initResultName(r));
  Serial.printf("    CHIP_ID  =  0x%02X (expected 0x%02X)\n",
                bno::lastChipId(), BNO_CHIP_ID_VALUE);
  if (r != bno::IMU_OK) {
    Serial.println(F("    STOP. Do not collect data until this reads OK."));
    return false;
  }
  Serial.println(F("    OPR_MODE and UNIT_SEL were written AND read back."));
  Serial.println(F("    Mode 0x08 is IMUPLUS: accel+gyro fusion, magnetometer OFF."));
  return true;
}

// ---------------------------------------------------------------------------
static bool testQuaternion() {
  hr("[3] FUSION LIVENESS");
  Serial.println(F("    The quaternion registers read as zeros if the fusion is"));
  Serial.println(F("    not actually running. Preprocessing silently substitutes"));
  Serial.println(F("    the identity for a zero quaternion, so a dead fusion does"));
  Serial.println(F("    not crash anything -- it just makes four of the twelve"));
  Serial.println(F("    tensor channels constant, and the wand classifies worse"));
  Serial.println(F("    for no visible reason."));
  const bool live = bno::quaternionIsLive();
  Serial.printf("    quaternion norm is unit: %s\n", live ? "YES -- PASS" : "NO -- FAIL");
  if (!live) Serial.println(F("    STOP. The chip is not in a fusion mode."));
  return live;
}

// ---------------------------------------------------------------------------
static void testScale() {
  hr("[4] SCALE VERIFICATION");
  Serial.println(F("    Lay the wand FLAT AND STILL on the table."));
  Serial.println(F("    Averaging for 3 seconds -- do not touch it..."));
  delay(1500);

  double sl[3] = {0, 0, 0}, sg[3] = {0, 0, 0};
  int n = 0;
  const uint32_t until = millis() + 3000;
  while ((int32_t)(millis() - until) < 0) {
    RawSample s{};
    if (bno::read(&s)) {
      sl[0] += s.lax; sl[1] += s.lay; sl[2] += s.laz;
      sg[0] += s.gx;  sg[1] += s.gy;  sg[2] += s.gz;
      ++n;
    }
    delay(10);
  }
  if (n < 100) { Serial.println(F("    too few samples -- the bus is unhealthy")); return; }
  for (int i = 0; i < 3; ++i) { sl[i] /= n; sg[i] /= n; }

  // The KEY difference from a raw-accelerometer bring-up: at rest the LINEAR
  // acceleration registers should read ZERO, not 1 g. If they read ~981 counts
  // the chip is reporting the raw accelerometer, which means it is not fusing,
  // and every gravity assumption downstream is wrong.
  const double lin = sqrt(sl[0] * sl[0] + sl[1] * sl[1] + sl[2] * sl[2]);
  Serial.printf("    mean LINEAR accel counts: %.1f, %.1f, %.1f   |a| = %.1f\n",
                sl[0], sl[1], sl[2], lin);
  Serial.printf("    |a| SHOULD be near 0 at rest (gravity already removed).\n");
  if (lin > 40.0) {
    Serial.printf("    FAIL -- %.0f counts is %.2f g of 'linear' acceleration on a\n",
                  lin, lin / LINACC_LSB_PER_G);
    Serial.println(F("    stationary table. Near 981 means these are RAW accel"));
    Serial.println(F("    registers, i.e. the chip is not in a fusion mode."));
  } else {
    Serial.println(F("    PASS -- the fusion is removing gravity."));
  }

  Serial.printf("    gyro residual: %.2f, %.2f, %.2f counts (%.2f, %.2f, %.2f dps)\n",
                sg[0], sg[1], sg[2],
                sg[0] / BNO_LSB_PER_DPS, sg[1] / BNO_LSB_PER_DPS, sg[2] / BNO_LSB_PER_DPS);
  Serial.println(F("    This should be near zero too -- the fusion removes gyro"));
  Serial.println(F("    bias continuously, which is what replaced the pre-roll"));
  Serial.println(F("    bias subtraction the MPU6050 build had to do in software."));

  Serial.println(F("\n    Now rotate the wand slowly through exactly 90 degrees"));
  Serial.println(F("    about its LONG AXIS over about 3 seconds, starting NOW..."));
  double integ = 0;
  const uint32_t t_end = millis() + 3500;
  while ((int32_t)(millis() - t_end) < 0) {
    RawSample s{};
    if (bno::read(&s)) integ += ((double)s.gx) / BNO_LSB_PER_DPS * DT_S;
    delay(10);
  }
  Serial.printf("    integrated X rotation = %.1f deg (expect roughly +/-90)\n", integ);
  Serial.println(F("    Within 75-105 confirms BNO_LSB_PER_DPS. A result near 16x"));
  Serial.println(F("    or 1/16x of 90 means UNIT_SEL's gyro bit is wrong."));
}

// ---------------------------------------------------------------------------
static void testGeometry() {
  hr("[5] MOUNTING GEOMETRY");
  Serial.printf("    The model assumes: wand %.0f cm, BNO %.0f cm below the tip\n",
                WAND_LENGTH_M * 100, SENSOR_FROM_TIP_M * 100);
  Serial.printf("    (%.0f cm up from the butt), sensor +X pointing at the TIP.\n",
                SENSOR_FROM_BUTT_M * 100);
  Serial.println(F("    Everything derived from the geometry -- the bootstrap"));
  Serial.println(F("    corpus, the tensor divisors, the Stage 1 thresholds -- is"));
  Serial.println(F("    describing a different wand if this is wrong, and the"));
  Serial.println(F("    captures give no hint of it."));

  Serial.println(F("\n    (a) POINT THE TIP STRAIGHT DOWN at the floor and hold."));
  Serial.println(F("        Reading in 3 s..."));
  delay(3000);
  // Gravity in the body frame, recovered from the quaternion: g_body = R^T * up.
  // With the tip down and +X towards the tip, +X should be pointing along -up,
  // so the X component of body-frame gravity should be close to -1.
  RawSample s{};
  double gx_sum = 0, gy_sum = 0, gz_sum = 0;
  int n = 0;
  for (int i = 0; i < 60; ++i) {
    if (bno::read(&s)) {
      const float q0 = s.qw / BNO_LSB_PER_QUAT, q1 = s.qx / BNO_LSB_PER_QUAT;
      const float q2 = s.qy / BNO_LSB_PER_QUAT, q3 = s.qz / BNO_LSB_PER_QUAT;
      // R^T * (0,0,1), i.e. world-up expressed in the body frame.
      gx_sum += 2.0f * (q1 * q3 - q0 * q2);
      gy_sum += 2.0f * (q2 * q3 + q0 * q1);
      gz_sum += q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
      ++n;
    }
    delay(15);
  }
  if (n < 20) { Serial.println(F("        too few samples")); return; }
  const double bx = gx_sum / n, by = gy_sum / n, bz = gz_sum / n;
  Serial.printf("        body-frame UP = (%+.2f, %+.2f, %+.2f)\n", bx, by, bz);
  if (bx < -0.80) {
    Serial.println(F("        PASS -- +X points at the tip, as the model assumes."));
  } else if (bx > 0.80) {
    Serial.println(F("        FAIL -- +X points at the BUTT. The board is on"));
    Serial.println(F("        backwards. Either turn it round, or negate X and Y"));
    Serial.println(F("        in bno::read() -- but if you negate, say so in the"));
    Serial.println(F("        report, because the geometry section will be wrong."));
  } else {
    Serial.println(F("        FAIL -- the wand's long axis is not the sensor's X."));
    Serial.printf("        The dominant component is %s. The board is rotated 90\n",
                  fabs(by) > fabs(bz) ? "Y" : "Z");
    Serial.println(F("        degrees from what the model assumes."));
  }

  Serial.println(F("\n    (b) Hold the wand HORIZONTAL, OLED facing you, and"));
  Serial.println(F("        SWING THE TIP UP through about 90 degrees in half a"));
  Serial.println(F("        second. Peak values in 4 s..."));
  delay(500);
  int16_t pw[3] = {0, 0, 0};
  float peak_a = 0;
  const uint32_t t_end = millis() + 4000;
  while ((int32_t)(millis() - t_end) < 0) {
    if (bno::read(&s)) {
      const int16_t w[3] = {s.gx, s.gy, s.gz};
      for (int i = 0; i < 3; ++i) if (abs(w[i]) > abs(pw[i])) pw[i] = w[i];
      const float a = sqrtf((float)s.lax * s.lax + (float)s.lay * s.lay +
                            (float)s.laz * s.laz) / LINACC_LSB_PER_G;
      if (a > peak_a) peak_a = a;
    }
    delay(10);
  }
  Serial.printf("        peak gyro (dps): X %+.0f  Y %+.0f  Z %+.0f\n",
                pw[0] / BNO_LSB_PER_DPS, pw[1] / BNO_LSB_PER_DPS,
                pw[2] / BNO_LSB_PER_DPS);
  Serial.printf("        peak |linear a| = %.2f g\n", peak_a);
  Serial.println(F("        EXPECT: Y dominant and NEGATIVE (tip-up is a negative"));
  Serial.println(F("        pitch about +Y), X and Z much smaller."));
  Serial.printf("        EXPECT |a| roughly 1.5-4 g. This is the LEVER ARM: the\n");
  Serial.printf("        sensor sits ~%.2f m from your wrist, so a flick that\n",
                PIVOT_WRIST_M + SENSOR_FROM_BUTT_M);
  Serial.println(F("        feels gentle still throws several g at it. A peak"));
  Serial.println(F("        under 0.5 g means the lever-arm model is wrong for"));
  Serial.println(F("        this build and the bootstrap corpus needs regenerating"));
  Serial.println(F("        with the geometry you actually have."));
}

// ---------------------------------------------------------------------------
static void testStress() {
  hr("[6] CLOCK-STRETCHING STRESS TEST -- 10 minutes at 100 Hz");
  Serial.println(F("    This is the test that decides whether the sensor is"));
  Serial.println(F("    demo-worthy. BNO055 clock stretching causes bus hangs that"));
  Serial.println(F("    pass every short test and fail during the demo, so the"));
  Serial.println(F("    duration is the point. PICK THE WAND UP AND WAVE IT"));
  Serial.println(F("    THROUGHOUT -- a hang under vibration is the failure mode."));
  Serial.println(F("    Turn the LED ring on too; the ~300 mA it draws is the"));
  Serial.println(F("    supply disturbance the bench test otherwise never sees."));
  Serial.println(F("      ok      err   max_us  mean_us"));

  uint32_t ok = 0, err = 0, max_us = 0;
  uint64_t sum_us = 0;
  uint32_t next = micros();
  const uint32_t t_end = millis() + 600000UL;
  uint32_t report = millis() + 15000;

  while ((int32_t)(millis() - t_end) < 0) {
    while ((int32_t)(micros() - next) < 0) {}
    next += DT_US;

    RawSample s{};
    if (bno::read(&s)) {
      ++ok;
      const uint32_t u = bno::lastReadMicros();
      sum_us += u;
      if (u > max_us) max_us = u;
    } else {
      ++err;
      bno::busRecover();
      next = micros() + DT_US;
    }

    if ((int32_t)(millis() - report) >= 0) {
      report += 15000;
      Serial.printf("    %6lu  %7lu  %7lu  %7.1f\n",
                    (unsigned long)ok, (unsigned long)err, (unsigned long)max_us,
                    ok ? (double)sum_us / ok : 0.0);
    }
  }

  Serial.printf("\n    RESULT: %lu reads, %lu errors (%.4f%%), max %lu us, mean %.1f us\n",
                (unsigned long)ok, (unsigned long)err,
                100.0 * err / (ok + err ? ok + err : 1),
                (unsigned long)max_us, ok ? (double)sum_us / ok : 0.0);
  if (err == 0 && max_us < 6000) {
    Serial.println(F("    PASS -- solder it, and go collect data."));
  } else if (err == 0) {
    Serial.println(F("    MARGINAL -- no errors, but a read approached the 10 ms"));
    Serial.println(F("    sample budget. Check that dt_us stays inside 10000 +/- 200"));
    Serial.println(F("    during real captures before trusting the dataset."));
  } else {
    Serial.println(F("    FAIL -- do not solder yet. In order: shorten and twist"));
    Serial.println(F("    the SDA/SCL pair; confirm 3V3 is clean under LED load;"));
    Serial.println(F("    try the spare BNO055; only then consider 4.7k pull-ups."));
  }
}

// ---------------------------------------------------------------------------
static void waitForKey(const char* what) {
  Serial.printf("\n    Send any character to run %s.\n", what);
  while (!Serial.available()) delay(50);
  while (Serial.available()) Serial.read();
}

void setup() {
  Serial.begin(921600);
  delay(300);
  Serial.println(F("\n=============================================="));
  Serial.println(F(" BNO055 acceptance suite -- Edge of Magic  R4"));
  Serial.println(F("=============================================="));

  testScan();
  if (!testConfig()) return;
  if (!testQuaternion()) return;
  waitForKey("[4] SCALE");
  testScale();
  waitForKey("[5] GEOMETRY");
  testGeometry();
  waitForKey("[6] the 10-minute STRESS test");
  testStress();

  Serial.println(F("\nDone. Flash the logger target to begin collection."));
}

void loop() { delay(1000); }
