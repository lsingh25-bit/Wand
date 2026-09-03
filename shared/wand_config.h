// shared/wand_config.h
// -----------------------------------------------------------------------------
// SINGLE SOURCE OF TRUTH for every constant shared between the ESP32 firmware
// and the Python training pipeline.  training/wand_config.py mirrors this file
// exactly and training/test_config_parity.py asserts they never diverge.
//
// RULE: no numeric literal that appears in this table may ever be re-typed
// anywhere else in the codebase.  Include this header instead.
//
// ============================== REVISION R3 ==================================
// Sensor changed from MPU-6050 to BNO055, running in IMU FUSION mode (0x08).
// This is not a drop-in swap; it deletes three stages of the signal chain.
//
//   * The BNO055 carries a Cortex-M0 running Bosch's fusion at 100 Hz and hands
//     back gravity-free linear acceleration, bias-compensated angular rate, and
//     an orientation quaternion.  Gyro bias removal, the complementary filter
//     and gravity projection are therefore DELETED from both implementations --
//     the chip already did all three, better.
//   * The tensor grows from 8 to 12 channels: linear acceleration and angular
//     rate are now expressed in a frame ANCHORED to the wand's orientation at
//     gesture start, and the relative quaternion itself becomes four channels.
//     This decouples translation from rotation, which the old body-frame
//     representation entangled.
//   * Accelerometer range is LOCKED to +/-4 g by the fusion algorithm, leaving
//     roughly 3 g of linear headroom after gravity.  The old 7.5 g "impact"
//     rejection is therefore unreachable and has been replaced by a spike-WIDTH
//     test (see REJ_SPIKE_* below).
//
// Class labels 0..5 are UNCHANGED from R2.  Any gesture recorded under R2 is,
// however, NOT reusable -- it was captured through a different sensor with a
// different signal chain.
// -----------------------------------------------------------------------------
#ifndef WAND_CONFIG_H
#define WAND_CONFIG_H

#include <stdint.h>

// The three softmax thresholds are MEASURED, not chosen, so they live in their
// own auto-generated file.  See the header comment there for why.
#include "wand_thresholds.h"

// ============================ BUILD FLAGS ====================================
// DEBUG_SERIAL MUST be 0 for every build flashed during evaluation.  The PS
// disqualifies any team whose device is seen using serial to offload work; the
// safest defence is firmware in which Serial.begin() is not compiled at all.
#ifndef DEBUG_SERIAL
#define DEBUG_SERIAL 0
#endif

// 0 = DTW template matcher (fallback), 1 = TFLite-Micro CNN (primary)
#ifndef USE_NN_PATH
#define USE_NN_PATH 1
#endif

// 0 = trigger button segments the gesture, 1 = energy-threshold auto-segmentation
#ifndef AUTO_SEGMENT
#define AUTO_SEGMENT 0
#endif

// ============================ PINOUT =========================================
constexpr int PIN_IMU_SDA      = 21;   // I2C bus 0 -- BNO055 only
constexpr int PIN_IMU_SCL      = 22;
constexpr int PIN_OLED_SDA     = 25;   // I2C bus 1 -- SSD1306 only (see report 4.2)
constexpr int PIN_OLED_SCL     = 26;
constexpr int PIN_TRIGGER      = 27;   // active-low, internal pull-up, NOT a strapping pin
constexpr int PIN_LED_DATA     =  4;   // WS2812 DIN via 470 R placed at the ring
constexpr int PIN_BUZZER       = 18;   // passive piezo
constexpr int PIN_VBAT_SENSE   = 34;   // ADC1_CH6, input-only pin, 100k/100k divider

constexpr uint8_t I2C_ADDR_BNO  = 0x28;   // ADR floating or low.  NOT 0x68 -- that was the MPU.
constexpr uint8_t I2C_ADDR_OLED = 0x3C;

// Bus 0 runs at 100 kHz, NOT 400 kHz.  The BNO055 stretches the I2C clock, and
// the failure mode of a too-fast bus with a short timeout is an intermittent
// hang -- it passes every short bench test and dies during the demo.  100 kHz
// costs 2.4 ms per 26-byte burst, which is affordable on a core that does
// nothing else.  Raise it only after the 3-minute continuous-read stress test
// passes at the higher speed.
constexpr uint32_t I2C_FREQ_IMU  = 100000;
constexpr uint32_t I2C_FREQ_OLED = 400000;
constexpr uint32_t I2C_TIMEOUT_MS = 1000;   // long, for the same clock-stretching reason

// ============================ BNO055 REGISTERS ===============================
constexpr uint8_t BNO_REG_CHIP_ID    = 0x00;   // reads 0xA0
// PAGE_ID.  The BNO055 has TWO register pages and this selects between them.
// We never write a page-1 register -- in fusion mode the algorithm owns the
// ranges -- but we must still force page 0 at boot, because UNIT_SEL (0x3B) and
// OPR_MODE (0x3D) are page-0 addresses. A chip left on page 1 by an earlier
// experiment or a mid-sequence reset would silently discard both writes, and the
// wand would run in whatever mode and units it happened to be left in.
constexpr uint8_t BNO_REG_PAGE_ID    = 0x07;
constexpr uint8_t BNO_CHIP_ID_VALUE  = 0xA0;
constexpr uint8_t BNO_REG_GYR_DATA   = 0x14;   // 6 bytes, start of the burst
constexpr uint8_t BNO_REG_EUL_DATA   = 0x1A;   // 6 bytes, read but discarded
constexpr uint8_t BNO_REG_QUA_DATA   = 0x20;   // 8 bytes
constexpr uint8_t BNO_REG_LIA_DATA   = 0x28;   // 6 bytes, gravity already removed
constexpr uint8_t BNO_REG_GRV_DATA   = 0x2E;   // 6 bytes, not read
constexpr uint8_t BNO_REG_CALIB_STAT = 0x35;
constexpr uint8_t BNO_REG_UNIT_SEL   = 0x3B;
constexpr uint8_t BNO_REG_OPR_MODE   = 0x3D;
constexpr uint8_t BNO_REG_PWR_MODE   = 0x3E;
constexpr uint8_t BNO_REG_SYS_TRIGGER= 0x3F;

constexpr uint8_t BNO_MODE_CONFIG    = 0x00;
// IMUPLUS: accelerometer + gyroscope fusion, MAGNETOMETER OFF.
//
// Not NDOF (0x0C).  NDOF adds the magnetometer to fix absolute heading, and the
// magnetometer is the one sensor that is actively hostile in a demo hall: steel
// tables, laptops, speakers, and a WS2812 ring drawing ~300 mA a few centimetres
// away.  We do not need absolute heading -- the gesture is anchored to its own
// starting orientation (see report 6.3), so heading cancels out. IMU mode also
// needs only GYRO calibration, which reaches full status after a few seconds of
// stillness, rather than the figure-of-eight magnetometer dance in front of judges.
constexpr uint8_t BNO_MODE_IMU       = 0x08;
constexpr uint8_t BNO_MODE_AMG       = 0x07;   // raw, non-fusion. Not used; see report 6.1.

// UNIT_SEL = 0x00: m/s^2, dps, degrees, Celsius, Windows orientation.
// This is the power-on default, written explicitly anyway so that a chip left in
// another state by earlier experiments cannot silently rescale the dataset.
constexpr uint8_t BNO_UNIT_SEL_VALUE = 0x00;

// Fixed by the datasheet and by UNIT_SEL above -- these are NOT range settings
// that someone can change, which is a real improvement over the MPU6050, where
// ACCEL_LSB_PER_G depended on a register nobody remembered to check.
constexpr float BNO_LSB_PER_MS2   = 100.0f;    // linear acceleration
constexpr float BNO_LSB_PER_DPS   = 16.0f;     // angular rate
constexpr float BNO_LSB_PER_QUAT  = 16384.0f;  // 2^14
constexpr float G_MS2             = 9.80665f;
constexpr float DEG2RAD           = 0.017453292519943295f;

// Derived, and the only conversion the signal chain performs.
constexpr float LINACC_LSB_PER_G  = BNO_LSB_PER_MS2 * G_MS2;   // 980.665

// The fusion algorithm locks the accelerometer to +/-4 g in every fusion mode.
// This is a hardware rail on the RAW signal, and it is the reason the impact
// test had to be redesigned around spike width rather than magnitude.
//
// R4 CORRECTION.  R3 reasoned that "gravity consumes 1 g of the 4, so linear
// acceleration tops out near 3 g" and set NORM_ACC_LIN to 3.0 accordingly.
// That reasoning had the sign backwards.  Gravity is SUBTRACTED by the fusion,
// not added to the budget: the rail applies to raw = linear + gravity, so
//
//     lin_axis = raw_axis - g_axis,   raw_axis in [-4, +4],  g_axis in [-1, +1]
//     => lin_axis in [-5, +5] g
//
// The measured per-axis maximum across the bootstrap corpus is exactly 5.00 g,
// which is the arithmetic above, reached and not exceeded. NORM_ACC_LIN is
// therefore 5.0 -- the hard physical bound, so nothing clips and no int8 code
// is spent on a value the sensor cannot produce.  At 3.0 the tensor railed on
// 2.4% of all samples and on the peak of every sharp flick, which is precisely
// the part of the trace that identifies the gesture.
constexpr float BNO_FUSION_ACC_RANGE_G = 4.0f;

// Minimum gyro calibration status (0..3) before the wand will arm.  The fusion's
// bias estimate is what replaced our pre-roll bias removal; below 3 it is not
// yet trustworthy and every capture taken there is quietly wrong.
constexpr uint8_t BNO_MIN_GYRO_CALIB = 3;

constexpr uint32_t BNO_POWERON_DELAY_MS  = 700;  // datasheet POR is ~650 ms; do not shorten
constexpr uint32_t BNO_MODE_SWITCH_MS    = 30;   // config -> fusion

// ======================= PHYSICAL WAND GEOMETRY ==============================
// Measured on the built wand, and the reason these numbers live in the shared
// config rather than in a comment: the synthetic bootstrap generator DERIVES the
// accelerometer signal from them.  A sensor bolted to a 50 cm lever does not
// measure the caster's hand; it measures the hand PLUS the lever arm, and the
// lever term dominates.  Getting the geometry wrong therefore does not shift the
// synthetic data slightly, it changes which channel carries the information.
//
// Stack-up along the tube, butt (0 cm) to tip (50 cm):
//
//     0 cm   ESP32 devkit + LiPo, charger, boost converter   <- mass, and the grip
//    ~22 cm  SSD1306 OLED, facing the caster
//     37 cm  BNO055                                          <- SENSOR_FROM_BUTT_M
//     50 cm  WS2812 ring, 8 LEDs, at the tip
//
// Sensor axes, as mounted: +X runs along the tube TOWARDS THE TIP, +Z points out
// of the PCB face (the wand's "back"), +Y completes the right-handed set and
// points to the caster's left when the wand is held with the OLED facing them.
constexpr float WAND_LENGTH_M         = 0.500f;  // butt to tip
constexpr float SENSOR_FROM_TIP_M     = 0.130f;  // BNO055 mount, measured from the tip
constexpr float SENSOR_FROM_BUTT_M    = 0.370f;  // = WAND_LENGTH_M - SENSOR_FROM_TIP_M
constexpr float SENSOR_RADIAL_OFF_M   = 0.012f;  // PCB sits against the tube wall, +Z

// Where the wand actually rotates about.  Not a property of the wand -- a
// property of the caster -- so these are three archetypes, measured backwards
// from the butt of the wand along its own axis.  A gesture is assigned one of
// them plus a per-caster scale, because arm lengths differ.
//
//   wrist     the hand pivots, forearm still            -- flicks, taps
//   elbow     forearm swings, upper arm still           -- loops, swirls, swishes
//   shoulder  whole arm swings                          -- walking, big sweeps
//
// The lever arm that matters is PIVOT + SENSOR_FROM_BUTT_M, so even the shortest
// of them puts the BNO055 45 cm from the centre of rotation.
constexpr float PIVOT_WRIST_M    = 0.080f;
constexpr float PIVOT_ELBOW_M    = 0.330f;
constexpr float PIVOT_SHOULDER_M = 0.620f;

// ============================ SAMPLING =======================================
constexpr int      FS_HZ        = 100;        // sample rate, Hz
constexpr float    DT_S         = 0.01f;      // 1 / FS_HZ, seconds
constexpr uint32_t DT_US        = 10000;      // timer ISR period, microseconds
constexpr uint32_t DT_TOL_US    = 200;        // acceptable jitter; data outside this is rejected

// The BNO055's fusion output is also 100 Hz, but its clock is not phase-locked
// to ours, so the two beat against each other and a fusion frame is occasionally
// read twice.  This is expected and harmless -- it is at most one sample period
// of sample-and-hold, far below the noise floor once the gesture is resampled to
// 64 steps.  The logger counts duplicates anyway, because a HIGH duplicate rate
// means something else is wrong (bus hang, mode fell back to CONFIG).
constexpr float MAX_DUP_FRACTION = 0.12f;

// ============================ CAPTURE WINDOW =================================
constexpr int RING_LEN            = 320;  // circular buffer depth = 3.2 s of history
constexpr int PREROLL_SAMPLES     = 20;   // 200 ms captured BEFORE the button press
constexpr int POSTROLL_SAMPLES    = 10;   // 100 ms captured AFTER release
constexpr int MIN_GESTURE_SAMPLES = 35;   // 350 ms -- below this it is not a spell
constexpr int MAX_GESTURE_SAMPLES = 200;  // 2.00 s -- above this the window is truncated

// ============================ FEATURE TENSOR =================================
// [N_RESAMPLE x N_CHANNELS], row-major (t, c), produced by the identical chain in
// training/preprocess.py and firmware/wand_infer/preprocess.cpp.
//
//   0..2   linear acceleration, ANCHORED frame       g       / NORM_ACC_LIN
//   3..5   angular rate,        ANCHORED frame       rad/s   / NORM_GYRO
//   6..9   relative quaternion  (w, x, y, z)         unit    already in [-1,1]
//   10     |linear acceleration|                     g       / NORM_AMAG
//   11     |angular rate|                            rad/s   / NORM_WMAG
//
// "Anchored" means rotated into the frame the wand occupied at sample 0.  The old
// body-frame representation entangled translation with rotation -- a constant
// world-frame push appeared as a time-varying signal because the frame itself was
// spinning.  Anchoring separates them: channels 0..2 are pure translation in a
// fixed frame, channels 3..9 are pure rotation.  See report 6.3.
constexpr int N_RESAMPLE  = 64;
constexpr int N_CHANNELS  = 12;
constexpr int TENSOR_LEN  = N_RESAMPLE * N_CHANNELS;   // 768

// Channel group boundaries, so no loop in either implementation hardcodes them.
constexpr int CH_ACC   = 0;   // 3 channels
constexpr int CH_GYRO  = 3;   // 3 channels
constexpr int CH_QUAT  = 6;   // 4 channels
constexpr int CH_AMAG  = 10;
constexpr int CH_WMAG  = 11;

// A quaternion whose norm falls below this is treated as invalid and replaced by
// the identity.  Both implementations must do this identically or parity fails.
constexpr float QUAT_MIN_NORM = 0.50f;

// FIXED normalisation divisors.  Constants, NOT per-sample statistics.
// Rationale (report 6.5): per-sample z-scoring destroys the amplitude difference
// between a gentle swish and a ballistic thrust, and would force a two-pass
// implementation on the MCU.  Fixed divisors keep the int8 quantisation scale
// stable across every capture.
//
// NORM_ACC_LIN is 5.0 = BNO_FUSION_ACC_RANGE_G + 1 g of gravity that the fusion
// subtracts back out.  See the derivation at BNO_FUSION_ACC_RANGE_G above; this
// is a hard bound, not a percentile, so it clips nothing and wastes nothing.
//
// NORM_GYRO stays at 12.0 rad/s even though the corpus only reaches 0.43 of that
// at the 99th percentile, and the temptation to tighten it to 9 or 10 should be
// resisted.  The samples out at the tail are the PEAKS OF THE FLICKS, and the
// presence and sharpness of a terminal flick is the entire difference between
// Alohomora and Lumos.  Buying 20% more int8 resolution on the quiet 99% by
// railing the 0.1% that carries the label is a bad trade.  Measured, considered,
// deliberately left alone.
constexpr float NORM_ACC_LIN = 5.0f;    // g   = 4 g rail + 1 g removed gravity
constexpr float NORM_GYRO    = 12.0f;   // rad/s (~687 dps, above any human wrist flick)
constexpr float NORM_AMAG    = 5.0f;    // g
constexpr float NORM_WMAG    = 12.0f;   // rad/s

// ============================ CLASSES ========================================
// NUM_SPELL_CLASSES are the castable spells.  NOISE is index 5 and is a real,
// trained output of the network -- not a threshold artefact.
enum WandGestureClass : int8_t {
  GESTURE_LUMOS               = 0,   // upward loop            [compulsory]
  GESTURE_ALOHOMORA           = 1,   // backward loop and flick [compulsory]
  GESTURE_EXPELLIARMUS        = 2,   // spiral swirl            [compulsory]
  GESTURE_WINGARDIUM_LEVIOSA  = 3,   // swish and flick         [compulsory]
  GESTURE_STUPEFY             = 4,   // ballistic forward thrust [bonus]
  GESTURE_NOISE               = 5,   // everything that is not a spell
  GESTURE_NONE                = -1   // cascade rejected the window
};

constexpr int NUM_SPELL_CLASSES = 5;   // indices 0..4
constexpr int NUM_CLASSES       = 6;   // indices 0..5, network output width

static const char* const GESTURE_NAMES[NUM_CLASSES] = {
  "LUMOS", "ALOHOMORA", "EXPELLIARMUS", "WINGARDIUM LEVIOSA", "STUPEFY", "NOISE"
};

// 24-bit RGB colours for the WS2812 ring, one per spell.
static const uint32_t GESTURE_COLORS[NUM_SPELL_CLASSES] = {
  0xFFFFC8,   // LUMOS              - warm white
  0xFF8C00,   // ALOHOMORA          - amber
  0xFF0000,   // EXPELLIARMUS       - red
  0x00A0FF,   // WINGARDIUM LEVIOSA - cyan
  0xFF00FF    // STUPEFY            - magenta
};

// ==================== REJECTION CASCADE THRESHOLDS ===========================
// Stage 1 -- kinematic plausibility gate, evaluated BEFORE the network runs.
//
// Duration bounds are on the WHOLE buffer, pre-roll and post-roll included
// (0.30 s of the two together), so 2.30 s here is a 2.00 s gesture.
// R5: lowered 0.60 -> 0.35 from calibrate.py's sweep_duration_gate() against the
// first real single-caster corpus (99 captures). At 0.60 this floor rejected
// 41.7% of genuine real casts (Wingardium Leviosa and Alohomora hardest hit,
// median real duration 0.42s / 0.53s -- both below the old floor); 0.35 is the
// largest floor the sweep found with zero real spell loss. This is a
// single-caster bootstrap number, not a converged one -- re-sweep once a
// second caster's data exists, the same way the softmax thresholds must be.
constexpr float REJ_MIN_DURATION_S   = 0.35f;  // provisional, single-caster real data
constexpr float REJ_MAX_DURATION_S   = 2.30f;  // MAX_GESTURE + PREROLL + POSTROLL

// Energy, as a DISJUNCTION: rotationally energetic OR ballistically energetic.
// A single rotation floor rejects Stupefy by construction, because Stupefy is
// DEFINED as a thrust that barely rotates.  See report 8.3.
//
// R4: these three are LOOSER than R3's 2.20 / 1.00 / 1.80, and deliberately so.
// Sweeping them jointly (calibrate.py::sweep_energy_gate) against a corpus with
// realistic lever-arm physics showed the whole gate is worth at most 17% of noise
// windows, and that every setting which buys more than about 11 of those points
// starts costing real casts. The R3 setting was in fact strictly dominated -- it
// lost 2.0% of spells to stop 14.4% of noise, where 0.90/0.30/1.00 loses NONE and
// still stops 11.1%.
//
// So the gate's job description shrank to what it is actually good at: deciding
// whether anything happened at all. It rejects a window in which the trigger was
// pressed and the wand did not move -- a fumbled button, a caster steadying
// themselves -- and passes everything else to the network, which has a trained
// NOISE class and is far better at this than three scalars can be.
// R5: swept jointly against the first real corpus with sweep_energy_gate().
// The old synthetic-calibrated 0.90/0.30/1.00 turned out to reject 9.5% of
// this caster's real spells for 0% real noise stopped -- pure cost, no
// benefit, because real casts (in particular real Stupefy, whose whole
// signature is a LOW-rotation ballistic thrust) land closer to these floors
// than the synthetic generator predicted. 0.45/0.10/1.00 is the largest
// triple the sweep found with zero real spell loss, with a small margin below
// its exact zero-loss ceiling (0.50/0.10/1.20) for single-caster headroom.
// Provisional until a second caster's data exists to re-sweep against.
constexpr float REJ_MIN_PEAK_W       = 0.45f;  // rad/s, rotational arm
constexpr float REJ_MIN_PATH_RAD     = 0.10f;  // integral |w| dt, total angle swept
constexpr float REJ_MIN_PEAK_A       = 1.00f;  // g, ballistic arm -- lets Stupefy through

// Impact rejection, by SPIKE WIDTH rather than magnitude.
//
// R3 change.  The MPU6050 at +/-8 g could tell a 7.5 g table knock from a 3 g
// cast by magnitude alone.  The BNO055's fusion locks the accelerometer to
// +/-4 g, so a knock and a hard thrust BOTH saturate near 3 g and magnitude can
// no longer separate them.  Duration can, and always could: a collision is an
// impulse a couple of samples wide, a thrust is a push a few hundred
// milliseconds wide.  A window is an impact if it reaches the spike level but
// never holds it.
// R4: the width narrowed from 4 samples to 3 after the sweep grid was extended
// upward. R3's grid stopped at 2.40 g because R3 believed 3 g was the ceiling;
// once the lever-arm physics put real casts at 5 g, the old grid's ceiling was
// INSIDE the search space rather than outside it, and a truncated sweep does not
// report that it was truncated -- it reports a winner. At 2.00 g / 3 samples the
// test catches 49.6% of noise windows for 0.7% of casts; the R3 setting caught
// 44%.  This is the second time this one subsystem has been improved by measuring
// something that had been reasoned about instead.
// R5: sweep_spike_test() against the first real corpus keeps the level at
// 2.00g but moves the width 3 -> 8 (80ms): at width 8 the real data shows the
// same 0% spell false-rejection but catches 6.7% of real noise instead of 0%.
constexpr float REJ_SPIKE_G          = 2.00f;  // g, "this is a large acceleration"
constexpr int   REJ_SPIKE_MIN_RUN    = 8;      // 80 ms -- below this it is a collision

// The free-fall / dropped-wand detector from R2 is GONE, deliberately.  In fusion
// mode the accelerometer's free-fall signature (|a| ~ 0) is not visible: linear
// acceleration is raw minus the fusion's gravity estimate, which coasts during a
// fall, so a dropped wand reads ~1 g of linear acceleration rather than zero.
// Two things replace it, and between them it is covered: a hold-to-record trigger
// means a dropped wand releases the button and ends its own capture, and the
// impact at the end of a drop is exactly the impulse the spike test catches.

// Stage 3 -- softmax confidence and margin: see shared/wand_thresholds.h.
// REJ_TAU_CONF, REJ_TAU_MARGIN and REJ_TAU_NOISE are defined there because they
// are calibration outputs, not design choices.

// Stage 4 -- temporal persistence, auto-segment mode only.
constexpr int REJ_PERSIST_WINDOWS = 3;
constexpr int REJ_PERSIST_AGREE   = 2;   // 2 of 3 consecutive windows must agree

// Stage 5 -- refractory period after any fire, prevents double-casting.
constexpr uint32_t REJ_REFRACTORY_MS = 500;

// Auto-segmentation energy gate (only compiled when AUTO_SEGMENT == 1).
constexpr float SEG_ENTER_W   = 2.80f;  // rad/s, rising edge of a gesture
constexpr float SEG_EXIT_W    = 0.90f;  // rad/s, falling edge
constexpr int   SEG_EXIT_HOLD = 12;     // 120 ms below SEG_EXIT_W ends the window

// ============================ FEEDBACK / UI ==================================
constexpr uint8_t  LED_MAX_BRIGHT   = 64;    // caps ring current to ~150 mA
constexpr int      NUM_LEDS         = 8;
constexpr uint16_t BUZZER_FREQ_HZ   = 2400;
constexpr uint16_t BUZZER_MS        = 60;
constexpr uint32_t FEEDBACK_HOLD_MS = 1200;  // how long the spell name stays on the OLED

// ============================ MODEL ==========================================
// Sized in report 7.6.  The arena is deliberately over-provisioned; SRAM is not
// the binding constraint on an ESP32, latency is.
constexpr int TFLM_ARENA_BYTES = 64 * 1024;

#endif  // WAND_CONFIG_H
