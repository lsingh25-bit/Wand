// firmware/wand_infer/wand_infer.cpp
// -----------------------------------------------------------------------------
// THE DEMO FIRMWARE. Everything happens on this chip.
//
// Core allocation:
//   Core 0  imuTask   -- 100 Hz sampling, highest priority, never blocked
//   Core 1  loop()    -- segmentation, inference, and all display/LED work
// -----------------------------------------------------------------------------
#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <FastLED.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "bno055_fusion.h"
#include "model_data.h"
#include "preprocess.h"
#include "reject.h"
#include "wand_config.h"

// ============================ 100 Hz acquisition =============================
static RawSample      s_ring[RING_LEN];
static volatile int   s_head = 0;
static volatile uint32_t s_seq = 0;
static volatile uint32_t s_dups = 0;
static hw_timer_t*    s_timer = nullptr;
static SemaphoreHandle_t s_tick;

void IRAM_ATTR onTimer() {
  BaseType_t hpw = pdFALSE;
  xSemaphoreGiveFromISR(s_tick, &hpw);
  if (hpw) portYIELD_FROM_ISR();
}

void imuTask(void*) {
  uint32_t last = micros();
  RawSample prev{};
  for (;;) {
    xSemaphoreTake(s_tick, portMAX_DELAY);
    RawSample s;
    if (!bno::read(&s)) continue;
    const uint32_t now = micros();
    s.dt_us = now - last;
    last = now;

    s.dup = (s.qw == prev.qw && s.qx == prev.qx && s.qy == prev.qy &&
             s.qz == prev.qz && s.lax == prev.lax && s.lay == prev.lay &&
             s.laz == prev.laz && s.gx == prev.gx && s.gy == prev.gy &&
             s.gz == prev.gz) ? 1 : 0;
    if (s.dup) ++s_dups;
    prev = s;

    s_ring[s_head] = s;
    s_head = (s_head + 1) % RING_LEN;
    ++s_seq;
  }
}

// ============================ feedback ======================================
static Adafruit_SSD1306 oled(128, 64, &Wire1, -1);
static CRGB leds[NUM_LEDS];
static bool s_have_oled = false;

static void ringSolid(uint32_t rgb, uint8_t bright) {
  fill_solid(leds, NUM_LEDS, CRGB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF));
  FastLED.setBrightness(bright);
  FastLED.show();
}

static void ringOff() { fill_solid(leds, NUM_LEDS, CRGB::Black); FastLED.show(); }

// ============================ TFLite Micro ==================================
namespace nn {

static uint8_t* s_arena = nullptr;
static tflite::MicroInterpreter* s_interp = nullptr;
static TfLiteTensor* s_in = nullptr;
static TfLiteTensor* s_out = nullptr;
static float s_in_scale, s_out_scale;
static int   s_in_zp, s_out_zp;

bool begin() {
  Serial.println("[NN] Starting TFLite initialization...");
  Serial.flush();

  // Dynamic heap allocation to prevent DRAM static overload
  if (!s_arena) {
    s_arena = (uint8_t*)malloc(TFLM_ARENA_BYTES);
    if (!s_arena) {
      Serial.println("[NN] ERROR: Heap memory allocation failed for s_arena!");
      Serial.flush();
      return false;
    }
  }

  const tflite::Model* model = tflite::GetModel(g_wand_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[NN] ERROR: Schema mismatch! Model: %d, Engine: %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    Serial.flush();
    return false;
  }

  static tflite::AllOpsResolver resolver;

  Serial.println("[NN] Allocating TFLite tensors...");
  Serial.flush();

  // The 5th argument is an ErrorReporter*, NOT optional in this library version --
  // it has no default. Passing nullptr here compiles fine (it's just a pointer)
  // but every internal error path in TFLite Micro calls it unconditionally via
  // TF_LITE_REPORT_ERROR(reporter, ...), which expands to
  // `static_cast<tflite::ErrorReporter*>(reporter)->Report(...)` with NO null
  // check (confirmed directly in this library's error_reporter.h). The very
  // first time AllocateTensors() hits anything worth reporting -- an op it
  // can't find, a tensor-shape mismatch, an arena that's a few bytes short --
  // it dereferences null and the board hard-crashes with "abort() was called",
  // instead of the readable message that would have said exactly what was
  // wrong. tflite::GetMicroErrorReporter() is the library's own always-valid
  // singleton; there is no reason to ever pass nullptr here.
  static tflite::MicroInterpreter interp(model, resolver, s_arena, TFLM_ARENA_BYTES,
                                          tflite::GetMicroErrorReporter());

  if (interp.AllocateTensors() != kTfLiteOk) {
    Serial.println("[NN] ERROR: AllocateTensors() failed! Check TFLM_ARENA_BYTES.");
    Serial.flush();
    return false;
  }

  s_interp = &interp;
  s_in  = interp.input(0);
  s_out = interp.output(0);

  if (s_in->bytes != (size_t)TENSOR_LEN) {
    Serial.printf("[NN] ERROR: Tensor len mismatch: expected %d, got %u\n", TENSOR_LEN, (unsigned)s_in->bytes);
    Serial.flush();
    return false;
  }

  s_in_scale  = s_in->params.scale;   s_in_zp  = s_in->params.zero_point;
  s_out_scale = s_out->params.scale;  s_out_zp = s_out->params.zero_point;

  Serial.println("[NN] TensorFlow Lite Model Loaded Successfully!");
  Serial.flush();
  return true;
}

bool infer(const float* tensor, float* probs) {
  int8_t* q = s_in->data.int8;
  for (int i = 0; i < TENSOR_LEN; ++i) {
    int v = (int)lroundf(tensor[i] / s_in_scale) + s_in_zp;
    q[i] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
  }
  if (s_interp->Invoke() != kTfLiteOk) return false;

  const int8_t* o = s_out->data.int8;
  for (int i = 0; i < NUM_CLASSES; ++i)
    probs[i] = ((float)o[i] - (float)s_out_zp) * s_out_scale;
  return true;
}

size_t arenaUsed() { return s_interp ? s_interp->arena_used_bytes() : 0; }

}  // namespace nn

// ============================ segmentation ==================================
constexpr int WINDOW_CAP = MAX_GESTURE_SAMPLES + PREROLL_SAMPLES + POSTROLL_SAMPLES;
static RawSample s_window[WINDOW_CAP];

static int snapshot(int start_seq, int count) {
  if (count < 1) return 0;
  if (count > WINDOW_CAP) return 0;
  if ((int)s_seq - start_seq > RING_LEN) return 0;
  int idx = ((start_seq % RING_LEN) + RING_LEN) % RING_LEN;
  for (int i = 0; i < count; ++i) {
    s_window[i] = s_ring[idx];
    idx = (idx + 1) % RING_LEN;
  }
  return count;
}

// ============================ display =======================================
static void showResult(const Decision& d, uint32_t latency_us, uint32_t infer_us) {
  if (!s_have_oled) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);

  if (d.cls != GESTURE_NONE) {
    oled.setCursor(0, 0);
    oled.println(F("* SPELL CAST *"));
    oled.setCursor(0, 16);
    oled.println(GESTURE_NAMES[d.cls]);
    oled.setCursor(0, 30);
    oled.printf("conf %.2f  m %.2f", d.p1, d.p1 - d.p2);
  } else {
    oled.setCursor(0, 0);
    oled.println(F("-- no spell --"));
    oled.setCursor(0, 16);
    oled.println(rejectReasonName(d.reason));
    oled.setCursor(0, 30);
    oled.printf("stage %u", d.stage);
  }
  oled.setCursor(0, 46);
  oled.printf("LATENCY_MS %.1f", latency_us / 1000.0f);
  oled.setCursor(0, 56);
  oled.printf("nn %.1f  arena %u", infer_us / 1000.0f, (unsigned)nn::arenaUsed());
  oled.display();
}

static void showIdle() {
  if (!s_have_oled) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println(F("WAND READY"));
  oled.setCursor(0, 20);
  oled.println(F("hold trigger, cast,"));
  oled.setCursor(0, 30);
  oled.println(F("release"));
  oled.display();
}

static void showCalibrating(uint8_t gyro_cal) {
  if (!s_have_oled) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println(F("CALIBRATING"));
  oled.setCursor(0, 20);
  oled.println(F("hold the wand still"));
  oled.setCursor(0, 40);
  oled.printf("gyro %u / %u", gyro_cal, BNO_MIN_GYRO_CALIB);
  oled.display();
}

// ============================ setup / loop ==================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=================================");
  Serial.println("   MAGIC WAND INFERENCE FIRMWARE ");
  Serial.println("=================================");

  pinMode(PIN_TRIGGER, INPUT_PULLUP);

  FastLED.addLeds<WS2812B, PIN_LED_DATA, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(LED_MAX_BRIGHT);
  ringOff();

  Wire1.begin(PIN_OLED_SDA, PIN_OLED_SCL, I2C_FREQ_OLED);
  s_have_oled = oled.begin(SSD1306_SWITCHCAPVCC, I2C_ADDR_OLED);
  if (s_have_oled) {
    Serial.println("[OLED] Initialized successfully.");
  } else {
    Serial.println("[OLED] WARNING: Display init failed!");
  }

  // Explicit initialization of I2C Bus 0 and BNO055 power-up delay
  Serial.println("[IMU] Initializing I2C Bus 0 and waiting for BNO055 startup...");
  Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL, I2C_FREQ_IMU);
  Wire.setClock(100000); // Clock frequency force 100kHz for BNO055 compatibility
  delay(800);

  // Read InitResult explicitly instead of using boolean negation
  const bno::InitResult imu_res = bno::begin();
  if (imu_res != bno::IMU_OK) {
    Serial.printf("[ERROR] BNO055 IMU Init Failed: %s\n", bno::initResultName(imu_res));
    for (;;) { ringSolid(0xFF0000, 32); delay(300); ringOff(); delay(300); }
  }
  Serial.println("[IMU] BNO055 Connected.");

  if (!bno::quaternionIsLive()) {
    Serial.println("[ERROR] BNO055 Quaternion Not Live!");
    for (;;) { ringSolid(0xFFFF00, 32); delay(150); ringOff(); delay(150); }
  }
  Serial.println("[IMU] Quaternion stream active.");

  if (!nn::begin()) {
    Serial.println("[ERROR] TFLite NN Model Allocation Failed!");
    for (;;) { ringSolid(0xFF00FF, 32); delay(120); ringOff(); delay(120); }
  }
  Serial.println("[NN] TensorFlow Lite Model Loaded.");

  s_tick = xSemaphoreCreateBinary();
  xTaskCreatePinnedToCore(imuTask, "imu", 4096, nullptr, configMAX_PRIORITIES - 1, nullptr, 0);
  s_timer = timerBegin(0, 80, true);
  timerAttachInterrupt(s_timer, &onTimer, true);
  timerAlarmWrite(s_timer, DT_US, true);
  timerAlarmEnable(s_timer);

  uint8_t cal = 0;
  Serial.println("[CALIB] Keep wand still for Gyro Calibration...");
  while ((cal = bno::gyroCalib()) < BNO_MIN_GYRO_CALIB) {
    Serial.printf("[CALIB] Gyro status: %u / %u\n", cal, BNO_MIN_GYRO_CALIB);
    ringSolid(0x201000, LED_MAX_BRIGHT / 2);
    showCalibrating(cal);
    delay(250);
  }
  Serial.println("[CALIB] Gyro Calibrated!");

  resetTemporal();
  showIdle();
  ringSolid(0x000820, LED_MAX_BRIGHT);
  Serial.println("\n=== WAND ARMED & READY TO CAST ===");
}

void loop() {
  static bool pressed = false;
  static uint32_t edge_ms = 0;
  static int start_seq = 0;

  const bool raw_down = (digitalRead(PIN_TRIGGER) == LOW);
  const uint32_t now = millis();

  if (raw_down != pressed) {
    if (edge_ms == 0) edge_ms = now;
    if (now - edge_ms < 20) return;
    edge_ms = 0;
    pressed = raw_down;

    if (pressed) {
      start_seq = (int)s_seq - PREROLL_SAMPLES;
      ringSolid(0x003000, LED_MAX_BRIGHT);
      Serial.println("\n[TRIGGER] Pressed! Recording gesture...");
      return;
    }

    Serial.println("[TRIGGER] Released! Processing capture...");
    const uint32_t t_end_us = micros();

    delay(POSTROLL_SAMPLES * 10);
    const int held = (int)s_seq - start_seq;

    float tensor[TENSOR_LEN];
    KinFeatures kin;
    Decision d{GESTURE_NONE, REJ_TOO_SHORT, 0.f, 0.f, 1};
    uint32_t infer_us = 0;

    const int n = (held > WINDOW_CAP) ? 0 : snapshot(start_seq, held);
    if (n == 0) d = Decision{GESTURE_NONE, REJ_TOO_LONG, 0.f, 0.f, 1};

    if (n > 0 && wandPreprocess(s_window, n, tensor, &kin)) {
      const RejectReason kr = gateKinematics(kin);
      if (kr != REJ_NONE) {
        d = Decision{GESTURE_NONE, kr, 0.f, 0.f, 1};
      } else {
        float probs[NUM_CLASSES];
        const uint32_t t0 = micros();
        const bool ok = nn::infer(tensor, probs);
        infer_us = micros() - t0;
        d = ok ? judgeProbs(probs, NUM_CLASSES)
               : Decision{GESTURE_NONE, REJ_LOW_CONF, 0.f, 0.f, 2};
        d = applyTemporal(d, now);
      }
    }

    const uint32_t latency_us = micros() - t_end_us;

    if (d.cls != GESTURE_NONE) {
      ringSolid(GESTURE_COLORS[d.cls], LED_MAX_BRIGHT);
      tone(PIN_BUZZER, BUZZER_FREQ_HZ, BUZZER_MS);
      Serial.printf("[RESULT] SPELL CAST: %s | Conf: %.2f | Latency: %.1f ms\n",
                    GESTURE_NAMES[d.cls], d.p1, latency_us / 1000.0f);
    } else {
      ringSolid(0x200000, LED_MAX_BRIGHT / 2);
      Serial.printf("[RESULT] NO SPELL | Reason: %s | Latency: %.1f ms\n",
                    rejectReasonName(d.reason), latency_us / 1000.0f);
    }
    showResult(d, latency_us, infer_us);

    delay(3000);
    ringSolid(0x000820, LED_MAX_BRIGHT);
    showIdle();
  } else {
    edge_ms = 0;
  }
}