// firmware/wand_logger/wand_logger.cpp
// -----------------------------------------------------------------------------
// DATA COLLECTION FIRMWARE.  Serial is ON here and nowhere else.
// -----------------------------------------------------------------------------
#include <Arduino.h>
#include <Wire.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <FastLED.h>

#include "bno055_fusion.h"
#include "wand_config.h"
#include "wand_types.h"

#ifndef DEBUG_SERIAL
#error "wand_logger requires DEBUG_SERIAL -- build the 'logger' environment"
#endif

// ============================ 100 Hz acquisition =============================
static RawSample         s_ring[RING_LEN];
static volatile uint32_t s_seq = 0;
static volatile int      s_head = 0;
static volatile uint32_t s_dups = 0;
static volatile uint32_t s_bad = 0;
static hw_timer_t*       s_timer = nullptr;
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
    RawSample s{};
    if (!bno::read(&s)) {
      ++s_bad;
      bno::busRecover();
      last = micros();
      continue;
    }
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

// ============================ session state =================================
static Adafruit_SSD1306 oled(128, 64, &Wire1, -1);
static CRGB leds[NUM_LEDS];
static bool s_have_oled = false;

static int   s_label = 0;
static int   s_rep[NUM_CLASSES] = {0};
static char  s_caster[16] = "unknown";

static void ringSolid(uint32_t rgb, uint8_t bright) {
  fill_solid(leds, NUM_LEDS, CRGB((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF));
  FastLED.setBrightness(bright);
  // FastLED.show(); // Disabled to prevent WS2812 interrupt delays during 100 Hz IMU sampling
}

static void showIdle(uint8_t cal) {
  if (!s_have_oled) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);  oled.println(F("LOGGING"));
  oled.setCursor(0, 14); oled.println(GESTURE_NAMES[s_label]);
  oled.setCursor(0, 28); oled.printf("caster %s", s_caster);
  oled.setCursor(0, 40); oled.printf("rep %d", s_rep[s_label]);
  oled.setCursor(0, 52); oled.printf("gyrocal %u  dup %lu%%", cal,
                                     (unsigned long)(s_seq ? 100 * s_dups / s_seq : 0));
  oled.display();
}

static void showCalibrating(uint8_t cal) {
  if (!s_have_oled) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);  oled.println(F("CALIBRATING"));
  oled.setCursor(0, 20); oled.println(F("hold the wand still"));
  oled.setCursor(0, 40); oled.printf("gyro %u / %u", cal, BNO_MIN_GYRO_CALIB);
  oled.display();
}

// ============================ emission ======================================
static void emit(int start_seq, int count, uint8_t cal) {
  Serial.printf("# GESTURE_START label=%d caster=%s rep=%d nsamples=%d "
                "calib=%u mode=0x%02X\n",
                s_label, s_caster, s_rep[s_label], count, cal, BNO_MODE_IMU);
  Serial.println(F("idx,qw,qx,qy,qz,lax,lay,laz,gx,gy,gz,dt_us,dup"));
  int idx = ((start_seq % RING_LEN) + RING_LEN) % RING_LEN;
  for (int i = 0; i < count; ++i) {
    const RawSample& s = s_ring[idx];
    Serial.printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lu,%u\n",
                  i, s.qw, s.qx, s.qy, s.qz, s.lax, s.lay, s.laz,
                  s.gx, s.gy, s.gz, (unsigned long)s.dt_us, s.dup);
    idx = (idx + 1) % RING_LEN;
  }
  Serial.println(F("# GESTURE_END"));
}

static void handleSerial() {
  static char line[64];
  static uint8_t len = 0;
  while (Serial.available()) {
    const int c = Serial.read();
    if (c != '\n' && c != '\r') {
      if (len < sizeof(line) - 1) line[len++] = (char)c;
      continue;
    }
    line[len] = '\0';
    len = 0;
    if (line[0] == '\0') continue;

    if (strncmp(line, "SET", 3) == 0) {
      const char* p = strstr(line, "label=");
      if (p) {
        const int v = atoi(p + 6);
        if (v >= 0 && v < NUM_CLASSES) s_label = v;
      }
      p = strstr(line, "caster=");
      if (p) {
        size_t i = 0;
        for (p += 7; *p > 32 && i < sizeof(s_caster) - 1; ++p) s_caster[i++] = *p;
        s_caster[i] = '\0';
      }
      Serial.printf("# ACK label=%d %s caster=%s\n",
                    s_label, GESTURE_NAMES[s_label], s_caster);
    } else if (strcmp(line, "?") == 0) {
      Serial.printf("# status label=%d caster=%s seq=%lu dup=%lu bad=%lu "
                    "gyrocal=%u\n", s_label, s_caster, (unsigned long)s_seq,
                    (unsigned long)s_dups, (unsigned long)s_bad, bno::gyroCalib());
    } else if (strcmp(line, "n") == 0) {
      s_label = (s_label + 1) % NUM_CLASSES;
      Serial.printf("# ACK label=%d %s\n", s_label, GESTURE_NAMES[s_label]);
    }
  }
}

// ============================ setup / loop ==================================
void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(PIN_TRIGGER, INPUT_PULLUP);

  FastLED.addLeds<WS2812B, PIN_LED_DATA, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(LED_MAX_BRIGHT);
  ringSolid(0x000000, 0);

  Wire1.begin(PIN_OLED_SDA, PIN_OLED_SCL, I2C_FREQ_OLED);
  s_have_oled = oled.begin(SSD1306_SWITCHCAPVCC, I2C_ADDR_OLED);
  if (s_have_oled) {
    oled.clearDisplay();
    oled.display();
  }

  const bno::InitResult r = bno::begin();
  Serial.printf("# bno055 begin -> %s (chip_id 0x%02X)\n",
                bno::initResultName(r), bno::lastChipId());
  if (r != bno::IMU_OK) {
    for (;;) { ringSolid(0xFF0000, 32); delay(300); ringSolid(0, 0); delay(300); }
  }
  if (!bno::quaternionIsLive()) {
    Serial.println(F("# FATAL quaternion registers are dead -- not in fusion mode"));
    for (;;) { ringSolid(0xFFFF00, 32); delay(150); ringSolid(0, 0); delay(150); }
  }

  s_tick = xSemaphoreCreateBinary();
  xTaskCreatePinnedToCore(imuTask, "imu", 4096, nullptr, configMAX_PRIORITIES - 1,
                          nullptr, 0);
  s_timer = timerBegin(0, 80, true);
  timerAttachInterrupt(s_timer, &onTimer, true);
  timerAlarmWrite(s_timer, DT_US, true);
  timerAlarmEnable(s_timer);

  uint8_t cal = 0;
  while ((cal = bno::gyroCalib()) < BNO_MIN_GYRO_CALIB) {
    ringSolid(0x201000, LED_MAX_BRIGHT / 2);
    showCalibrating(cal);
    delay(250);
  }

  Serial.println(F("# READY  commands: SET label=<0-5> caster=<name> | n | ?"));
  Serial.println(F("# hold the trigger, cast, release"));
  showIdle(cal);
  ringSolid(0x000820, LED_MAX_BRIGHT);
}

void loop() {
  static bool pressed = false;
  static uint32_t edge_ms = 0;
  static int start_seq = 0;

  handleSerial();

  const bool raw_down = (digitalRead(PIN_TRIGGER) == LOW);
  const uint32_t now = millis();

  if (raw_down == pressed) { edge_ms = 0; return; }
  if (edge_ms == 0) edge_ms = now;
  if (now - edge_ms < 20) return;            // 20 ms software debounce
  edge_ms = 0;
  pressed = raw_down;

  if (pressed) {
    start_seq = (int)s_seq - PREROLL_SAMPLES;
    ringSolid(0x003000, LED_MAX_BRIGHT);     // green: capturing
    return;                                  
  }

  delay(POSTROLL_SAMPLES * 10);              // let the post-roll accumulate
  const int held = (int)s_seq - start_seq;
  const uint8_t cal = bno::gyroCalib();

  if (held < MIN_GESTURE_SAMPLES) {
    Serial.printf("# REJECT too short (%d samples)\n", held);
    ringSolid(0x200000, LED_MAX_BRIGHT / 2);
  } else if (held > MAX_GESTURE_SAMPLES + PREROLL_SAMPLES + POSTROLL_SAMPLES ||
             held > RING_LEN) {
    Serial.printf("# REJECT too long (%d samples)\n", held);
    ringSolid(0x200000, LED_MAX_BRIGHT / 2);
  } else {
    emit(start_seq, held, cal);
    ++s_rep[s_label];
    ringSolid(GESTURE_COLORS[s_label < NUM_SPELL_CLASSES ? s_label : 0],
              LED_MAX_BRIGHT);
  }

  delay(400);
  showIdle(cal);
  ringSolid(0x000820, LED_MAX_BRIGHT);
}