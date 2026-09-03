// tools/hostcheck/stubs/Arduino.h -- COMPILE-ONLY stub. Not a simulator.
// Just enough of the Arduino/ESP32 surface for g++ to type-check the firmware.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#define F(x) (x)
#define IRAM_ATTR
#define LOW 0
#define HIGH 1
#define INPUT_PULLUP 2
typedef uint8_t byte;
inline void pinMode(int, int) {}
inline int  digitalRead(int) { return 1; }
inline void delay(uint32_t) {}
inline void delayMicroseconds(uint32_t) {}
inline uint32_t millis() { return 0; }
inline uint32_t micros() { return 0; }
inline void tone(int, int, int) {}
struct SerialStub {
  void begin(unsigned long) {}
  int  available() { return 0; }
  int  read() { return -1; }
  size_t readBytesUntil(char, char*, size_t) { return 0; }
  template <class... A> void printf(const char*, A...) {}
  template <class T> void println(T) {}
  void println() {}
  template <class T> void print(T) {}
  void flush() {}
};
static SerialStub Serial;
// --- FreeRTOS ---
typedef int BaseType_t;
typedef void* SemaphoreHandle_t;
#define pdFALSE 0
#define portMAX_DELAY 0xFFFFFFFF
#define configMAX_PRIORITIES 25
inline void portYIELD_FROM_ISR() {}
inline SemaphoreHandle_t xSemaphoreCreateBinary() { return nullptr; }
inline void xSemaphoreGiveFromISR(SemaphoreHandle_t, BaseType_t*) {}
inline int  xSemaphoreTake(SemaphoreHandle_t, uint32_t) { return 1; }
inline void xTaskCreatePinnedToCore(void (*)(void*), const char*, int, void*, int, void*, int) {}
// --- ESP32 hardware timer ---
struct hw_timer_t;
inline hw_timer_t* timerBegin(int, int, bool) { return nullptr; }
inline void timerAttachInterrupt(hw_timer_t*, void (*)(), bool) {}
inline void timerAlarmWrite(hw_timer_t*, uint32_t, bool) {}
inline void timerAlarmEnable(hw_timer_t*) {}
