#pragma once
#include <Arduino.h>
struct CRGB {
  uint8_t r=0,g=0,b=0;
  CRGB() {}
  CRGB(int, int, int) {}
  enum Named { Black = 0 };
  CRGB(Named) {}
};
template <int T, int P, int O> struct _Chipset {};
#define WS2812B 1
#define GRB 2
inline void fill_solid(CRGB*, int, CRGB) {}
struct FastLEDStub {
  template <int C, int P, int O> void addLeds(CRGB*, int) {}
  void setBrightness(uint8_t) {}
  void show() {}
};
static FastLEDStub FastLED;
