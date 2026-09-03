#pragma once
#include <Arduino.h>
struct TwoWire {
  void begin(int, int, uint32_t) {}
  void begin() {}
  void end() {}
  void setTimeOut(uint32_t) {}
  void setClock(uint32_t) {}
  void beginTransmission(uint8_t) {}
  void write(uint8_t) {}
  uint8_t endTransmission(bool = true) { return 0; }
  int requestFrom(int, int) { return 0; }
  int available() { return 0; }
  uint8_t read() { return 0; }
};
static TwoWire Wire, Wire1;
