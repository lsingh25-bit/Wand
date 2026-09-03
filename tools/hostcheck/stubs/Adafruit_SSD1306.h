#pragma once
#include <Arduino.h>
#include <Wire.h>
#define SSD1306_WHITE 1
#define SSD1306_SWITCHCAPVCC 2
struct Adafruit_SSD1306 {
  Adafruit_SSD1306(int, int, TwoWire*, int) {}
  bool begin(int, uint8_t) { return true; }
  void clearDisplay() {}
  void display() {}
  void setTextColor(int) {}
  void setTextSize(int) {}
  void setCursor(int, int) {}
  template <class T> void println(T) {}
  template <class... A> void printf(const char*, A...) {}
};
