// COMPILE-ONLY stub. Added when wand_infer.cpp started passing a real
// ErrorReporter* instead of nullptr (see wand_infer.cpp nn::begin() for why
// nullptr was never safe here). Mirrors just enough of the real API for a
// mismatched call to fail here, on the host, rather than on the ESP32.
#pragma once
namespace tflite {
class ErrorReporter {};
inline ErrorReporter* GetMicroErrorReporter() {
  static ErrorReporter r;
  return &r;
}
}  // namespace tflite
