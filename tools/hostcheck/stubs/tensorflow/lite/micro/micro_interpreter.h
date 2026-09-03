// COMPILE-ONLY stub of the slice of TFLite Micro that wand_infer.cpp uses.
// Mirrors the real signatures so a mismatched call fails here rather than on the
// ESP32 toolchain. It computes nothing.
#pragma once
#include <cstddef>
#include <cstdint>
#define TFLITE_SCHEMA_VERSION 3
enum TfLiteStatus { kTfLiteOk = 0, kTfLiteError = 1 };
struct TfLiteQuantizationParams { float scale; int zero_point; };
struct TfLiteTensorData { int8_t* int8; float* f; };
struct TfLiteTensor { size_t bytes; TfLiteQuantizationParams params; TfLiteTensorData data; };
namespace tflite {
struct Model { int version() const { return TFLITE_SCHEMA_VERSION; } };
inline const Model* GetModel(const void*) { return nullptr; }
class MicroOpResolver {};
class ErrorReporter;          // real declaration in micro_error_reporter.h
class MicroResourceVariables;
class MicroProfiler;
class MicroInterpreter {
 public:
  // Mirrors the real 4-required + 3-optional signature (model, resolver,
  // arena, arena_size, error_reporter, resource_variables=nullptr,
  // profiler=nullptr) -- error_reporter has NO default in the real library,
  // which is exactly the bug this stub exists to catch: a call that leaves
  // it out, or passes nullptr where the real header would refuse to compile
  // without an explicit argument, should fail here rather than surface as a
  // runtime abort() on the device.
  MicroInterpreter(const Model*, const MicroOpResolver&, uint8_t*, size_t,
                    ErrorReporter*, MicroResourceVariables* = nullptr,
                    MicroProfiler* = nullptr) {}
  TfLiteStatus AllocateTensors() { return kTfLiteOk; }
  TfLiteStatus Invoke() { return kTfLiteOk; }
  TfLiteTensor* input(int) { return nullptr; }
  TfLiteTensor* output(int) { return nullptr; }
  size_t arena_used_bytes() const { return 0; }
};
}  // namespace tflite
