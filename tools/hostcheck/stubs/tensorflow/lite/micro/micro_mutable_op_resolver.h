#pragma once
#include <tensorflow/lite/micro/micro_interpreter.h>
namespace tflite {
template <unsigned N>
class MicroMutableOpResolver : public MicroOpResolver {
 public:
  TfLiteStatus AddConv2D() { return kTfLiteOk; }
  TfLiteStatus AddDepthwiseConv2D() { return kTfLiteOk; }
  TfLiteStatus AddFullyConnected() { return kTfLiteOk; }
  TfLiteStatus AddMaxPool2D() { return kTfLiteOk; }
  TfLiteStatus AddReshape() { return kTfLiteOk; }
  TfLiteStatus AddSoftmax() { return kTfLiteOk; }
  TfLiteStatus AddRelu() { return kTfLiteOk; }
};
}  // namespace tflite
