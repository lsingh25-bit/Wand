// COMPILE-ONLY stub. wand_infer.cpp uses tflite::AllOpsResolver rather than a
// hand-named MicroMutableOpResolver<N> list -- deliberately, after a real
// retrain showed the exact op set a given architecture lowers to (EXPAND_DIMS,
// PACK, SHAPE, STRIDED_SLICE alongside the obvious CONV_2D/etc.) is a function
// of the TF/TFLite converter version, not just the layer list, and is too
// fragile to hand-curate safely. AllOpsResolver costs flash, not correctness.
#pragma once
#include "tensorflow/lite/micro/micro_interpreter.h"
namespace tflite {
class AllOpsResolver : public MicroOpResolver {};
}  // namespace tflite
