#pragma once

#include "mruntime/dtype.h"

namespace mruntime {

// Centralizes the tensor storage formats used by the runtime.
//
// Today:
// - Activations + KV-cache are stored in FP16.
// - Logits are stored in FP16.
// - Reduction-heavy ops may compute/accumulate in FP32 internally.
//
// Future formats (FP8/INT8/NF4/etc.) can be introduced by extending `DType`
// and evolving this struct (or replacing it with a richer "format policy"
// that also carries quantization metadata).
struct RuntimeFormats {
    static constexpr DType kActivation = DType::FP16;
    static constexpr DType kKvCache = DType::FP16;
    static constexpr DType kLogits = DType::FP16;

    // NOTE: FP32 is considered a compute/accumulation type, not a mainstream
    // tensor storage format in this runtime.
    static constexpr DType kAccum = DType::FP32;
};

}  // namespace mruntime
