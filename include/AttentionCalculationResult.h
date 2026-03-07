#pragma once

#include <torch/torch.h>

struct AttentionCalculationResult
{
    torch::Tensor CalculatedAttention;
    bool IsKVCacheCapacityValid = true;
};