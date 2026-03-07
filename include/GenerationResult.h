#pragma once

#include <torch/torch.h>
#include "GenerationError.h"

struct GenerationResult
{
    torch::Tensor Logits;
    bool IsSuccess = true;
    GenerationError Error = GenerationError::None;

};