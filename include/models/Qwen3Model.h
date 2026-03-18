#pragma once

#include "ModelBase.h"
#include <torch/torch.h>

class Qwen3Model : public ModelBase
{
public:

    Qwen3Model();

protected:

    void PopulateWeightNames() override;
    torch::Tensor CalculatePostProjectionQ(const torch::Tensor &q, int layer) override;
    torch::Tensor CalculatePostProjectionK(const torch::Tensor &k, int layer) override;
    virtual torch::Tensor ApplyHeadDimRMSNorm(const torch::Tensor &x, const torch::Tensor* weight);
};