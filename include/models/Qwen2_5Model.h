#pragma once

#include "ModelBase.h"

class Qwen2_5Model : public ModelBase
{
protected:
  
  void PopulateWeightNames() override;
  torch::Tensor CalculateProjectionQ(const torch::Tensor &x, int layer) override;
  torch::Tensor CalculateProjectionK(const torch::Tensor &x, int layer) override;
  torch::Tensor CalculateProjectionV(const torch::Tensor &x, int layer) override;

public:
  Qwen2_5Model();
};
