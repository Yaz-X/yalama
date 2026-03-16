#pragma once

#include "LlamaModel.h"

class QwenModel : public LlamaModel
{
protected:
  
  void PopulateWeightNames() override;
  torch::Tensor CalculateProjectionQ(const torch::Tensor &x, int layer) override;
  torch::Tensor CalculateProjectionK(const torch::Tensor &x, int layer) override;
  torch::Tensor CalculateProjectionV(const torch::Tensor &x, int layer) override;

public:
  QwenModel();
};
