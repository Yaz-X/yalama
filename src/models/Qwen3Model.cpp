#include "Qwen3Model.h"
#include "ConfigManager.h"

Qwen3Model::Qwen3Model()
{
}

void Qwen3Model::PopulateWeightNames()
{
    ModelBase::PopulateWeightNames();

    _WeightNames[WeightType::QNormWeight] = "model.layers.%d.self_attn.q_norm.weight";
    _WeightNames[WeightType::KNormWeight] = "model.layers.%d.self_attn.k_norm.weight";
}

torch::Tensor Qwen3Model::CalculatePostProjectionQ(const torch::Tensor &q, int layer)
{
     return ApplyHeadDimRMSNorm(q, _QNormWeights[layer]);
}

torch::Tensor Qwen3Model::CalculatePostProjectionK(const torch::Tensor &k, int layer)
{
    return ApplyHeadDimRMSNorm(k, _KNormWeights[layer]);
}

torch::Tensor Qwen3Model::ApplyHeadDimRMSNorm(const torch::Tensor &x, const torch::Tensor *weight)
{
    auto mean_sq = torch::mean(x * x, -1, true).to(torch::kFloat32);
    auto denom = torch::rsqrt(mean_sq + ConfigManager::Eps).to(x.dtype());
    auto normed = x * denom;

    return normed * weight->view({1, 1, 1, -1});
}