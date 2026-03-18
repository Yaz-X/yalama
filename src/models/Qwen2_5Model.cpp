/*
    YALAMA Runtime
    Copyright 2026 Yazeed Hamdan
    Licensed under the Apache License, Version 2.0
    See LICENSE file in project root.
*/

#include "Qwen2_5Model.h"
#include "ConfigManager.h"
#include "TraceLogger.h"

Qwen2_5Model::Qwen2_5Model()
{
}

void Qwen2_5Model::PopulateWeightNames()
{
    ModelBase::PopulateWeightNames();

    _WeightNames[WeightType::BiasQ] = "model.layers.%d.self_attn.q_proj.bias";
    _WeightNames[WeightType::BiasK] = "model.layers.%d.self_attn.k_proj.bias";
    _WeightNames[WeightType::BiasV] = "model.layers.%d.self_attn.v_proj.bias";
}

torch::Tensor Qwen2_5Model::CalculateProjectionQ(const torch::Tensor &x, int layer)
{
    auto out = ModelBase::CalculateProjectionQ(x, layer);
    out = out + *_QBiasWeightKeys[layer];

    return out;
}

torch::Tensor Qwen2_5Model::CalculateProjectionK(const torch::Tensor &x, int layer)
{

    auto out = ModelBase::CalculateProjectionK(x, layer);
    out = out + *_KBiasWeightKeys[layer];

    return out;
}
torch::Tensor Qwen2_5Model::CalculateProjectionV(const torch::Tensor &x, int layer)
{
    auto out = ModelBase::CalculateProjectionV(x, layer);
    out = out + *_VBiasWeightKeys[layer];

    return out;
}