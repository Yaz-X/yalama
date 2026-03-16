/*
    YALAMA Runtime
    Copyright 2026 Yazeed Hamdan
    Licensed under the Apache License, Version 2.0
    See LICENSE file in project root.
*/

#include "QwenModel.h"
#include "ConfigManager.h"
#include "TraceLogger.h"

QwenModel::QwenModel()
{
}

void QwenModel::PopulateWeightNames()
{
    LlamaModel::PopulateWeightNames();

    _WeightNames[WeightType::BiasQ] = "model.layers.%d.self_attn.q_proj.bias";
    _WeightNames[WeightType::BiasK] = "model.layers.%d.self_attn.k_proj.bias";
    _WeightNames[WeightType::BiasV] = "model.layers.%d.self_attn.v_proj.bias";
}

torch::Tensor QwenModel::CalculateProjectionQ(const torch::Tensor &x, int layer)
{
    auto out = LlamaModel::CalculateProjectionQ(x, layer);
    out = out + *_QBiasWeightKeys[layer];

    return out;
}

torch::Tensor QwenModel::CalculateProjectionK(const torch::Tensor &x, int layer)
{

    auto out = LlamaModel::CalculateProjectionK(x, layer);
    out = out + *_KBiasWeightKeys[layer];

    return out;
}
torch::Tensor QwenModel::CalculateProjectionV(const torch::Tensor &x, int layer)
{
    auto out = LlamaModel::CalculateProjectionV(x, layer);
    out = out + *_VBiasWeightKeys[layer];

    return out;
}