/*
    YALAMA Runtime
    Copyright 2026 Yazeed Hamdan
    Licensed under the Apache License, Version 2.0
    See LICENSE file in project root.
*/

#include "LlamaModel.h"

LlamaModel::LlamaModel()
{     
}

void LlamaModel::PopulateWeightNames()
{
    _WeightNames[WeightType::Embedding] = "model.embed_tokens.weight";
    _WeightNames[WeightType::PreAttentionNorm] = "model.layers.%d.input_layernorm.weight";
    _WeightNames[WeightType::QProj] = "model.layers.%d.self_attn.q_proj.weight";
    _WeightNames[WeightType::KProj] = "model.layers.%d.self_attn.k_proj.weight";
    _WeightNames[WeightType::VProj] = "model.layers.%d.self_attn.v_proj.weight";
    _WeightNames[WeightType::OProj] = "model.layers.%d.self_attn.o_proj.weight";
    _WeightNames[WeightType::PostAttentionNorm] = "model.layers.%d.post_attention_layernorm.weight";
    _WeightNames[WeightType::GateProj] = "model.layers.%d.mlp.gate_proj.weight";
    _WeightNames[WeightType::UpProj] = "model.layers.%d.mlp.up_proj.weight";
    _WeightNames[WeightType::DownProj] = "model.layers.%d.mlp.down_proj.weight";
    _WeightNames[WeightType::FinalNorm] = "model.norm.weight";
    _WeightNames[WeightType::LMHead] = "lm_head.weight";
}