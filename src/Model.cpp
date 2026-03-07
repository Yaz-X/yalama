/*
    YALAMA Runtime
    Copyright 2026 Yazeed Hamdan
    Licensed under the Apache License, Version 2.0
    See LICENSE file in project root.
*/
#include "Helpers.h"
#include "TorchChecker.h"
#include "Model.h"
#include "ConfigManager.h"
#include "TraceLogger.h"
#include <cmath>
#include <filesystem>

int TraceLogger::_TraceStep = 0;

Model::Model()
{
    Load();
}

void Model::Load()
{
    std::filesystem::path modelPath = ConfigManager::ModelPath;

    if (!std::filesystem::exists(modelPath) || !std::filesystem::is_directory(modelPath))
        throw std::runtime_error("Invalid model folder: " + modelPath.string());

    if (std::filesystem::is_empty(modelPath))
        throw std::runtime_error("Folder (" + modelPath.string() + ") is Empty, make sure the model path has safetensor files");

    std::vector<std::string> weightFiles;
    size_t totalTensorCount = 0;
    size_t loadedTensorCount = 0;

    for (const auto &entry : std::filesystem::directory_iterator(ConfigManager::ModelPath))
    {
        if (entry.is_regular_file())
        {
            auto path = entry.path();

            if (path.extension() == ".safetensors")
            {
                std::ifstream file(path, std::ios::binary);

                if (file)
                {
                    uint64_t header_len;
                    file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));

                    std::vector<char> header(header_len);
                    file.read(header.data(), header_len);

                    std::string headerStr(header.begin(), header.end());

                    if (headerStr.find("\"data_offsets\"") != std::string::npos)
                        weightFiles.push_back(path.string());
                }
            }
        }
    }

    if (weightFiles.empty())
        throw std::runtime_error("Folder (" + modelPath.string() + ") has no model weight files, make sure the model path has safetensor files");

    std::sort(weightFiles.begin(), weightFiles.end());
    std::vector<SafeTensors> shards;

    // Load all shards
    for (const auto &path : weightFiles)
    {
        std::cout << "Loading Shard from: " + path << std::endl;
        shards.push_back(load_safetensors(path));
    }

    if (!ConfigManager::IsShowLoadedWeights.value())
    {
        std::cout << "Please wait while loading Weights from shards, if you want to see the weight names while loading, supply (--isShowLoadingWeights 1) arg" << std::endl
                  << std::endl;

        for (auto &st : shards)
            totalTensorCount += st.tensors.size();
    }

    for (auto &st : shards)
    {
        for (auto &[name, meta] : st.tensors)
        {
            char *base = reinterpret_cast<char *>(st.data);
            char *ptr = base + meta.offset0;

            std::vector<int64_t> dims = meta.shape;

            if (ConfigManager::IsShowLoadedWeights.value())
                std::cout << "Loading Weight: " << name << std::endl;

            auto cpu = torch::from_blob(
                           (void *)ptr,
                           torch::IntArrayRef(dims),
                           torch::kBFloat16)
                           .clone();

            if (ConfigManager::IsShowLoadedWeights.value())
                std::cout << "Move " << name << " to GPU" << std::endl;

            loadedTensorCount++;

            _Weights[name] = cpu.to(torch::kCUDA);

            if (!ConfigManager::IsShowLoadedWeights.value())
            {
                float progress =
                    static_cast<float>(loadedTensorCount) /
                    static_cast<float>(totalTensorCount);

                ShowProgressBar(progress);
            }
        }
    }

    if (!ConfigManager::IsShowLoadedWeights.value())
        std::cout << std::endl;

    _FinalRMSNormWeight = &_Weights.at("model.norm.weight");

    if (_Weights.contains("lm_head.weight"))
        _LMHeadWeight = &_Weights.at("lm_head.weight");
    else
        _LMHeadWeight = &_Weights.at("model.embed_tokens.weight");

    TORCH_CHECKER(_LMHeadWeight);

    _RMSNormPreAttentionWeightKeys.resize(ConfigManager::NumLayers);
    _RMSNormPostAttentionWeightKeys.resize(ConfigManager::NumLayers);
    _QWeightKeys.resize(ConfigManager::NumLayers);
    _KWeightKeys.resize(ConfigManager::NumLayers);
    _VWeightKeys.resize(ConfigManager::NumLayers);
    _WoWeightKeys.resize(ConfigManager::NumLayers);
    _Wgate.resize(ConfigManager::NumLayers);
    _WUp.resize(ConfigManager::NumLayers);
    _Wdown.resize(ConfigManager::NumLayers);

    for (int i = 0; i < ConfigManager::NumLayers; i++)
    {
        _RMSNormPreAttentionWeightKeys[i] = &_Weights.at("model.layers." + std::to_string(i) + ".input_layernorm.weight");
        _RMSNormPostAttentionWeightKeys[i] = &_Weights.at("model.layers." + std::to_string(i) + ".post_attention_layernorm.weight");
        _QWeightKeys[i] = &_Weights.at("model.layers." + std::to_string(i) + ".self_attn.q_proj.weight");
        _KWeightKeys[i] = &_Weights.at("model.layers." + std::to_string(i) + ".self_attn.k_proj.weight");
        _VWeightKeys[i] = &_Weights.at("model.layers." + std::to_string(i) + ".self_attn.v_proj.weight");
        _WoWeightKeys[i] = &_Weights.at("model.layers." + std::to_string(i) + ".self_attn.o_proj.weight");

        _Wgate[i] = &_Weights.at("model.layers." + std::to_string(i) + ".mlp.gate_proj.weight");
        _WUp[i] = &_Weights.at("model.layers." + std::to_string(i) + ".mlp.up_proj.weight");
        _Wdown[i] = &_Weights.at("model.layers." + std::to_string(i) + ".mlp.down_proj.weight");
    }

    std::cout << "Model is Running..." << std::endl;

    InitKVCache();
    BuildRopeCache();

    torch::globalContext().setSDPUseFlash(true);
    torch::globalContext().setSDPUseMemEfficient(true);
    torch::globalContext().setSDPUseMath(false);

    torch::NoGradGuard no_grad;
}

void Model::BeginInfer()
{
    TraceLogger::_TraceStep = 0;
    _IsInferReady = true;
    _LastKVCacheTokenIndex = 0;

    if (ConfigManager::IsKVCacheEnabled.value())
    {
        for (int i = 0; i < ConfigManager::NumLayers; i++)
        {
            _Kcache[i].zero_();
            _Vcache[i].zero_();
        }
    }
}

void Model::InitKVCache()
{

    if (ConfigManager::IsKVCacheEnabled.value())
    {
        float gb = ConfigManager::KVCacheSizeInGB;

        int B = 1;
        int H = ConfigManager::NumKVHeads;
        int D = ConfigManager::HiddenSize / ConfigManager::NumHeads;

        size_t bytes_per_token =
            B * H * D * sizeof(torch::BFloat16);

        ssize_t total_bytes = (gb * 1024ULL * 1024ULL * 1024ULL);

        _KVCapacityInTokens = (total_bytes / bytes_per_token) / ConfigManager::NumLayers;
        _KVCapacityInTokens = _KVCapacityInTokens / 2; // per each K and each V

        auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        _Kcache.resize(ConfigManager::NumLayers);
        _Vcache.resize(ConfigManager::NumLayers);

        for (int i = 0; i < ConfigManager::NumLayers; i++)
        {
            _Kcache[i] = torch::zeros({B, H, _KVCapacityInTokens, D}, opts);
            _Vcache[i] = torch::zeros({B, H, _KVCapacityInTokens, D}, opts);
        }
    }
}

void Model::BuildRopeCache()
{
    int maxT = ConfigManager::MaxSequenceLength;
    int D = ConfigManager::HeadDim;
    int half = D / 2;
    float base = ConfigManager::RopeTheta;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto idx = torch::arange(0, half, opts);
    auto pos = torch::arange(0, maxT, opts);

    auto inv_freq = 1.0f / torch::pow(base, idx / (float)half);

    auto freqs = torch::matmul(
        pos.unsqueeze(1),
        inv_freq.unsqueeze(0));

    auto emb = torch::cat({freqs, freqs}, -1);

    _RopeCos = emb.cos().to(torch::kBFloat16).unsqueeze(0).unsqueeze(0);
    _RopeSin = emb.sin().to(torch::kBFloat16).unsqueeze(0).unsqueeze(0);
}

GenerationResult Model::Infer(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt)
{
    GenerationResult result;
    AttentionCalculationResult attnCalculationResult;

    if (!_IsInferReady)
        throw std::runtime_error("Call BeginInfer() first");

    bool isValid = EnsureSequenceLength(tokenVectors, eosPerPrompt);

    if (!isValid)
        result.Error = GenerationError::SequenceLengthExceeded;
    else
    {
        torch::Tensor x = GetEmbeddings(tokenVectors);

        TORCH_CHECKER(x.dim() == 3);

        for (int layer = 0; layer < ConfigManager::NumLayers; layer++)
        {
            TraceLogger::Dump("layer_input", x, TraceLogger::_TraceStep);

            auto xNorm = CalculatePreAttentionRMSNorm(x, layer);

            TraceLogger::Dump("rmsnorm_pre_attn", xNorm, TraceLogger::_TraceStep);

            attnCalculationResult = CalculateInferenceAttention(xNorm, layer);

            if (!attnCalculationResult.IsKVCacheCapacityValid)
            {
                result.Error = GenerationError::KVCacheExceeded;
                break;
            }

            x = x + attnCalculationResult.CalculatedAttention;
            TraceLogger::Dump("resid_attn", x, TraceLogger::_TraceStep);

            auto xNorm2 = CalculatePostAttentionRMSNorm(x, layer);

            TraceLogger::Dump("rmsnorm_post_attn", xNorm2, TraceLogger::_TraceStep);

            auto ffnOut = CalculateFFN(xNorm2, layer); // BF16

            TraceLogger::Dump("ffn_out", ffnOut, TraceLogger::_TraceStep);

            x = x + ffnOut;

            TraceLogger::Dump("resid_ffn_out", x, TraceLogger::_TraceStep);
        }

        if (attnCalculationResult.IsKVCacheCapacityValid)
        {

            TraceLogger::Dump("before_final_rms", x, TraceLogger::_TraceStep);

            x = CalculateFinalRMSNorm(x);

            TraceLogger::Dump("final_rms_norm", x, TraceLogger::_TraceStep);

            auto logits = torch::matmul(x, _LMHeadWeight->t());

            TraceLogger::Dump("final_logits", logits, TraceLogger::_TraceStep);

            if (ConfigManager::IsKVCacheEnabled.value())
            {
                // Prefill
                if (_LastKVCacheTokenIndex == 0)
                    _LastKVCacheTokenIndex = tokenVectors.size(1) - 1;
                else // Decode
                    _LastKVCacheTokenIndex++;
            }

            result.Logits = logits;
        }
    }

    if (result.Error != GenerationError::None)
        result.IsSuccess = false;

    return result;
}

bool Model::EnsureSequenceLength(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt)
{
    bool isValid = true;

    while (tokenVectors.size(1) > ConfigManager::MaxSequenceLength)
    {
        if (tokenVectors.size(1) == 0)
        {
            isValid = false;
            break;
        }

        if (eosPerPrompt.empty())
        {
            tokenVectors = tokenVectors.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});
        }
        else
        {
            int dropUntil = eosPerPrompt.front() + 1;

            tokenVectors = tokenVectors.index({torch::indexing::Slice(), torch::indexing::Slice(dropUntil, torch::indexing::None)});

            eosPerPrompt.erase(eosPerPrompt.begin());

            for (int &eos : eosPerPrompt)
                eos -= dropUntil;
        }
    }

    return isValid;
}

AttentionCalculationResult Model::CalculateInferenceAttention(const torch::Tensor &x, int layer)
{
    AttentionCalculationResult result;

    auto q = CalculateProjectionQ(x, layer);
    auto k = CalculateProjectionK(x, layer);
    auto v = CalculateProjectionV(x, layer);

    TraceLogger::Dump("q_raw", q, TraceLogger::_TraceStep);
    TraceLogger::Dump("k_raw", k, TraceLogger::_TraceStep);
    TraceLogger::Dump("v_raw", v, TraceLogger::_TraceStep);

    q = q.view({q.size(0), q.size(1), ConfigManager::NumHeads, ConfigManager::HeadDim})
            .transpose(1, 2);

    k = k.view({k.size(0), k.size(1), ConfigManager::NumKVHeads, ConfigManager::HeadDim})
            .transpose(1, 2);

    v = v.view({v.size(0), v.size(1), ConfigManager::NumKVHeads, ConfigManager::HeadDim})
            .transpose(1, 2);

    TraceLogger::Dump("q_t", q, TraceLogger::_TraceStep);
    TraceLogger::Dump("k_t", k, TraceLogger::_TraceStep);
    TraceLogger::Dump("v_t", v, TraceLogger::_TraceStep);

    TORCH_CHECKER(k.size(1) == ConfigManager::NumKVHeads);
    TORCH_CHECKER(v.size(1) == ConfigManager::NumKVHeads);

    if (_LastKVCacheTokenIndex == 0) // KVCache disabled or Prefill with KVCache
    {
        q = ApplyInferencingRope(q, 0, "q");
        k = ApplyInferencingRope(k, 0, "k");
    }

    if (ConfigManager::IsKVCacheEnabled.value())
    {
        if (_LastKVCacheTokenIndex == 0) // Prefill KVCache Write
        {
            TraceLogger::DumpStr("Prefill_CurPos_Layer" + std::to_string(layer), std::to_string(_LastKVCacheTokenIndex));

            int T0 = k.size(2);

            result.IsKVCacheCapacityValid = EnsureKVCapacity(T0);

            if (result.IsKVCacheCapacityValid)
            {
                _Kcache[layer]
                    .narrow(2, 0, T0)
                    .copy_(k);

                _Vcache[layer]
                    .narrow(2, 0, T0)
                    .copy_(v);
            }
        }
        else // Decode
        {
            TraceLogger::DumpStr("Prefill_CurPos_Layer" + std::to_string(layer), std::to_string(_LastKVCacheTokenIndex));

            result.IsKVCacheCapacityValid = EnsureKVCapacity(_LastKVCacheTokenIndex + 1);

            if (result.IsKVCacheCapacityValid)
            {
                auto k_last = k;
                auto v_last = v;

                q = ApplyInferencingRope(q, _LastKVCacheTokenIndex, "q");
                k_last = ApplyInferencingRope(k_last, _LastKVCacheTokenIndex, "k");

                _Kcache[layer].select(2, _LastKVCacheTokenIndex + 1).copy_(k_last.squeeze(2));
                _Vcache[layer].select(2, _LastKVCacheTokenIndex + 1).copy_(v_last.squeeze(2));

                k = _Kcache[layer].narrow(2, 0, _LastKVCacheTokenIndex + 2);
                v = _Vcache[layer].narrow(2, 0, _LastKVCacheTokenIndex + 2);
            }
        }
    }

    if (result.IsKVCacheCapacityValid)
    {
        TraceLogger::Dump("q_rope", q, TraceLogger::_TraceStep);
        TraceLogger::Dump("k_rope", k, TraceLogger::_TraceStep);

        int rep = ConfigManager::NumHeads / ConfigManager::NumKVHeads;

        k = k.unsqueeze(2);
        v = v.unsqueeze(2);

        k = k.expand({k.size(0), k.size(1), rep, k.size(3), k.size(4)});
        v = v.expand({v.size(0), v.size(1), rep, v.size(3), v.size(4)});

        k = k.reshape({k.size(0), ConfigManager::NumHeads, k.size(3), k.size(4)});
        v = v.reshape({v.size(0), ConfigManager::NumHeads, v.size(3), v.size(4)});

        TraceLogger::Dump("q_exp", q, TraceLogger::_TraceStep);
        TraceLogger::Dump("k_exp", k, TraceLogger::_TraceStep);
        TraceLogger::Dump("v_exp", v, TraceLogger::_TraceStep);

        TORCH_CHECKER(k.size(1) == ConfigManager::NumHeads);
        TORCH_CHECKER(v.size(1) == ConfigManager::NumHeads);

        auto B = q.size(0);
        auto Hq = q.size(1);
        auto D = q.size(3);

        auto Hkv = k.size(1);

        TORCH_CHECKER(Hq == ConfigManager::NumHeads);
        TORCH_CHECKER(Hkv == ConfigManager::NumHeads);
        TORCH_CHECKER(k.size(0) == B && k.size(3) == D);
        TORCH_CHECKER(v.size(0) == B && v.size(3) == D);

        auto kt = k.transpose(2, 3);
        TraceLogger::Dump("k_t2", kt, TraceLogger::_TraceStep);

        auto scores = CalculateScores(q, kt, layer);

        TORCH_CHECKER(kt.size(0) == B && kt.size(1) == Hq && kt.size(2) == D);
        TORCH_CHECKER(scores.size(0) == B && scores.size(1) == Hq);

        auto probs = ApplySoftMaxAndGetProbs(scores);

        TraceLogger::Dump("attn_softmax", probs, TraceLogger::_TraceStep);

        auto attOut = MultiplyProbsWithV(probs, v);

        TraceLogger::Dump("attn_heads", attOut, TraceLogger::_TraceStep);

        auto attOut2 = MergeHeadsAttentionOut(attOut);

        TraceLogger::Dump("attn_merged", attOut2, TraceLogger::_TraceStep);

        auto finalAttOut = MultiplyWithWo(attOut2, layer);

        TraceLogger::Dump("attOut_Wo", finalAttOut, TraceLogger::_TraceStep);

        result.CalculatedAttention = finalAttOut;
    }

    return result;
}

torch::Tensor Model::CalculatePreAttentionRMSNorm(const torch::Tensor &x, int layer)
{
    return CalculateRMSNorm(x, _RMSNormPreAttentionWeightKeys[layer]);
}

torch::Tensor Model::CalculatePostAttentionRMSNorm(const torch::Tensor &x, int layer)
{
    return CalculateRMSNorm(x, _RMSNormPostAttentionWeightKeys[layer]);
}

torch::Tensor Model::CalculateFinalRMSNorm(const torch::Tensor &x)
{
    return CalculateRMSNorm(x, _FinalRMSNormWeight);
}

torch::Tensor Model::CalculateRMSNorm(const torch::Tensor &x, const torch::Tensor *weight)
{

    TORCH_CHECKER(weight);
    TORCH_CHECKER(weight->dim() == 1);
    TORCH_CHECKER(weight->size(0) == x.size(-1));

    auto mean_sq = torch::mean(x * x, -1, true).to(torch::kFloat32);
    auto denom = torch::rsqrt(mean_sq + ConfigManager::Eps).to(x.dtype());
    auto normed = x * denom;

    return normed * (*weight).t();
}

torch::Tensor Model::CalculateFFN(const torch::Tensor &x, int layer)
{
    const auto &Wgate = *_Wgate[layer];
    const auto &Wup = *_WUp[layer];
    const auto &Wdown = *_Wdown[layer];

    auto gate = torch::matmul(x, Wgate.t());
    auto up = torch::matmul(x, Wup.t());

    auto activation = torch::silu(gate);
    auto h = activation * up;

    auto down = torch::matmul(h, Wdown.t());

    return down;
}

torch::Tensor Model::CalculateProjectionQ(const torch::Tensor &x, int layer)
{
    TORCH_CHECKER(_QWeightKeys[layer]->dim() == 2);
    TORCH_CHECKER(_QWeightKeys[layer]->size(0) == x.size(-1));

    auto &W = *_QWeightKeys[layer];
    return torch::matmul(x, W.t());
}

torch::Tensor Model::CalculateProjectionK(const torch::Tensor &x, int layer)
{

    auto &W = *_KWeightKeys[layer];
    return torch::matmul(x, W.t());
}

torch::Tensor Model::CalculateProjectionV(const torch::Tensor &x, int layer)
{
    auto &W = *_VWeightKeys[layer];
    return torch::matmul(x, W.t());
}

torch::Tensor Model::ApplyInferencingRope(const torch::Tensor &x, int start_pos, std::string name)
{
    auto T = x.size(2);
    auto D = x.size(3);

    int half = D / 2;

    auto cos = _RopeCos.narrow(2, start_pos, T);
    auto sin = _RopeSin.narrow(2, start_pos, T);

    TraceLogger::Dump("rope_sin", sin, TraceLogger::_TraceStep);
    TraceLogger::Dump("rope_cos", cos, TraceLogger::_TraceStep);

    auto xcos = (x * cos);

    TraceLogger::Dump(name + "cos", sin, TraceLogger::_TraceStep);

    auto x1 = x.narrow(3, 0, half);
    auto x2 = x.narrow(3, half, D - half);

    TraceLogger::Dump("rotate_half_x1_" + name, x1, TraceLogger::_TraceStep);
    TraceLogger::Dump("rotate_half_x2_" + name, x2, TraceLogger::_TraceStep);

    auto rot = torch::cat({-x2, x1}, -1);

    TraceLogger::Dump("rotate_half_rot_" + name, rot, TraceLogger::_TraceStep);

    auto xsin = (rot * sin);

    TraceLogger::Dump(name + "sin", xsin, TraceLogger::_TraceStep);

    auto out = xcos + xsin;

    return out;
}

torch::Tensor Model::CalculateScores(const torch::Tensor &q, const torch::Tensor &kt, int layer)
{
    // QK^T
    auto scores = torch::matmul(q, kt);

    TraceLogger::Dump("attn_scores_before_scale", scores, TraceLogger::_TraceStep);

    // scale ONCE
    float scale = 1.0f / std::sqrt((float)ConfigManager::HeadDim);
    scores = scores * scale;

    TraceLogger::Dump("attn_scores_after_scale", scores, TraceLogger::_TraceStep);

    if (!ConfigManager::IsKVCacheEnabled.value() || _LastKVCacheTokenIndex == 0)
    {
        // scores_f: [B, H, Tq, Tk]
        int64_t Tq = scores.size(-2);
        int64_t Tk = scores.size(-1);

        auto mask = torch::triu(
                        torch::ones({Tk, Tk}, scores.options()),
                        1) *
                    scores.new_full({}, -65504.0);

        mask = mask.slice(1, 0, Tq)
                   .unsqueeze(0)
                   .unsqueeze(0);

        scores = scores + mask;

        TraceLogger::Dump("attn_scores_with_mask", scores, TraceLogger::_TraceStep);
    }

    return scores;
}
torch::Tensor Model::ApplySoftMaxAndGetProbs(const torch::Tensor &scores)
{
    auto probs = torch::softmax(scores.to(torch::kFloat32), -1).to(scores.dtype());

    return probs;
}
torch::Tensor Model::MultiplyProbsWithV(const torch::Tensor &probs, const torch::Tensor &v)
{
    auto v2 = v;

    auto out = torch::matmul(probs, v2);

    return out;
}

torch::Tensor Model::MergeHeadsAttentionOut(const torch::Tensor &attentionOut)
{
    // [B, H, T, D]
    auto y = attentionOut.transpose(1, 2).contiguous();

    // [B, T, H, D]
    auto merged = y.reshape({y.size(0), y.size(1), y.size(2) * y.size(3)});

    // [B, T, D]
    return merged;
}

torch::Tensor Model::MultiplyWithWo(const torch::Tensor &mergedAttentionOut, int layer)
{
    TORCH_CHECKER(_WoWeightKeys[layer]->size(0) == mergedAttentionOut.size(-1));

    return torch::matmul(mergedAttentionOut, _WoWeightKeys[layer]->t());
}

torch::Tensor Model::GetEmbeddings(const torch::Tensor &tokensIds)
{
    TraceLogger::Dump("token_ids", tokensIds, TraceLogger::_TraceStep);

    auto W = _Weights.at("model.embed_tokens.weight");
    auto out = torch::embedding(W, tokensIds).to(torch::kBFloat16);

    TraceLogger::Dump("embeddings", out, TraceLogger::_TraceStep);

    return out;
}

bool Model::EnsureKVCapacity(int needed)
{
    bool isValid = true;

    if (ConfigManager::IsKVCacheEnabled.value())
    {
        int availableSlots = _KVCapacityInTokens - _LastKVCacheTokenIndex;

        if (needed > availableSlots)
            isValid = false;
    }

    return isValid;
}
