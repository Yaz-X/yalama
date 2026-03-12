/*
    YALAMA Runtime
    Copyright 2026 Yazeed Hamdan
    Licensed under the Apache License, Version 2.0
    See LICENSE file in project root.
*/
#include "Helpers.h"
#include "TorchChecker.h"
#include "ModelBase.h"
#include "ConfigManager.h"
#include "TraceLogger.h"
#include <cmath>
#include <filesystem>
#include <cuda_runtime.h>
#include <algorithm>

int TraceLogger::_TraceStep = 0;

void ModelBase::Load()
{

    if (!_Weights.empty())
        return;

    std::cout << "Load Called..." << std::endl
              << std::flush;

    std::filesystem::path modelPath = ConfigManager::ModelPath;

    if (!std::filesystem::exists(modelPath) || !std::filesystem::is_directory(modelPath))
        throw std::runtime_error("Invalid model folder: " + modelPath.string());

    if (std::filesystem::is_empty(modelPath))
        throw std::runtime_error("Folder (" + modelPath.string() + ") is Empty, make sure the model path has safetensor files");

    PopulateWeightNames();

    if (_WeightNames.empty())
        throw std::runtime_error("_WeightNames map is empty, this needs to be populated in dervied class");

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

                    if (headerStr.find("\"data_offsets\"") != std::string::npos &&
                        headerStr.find("\"__metadata__\"") != std::string::npos &&
                        headerStr.find("\"format\":\"pt\"") != std::string::npos)
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
        auto safeTensor = load_safetensors(path);

        std::cout << "Loading Shard from: " + path << std::endl;
        shards.push_back(safeTensor);
    }

    if (!ConfigManager::IsShowLoadedWeights.value())
    {
        std::cout << "Please wait while loading Weights from shards, if you want to see the weight names while loading, supply (--showloadedweights 1) arg" << std::endl
                  << std::endl;

        for (auto &st : shards)
            totalTensorCount += st.tensors.size();
    }

    for (auto &st : shards)
    {
        for (auto &[name, meta] : st.tensors)
        {
            if (!_Weights.contains(name) && IsLoadWeight(name))
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
    }

    if (!ConfigManager::IsShowLoadedWeights.value())
        std::cout << std::endl;

    _FinalRMSNormWeight = &_Weights.at(_WeightNames.at(WeightType::FinalNorm));

    if (_Weights.contains(_WeightNames.at(WeightType::LMHead)))
    {
        _LMHeadWeight = &_Weights.at(_WeightNames.at(WeightType::LMHead));
    }
    else
    {
        _LMHeadWeight = &_Weights.at(_WeightNames.at(WeightType::Embedding));
    }

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
        _RMSNormPreAttentionWeightKeys[i] = &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::PreAttentionNorm), i));

        _RMSNormPostAttentionWeightKeys[i] = &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::PostAttentionNorm), i));

        _QWeightKeys[i] =
            &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::QProj), i));

        _KWeightKeys[i] =
            &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::KProj), i));

        _VWeightKeys[i] =
            &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::VProj), i));

        _WoWeightKeys[i] =
            &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::OProj), i));

        _Wgate[i] =
            &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::GateProj), i));

        _WUp[i] =
            &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::UpProj), i));

        _Wdown[i] =
            &_Weights.at(FormatWeightNameForLoading(_WeightNames.at(WeightType::DownProj), i));
    }

    InitKVCache();
    BuildRopeCache();
    CalculateMaxSequenceLength();

    torch::globalContext().setSDPUseFlash(true);
    torch::globalContext().setSDPUseMemEfficient(true);
    torch::globalContext().setSDPUseMath(false);

    torch::NoGradGuard no_grad;
}

bool ModelBase::IsLoadWeight(const std::string &name)
{
    bool isFound = false;

    for (auto &[type, templ] : _WeightNames)
    {
        if (templ.find("%d") != std::string::npos)
        {
            for (int i = 0; i < ConfigManager::NumLayers; i++)
            {
                std::string expanded = FormatWeightNameForLoading(templ, i);

                if (expanded == name)
                {
                    isFound = true;
                    break;
                }
            }
        }
        else
        {
            if (templ == name)
            {
                isFound = true;
            }
        }

        if (isFound)
        {
            break;
        }
    }

    return isFound;
}

std::string ModelBase::FormatWeightNameForLoading(const std::string &weightNameTemplate, int layer)
{
    return Replace(weightNameTemplate, "%d", std::to_string(layer));
}

void ModelBase::BeginInfer()
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

void ModelBase::CalculateMaxSequenceLength()
{
    size_t freeBytes = 0;
    size_t totalBytes = 0;

    cudaMemGetInfo(&freeBytes, &totalBytes);

    size_t bytesPerToken =
        ConfigManager::NumHeads *
        ConfigManager::HeadDim *
        sizeof(torch::BFloat16) *
        3 *
        ConfigManager::NumLayers;

    auto tokensPerAvailableMemory = static_cast<int>((freeBytes * 0.8) / bytesPerToken); // 0.8 = Use 80% of free memory

    ConfigManager::MaxSequenceLength = std::min(ConfigManager::MaxSequenceLength, tokensPerAvailableMemory);

    if (ConfigManager::MaxSequenceLength < 512)    
        throw std::runtime_error(
            "GPU memory too low: runtime cannot support the minimum context of 512 tokens");    

    std::cout
        << "Runtime Max Sequence Length After Calculation: "
        << ConfigManager::MaxSequenceLength
        << std::endl;
}

void ModelBase::InitKVCache()
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

void ModelBase::BuildRopeCache()
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

GenerationResult ModelBase::Infer(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt)
{
    GenerationResult result;
    AttentionCalculationResult attnCalculationResult;

    if (!_IsInferReady)
        throw std::runtime_error("Call BeginInfer() first");

    EnsureSequenceLength(tokenVectors, eosPerPrompt);

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

    if (result.Error != GenerationError::None)
        result.IsSuccess = false;

    return result;
}

void ModelBase::EnsureSequenceLength(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt)
{
    int64_t numOfTokens = tokenVectors.size(1);

    while (numOfTokens > ConfigManager::MaxSequenceLength)
    {
        if (eosPerPrompt.empty())
        {
            tokenVectors = tokenVectors.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});

            if (ConfigManager::IsServiceLoggingEnabled.value())
                std::cout << "Max Sequence Length of " << std::to_string(ConfigManager::MaxSequenceLength) << " exceeded, tokens size: "
                          << std::to_string(numOfTokens) << ", dropping oldes token" << std::endl;
        }
        else
        {
            int dropUntil = eosPerPrompt.front() + 1;

            tokenVectors = tokenVectors.index({torch::indexing::Slice(), torch::indexing::Slice(dropUntil, torch::indexing::None)});

            eosPerPrompt.erase(eosPerPrompt.begin());

            for (int &eos : eosPerPrompt)
                eos -= dropUntil;

            if (ConfigManager::IsServiceLoggingEnabled.value())
                std::cout << "Max Sequence Length of " << std::to_string(ConfigManager::MaxSequenceLength) << " exceeded, tokens size: "
                          << std::to_string(numOfTokens) << ", dropping oldes prompt" << std::endl;
        }

        numOfTokens = tokenVectors.size(1);
    }
}

AttentionCalculationResult ModelBase::CalculateInferenceAttention(const torch::Tensor &x, int layer)
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

        torch::Tensor attOut;

        if (!ConfigManager::IsKVCacheEnabled.value() || _LastKVCacheTokenIndex == 0)
        {
            int64_t Tq = q.size(2);
            int64_t Tk = k.size(2);            
            
            auto mask =
                torch::triu(
                    torch::ones({Tk, Tk}, q.options()),
                    1) *
                q.new_full({}, -65504.0);

            mask = mask.slice(1, 0, Tq)
                       .unsqueeze(0)
                       .unsqueeze(0);

            TraceLogger::Dump("attn_scores_with_mask", mask, TraceLogger::_TraceStep);

            attOut =
                torch::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    mask,
                    0.0,
                    false);
        }
        else
        {
            attOut =
                torch::scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    c10::nullopt,
                    0.0,
                    false);
        }

        TraceLogger::Dump("attn_heads", attOut, TraceLogger::_TraceStep);

        auto attOut2 = MergeHeadsAttentionOut(attOut);

        TraceLogger::Dump("attn_merged", attOut2, TraceLogger::_TraceStep);

        auto finalAttOut = MultiplyWithWo(attOut2, layer);

        TraceLogger::Dump("attOut_Wo", finalAttOut, TraceLogger::_TraceStep);

        result.CalculatedAttention = finalAttOut;
    }

    return result;
}

torch::Tensor ModelBase::CalculatePreAttentionRMSNorm(const torch::Tensor &x, int layer)
{
    return CalculateRMSNorm(x, _RMSNormPreAttentionWeightKeys[layer]);
}

torch::Tensor ModelBase::CalculatePostAttentionRMSNorm(const torch::Tensor &x, int layer)
{
    return CalculateRMSNorm(x, _RMSNormPostAttentionWeightKeys[layer]);
}

torch::Tensor ModelBase::CalculateFinalRMSNorm(const torch::Tensor &x)
{
    return CalculateRMSNorm(x, _FinalRMSNormWeight);
}

torch::Tensor ModelBase::CalculateRMSNorm(const torch::Tensor &x, const torch::Tensor *weight)
{

    TORCH_CHECKER(weight);
    TORCH_CHECKER(weight->dim() == 1);
    TORCH_CHECKER(weight->size(0) == x.size(-1));

    auto mean_sq = torch::mean(x * x, -1, true).to(torch::kFloat32);
    auto denom = torch::rsqrt(mean_sq + ConfigManager::Eps).to(x.dtype());
    auto normed = x * denom;

    return normed * (*weight).t();
}

torch::Tensor ModelBase::CalculateFFN(const torch::Tensor &x, int layer)
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

torch::Tensor ModelBase::CalculateProjectionQ(const torch::Tensor &x, int layer)
{
    TORCH_CHECKER(_QWeightKeys[layer]->dim() == 2);
    TORCH_CHECKER(_QWeightKeys[layer]->size(0) == x.size(-1));

    auto &W = *_QWeightKeys[layer];
    return torch::matmul(x, W.t());
}

torch::Tensor ModelBase::CalculateProjectionK(const torch::Tensor &x, int layer)
{

    auto &W = *_KWeightKeys[layer];
    return torch::matmul(x, W.t());
}

torch::Tensor ModelBase::CalculateProjectionV(const torch::Tensor &x, int layer)
{
    auto &W = *_VWeightKeys[layer];
    return torch::matmul(x, W.t());
}

torch::Tensor ModelBase::ApplyInferencingRope(const torch::Tensor &x, int start_pos, std::string name)
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

torch::Tensor ModelBase::MergeHeadsAttentionOut(const torch::Tensor &attentionOut)
{
    // [B, H, T, D]
    auto y = attentionOut.transpose(1, 2).contiguous();

    // [B, T, H, D]
    auto merged = y.reshape({y.size(0), y.size(1), y.size(2) * y.size(3)});

    // [B, T, D]
    return merged;
}

torch::Tensor ModelBase::MultiplyWithWo(const torch::Tensor &mergedAttentionOut, int layer)
{
    TORCH_CHECKER(_WoWeightKeys[layer]->size(0) == mergedAttentionOut.size(-1));

    return torch::matmul(mergedAttentionOut, _WoWeightKeys[layer]->t());
}

torch::Tensor ModelBase::GetEmbeddings(const torch::Tensor &tokensIds)
{
    TraceLogger::Dump("token_ids", tokensIds, TraceLogger::_TraceStep);

    auto W = _Weights.at("model.embed_tokens.weight");
    auto out = torch::embedding(W, tokensIds).to(torch::kBFloat16);

    TraceLogger::Dump("embeddings", out, TraceLogger::_TraceStep);

    return out;
}

bool ModelBase::EnsureKVCapacity(int needed)
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
