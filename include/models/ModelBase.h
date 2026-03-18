#pragma once

#include <string>
#include <unordered_map>
#include <torch/torch.h>
#include "GenerationResult.h"
#include "AttentionCalculationResult.h"
#include "safetensors.h"
#include "WeightType.h"

class ModelBase
{

public:
    ModelBase() = default;
    virtual ~ModelBase() = default;

    void BeginInfer();
    GenerationResult Infer(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt);
    virtual void Load();

protected:
    bool _IsInferReady = false;
    int _LastKVCacheTokenIndex = 0;
    int _KVCapacityInTokens;
    torch::Tensor _RopeCos;
    torch::Tensor _RopeSin;
    std::unordered_map<WeightType, std::string> _WeightNames;

    std::vector<torch::Tensor> _Kcache;
    std::vector<torch::Tensor> _Vcache;
    std::vector<const torch::Tensor *> _RMSNormPreAttentionWeightKeys;
    std::vector<const torch::Tensor *> _RMSNormPostAttentionWeightKeys;
    std::vector<const torch::Tensor *> _QWeightKeys;
    std::vector<const torch::Tensor *> _KWeightKeys;
    std::vector<const torch::Tensor *> _VWeightKeys;

    std::vector<const torch::Tensor *> _QBiasWeightKeys;
    std::vector<const torch::Tensor *> _KBiasWeightKeys;
    std::vector<const torch::Tensor *> _VBiasWeightKeys;

    std::vector<const torch::Tensor *> _QNormWeights;
    std::vector<const torch::Tensor *> _KNormWeights;

    std::vector<const torch::Tensor *> _WoWeightKeys;
    std::vector<const torch::Tensor *> _Wgate;
    std::vector<const torch::Tensor *> _WUp;
    std::vector<const torch::Tensor *> _Wdown;

    std::unordered_map<std::string, torch::Tensor> _Weights;
    const torch::Tensor *_LMHeadWeight;
    const torch::Tensor *_FinalRMSNormWeight;

    virtual void PopulateWeightNames();
    virtual void BuildRopeCache();

    virtual bool IsLoadWeight(const std::string &name);
    virtual void InitKVCache();
    virtual void CalculateMaxSequenceLength();
    virtual bool EnsureKVCapacity(int memoryNeeded);
    virtual void EnsureSequenceLength(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt);
    virtual AttentionCalculationResult CalculateInferenceAttention(const torch::Tensor &x, int layer);
    virtual void ExpandKV(torch::Tensor &k, torch::Tensor &v);
    virtual torch::Tensor ApplyScaledDotProductAttention(torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &mask);
    virtual torch::Tensor GetEmbeddings(const torch::Tensor &ids);
    virtual torch::Tensor CalculatePreAttentionRMSNorm(const torch::Tensor &x, int layer);
    virtual torch::Tensor CalculatePostAttentionRMSNorm(const torch::Tensor &x, int layer);
    virtual torch::Tensor CalculateFinalRMSNorm(const torch::Tensor &x);
    virtual torch::Tensor CalculateRMSNorm(const torch::Tensor &x, const torch::Tensor *weight);
    virtual torch::Tensor CalculateProjectionQ(const torch::Tensor &x, int layer);
    virtual torch::Tensor CalculateProjectionK(const torch::Tensor &x, int layer);
    virtual torch::Tensor CalculateProjectionV(const torch::Tensor &x, int layer);

    virtual torch::Tensor CalculatePostProjectionQ(const torch::Tensor &q, int layer);
    virtual torch::Tensor CalculatePostProjectionK(const torch::Tensor &k, int layer);
    virtual torch::Tensor CalculatePostProjectionV(const torch::Tensor &v, int layer);

    virtual torch::Tensor ApplyInferencingRope(const torch::Tensor &x, int start_pos, std::string name);
    virtual torch::Tensor MergeHeadsAttentionOut(const torch::Tensor &attentionOut);
    virtual torch::Tensor MultiplyWithWo(const torch::Tensor &mergedAttentionOut, int layer);
    virtual torch::Tensor CalculateFFN(const torch::Tensor &x, int layer);
    virtual std::string FormatWeightNameForLoading(const std::string &weightNameTemplate, int layer);
};
