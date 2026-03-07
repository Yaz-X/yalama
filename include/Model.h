#pragma once

#include <string>
#include <unordered_map>
#include <torch/torch.h>
#include "GenerationResult.h"
#include "AttentionCalculationResult.h"
#include "safetensors.h"

class Model
{
public:
    Model();
    void BeginInfer();
    GenerationResult Infer(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt);
    
private:
    bool _IsInferReady = false;
    int _LastKVCacheTokenIndex = 0;
    int _KVCapacityInTokens;
    torch::Tensor _RopeCos;
    torch::Tensor _RopeSin;
    
    void BuildRopeCache();
    std::vector<torch::Tensor> _Kcache;
    std::vector<torch::Tensor> _Vcache;
    std::vector<const torch::Tensor *> _RMSNormPreAttentionWeightKeys;
    std::vector<const torch::Tensor *> _RMSNormPostAttentionWeightKeys;
    std::vector<const torch::Tensor *> _QWeightKeys;
    std::vector<const torch::Tensor *> _KWeightKeys;
    std::vector<const torch::Tensor *> _VWeightKeys;
    std::vector<const torch::Tensor *> _WoWeightKeys;
    std::vector<const torch::Tensor *> _Wgate;
    std::vector<const torch::Tensor *> _WUp;
    std::vector<const torch::Tensor *> _Wdown;

    std::unordered_map<std::string, torch::Tensor> _Weights;
    const torch::Tensor *_LMHeadWeight;
    const torch::Tensor *_FinalRMSNormWeight;

    void Load();
    void InitKVCache();
    bool EnsureKVCapacity(int memoryNeeded);
    bool EnsureSequenceLength(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt);

    AttentionCalculationResult CalculateInferenceAttention(const torch::Tensor &x, int layer);
    torch::Tensor GetEmbeddings(const torch::Tensor &ids);
    torch::Tensor CalculatePreAttentionRMSNorm(const torch::Tensor &x, int layer);
    torch::Tensor CalculatePostAttentionRMSNorm(const torch::Tensor &x, int layer);
    torch::Tensor CalculateFinalRMSNorm(const torch::Tensor &x);
    torch::Tensor CalculateRMSNorm(const torch::Tensor &x, const torch::Tensor *weight);
    torch::Tensor CalculateProjectionQ(const torch::Tensor &x, int layer);
    torch::Tensor CalculateProjectionK(const torch::Tensor &x, int layer);
    torch::Tensor CalculateProjectionV(const torch::Tensor &x, int layer);
    torch::Tensor ApplyInferencingRope(const torch::Tensor &x, int start_pos, std::string name);
    torch::Tensor CalculateScores(const torch::Tensor &q, const torch::Tensor &kt, int layer);
    torch::Tensor ApplySoftMaxAndGetProbs(const torch::Tensor &scores);
    torch::Tensor MultiplyProbsWithV(const torch::Tensor &probs, const torch::Tensor &v); // Attention Out
    torch::Tensor MergeHeadsAttentionOut(const torch::Tensor &attentionOut);
    torch::Tensor MultiplyWithWo(const torch::Tensor &mergedAttentionOut, int layer);
    torch::Tensor CalculateFFN(const torch::Tensor &x, int layer);    
};
