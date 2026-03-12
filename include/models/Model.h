#pragma once

#include <memory>
#include "ModelBase.h"
#include "GenerationResult.h"

class Model
{
public:
    static void Init();
    static void BeginInfer();
    static GenerationResult Infer(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt);

private:
    static void EnsureModelIsInitialized();
    static std::unique_ptr<ModelBase> _model;
};