#include "Model.h"
#include "LlamaModel.h"
#include "MistralModel.h"
#include "Qwen2_5Model.h"
#include "Qwen3Model.h"
#include "ConfigManager.h"

std::unique_ptr<ModelBase> Model::_model = nullptr;

void Model::Init()
{
    switch (ConfigManager::ModelLoadedType)
    {
    case ModelType::LLama:
        _model = std::make_unique<LlamaModel>();
        break;

    case ModelType::Mistral:
        _model = std::make_unique<MistralModel>();
        break;

    case ModelType::Qwen2_5:
        _model = std::make_unique<Qwen2_5Model>();
        break;
    case ModelType::Qwen3:
        _model = std::make_unique<Qwen3Model>();
        break;

    default:
        throw std::runtime_error("Unrecognized/unsupported model type. Supported: Llama 3+, Mistral");
    }

    _model->Load();
}

void Model::BeginInfer()
{
    EnsureModelIsInitialized();

    _model->BeginInfer();
}

GenerationResult Model::Infer(torch::Tensor &tokenVectors, std::vector<int> &eosPerPrompt)
{
    EnsureModelIsInitialized();

    return _model->Infer(tokenVectors, eosPerPrompt);
}

void Model::EnsureModelIsInitialized()
{
    if (_model == nullptr)
        throw std::runtime_error("You must explicitly call Init() function at program startup once before calling any other functions");
}