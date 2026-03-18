#include "Tokenizer.h"
#include "ConfigManager.h"
#include "LlamaTokenizer.h"
#include "MistralTokenizer.h"
#include "QwenTokenizer.h"

std::unique_ptr<TokenizerBase> Tokenizer::_tokenizer = nullptr;

void Tokenizer::Init()
{
    switch (ConfigManager::ModelLoadedType)
    {
    case ModelType::LLama:
        _tokenizer = std::make_unique<LlamaTokenizer>();
        break;

    case ModelType::Mistral:
        _tokenizer = std::make_unique<MistralTokenizer>();
        break;
    case ModelType::Qwen2_5:
    case ModelType::Qwen3:
        _tokenizer = std::make_unique<QwenTokenizer>();
        break;
        
    default:
        throw std::runtime_error("Unsupported model type for tokenizer");
    }

    _tokenizer->Init();
}

std::vector<int64_t> Tokenizer::EncodeWithChatTemplate(std::string &text, bool isAddSystemHeaderInChatTemplate)
{
    return _tokenizer->EncodeWithChatTemplate(text, isAddSystemHeaderInChatTemplate);
}

std::vector<int64_t> Tokenizer::EncodeWithoutChatTemplate(std::string &text)
{
    return _tokenizer->EncodeWithoutChatTemplate(text);
}

std::string Tokenizer::Decode(const int64_t tokenID)
{
    return _tokenizer->Decode(tokenID);
}

std::unordered_map<int, std::string> Tokenizer::GetSpecialTokens()
{
    return _tokenizer->GetSpecialTokens();
}

std::unordered_map<std::string, std::vector<int64_t>> Tokenizer::GetEncodedSpecialTokens()
{
    return _tokenizer->GetEncodedSpecialTokens();
}
