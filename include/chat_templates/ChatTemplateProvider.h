#pragma once
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "ConfigManager.h"
#include "Tokenizer.h"
#include "IChatTemplateProvider.h"
#include "LlamaInstructChatTemplateProvider.h"
#include "LlamaChatTemplateProvider.h"
#include "MistralChatTemplateProvider.h"

class ChatTemplateProvider
{
private:

    inline static std::vector<std::string> SoftStops = {
        "**question**:",
        "User:",
        "Assistant:",
        "**answer**:",
        "<|",
        "|>",
        "</s>",
        "[INST]",
        "[/INST]"
    };

    inline static std::once_flag providerInitFlag;
    inline static std::once_flag softStopInitFlag;

    inline static void InitSoftStops()
    {
        std::call_once(softStopInitFlag, []()
        {
            for (const auto &stop : SoftStops)
            {
                std::string stopString = stop;

                SoftStopTokenSeqs.push_back(
                    Tokenizer::EncodeWithoutChatTemplate(stopString));
            }
        });
    }

    inline static void InitProvider()
    {
        std::call_once(providerInitFlag, []()
        {
            InitSoftStops();

            switch (ConfigManager::ModelLoadedType)
            {
            case ModelType::LLama:
            {
                if (ConfigManager::HasChatTemplate)
                    provider = std::make_unique<LlamaInstructChatTemplateProvider>();
                else
                    provider = std::make_unique<LlamaChatTemplateProvider>();

                break;
            }

            case ModelType::Mistral:
            {
                provider = std::make_unique<MistralChatTemplateProvider>();
                break;
            }

            default:
            {
                std::string error =
                    "Unsupported model type: " +
                    std::to_string(static_cast<int>(ConfigManager::ModelLoadedType));

                throw std::runtime_error(error);
            }
            }
        });
    }

public:

    inline static std::vector<std::vector<int64_t>> SoftStopTokenSeqs;

    inline static std::unique_ptr<IChatTemplateProvider> provider = nullptr;

    static std::string Format(const std::string &text, bool isAddSystemHeaderInChatTemplate)
    {
        InitProvider();

        return provider->Format(text, isAddSystemHeaderInChatTemplate);
    }

    static std::string GetEOSTokenString()
    {
        InitProvider();

        return provider->GetEOSTokenString();
    }
};