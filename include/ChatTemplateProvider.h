#pragma once
#include <memory>
#include <string>
#include <vector>
#include "Tokenizer.h"
#include "IChatTemplateProvider.h"
#include "LlamaInstructChatTemplateProvider.h"
#include "LlamaChatTemplateProvider.h"

class ChatTemplateProvider
{
private:
    inline static std::vector<std::string> SoftStops = {
        "**question**:",
        "User:",
        "Assistant:",
        "**answer**:",
        "<|",
        "|>"};

    inline static void InitSoftStops()
    {
        if (SoftStopTokenSeqs.empty())
        {
            for (const auto &stop : SoftStops)
            {
                std::string stopString = stop;

                SoftStopTokenSeqs.push_back(
                    Tokenizer::EncodeWithoutChatTemplate(stopString));
            }
        }
    }

public:
    inline static std::vector<std::vector<int64_t>> SoftStopTokenSeqs;

    static std::string Format(const std::string &text, bool isAddSystemHeaderInChatTemplate)
    {
        InitSoftStops();

        static std::unique_ptr<IChatTemplateProvider> provider;

        if (Tokenizer::HasChatTemplate())
            provider = std::make_unique<LlamaInstructChatTemplateProvider>();
        else
            provider = std::make_unique<LlamaChatTemplateProvider>();

        return provider->Format(text, isAddSystemHeaderInChatTemplate);
    }
};