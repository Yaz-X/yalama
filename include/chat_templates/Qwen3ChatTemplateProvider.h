#pragma once
#include "IChatTemplateProvider.h"
#include <sstream>

class Qwen3ChatTemplateProvider : public IChatTemplateProvider
{
public:
    std::string Format(const std::string &text, bool isAddSystemHeaderInChatTemplate) override
    {
        std::ostringstream chatTemplate;
        std::string thinkString;

        if (ConfigManager::IsModelSupportThinking)
        {
            if (ConfigManager::IsThinkingEnabled.value())
                thinkString = " /think";
            else
                thinkString = " /no_think";
        }

        if (isAddSystemHeaderInChatTemplate)
        {
            chatTemplate
                << "<|im_start|>system\n"
                << "You are a helpful assistant."
                << "<|im_end|>";
        }

        chatTemplate
            << "\n<|im_start|>user\n"
            << text
            << thinkString
            << "<|im_end|>\n"
            << "<|im_start|>assistant\n";

        return chatTemplate.str();
    }

    std::string GetEOSTokenString() override
    {
        return ConfigManager::EosTokenString;
    }

    std::string GetThinkString() override
    {
        return {};
    }
};