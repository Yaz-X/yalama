#pragma once
#include "IChatTemplateProvider.h"
#include <sstream>

class Qwen2_5ChatTemplateProvider : public IChatTemplateProvider
{
public:
    std::string Format(const std::string &text, bool isAddSystemHeaderInChatTemplate) override
    {
        std::ostringstream chatTemplate;
        
        if (isAddSystemHeaderInChatTemplate)
        {
            chatTemplate
                << "<|im_start|>system\n"
                << "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                << "<|im_end|>\n";
        }

        chatTemplate
            << "<|im_start|>user\n"
            << text            
            << "<|im_end|>\n"
            << "<|im_start|>assistant\n";

        return chatTemplate.str();
    }

    std::string GetEOSTokenString() override
    {
        return ConfigManager::EosTokenString;
    }
};