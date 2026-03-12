#pragma once
#include "IChatTemplateProvider.h"
#include <sstream>

class MistralChatTemplateProvider : public IChatTemplateProvider
{
public:
    std::string Format(const std::string &text, bool isAddSystemHeaderInChatTemplate) override
    {
        std::ostringstream chatTemplate;

        if (isAddSystemHeaderInChatTemplate)
            chatTemplate << "<s>";
                
            chatTemplate
                << "[INST] "
                << text
                << "[/INST] ";

        return chatTemplate.str();
    }

    std::string GetEOSTokenString() override
    {
        return ConfigManager::EosTokenString;
    }
};