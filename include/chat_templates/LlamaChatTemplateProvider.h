#pragma once
#include "IChatTemplateProvider.h"
#include <sstream>

class LlamaChatTemplateProvider : public IChatTemplateProvider
{
public:
    std::string Format(const std::string &text, bool isAddSystemHeaderInChatTemplate) override
    {
        std::ostringstream chatTemplate;

        chatTemplate
            << "<|begin_of_text|>"         
            << text;
            
        return chatTemplate.str();
    }

    std::string GetEOSTokenString() override
    {
        return ConfigManager::EosTokenString;
    }    

};