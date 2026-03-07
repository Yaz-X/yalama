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
            << "**question**:"
            << text
            << "\n\n"
            << "**answer**:";
            
        return chatTemplate.str();
    }
};