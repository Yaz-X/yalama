#pragma once

#include "ConfigManager.h" // used by derived classes
#include <string>

class IChatTemplateProvider
{
public:
    virtual std::string Format(const std::string& text, bool isAddSystemHeaderInChatTemplate) = 0;    
    virtual std::string GetEOSTokenString() = 0;    
    virtual ~IChatTemplateProvider() = default;
    virtual std::string GetThinkString()
    {
        return {};
    }
};