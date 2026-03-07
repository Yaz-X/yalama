#pragma once
#include <string>

class IChatTemplateProvider
{
public:
    virtual std::string Format(const std::string& text, bool isAddSystemHeaderInChatTemplate) = 0;
    virtual ~IChatTemplateProvider() = default;
};