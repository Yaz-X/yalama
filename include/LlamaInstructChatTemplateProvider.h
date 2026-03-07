#pragma once
#include "IChatTemplateProvider.h"
#include <sstream>

class LlamaInstructChatTemplateProvider : public IChatTemplateProvider
{
public:
    std::string Format(const std::string &text, bool isAddSystemHeaderInChatTemplate) override
    {
        std::ostringstream chatTemplate;

        if (isAddSystemHeaderInChatTemplate)
        {
            chatTemplate
                << "<|begin_of_text|>"
                << "<|start_header_id|>system<|end_header_id|>\n\n"
                << "Cutting Knowledge Date: December 2023\n"
                << "Today Date: 23 July 2024\n\n"
                << "You are a helpful assistant.<|eot_id|>";
        }

        chatTemplate
            << "<|start_header_id|>user<|end_header_id|>\n\n"
            << text
            << "<|eot_id|>"
            << "<|start_header_id|>assistant<|end_header_id|>\n\n";

        
        return chatTemplate.str();
    }
};