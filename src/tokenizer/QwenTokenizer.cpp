#include "QwenTokenizer.h"
#include "ChatTemplateProvider.h"

QwenTokenizer::QwenTokenizer()
{    
}

std::vector<std::string> QwenTokenizer::SplitText(const std::string &text)
{
    std::vector<std::string> result;

    result.push_back(text);

    return result;
}

