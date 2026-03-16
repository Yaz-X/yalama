#pragma once

#include "MistralTokenizer.h"

class QwenTokenizer : public TokenizerBase
{
public:
    QwenTokenizer();
protected:
    std::vector<std::string> SplitText(const std::string &text) override;             
};