#pragma once

#include "TokenizerBase.h"

class MistralTokenizer : public TokenizerBase
{
protected:
    std::vector<std::string> SplitText(const std::string &text) override;
    
    void BuildDecoderRules() override;
    std::string EncodeBytes(const std::string &input) override;
    std::string ByteToToken(unsigned char c) override;
    std::unordered_map<int, std::string> BuildByteEncoder() override;
    std::vector<std::string> ApplyBPE(const std::vector<std::string> &inputChars) override;

public:
    MistralTokenizer();
   
};