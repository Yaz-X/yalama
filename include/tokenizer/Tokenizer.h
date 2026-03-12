#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "TokenizerBase.h"

class Tokenizer
{
public:

    static void Init();
    static std::vector<int64_t> EncodeWithChatTemplate(std::string &text, bool isAddSystemHeaderInChatTemplate);
    static std::vector<int64_t> EncodeWithoutChatTemplate(std::string &text);
    static std::string Decode(const std::vector<int64_t> &ids);
    static std::unordered_map<int, std::string> GetSpecialTokens();
    static std::unordered_map<std::string, std::vector<int64_t>> GetEncodedSpecialTokens();    

private:

    static std::unique_ptr<TokenizerBase> _tokenizer;
};