#pragma once

#include "Model.h"
#include <torch/torch.h>
#include <regex>
#include "GenerationResult.h"

class ChatSession
{
public:
    ChatSession();
    GenerationResult Generate(std::string &openaiJson, std::function<void(const std::string &)> onTokenDecoded, bool &isCancel);

private:    
    static bool _isTokenizerInitialized;
    static std::regex _NewLineRegex;
    static std::regex _NewLineRegex2;
    static int _DecodeSafetyWindowToEmitTokens;
    
    std::shared_ptr<Model> _model;
    std::vector<int64_t> _Tokens;
    std::vector<int> _EosPerPrompt;
    std::vector<int64_t> _MaskedTokenIds;
    
    int EmitToken(const int64_t &tokenID, const int64_t &nextTokenID, int &emittedTokensCount, const std::function<void(const std::string &)> &onTokenDecoded);
    bool IsRepeatDetected(std::deque<int64_t> &tokenQueue, const std::vector<int64_t> &stopSeq);
    bool IsReaptedStringDetected(std::deque<std::string> &tokensEmitQueue, const std::vector<std::string> &tokensHistory);
    bool FormatInput(const std::string &openaiJson);    
    int64_t Sample(torch::Tensor &lastLogitsToken, bool isGreedy);
};