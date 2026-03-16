/*
    YALAMA Runtime
    Copyright 2026 Yazeed Hamdan
    Licensed under the Apache License, Version 2.0
    See LICENSE file in project root.
*/
#include "ChatSession.h"
#include "Helpers.h"
#include "Tokenizer.h"
#include "ConfigManager.h"
#include "TraceLogger.h"
#include "ChatTemplateProvider.h"
#include <iomanip>
#include <string>
#include <torch/torch.h>

using hr_clock = std::chrono::steady_clock;

static std::timed_mutex inferMutex;
std::regex ChatSession::_NewLineRegex = std::regex("Ċ");
std::regex ChatSession::_NewLineRegex2 = std::regex("<br>");
int ChatSession::_DecodeSafetyWindowToEmitTokens = 10;
bool ChatSession::_isTokenizerInitialized = false;
static std::mutex tokenizerMutex;

ChatSession::ChatSession()
{
    if (!_isTokenizerInitialized)
    {
        std::lock_guard<std::mutex> guard(tokenizerMutex);
        {
            if (!_isTokenizerInitialized)
            {
                Tokenizer::Init();
                _isTokenizerInitialized = true;
            }
        }
    }

    if (_MaskedTokenIds.empty())
    {
        for (const auto &[id, token] : Tokenizer::GetSpecialTokens())
        {
            bool isEos =
                std::find(
                    ConfigManager::EosIds.begin(),
                    ConfigManager::EosIds.end(),
                    id) != ConfigManager::EosIds.end();

            if (!isEos)
            {
                _MaskedTokenIds.push_back(id);

                // std::string specialToken = token;
                // specialToken.erase(std::remove(specialToken.begin(), specialToken.end(), '<'), specialToken.end());
                // specialToken.erase(std::remove(specialToken.begin(), specialToken.end(), '>'), specialToken.end());
                // specialToken.erase(std::remove(specialToken.begin(), specialToken.end(), '|'), specialToken.end());
                //
                // auto textIds = Tokenizer::GetEncodedSpecialTokens()[specialToken];
                //
                // for (auto tid : textIds)
                //{
                //    _MaskedTokenIds.push_back(tid);
                //}
            }
        }
    }

    std::sort(_MaskedTokenIds.begin(), _MaskedTokenIds.end());
    _MaskedTokenIds.erase(
        std::unique(_MaskedTokenIds.begin(), _MaskedTokenIds.end()),
        _MaskedTokenIds.end());
}

GenerationResult ChatSession::Generate(std::string &openaiJson, std::function<void(const std::string &)> onTokenDecoded, bool &isCancel)
{
    int emittedTokens = 0;
    int generatedTokens = 0;
    bool end = false;
    double tokenTimes = 0;
    std::deque<int64_t> tokensEmitQueue;
    std::vector<int64_t> tokensHistory;
    GenerationResult generationResult;
    torch::Tensor input;
    torch::Tensor last;
    torch::Tensor logits;
    hr_clock::time_point startTime;
    hr_clock::time_point endTime;
    int64_t nextID;

    if (ConfigManager::IsServicesRunMode.value() && ConfigManager::IsServiceLoggingEnabled.value())
        std::cout << "[HTTP] Thread " << std::this_thread::get_id() << " waiting for GPU inference..." << std::endl;

    auto waitStart = std::chrono::steady_clock::now();

    if (!inferMutex.try_lock_for(std::chrono::seconds(ConfigManager::HttpMaxQueueWaitSeconds)))
    {
        generationResult.IsSuccess = false;
        generationResult.Error = GenerationError::GPUQueueTimedOut;

        return generationResult;
    }

    std::unique_lock<std::timed_mutex> lock(inferMutex, std::adopt_lock);

    if (ConfigManager::IsServicesRunMode.value() && ConfigManager::IsServiceLoggingEnabled.value())
    {
        auto waitEnd = std::chrono::steady_clock::now();

        double waitSeconds =
            std::chrono::duration<double>(waitEnd - waitStart).count();

        std::cout << "[HTTP] Thread " << std::this_thread::get_id() << " acquired GPU for inference after "
                  << std::fixed << std::setprecision(6) << waitSeconds << " seconds..." << std::endl;
    }

    generationResult.IsSuccess = FormatInput(openaiJson);

    if (generationResult.IsSuccess)
    {
        TraceLogger::DumpLine("Max Sequence Length: " + std::to_string(ConfigManager::MaxSequenceLength));
        TraceLogger::DumpLine("Tokens Length: " + std::to_string(_Tokens.size()));

        torch::Tensor next;

        input = torch::tensor(_Tokens, torch::kInt64).unsqueeze(0).to(torch::kCUDA);

        Model::BeginInfer();

        while (!end)
        {
            if (isCancel)
            {
                end = true;
                generationResult.IsSuccess = false;
                generationResult.Error = GenerationError::Canceled;
            }
            else
            {
                startTime = hr_clock::now();

                generationResult = Model::Infer(input, _EosPerPrompt);

                if (!generationResult.IsSuccess)
                    break;

                logits = generationResult.Logits;

                last = logits.select(1, input.size(1) - 1);

                nextID = Sample(last, ConfigManager::IsGreedy.value() || generatedTokens == 0);

                endTime = hr_clock::now();

                if (generatedTokens > 0)
                {
                    tokenTimes += std::chrono::duration<double>(endTime - startTime).count();
                }

                generatedTokens++;

                for (const auto eos : ConfigManager::EosIds)
                {
                    if (nextID == eos)
                    {
                        end = true;
                        break;
                    }
                }

                if (!end && (_Tokens.size() + generatedTokens) >= ConfigManager::MaxSequenceLength)
                    end = true;

                if (!end)
                {
                    tokensEmitQueue.push_back(nextID);

                    if (tokensHistory.size() > ConfigManager::MaxSequenceLength)
                    {
                        tokensHistory.erase(tokensHistory.begin());
                    }

                    for (const auto &stopSeq : ChatTemplateProvider::SoftStopTokenSeqs)
                    {
                        if (tokensEmitQueue.size() >= stopSeq.size())
                        {
                            for (size_t start = 0; start + stopSeq.size() <= tokensEmitQueue.size(); ++start)
                            {
                                bool match = true;

                                for (size_t i = 0; i < stopSeq.size(); ++i)
                                {
                                    if (tokensEmitQueue[start + i] != stopSeq[i])
                                    {
                                        match = false;
                                        break;
                                    }
                                }

                                if (match)
                                {
                                    tokensEmitQueue.erase(
                                        tokensEmitQueue.begin() + start,
                                        tokensEmitQueue.end());

                                    end = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (emittedTokens > _DecodeSafetyWindowToEmitTokens)
                        end = IsRepeatDetected(tokensEmitQueue, tokensHistory);

                    if (!tokensEmitQueue.empty() && generatedTokens - emittedTokens >= _DecodeSafetyWindowToEmitTokens)
                    {
                        if (!end)
                        {
                            int64_t token = tokensEmitQueue[0];
                            int64_t nextToken = tokensEmitQueue.size() >= 2 ? tokensEmitQueue[1] : -1;

                            int numOfEmittedTokens = EmitToken(token, nextToken, emittedTokens, onTokenDecoded);

                            for (int i = 0; i < numOfEmittedTokens && !tokensEmitQueue.empty(); i++)
                            {
                                tokensHistory.push_back(tokensEmitQueue.front());
                                tokensEmitQueue.pop_front();
                            }
                        }
                    }

                    if (!end)
                    {

                        next = torch::tensor({nextID}, torch::TensorOptions()
                                                           .dtype(torch::kInt64)
                                                           .device(torch::kCUDA))
                                   .unsqueeze(0);

                        if (!ConfigManager::IsKVCacheEnabled.value())
                            input = torch::cat({input, next}, 1);
                        else
                            input = next;

                        TraceLogger::_TraceStep++;
                    }
                }
            }
        }

        if (generationResult.IsSuccess)
        {
            if (!tokensEmitQueue.empty())
            {
                int64_t token;
                int64_t nextToken;

                for (size_t i = 0; i < tokensEmitQueue.size();)
                {
                    token = tokensEmitQueue[i];
                    nextToken = (i + 1 < tokensEmitQueue.size()) ? tokensEmitQueue[i + 1] : -1;

                    i += EmitToken(token, nextToken, emittedTokens, onTokenDecoded);
                }
            }

            double tps = (generatedTokens == 0 ? 0 : generatedTokens - 1) / (tokenTimes == 0 ? 1 : tokenTimes);

            if (!ConfigManager::IsServicesRunMode.value() || (ConfigManager::IsServicesRunMode.value() && ConfigManager::IsServiceLoggingEnabled.value()))
                std::cout << "\nTokens Per Second: " << std::to_string(tps) << std::endl;
        }
        else
        {
            if (!ConfigManager::IsServicesRunMode)
            {
                if (generationResult.Error == GenerationError::InvalidPrompt)
                    std::cout << "Invalid request format" << std::endl;
                else if (generationResult.Error == GenerationError::KVCacheExceeded)
                    std::cout << "OOM, KV Cache Capacity Exceeded, consider increasing KV Cache Capacity from args or yalam_config.json file" << std::endl;
                else if (generationResult.Error == GenerationError::SequenceLengthExceeded)
                    std::cout << "Max Sequence Length Exceeded, Make sure your prompt is less or equal to model max sequence length" << std::endl;
                else if (generationResult.Error == GenerationError::Canceled)
                    std::cout << "Generation Canceled by the user" << std::endl;
                else if (generationResult.Error == GenerationError::GPUQueueTimedOut)
                    std::cout << "Failed to acquire a GPU within the waiting time of: " << ConfigManager::HttpMaxQueueWaitSeconds << std::endl;
            }
        }
    }
    else
        generationResult.Error = GenerationError::InvalidPrompt;

    return generationResult;
}

bool ChatSession::FormatInput(const std::string &openaiJson)
{
    std::string fullText;
    _Tokens.clear();
    _EosPerPrompt.clear();

    nlohmann::json requestJson;
    bool isValid = true;

    try
    {
        requestJson = nlohmann::json::parse(openaiJson);
    }
    catch (...)
    {
        isValid = false;
    }

    if (isValid)
    {
        if (requestJson.contains("messages") == false)
        {
            if (!ConfigManager::IsServicesRunMode.value() || (ConfigManager::IsServicesRunMode.value() && ConfigManager::IsServiceLoggingEnabled.value()))
                std::cout << "Invalid request: missing 'messages' field\n";

            isValid = false;
        }
        else if (requestJson["messages"].is_array() == false)
        {
            if (!ConfigManager::IsServicesRunMode.value() || (ConfigManager::IsServicesRunMode.value() && ConfigManager::IsServiceLoggingEnabled.value()))
                std::cout << "Invalid request: 'messages' must be an array\n";

            isValid = false;
        }
        else if (requestJson["messages"].empty())
        {
            if (!ConfigManager::IsServicesRunMode.value() || (ConfigManager::IsServicesRunMode.value() && ConfigManager::IsServiceLoggingEnabled.value()))
                std::cout << "Invalid request: 'messages' cannot be empty\n";

            isValid = false;
        }

        if (isValid)
        {
            auto messages = requestJson["messages"];

            for (auto message : messages)
            {
                nlohmann::json normalized;

                for (auto &[key, value] : message.items())
                {
                    normalized[TrimToLower(key)] = value;
                }

                message = normalized;
            }

            std::string userMessage;
            std::string assistantMessage;

            // if base model and non-instruct, take last prompt, no turn-in chat for it
            if (!ConfigManager::HasChatTemplate)
            {
                if (!messages.empty())
                {
                    nlohmann::json lastMessage = messages.back();
                    messages = nlohmann::json::array();
                    messages.push_back(lastMessage);
                }
            }

            std::string errorMessage;

            int messageIndex = 0;
            for (auto message : messages)
            {

                if (message.contains("role") == false ||
                    message.contains("content") == false)
                {
                    isValid = false;
                    errorMessage = "Invalid request: message missing role/content";
                }
                else if (message["role"].is_string() == false ||
                         message["content"].is_string() == false)
                {
                    isValid = false;
                    errorMessage = "Invalid request: role/content must be string";
                }
                else
                {
                    std::string role = message["role"];

                    if (role != "user" &&
                        role != "assistant")
                    {
                        isValid = false;
                        errorMessage = "Invalid request: unsupported role";
                    }
                }

                if (!isValid)
                    break;

                if (message["role"] == "user")
                {
                    userMessage = message["content"];

                    fullText += ChatTemplateProvider::Format(userMessage, messageIndex == 0);

                    auto userTokens = Tokenizer::EncodeWithChatTemplate(userMessage, messageIndex == 0);
                    _Tokens.insert(_Tokens.end(), userTokens.begin(), userTokens.end());
                }
                else if (message["role"] == "assistant" && ConfigManager::HasChatTemplate) // only for instruct
                {
                    assistantMessage = message["content"];

                    if (!assistantMessage.empty())
                    {
                        fullText += assistantMessage;

                        auto assistantTokens = Tokenizer::EncodeWithoutChatTemplate(assistantMessage);
                        _Tokens.insert(_Tokens.end(), assistantTokens.begin(), assistantTokens.end());

                        if (std::find(
                                ConfigManager::EosIds.begin(),
                                ConfigManager::EosIds.end(),
                                _Tokens.back()) == ConfigManager::EosIds.end())
                        {
                            std::string eosString = ChatTemplateProvider::GetEOSTokenString();
                            auto eosIds = Tokenizer::EncodeWithoutChatTemplate(eosString);

                            fullText += eosString;

                            _Tokens.insert(_Tokens.end(), eosIds.begin(), eosIds.end());
                        }

                        _EosPerPrompt.push_back(_Tokens.size() - 1);
                    }
                }
                else
                {
                    std::cout << "Invalid OpenAI request format. Expected one of:\n"
                                 "1) Turn-based with history:\n"
                                 "   {\"messages\": [\n"
                                 "       {\"role\": \"user\", \"content\": \"...\"},\n"
                                 "       {\"role\": \"assistant\", \"content\": \"...\"}\n"
                                 "       {\"role\": \"user\", \"content\": \"...\"}\n"
                                 "   ]}\n"
                                 "2) Single-turn:\n"
                                 "   {\"messages\": [\n"
                                 "       {\"role\": \"user\", \"content\": \"...\"}\n"
                                 "   ]}"
                              << std::endl;

                    isValid = false;
                }

                messageIndex++;
            }
        }

        if (isValid && _Tokens.empty())
        {
            isValid = false;
            std::cout << "Invalid OpenAI request: input produced zero tokens, Ensure at least one user message with non-empty content...";
        }
    }

    return isValid;
}

bool ChatSession::IsRepeatDetected(std::deque<int64_t> &tokensEmitQueue, const std::vector<int64_t> &tokensHistory)
{
    bool isLoop = false;
    size_t firstMatchedTokenPos = static_cast<size_t>(-1);

    if (tokensEmitQueue.size() >= _DecodeSafetyWindowToEmitTokens &&
        tokensHistory.size() >= _DecodeSafetyWindowToEmitTokens &&
        tokensHistory.size() >= tokensEmitQueue.size())
    {
        for (size_t start = 0; start < tokensHistory.size(); ++start)
        {
            if (tokensHistory[start] == tokensEmitQueue[0])
            {
                bool match = true;

                for (size_t i = 1; i < tokensEmitQueue.size(); ++i)
                {
                    if (tokensHistory.size() <= start + i ||
                        tokensHistory[start + i] != tokensEmitQueue[i])
                    {
                        match = false;
                        break;
                    }
                    else
                    {
                        if (firstMatchedTokenPos == static_cast<size_t>(-1))
                        {
                            firstMatchedTokenPos = i;
                        }
                    }
                }

                if (match)
                {
                    isLoop = true;

                    tokensEmitQueue.erase(
                        tokensEmitQueue.begin() + firstMatchedTokenPos,
                        tokensEmitQueue.end());

                    break;
                }
            }
        }
    }

    return isLoop;
}

int ChatSession::EmitToken(const int64_t &tokenID, const int64_t &nextTokenID, int &emittedTokensCount, const std::function<void(const std::string &)> &onTokenDecoded)
{
    bool isValidDecodedOutput = true;
    int numOfEmittedTokens = 1;

    auto decoded = Tokenizer::Decode(tokenID);

    if (!IsValidUTF8(decoded))
    {
        if (nextTokenID != -1)
        {
            auto nextDecoded = Tokenizer::Decode(nextTokenID);

            decoded += nextDecoded;
        }

        isValidDecodedOutput = IsValidUTF8(decoded);

        if (isValidDecodedOutput)
            numOfEmittedTokens += 1;
    }

    if (!isValidDecodedOutput)
        decoded = "�";
    else
    {
        decoded = std::regex_replace(decoded, _NewLineRegex, "\n");
        decoded = std::regex_replace(decoded, _NewLineRegex2, "\n");
    }

    onTokenDecoded(decoded);

    emittedTokensCount += numOfEmittedTokens;

    return numOfEmittedTokens;
}

int64_t ChatSession::Sample(torch::Tensor &lastLogitsToken, bool isGreedy)
{
    torch::Tensor nextToken;

    if (isGreedy)
    {
        nextToken = lastLogitsToken.argmax(-1);
    }
    else
    {
        torch::Tensor scaled = lastLogitsToken / ConfigManager::Temp;

        // Mask special tokens
        for (auto id : _MaskedTokenIds)
        {
            scaled[0][id] = -1e9f;
        }

        auto topk = torch::topk(scaled, ConfigManager::TopK, -1);

        torch::Tensor topk_logits = std::get<0>(topk);
        torch::Tensor topk_indices = std::get<1>(topk);

        torch::Tensor topk_probs = torch::softmax(topk_logits, -1);

        torch::Tensor sampled = torch::multinomial(topk_probs, 1);

        nextToken = topk_indices.gather(-1, sampled);
    }

    int64_t result = nextToken.to(torch::kCPU, false, true).item<int64_t>();

    return result;
}