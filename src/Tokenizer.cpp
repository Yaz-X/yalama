#include "Tokenizer.h"
#include "ConfigManager.h"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include "ChatTemplateProvider.h"

using json = nlohmann::json;

void Tokenizer::Init()
{
    LoadFromJson();
    _bpeCache.reserve(50000);
}

bool Tokenizer::HasChatTemplate()
{
    return _hasChatTemplate;
}

std::unordered_map<int, std::string> Tokenizer::BuildByteEncoder()
{
    std::unordered_map<int, std::string> result;

    bool safe[256] = {false};

    for (int i = 33; i <= 126; ++i)
        safe[i] = true;

    for (int i = 161; i <= 172; ++i)
        safe[i] = true;

    for (int i = 174; i <= 255; ++i)
        safe[i] = true;

    int placeholderIndex = 0;

    for (int byte = 0; byte < 256; ++byte)
    {
        int character;

        if (safe[byte])
        {
            character = byte;
        }
        else
        {
            character = 256 + placeholderIndex;
            placeholderIndex++;
        }

        std::string charString;

        if (character <= 0x7F)
        {
            charString = std::string(1, static_cast<char>(character));
        }
        else
        {
            char first = static_cast<char>(0xC0 | (character >> 6));
            char second = static_cast<char>(0x80 | (character & 0x3F));

            charString = std::string({first, second});
        }

        result[byte] = charString;
    }

    return result;
}

void Tokenizer::LoadFromJson()
{
    std::filesystem::path modelTokenizerFilePath = ConfigManager::ModelPath;
    std::filesystem::path modelTokenizerConfigFilePath = ConfigManager::ModelPath;

    modelTokenizerFilePath = modelTokenizerFilePath / "tokenizer.json";
    modelTokenizerConfigFilePath = modelTokenizerConfigFilePath / "tokenizer_config.json";

    if (!std::filesystem::exists(modelTokenizerFilePath) || !std::filesystem::is_regular_file(modelTokenizerFilePath))
        throw std::runtime_error("File at Path (" + modelTokenizerFilePath.string() + ") doesnt exist");

    if (!std::filesystem::exists(modelTokenizerConfigFilePath) || !std::filesystem::is_regular_file(modelTokenizerConfigFilePath))
        throw std::runtime_error("File at Path (" + modelTokenizerConfigFilePath.string() + ") doesnt exist");

    std::string path = modelTokenizerFilePath.string();

    std::ifstream tokenizerFilePath(path);

    auto json = nlohmann::json::parse(tokenizerFilePath);

    auto vocab = json["model"]["vocab"];
    _vocab.reserve(vocab.size());
    _idToToken.reserve(vocab.size());

    for (auto &[token, id] : vocab.items())
    {
        _vocab[token] = id;
        _idToToken[id] = token;
    }

    auto addedTokens = json["added_tokens"];

    for (auto &tok : addedTokens)
    {
        if (tok.contains("special") &&
            tok["special"].get<bool>() == true &&
            tok["content"].get<std::string>().rfind("<|", 0) == 0)
        {
            int id = tok["id"].get<int>();
            std::string content = tok["content"].get<std::string>();

            _specialTokens[id] = content;
            _idToToken[id] = content;
            _vocab[content] = id;
        }
    }

    _specialTokens[_vocab["<|eot_id|>"]] = "<|eot_id|>";
    _specialTokens[_vocab["<|end_of_text|>"]] = "<|end_of_text|>";

    _idToToken[_vocab["<|eot_id|>"]] = "<|eot_id|>";
    _idToToken[_vocab["<|end_of_text|>"]] = "<|end_of_text|>";

    _byteEncoder = BuildByteEncoder();
    _byteFallbackVocab.reserve(256);

    // build byte decoder
    for (auto &kv : _byteEncoder)
    {
        unsigned char byte = kv.first;
        std::string token = kv.second;

        _byteDecoder[token] = byte;
    }

    for (auto &kv : _byteEncoder)
    {
        std::string token = kv.second;

        auto it = _vocab.find(token);

        if (it != _vocab.end())
        {
            // performance optimization: fast lookup for byte tokens instead of full vocab search
            _byteFallbackVocab[token] = it->second;
        }
    }

    _decoderRules.emplace_back(std::regex("Ġ"), " ");
    _decoderRules.emplace_back(std::regex("Ċ"), "\n");
    _decoderRules.emplace_back(std::regex("ĊĊ"), "\n\n");
    _decoderRules.emplace_back(std::regex("Ċ([0-9])"), "\n$1");
    _decoderRules.emplace_back(std::regex("<\\|[^>]*\\|>"), "");

    auto merges = json["model"]["merges"];
    _mergeRanks.reserve(merges.size());

    std::string pair;

    for (int i = 0; i < merges.size(); ++i)
    {
        const std::string &m = merges[i];

        auto pos = m.find(' ');

        pair.clear();
        pair.append(m, 0, pos);
        pair += ' ';
        pair.append(m, pos + 1, m.size() - pos - 1);

        _mergeRanks[pair] = i;
    }

    std::cout << "Tokenizer vocab loaded: " << _vocab.size() << std::endl;

    path = modelTokenizerConfigFilePath.string();

    std::ifstream tokenizerConfigFileStream(path);
    json = nlohmann::json::parse(tokenizerConfigFileStream);

    _hasChatTemplate = json.contains("chat_template");

    //Create a cash for encoded special tokens
    for (auto &[id, token] : _specialTokens)
    {        
        _encodedSpecialTokens[token] = EncodeWithoutChatTemplate(token);
    }
}

std::vector<std::string> Tokenizer::LlamaSplit(const std::string &text)
{
    std::vector<std::string> result;
    std::string current;

    for (size_t i = 0; i < text.size();)
    {
        unsigned char c = static_cast<unsigned char>(text[i]);

        size_t charLen = 1;

        if ((c & 0x80) == 0)
            charLen = 1;
        else if ((c & 0xE0) == 0xC0)
            charLen = 2;
        else if ((c & 0xF0) == 0xE0)
            charLen = 3;
        else if ((c & 0xF8) == 0xF0)
            charLen = 4;

        std::string ch = text.substr(i, charLen);

        bool isSpace = false;
        bool isPunct = false;

        if (charLen == 1)
        {
            isSpace = std::isspace(c) != 0;
            isPunct = std::ispunct(c) != 0;
        }

        if (isSpace || isPunct)
        {
            if (current.empty() == false)
            {
                result.push_back(current);
                current.clear();
            }

            result.push_back(ch);
        }
        else
            current += ch;

        i += charLen;
    }

    if (current.empty() == false)
    {
        result.push_back(current);
    }

    return result;
}

std::string Tokenizer::ByteToToken(unsigned char c)
{
    std::string value;
    auto it = _byteEncoder.find(c);

    if (it != _byteEncoder.end())
        value = it->second;
    else
        value = std::string(1, c);

    return value;
}

std::vector<std::string> Tokenizer::ApplyBPE(const std::vector<std::string> &inputChars)
{
    std::vector<std::string> result;

    auto cacheIt = _bpeCache.find(inputChars);

    bool isCached = (cacheIt != _bpeCache.end());

    if (isCached)
    {
        result = cacheIt->second;
    }
    else
    {
        std::vector<std::string> chars;
        chars.reserve(inputChars.size());
        chars = inputChars;

        std::vector<int> pairRanks;
        pairRanks.reserve(chars.size());

        std::string pair;
        pair.reserve(64);

        int charCount = (int)chars.size();

        for (int i = 0; i + 1 < charCount; ++i)
        {
            const std::string &left = chars[i];
            const std::string &right = chars[i + 1];

            pair.clear();
            pair.append(left);
            pair.push_back(' ');
            pair.append(right);

            auto it = _mergeRanks.find(pair);

            int rank = INT_MAX;

            if (it != _mergeRanks.end())
            {
                rank = it->second;
            }

            pairRanks.push_back(rank);
        }

        bool merging = true;

        while (merging)
        {
            int bestRank = INT_MAX;
            int bestIndex = -1;

            int pairCount = (int)pairRanks.size();

            for (int i = 0; i < pairCount; ++i)
            {
                int rank = pairRanks[i];

                if (rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex >= 0 && bestRank != INT_MAX)
            {
                chars[bestIndex] += chars[bestIndex + 1];

                auto erasePos = chars.begin() + bestIndex + 1;
                chars.erase(erasePos);

                int pairCount = (int)pairRanks.size();

                for (int i = bestIndex; i + 1 < pairCount; ++i)
                {
                    pairRanks[i] = pairRanks[i + 1];
                }

                pairRanks.pop_back();

                int charCount = (int)chars.size();

                if (bestIndex > 0)
                {
                    const std::string &left = chars[bestIndex - 1];
                    const std::string &right = chars[bestIndex];

                    pair.clear();
                    pair.append(left);
                    pair.push_back(' ');
                    pair.append(right);

                    auto it = _mergeRanks.find(pair);

                    int rank = INT_MAX;

                    if (it != _mergeRanks.end())
                    {
                        rank = it->second;
                    }

                    pairRanks[bestIndex - 1] = rank;
                }

                if (bestIndex < charCount - 1)
                {
                    const std::string &left = chars[bestIndex];
                    const std::string &right = chars[bestIndex + 1];

                    pair.clear();
                    pair.append(left);
                    pair.push_back(' ');
                    pair.append(right);

                    auto it = _mergeRanks.find(pair);

                    int rank = INT_MAX;

                    if (it != _mergeRanks.end())
                    {
                        rank = it->second;
                    }

                    if (bestIndex < (int)pairRanks.size())
                    {
                        pairRanks[bestIndex] = rank;
                    }
                    else
                    {
                        pairRanks.push_back(rank);
                    }
                }
            }
            else
            {
                merging = false;
            }
        }

        result = chars;

        if (_bpeCache.size() > 50000)
        {
            _bpeCache.clear();
        }

        _bpeCache[inputChars] = result;
    }

    return result;
}

std::vector<int64_t> Tokenizer::Encode(std::string &text, bool isApplyChatTemplate, bool isAddSystemHeaderInChatTemplate)
{
    std::string input = text;

    if (isApplyChatTemplate)
    {
        input = ChatTemplateProvider::Format(text, isAddSystemHeaderInChatTemplate);
    }

    std::vector<std::string> tokens;

    std::string encodedInput;

    for (size_t i = 0; i < input.size(); ++i)
    {
        unsigned char c = static_cast<unsigned char>(input[i]);
        encodedInput += ByteToToken(c);
    }

    std::vector<std::string> allPiecesAfterSegmentation;

    size_t segmentStart = 0;
    size_t segmentEnd = 0;
    std::string before;
    std::string specialToken;
    bool isDoneSegmentation = false;

    std::vector<std::string> split;

    while (!encodedInput.empty())
    {
        split.clear();
        specialToken.clear();
        before.clear();
        segmentStart = 0;
        segmentEnd = 0;

        segmentStart = encodedInput.find("<|", 0);

        if (segmentStart == std::string::npos)
        {
            split = LlamaSplit(encodedInput);
            encodedInput.clear();
        }
        else
        {
            before = encodedInput.substr(0, segmentStart);

            segmentEnd = encodedInput.find("|>", segmentStart);

            if (segmentEnd != std::string::npos)
                segmentEnd += 2;

            // if this is a damaged/invlid special token like <|.... without |>, then find the next proper <|... token
            // if found, split the text till the valid special token, split it like normal text, if no remaining special tokens
            //  then do nothing and the next turn in loop
            if (segmentEnd == std::string::npos)
            {
                auto nextSegmentStart = encodedInput.find("<|", segmentStart + 2);

                if (nextSegmentStart != std::string::npos)
                    segmentEnd = nextSegmentStart;
            }

            if (segmentEnd != std::string::npos)
            {
                specialToken = encodedInput.substr(segmentStart, (segmentEnd - segmentStart));

                if (!before.empty())
                    split = LlamaSplit(before);

                encodedInput = encodedInput.substr(segmentEnd);
            }
            else
            {
                encodedInput = encodedInput.substr(segmentStart);
                split = LlamaSplit(encodedInput);
                encodedInput.clear();
            }
        }

        if (!split.empty())
            allPiecesAfterSegmentation.insert(allPiecesAfterSegmentation.end(), split.begin(), split.end());

        if (!specialToken.empty())
            allPiecesAfterSegmentation.push_back(specialToken);
    }

    for (auto &piece : allPiecesAfterSegmentation)
    {
        if (_vocab.find(piece) != _vocab.end() || piece.find("<|") == 0)
        {
            tokens.push_back(piece);
        }
        else
        {
            if (!piece.empty())
            {
                std::vector<std::string> chars;
                std::string encoded = piece;

                for (size_t i = 0; i < encoded.size();)
                {
                    unsigned char c = encoded[i];

                    size_t len = 1;

                    if ((c & 0x80) == 0)
                        len = 1;
                    else if ((c & 0xE0) == 0xC0)
                        len = 2;
                    else if ((c & 0xF0) == 0xE0)
                        len = 3;
                    else if ((c & 0xF8) == 0xF0)
                        len = 4;

                    chars.push_back(encoded.substr(i, len));
                    i += len;
                }

                chars = ApplyBPE(chars);

                tokens.insert(tokens.end(), chars.begin(), chars.end());
            }
        }
    }

    std::vector<int64_t> ids;

    for (auto &token : tokens)
    {
        if (token.empty() == false)
        {
            auto vocabKV = _vocab.find(token);

            if (vocabKV != _vocab.end())
            {
                ids.push_back(vocabKV->second);
            }
            else
            {
                for (size_t i = 0; i < token.size(); ++i)
                {
                    unsigned char character = static_cast<unsigned char>(token[i]);

                    auto encodedCharacterByteKV = _byteEncoder.find(character);

                    if (encodedCharacterByteKV != _byteEncoder.end())
                    {
                        std::string byteToken = encodedCharacterByteKV->second;

                        auto vocabIt = _byteFallbackVocab.find(byteToken);

                        if (vocabIt != _byteFallbackVocab.end())
                        {
                            ids.push_back(vocabIt->second);
                        }
                        else
                        {
                            ids.push_back(_vocab["<unk>"]);
                        }
                    }
                    else
                    {
                        ids.push_back(_vocab["<unk>"]);
                    }
                }
            }
        }
    }

    return ids;
}

std::vector<int64_t> Tokenizer::EncodeWithChatTemplate(std::string &text, bool isAddSystemHeaderInChatTemplate)
{
    return Encode(text, true, isAddSystemHeaderInChatTemplate);
}

std::vector<int64_t> Tokenizer::EncodeWithoutChatTemplate(std::string &text)
{
    return Encode(text);
}

std::string Tokenizer::Decode(const std::vector<int64_t> &ids)
{
    std::ostringstream builder;
    bool stopDecoding = false;

    for (auto id : ids)
    {
        auto tokenIt = _idToToken.find(id);
        bool isSpecial = (_specialTokens.count(id) > 0);

        if (tokenIt != _idToToken.end())
        {
            if (isSpecial == true)
            {
                bool isEos =
                    std::find(
                        ConfigManager::EosIds.begin(),
                        ConfigManager::EosIds.end(),
                        id) != ConfigManager::EosIds.end();

                if (isEos == true)
                    stopDecoding = true;
            }

            if (!stopDecoding)
            {
                std::string tokenValue = tokenIt->second;
                if (isSpecial == false)
                {
                    bool needsDecode =
                        tokenValue.find("Ġ") != std::string::npos ||
                        tokenValue.find("Ċ") != std::string::npos;

                    if (needsDecode)
                    {
                        for (auto &[pattern, repl] : _decoderRules)
                        {
                            tokenValue = std::regex_replace(tokenValue, pattern, repl);
                        }
                    }
                }

                for (size_t i = 0; i < tokenValue.size();)
                {
                    unsigned char c = static_cast<unsigned char>(tokenValue[i]);

                    size_t len = 1;

                    if ((c & 0x80) == 0)
                        len = 1;
                    else if ((c & 0xE0) == 0xC0)
                        len = 2;
                    else if ((c & 0xF0) == 0xE0)
                        len = 3;
                    else if ((c & 0xF8) == 0xF0)
                        len = 4;

                    std::string piece = tokenValue.substr(i, len);

                    auto it = _byteDecoder.find(piece);

                    if (it != _byteDecoder.end())
                    {
                        builder << static_cast<char>(it->second);
                    }
                    else
                    {
                        builder << piece;
                    }

                    i += len;
                }
            }
        }

        if (stopDecoding)
            break;
    }

    return builder.str();
}

std::unordered_map<int, std::string> Tokenizer::GetSpecialTokens()
{
    return _specialTokens;
}

std::unordered_map<std::string, std::vector<int64_t>> Tokenizer::GetEncodedSpecialTokens()
{
    return _encodedSpecialTokens;
}
