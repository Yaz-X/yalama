#include "TokenizerBase.h"
#include "ConfigManager.h"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include "Helpers.h"
#include "ChatTemplateProvider.h"

using json = nlohmann::json;

void TokenizerBase::Init()
{
    LoadFromJson();
    _bpeCache.reserve(50000);
}

std::unordered_map<int, std::string> TokenizerBase::BuildByteEncoder()
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

void TokenizerBase::LoadFromJson()
{
    std::filesystem::path modelTokenizerFilePath = ConfigManager::ModelPath;

    modelTokenizerFilePath = modelTokenizerFilePath / "tokenizer.json";

    if (!std::filesystem::exists(modelTokenizerFilePath) || !std::filesystem::is_regular_file(modelTokenizerFilePath))
        throw std::runtime_error("File at Path (" + modelTokenizerFilePath.string() + ") doesnt exist");

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
            tok.contains("content"))
        {
            int id = tok["id"].get<int>();
            std::string content = tok["content"].get<std::string>();

            if (content.find_first_of("\x00\x01\x02\x03\x04\x05\x06\x07") == std::string::npos &&
                content.rfind("<|reserved_special_token", 0) != 0 &&
                content.rfind("[control_", 0) != 0)
            {
                _specialTokens[id] = content;
                _idToToken[id] = content;
                _vocab[content] = id;
            }
        }
    }

    if (ConfigManager::IsDebuggingEnabled.value())
    {
        std::cout << "Special Tokens Count: " << _specialTokens.size() << std::endl
                  << std::flush;
        for (auto &[id, token] : _specialTokens)
        {
            std::cout << "SPECIAL TOKEN: " << id << " -> " << token << std::endl
                      << std::flush;
        }
    }

    _byteEncoder = BuildByteEncoder();
    _byteFallbackVocab.reserve(256);

    // build byte decoder
    for (auto &kv : _byteEncoder)
    {
        unsigned char byte = kv.first;
        std::string token = kv.second;

        _byteDecoder[token] = byte;

        if (_IsByteFallbackEnabled)
        {
            auto it = _vocab.find(token);

            if (it != _vocab.end())
            {
                // performance optimization: fast lookup for byte tokens instead of full vocab search
                _byteFallbackVocab[token] = it->second;
            }
        }
    }

    BuildDecoderRules();

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

    // Create a cash for encoded special tokens
    for (auto &[id, token] : _specialTokens)
    {
        _encodedSpecialTokens[token] = EncodeWithoutChatTemplate(token);
    }
}

void TokenizerBase::BuildDecoderRules()
{
    _postDecoderRules.emplace_back(std::regex("Ġ"), " ");
    _postDecoderRules.emplace_back(std::regex("Ċ"), "\n");
    _postDecoderRules.emplace_back(std::regex("ĊĊ"), "\n\n");
    _postDecoderRules.emplace_back(std::regex("Ċ([0-9])"), "\n$1");
    _postDecoderRules.emplace_back("Ġ", " ");
}

std::vector<std::string> TokenizerBase::SplitText(const std::string &text)
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

std::string TokenizerBase::ByteToToken(unsigned char c)
{
    std::string value;
    auto it = _byteEncoder.find(c);

    if (it != _byteEncoder.end())
        value = it->second;
    else
        value = std::string(1, c);

    return value;
}

std::vector<std::string> TokenizerBase::ApplyBPE(const std::vector<std::string> &inputChars)
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

std::vector<int64_t> TokenizerBase::EncodeWithChatTemplate(std::string &text, bool isAddSystemHeaderInChatTemplate)
{
    return Encode(text, true, isAddSystemHeaderInChatTemplate);
}

std::vector<int64_t> TokenizerBase::EncodeWithoutChatTemplate(std::string &text)
{
    return Encode(text);
}

std::vector<int64_t> TokenizerBase::Encode(std::string &text, bool isApplyChatTemplate, bool isAddSystemHeaderInChatTemplate)
{
    std::string input = text;

    if (isApplyChatTemplate)
    {
        input = ChatTemplateProvider::Format(text, isAddSystemHeaderInChatTemplate);
    }

    if (ConfigManager::IsDebuggingEnabled.value())
    {
        size_t i = 0;

        std::cout << "Before:";

        while (i < input.size())
        {
            unsigned char c = static_cast<unsigned char>(input[i]);

            size_t len = 1;

            if ((c & 0x80) == 0)
                len = 1;
            else if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;

            std::cout << " [" << input.substr(i, len) << "] ";

            i += len;
        }

        std::cout << std::endl;
    }

    std::string encodedInput = EncodeBytes(input);

    if (ConfigManager::IsDebuggingEnabled.value())
    {
        std::cout << "After:";

        size_t i = 0;
        while (i < encodedInput.size())
        {
            unsigned char c = static_cast<unsigned char>(encodedInput[i]);

            size_t len = 1;

            if ((c & 0x80) == 0)
                len = 1;
            else if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;

            std::cout << " [" << encodedInput.substr(i, len) << "] ";

            i += len;
        }

        std::cout << std::endl;
    }

    std::vector<std::string> allPiecesAfterSegmentation = SegmentInput(encodedInput);

    if (ConfigManager::IsDebuggingEnabled.value())
    {
        for (auto &p : allPiecesAfterSegmentation)
        {
            std::cout << "SEG: " << p << std::endl;
        }
    }

    std::vector<std::string> tokens;

    for (auto &piece : allPiecesAfterSegmentation)
    {
        if (_vocab.find(piece) != _vocab.end())
        {
            tokens.push_back(piece);
        }
        else
        {
            if (!piece.empty())
            {
                std::vector<std::string> chars;

                for (size_t i = 0; i < piece.size();)
                {
                    unsigned char c = piece[i];

                    size_t len = 1;

                    if ((c & 0x80) == 0)
                        len = 1;
                    else if ((c & 0xE0) == 0xC0)
                        len = 2;
                    else if ((c & 0xF0) == 0xE0)
                        len = 3;
                    else if ((c & 0xF8) == 0xF0)
                        len = 4;

                    chars.push_back(piece.substr(i, len));
                    i += len;
                }

                if (ConfigManager::IsDebuggingEnabled.value())
                {
                    std::cout << "UTF8 SPLIT: ";
                    for (auto &c : chars)
                    {
                        std::cout << "[" << c << "]";
                    }
                    std::cout << std::endl;
                }
                
                chars = ApplyBPE(chars);

                tokens.insert(tokens.end(), chars.begin(), chars.end());
            }
        }
    }

    std::vector<int64_t> ids;

    for (auto &token : tokens)
    {
        if (!token.empty())
        {
            auto vocabKV = _vocab.find(token);

            if (vocabKV != _vocab.end())
            {
                ids.push_back(vocabKV->second);
            }
            else if (_IsByteFallbackEnabled)
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
            else
            {
                auto unkIt = _vocab.find("<unk>");

                if (unkIt != _vocab.end())
                {
                    ids.push_back(unkIt->second);
                }
            }
        }
    }

    return ids;
}

std::string TokenizerBase::EncodeBytes(const std::string &input)
{
    std::string encodedInput;

    for (size_t i = 0; i < input.size(); ++i)
    {
        unsigned char c = static_cast<unsigned char>(input[i]);
        encodedInput += ByteToToken(c);
    }

    return encodedInput;
}

std::vector<std::string> TokenizerBase::SegmentInput(std::string &encodedInput)
{
    std::vector<std::string> allPiecesAfterSegmentation;

    size_t segmentStart = 0;
    size_t segmentEnd = 0;

    std::string before;
    std::string specialToken;

    std::vector<std::string> split;

    while (!encodedInput.empty())
    {
        split.clear();
        specialToken.clear();
        before.clear();

        segmentStart = std::string::npos;
        segmentEnd = std::string::npos;

        for (auto &[id, token] : _specialTokens)
        {
            size_t pos = encodedInput.find(token);

            if (pos != std::string::npos)
            {
                if (segmentStart == std::string::npos || pos < segmentStart)
                {
                    segmentStart = pos;
                    specialToken = token;
                }
            }
        }

        if (segmentStart == std::string::npos)
        {
            split = SplitText(encodedInput);
            encodedInput.clear();
        }
        else
        {
            before = encodedInput.substr(0, segmentStart);

            segmentEnd = segmentStart + specialToken.length();

            if (!before.empty())
                split = SplitText(before);

            encodedInput = encodedInput.substr(segmentEnd);
        }

        if (!split.empty())
            allPiecesAfterSegmentation.insert(allPiecesAfterSegmentation.end(), split.begin(), split.end());

        if (!specialToken.empty())
            allPiecesAfterSegmentation.push_back(specialToken);
    }

    return allPiecesAfterSegmentation;
}

std::string TokenizerBase::Decode(const int64_t tokenID)
{
    std::ostringstream builder;

    auto tokenIt = _idToToken.find(tokenID);
    bool isSpecial = (_specialTokens.count(tokenID) > 0);

    if (!isSpecial && tokenIt != _idToToken.end())
    {
        std::string tokenValue = tokenIt->second;

        if (_IsByteFallbackEnabled)
        {
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
                    builder.write(reinterpret_cast<const char *>(&it->second), 1);
                }
                else
                {
                    builder << piece;
                }

                i += len;
            }
        }
        else
            builder << tokenValue;
    }

    std::string decoded;

    if (isSpecial && tokenIt != _idToToken.end())
        decoded = tokenIt->second;
    else
    {
        decoded = builder.str();

        if (!decoded.empty())
        {
            for (auto &[pattern, repl] : _postDecoderRules)
            {
                decoded = std::regex_replace(decoded, pattern, repl);
            }
        }
    }

    return decoded;
}

std::unordered_map<int, std::string> TokenizerBase::GetSpecialTokens()
{
    return _specialTokens;
}

std::unordered_map<std::string, std::vector<int64_t>> TokenizerBase::GetEncodedSpecialTokens()
{
    return _encodedSpecialTokens;
}
