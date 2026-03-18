#pragma once

#include <regex>
#include <string>
#include <vector>
#include <unordered_map>

struct VectorHash
{
    size_t operator()(const std::vector<std::string> &v) const
    {
        size_t hash = 0;

        for (const auto &s : v)
        {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }

        return hash;
    }
};

class TokenizerBase
{

protected:
    TokenizerBase() = default;
    
    bool _isByteFallbackEnabled = true;    
    std::vector<std::string> _ignoredAddedTokens;
    std::unordered_map<std::string, int> _byteFallbackVocab;
    std::unordered_map<int, std::string> _byteEncoder;
    std::unordered_map<std::string, unsigned char> _byteDecoder;    
    std::vector<std::pair<std::regex, std::string>> _postDecoderRules;
    
    std::unordered_map<std::vector<std::string>, std::vector<std::string>, VectorHash> _bpeCache;
    std::unordered_map<int, std::string> _specialTokens;
    std::unordered_map<std::string, std::vector<int64_t>> _encodedSpecialTokens;
    std::unordered_map<std::string, int> _vocab;
    std::unordered_map<int, std::string> _idToToken;
    std::unordered_map<std::string, int> _mergeRanks;

    virtual void BuildDecoderRules();
    virtual std::string EncodeBytes(const std::string& input);
    virtual size_t GetCharByteLength(unsigned char c);
    virtual std::string ByteToToken(unsigned char c);
    virtual std::vector<std::string> SplitText(const std::string &text);
    virtual void LoadFromJson();    
    virtual std::unordered_map<int, std::string> BuildByteEncoder();
    virtual std::vector<std::string> ApplyBPE(const std::vector<std::string> &inputChars);

    virtual std::vector<int64_t> Encode(std::string &text, bool isApplyChatTemplate = false, bool isAddSystemHeaderInChatTemplate = false);
    virtual std::vector<std::string> SegmentInput(std::string& encodedInput);
    
public:
    void Init();
    
    std::vector<int64_t> EncodeWithChatTemplate(std::string &text, bool isAddSystemHeaderInChatTemplate);
    std::vector<int64_t> EncodeWithoutChatTemplate(std::string &text);
    virtual std::string Decode(const int64_t tokenID);
    std::unordered_map<int, std::string> GetSpecialTokens();
    std::unordered_map<std::string, std::vector<int64_t>> GetEncodedSpecialTokens();
    virtual ~TokenizerBase() = default;
};
