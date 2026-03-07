#pragma once

#include <regex>
#include <string>
#include <vector>
#include <unordered_map>

struct VectorHash
{
    size_t operator()(const std::vector<std::string>& v) const
    {
        size_t hash = 0;

        for (const auto& s : v)
        {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }

        return hash;
    }
};

class Tokenizer
{
public:
 
    static void Init();
    static std::vector<int64_t> EncodeWithChatTemplate(std::string &text, bool isAddSystemHeaderInChatTemplate);
    static std::vector<int64_t> EncodeWithoutChatTemplate(std::string &text);
    static std::string Decode(const std::vector<int64_t> &ids);    
    static std::unordered_map<int, std::string> GetSpecialTokens();
    static std::unordered_map<std::string, std::vector<int64_t>> GetEncodedSpecialTokens();
    static bool HasChatTemplate();
   

private:
    inline static bool _hasChatTemplate = false;
    inline static std::unordered_map<std::string, int> _byteFallbackVocab;
    inline static std::unordered_map<int, std::string> _byteEncoder;
    inline static std::unordered_map<std::string, unsigned char> _byteDecoder;
    inline static std::vector<std::pair<std::regex, std::string>> _decoderRules;
    inline static std::unordered_map<std::vector<std::string>, std::vector<std::string>, VectorHash> _bpeCache;
    inline static std::unordered_map<int, std::string> _specialTokens;    
    inline static std::unordered_map<std::string, std::vector<int64_t>> _encodedSpecialTokens;        
    inline static std::unordered_map<std::string, int> _vocab;
    inline static std::unordered_map<int, std::string> _idToToken;
    inline static std::unordered_map<std::string, int> _mergeRanks;

    static std::string ByteToToken(unsigned char c);
    static std::vector<std::string> LlamaSplit(const std::string &text);
    static void LoadFromJson();
    static std::unordered_map<int, std::string> BuildByteEncoder();
    static std::vector<std::string> ApplyBPE(const std::vector<std::string>& inputChars);
    static std::vector<int64_t> Encode(std::string &text, bool isApplyChatTemplate = false, bool isAddSystemHeaderInChatTemplate = false);

};
