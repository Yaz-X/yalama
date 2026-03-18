#include "MistralTokenizer.h"
#include "ChatTemplateProvider.h"
#include "Helpers.h"

MistralTokenizer::MistralTokenizer()
{
    _isByteFallbackEnabled = false;
}
std::vector<std::string> MistralTokenizer::SplitText(const std::string &text)
{
    std::vector<std::string> result;

    result.push_back(text);

    return result;
}

void MistralTokenizer::BuildDecoderRules()
{
    _postDecoderRules.emplace_back("▁", " ");
    _postDecoderRules.emplace_back("<0x0A>", "\n");
}

std::string MistralTokenizer::EncodeBytes(const std::string &input)
{
    std::string encodedInput = input;

    ReplaceAll(encodedInput, " ", "▁");

    return encodedInput;
}
std::string MistralTokenizer::ByteToToken(unsigned char c)
{
    std::string value;

    if ((c & 0x80) == 0)
    {
        auto it = _byteEncoder.find(c);

        if (it != _byteEncoder.end())
            value = it->second;
        else
            value = std::string(1, static_cast<char>(c));
    }
    else
    {
        value = std::string(1, static_cast<char>(c));
    }

    return value;
}

std::unordered_map<int, std::string> MistralTokenizer::BuildByteEncoder()
{
    std::unordered_map<int, std::string> result;

    for (int byte = 0; byte < 256; byte++)
    {
        std::string value(1, static_cast<char>(byte));
        result[byte] = value;
    }

    return result;
}

std::vector<std::string> MistralTokenizer::ApplyBPE(const std::vector<std::string> &inputChars)
{
    std::vector<std::string> result;

    if (!inputChars.empty())
    {
        result = TokenizerBase::ApplyBPE(inputChars);
    }

    return result;
}