#include "QwenTokenizer.h"
#include "ChatTemplateProvider.h"

QwenTokenizer::QwenTokenizer()
{    
}

std::vector<std::string> QwenTokenizer::SplitText(const std::string &text)
{
    std::vector<std::string> result;

    result.push_back(text);

    return result;
}

std::string QwenTokenizer::Decode(const int64_t tokenID)
{
    std::ostringstream builder;

    auto tokenIt = _idToToken.find(tokenID);
    bool isSpecial = (_specialTokens.count(tokenID) > 0);

    if (!isSpecial && tokenIt != _idToToken.end())
    {
        const std::string &tokenValue = tokenIt->second;

        if (_isByteFallbackEnabled)
        {
            size_t i = 0;

            while (i < tokenValue.size())
            {
                unsigned char c = static_cast<unsigned char>(tokenValue[i]);

                // ASCII → 1 byte
                if (c < 128)
                {
                    std::string piece(1, tokenValue[i]);

                    auto it = _byteDecoder.find(piece);

                    if (it != _byteDecoder.end())
                    {
                        builder.write(reinterpret_cast<const char *>(&it->second), 1);
                    }
                    else
                    {
                        builder << piece;
                    }

                    i += 1;
                }
                else
                {
                    // UTF-8 multi-byte
                    int len = GetCharByteLength(c);

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
        }
        else
        {
            builder << tokenValue;
        }
    }

    std::string decoded;

    if (isSpecial && tokenIt != _idToToken.end())
    {
        decoded = tokenIt->second;
    }
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