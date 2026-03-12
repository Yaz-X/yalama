#include "MistralTokenizer.h"
#include "ChatTemplateProvider.h"

//std::vector<std::string> MistralTokenizer::SegmentInput(std::string &encodedInput)
//{
//    std::vector<std::string> allPiecesAfterSegmentation;
//
//    size_t segmentStart = 0;
//    size_t segmentEnd = 0;
//
//    std::string before;
//    std::string specialToken;
//
//    std::vector<std::string> split;
//
//    while (!encodedInput.empty())
//    {
//        split.clear();
//        specialToken.clear();
//        before.clear();
//
//        segmentStart = std::string::npos;
//        segmentEnd = std::string::npos;
//
//        for (auto &[id, token] : _specialTokens)
//        {
//            size_t pos = encodedInput.find(token);
//
//            if (pos != std::string::npos)
//            {
//                if (segmentStart == std::string::npos || pos < segmentStart)
//                {
//                    segmentStart = pos;
//                    specialToken = token;
//                }
//            }
//        }
//
//        if (segmentStart == std::string::npos)
//        {
//            split = SplitText(encodedInput);
//            encodedInput.clear();
//        }
//        else
//        {
//            before = encodedInput.substr(0, segmentStart);
//
//            segmentEnd = segmentStart + specialToken.length();
//
//            if (!before.empty())
//                split = SplitText(before);
//
//            encodedInput = encodedInput.substr(segmentEnd);
//        }
//
//        if (!split.empty())
//            allPiecesAfterSegmentation.insert(allPiecesAfterSegmentation.end(), split.begin(), split.end());
//
//        if (!specialToken.empty())
//            allPiecesAfterSegmentation.push_back(specialToken);
//    }
//
//    return allPiecesAfterSegmentation;
//}