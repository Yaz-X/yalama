#include "LlamaTokenizer.h"
#include "ChatTemplateProvider.h"

//std::vector<std::string> LlamaTokenizer::SegmentInput(std::string &encodedInput)
//{
//     std::vector<std::string> allPiecesAfterSegmentation;
//
//    size_t segmentStart = 0;
//    size_t segmentEnd = 0;
//    std::string before;
//    std::string specialToken;
//    std::vector<std::string> split;
//    bool isDoneSegmentation = false;
//
//    while (!encodedInput.empty())
//    {
//        split.clear();
//        specialToken.clear();
//        before.clear();
//        segmentStart = 0;
//        segmentEnd = 0;
//
//        segmentStart = encodedInput.find("<|", 0);
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
//            segmentEnd = encodedInput.find("|>", segmentStart);
//
//            if (segmentEnd != std::string::npos)
//                segmentEnd += 2;
//
//            // if this is a damaged/invlid special token like <|.... without |>, then find the next proper <|... token
//            // if found, split the text till the valid special token, split it like normal text, if no remaining special tokens
//            //  then do nothing and the next turn in loop
//            if (segmentEnd == std::string::npos)
//            {
//                auto nextSegmentStart = encodedInput.find("<|", segmentStart + 2);
//
//                if (nextSegmentStart != std::string::npos)
//                    segmentEnd = nextSegmentStart;
//            }
//
//            if (segmentEnd != std::string::npos)
//            {
//                specialToken = encodedInput.substr(segmentStart, (segmentEnd - segmentStart));
//
//                if (!before.empty())
//                    split = SplitText(before);
//
//                encodedInput = encodedInput.substr(segmentEnd);
//            }
//            else
//            {
//                encodedInput = encodedInput.substr(segmentStart);
//                split = SplitText(encodedInput);
//                encodedInput.clear();
//            }
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