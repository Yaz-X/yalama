#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <optional>

class ConfigManager
{
public:
    // Load once at startup
    static void Load(std::string configFolderPath);

    // Model Configs
    inline static int NumLayers = 0;
    inline static int HiddenSize = 0;
    inline static int NumHeads = 0;
    inline static int NumKVHeads = 0;
    inline static int64_t VocabSize = 0;
    inline static int HeadDim = 0;
    inline static float Eps = 0.0f;
    inline static float RopeTheta = 0.0f;
    inline static float RopeFactor = 1.0f;
    inline static float RopeHighFreq = 1.0f;
    inline static float RopeLowFreq = 1.0f;
    inline static int RopeOrigMaxPos = 0;
    inline static int MaxSequenceLength = 0;
    inline static std::vector<int64_t> EosIds;

    // yalama Configs
    inline static std::string ModelPath = "";
    inline static std::string ConfigPath = "";
    inline static std::string LogsPath = "";
    inline static int ServicePort = 0;
    inline static std::optional<bool> IsServicesRunMode;
    inline static std::optional<bool> IsShowLoadedWeights;
    inline static std::optional<bool> IsDebuggingEnabled;
    inline static std::optional<bool> IsKVCacheEnabled;
    inline static std::optional<bool> IsGreedy;
    inline static std::optional<bool> IsTorchChecksEnabled;
    inline static int KVCacheSizeInGB = 0;    
    inline static int TopK = 0;
    inline static float Temp = 0;
    inline static int MaxGeneratedTokensPerNonInstruct = 500;
    inline static int HttpThreadsPoolSize  = 0;
    inline static int HttpMaxQueueWaitSeconds  = 60;        
    
private:
    inline static bool _isLoaded = false;
    static void Validate();
    static void PrintPropertyLoaded(std::string key, std::string value);
    static void PrintPropertyNotFound(std::string key, std::string defaultValue);
};
