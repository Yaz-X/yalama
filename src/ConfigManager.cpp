#include "ConfigManager.h"
#include "Constants.h"
#include "Helpers.h"
#include <fstream>
#include <iostream>
#include "json.hpp"
#include "Helpers.h"

using json = nlohmann::json;

void ConfigManager::Load(std::string configFolderPath)
{
    if (!_isLoaded)
    {

        configFolderPath = ResolvePath(configFolderPath);

        json jsonDeserializer;
        std::filesystem::path yalamaConfigFilePath = configFolderPath;
        yalamaConfigFilePath = yalamaConfigFilePath / Constants::CONFIG_FILE_NAME;

        if (std::filesystem::exists(yalamaConfigFilePath))
        {
            std::ifstream yalamaConfigFile(yalamaConfigFilePath);

            if (!yalamaConfigFile.is_open())
            {
                std::cout << "Couldnt open file " << yalamaConfigFilePath.string() << ", settings default properties" << std::endl;
            }
            else
            {
                std::cout << "Loading yalama Configs from  " << yalamaConfigFilePath.string() << std::endl;

                yalamaConfigFile >> jsonDeserializer;
                std::string key;

                if (ModelPath.empty())
                {
                    key = "model";
                    if (jsonDeserializer.contains(key))
                    {
                        std::string value = jsonDeserializer.at(key).get<std::string>();

                        if (!value.empty())
                        {
                            ModelPath = value;
                            PrintPropertyLoaded(key, ModelPath);
                        }
                        else
                            std::cout << "Cofing property model is empty, ignoring..." << std::endl;
                    }
                }

                if (LogsPath.empty())
                {
                    key = "logs";
                    if (jsonDeserializer.contains(key))
                    {
                        LogsPath = jsonDeserializer.at(key).get<std::string>();
                        PrintPropertyLoaded(key, LogsPath);
                    }
                    else
                    {
                        LogsPath = "";
                        PrintPropertyNotFound(key, "current folder (.)");
                    }
                }

                LogsPath = ResolvePath(LogsPath);

                if (!IsServicesRunMode.has_value())
                {
                    key = "serviceMode";
                    if (jsonDeserializer.contains(key))
                    {
                        IsServicesRunMode = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsServicesRunMode.value()));
                    }
                    else
                    {
                        IsServicesRunMode = true;
                        PrintPropertyNotFound(key, "true");
                    }
                }

                if (HttpThreadsPoolSize == 0)
                {
                    key = "httpThreadsPoolSize";
                    if (jsonDeserializer.contains(key))
                    {
                        HttpThreadsPoolSize = jsonDeserializer.at(key).get<int>();

                        if (HttpThreadsPoolSize < 4 || HttpThreadsPoolSize > 64)
                        {
                            HttpThreadsPoolSize = 32;
                            std::cout << "property httpThreadsPoolSize should be between 4-64, it will be set to 32" << std::endl;
                        }
                        else
                            PrintPropertyLoaded(key, std::to_string(HttpThreadsPoolSize));
                    }
                    else
                    {
                        HttpThreadsPoolSize = 32;
                        PrintPropertyNotFound(key, "32");
                    }
                }

                if (!IsShowLoadedWeights.has_value())
                {
                    key = "showloadedweights";
                    if (jsonDeserializer.contains(key))
                    {
                        IsShowLoadedWeights = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsShowLoadedWeights.value()));
                    }
                    else
                    {
                        IsShowLoadedWeights = false;
                        PrintPropertyNotFound(key, "false");
                    }
                }

                if (!IsDebuggingEnabled.has_value())
                {
                    key = "debug";
                    if (jsonDeserializer.contains(key))
                    {
                        IsDebuggingEnabled = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsDebuggingEnabled.value()));
                    }
                    else
                    {
                        IsDebuggingEnabled = false;
                        PrintPropertyNotFound(key, "false");
                    }
                }

                if (!IsTorchChecksEnabled.has_value())
                {
                    key = "isTorchValidationsEnabled";
                    if (jsonDeserializer.contains(key))
                    {
                        IsTorchChecksEnabled = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsTorchChecksEnabled.value()));
                    }
                    else
                    {
                        IsTorchChecksEnabled = false;
                        PrintPropertyNotFound(key, "false");
                    }
                }

                if (!IsServiceLoggingEnabled.has_value())
                {
                    key = "isServiceLoggingEnabled";
                    if (jsonDeserializer.contains(key))
                    {
                        IsServiceLoggingEnabled = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsServiceLoggingEnabled.value()));
                    }
                    else
                    {
                        IsServiceLoggingEnabled = false;
                        PrintPropertyNotFound(key, "false");
                    }
                }

                if (!IsKVCacheEnabled.has_value())
                {
                    key = "isKVCacheEnabled";
                    if (jsonDeserializer.contains(key))
                    {
                        IsKVCacheEnabled = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsKVCacheEnabled.value()));
                    }
                    else
                    {
                        IsKVCacheEnabled = true;
                        PrintPropertyNotFound(key, "true");
                    }
                }

                if (ServicePort == 0)
                {
                    key = "port";
                    if (jsonDeserializer.contains(key))
                    {
                        ServicePort = jsonDeserializer.at(key).get<int>();
                        PrintPropertyLoaded(key, std::to_string(ServicePort));
                    }
                    else
                        PrintPropertyNotFound(key, "5067");
                }

                if (KVCacheSizeInGB == 0)
                {
                    key = "kvCacheSizeInGB";
                    if (jsonDeserializer.contains(key))
                    {
                        KVCacheSizeInGB = jsonDeserializer.at(key).get<int>();

                        if (KVCacheSizeInGB < 1)
                        {
                            KVCacheSizeInGB = 1;
                            std::cout << "property kvCacheSizeInGB has less than 1GB, it will be set to 1" << std::endl;
                        }
                        else
                            PrintPropertyLoaded(key, std::to_string(KVCacheSizeInGB));
                    }
                    else
                    {
                        KVCacheSizeInGB = 2;
                        PrintPropertyNotFound(key, "2GB");
                    }
                }

                if (!IsGreedy.has_value())
                {
                    key = "isGreedy";
                    if (jsonDeserializer.contains(key))
                    {
                        IsGreedy = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsGreedy.value()));
                    }
                    else
                    {
                        IsGreedy = true;
                        PrintPropertyNotFound(key, "true");
                    }
                }

                if (!IsThinkingEnabled.has_value())
                {
                    key = "isThinkingEnabled";
                    if (jsonDeserializer.contains(key))
                    {
                        IsThinkingEnabled = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(IsThinkingEnabled.value()));
                    }
                    else
                    {
                        IsThinkingEnabled = true;
                        PrintPropertyNotFound(key, "true");
                    }
                }

                if (!isPrintChatTemplateOutput.has_value())
                {
                    key = "isPrintChatTemplateOutput";
                    if (jsonDeserializer.contains(key))
                    {
                        isPrintChatTemplateOutput = jsonDeserializer.at(key).get<bool>();
                        PrintPropertyLoaded(key, std::to_string(isPrintChatTemplateOutput.value()));
                    }
                    else
                    {
                        isPrintChatTemplateOutput = true;
                        PrintPropertyNotFound(key, "false");
                    }
                }

                if (!IsGreedy.value_or(true))
                {
                    if (TopK == 0)
                    {
                        key = "topk";
                        if (jsonDeserializer.contains(key))
                        {
                            TopK = jsonDeserializer.at(key).get<int>();

                            if (TopK < 2 || TopK > 40)
                            {
                                TopK = 40;
                                std::cout << "property topk must be between 2 and 40, it will be set to 40" << std::endl;
                            }
                            else
                                PrintPropertyLoaded(key, std::to_string(TopK));
                        }
                        else
                        {
                            TopK = 40;
                            PrintPropertyNotFound(key, std::to_string(TopK));
                        }
                    }

                    if (Temp == 0)
                    {
                        key = "temp";
                        if (jsonDeserializer.contains(key))
                        {
                            Temp = jsonDeserializer.at(key).get<float>();

                            if (Temp < 0.05 || Temp > 0.7)
                            {
                                Temp = 0.6;
                                std::cout << "property temp must be between 0.05 and 0.6, it will be set to 0.6" << std::endl;
                            }
                            else
                                PrintPropertyLoaded(key, std::to_string(Temp));
                        }
                        else
                        {
                            Temp = 0.6;
                            PrintPropertyNotFound(key, std::to_string(Temp));
                        }
                    }
                }
                else
                    std::cout << "Sampling is set to Greedy, temp and topk properties will be ignored" << std::endl;
            }
        }
        else
            std::cout << "Couldnt find file " << yalamaConfigFilePath.string() << ", settings default properties" << std::endl;

        if (ModelPath.empty())
            throw std::runtime_error("Model path is not supplied in args nor in config, please set model safetensors path either as --model arg or in yalama_config.json and supply --config arg for folder path that contains the config file");
        else
            ModelPath = ResolvePath(ModelPath);

        // Resolved ModelPath from refs and find safetensors
        std::filesystem::path repoRoot = ModelPath;
        std::filesystem::path refFile = repoRoot / "refs" / "main";

        if (!std::filesystem::exists(refFile))
            throw std::runtime_error("Invalid HuggingFace repo: refs/main not found in " + repoRoot.string());

        std::ifstream ref(refFile);

        if (!ref.is_open())
            throw std::runtime_error("Failed to open HF ref file: " + refFile.string());

        std::string snapshotHash;
        std::getline(ref, snapshotHash);

        std::filesystem::path snapshotPath = repoRoot / "snapshots" / snapshotHash;

        if (!std::filesystem::exists(snapshotPath))
            throw std::runtime_error("Snapshot folder does not exist: " + snapshotPath.string());

        ModelPath = snapshotPath.string();

        std::cout << "Resolved snapshot: " << ModelPath << std::endl;

        if (!LogsPath.empty())
            LogsPath = ResolvePath(LogsPath);

        if (ServicePort == 0)
            ServicePort = 5067;

        if (!IsServicesRunMode.has_value())
            IsServicesRunMode = true;

        if (HttpThreadsPoolSize < 4 || HttpThreadsPoolSize > 64)
            HttpThreadsPoolSize = 32;

        if (KVCacheSizeInGB < 1)
            KVCacheSizeInGB = 2;

        if (!IsShowLoadedWeights.has_value())
            IsShowLoadedWeights = false;

        if (!IsDebuggingEnabled.has_value())
            IsDebuggingEnabled = false;

        if (!IsKVCacheEnabled.has_value())
            IsKVCacheEnabled = true;

        if (!IsGreedy.has_value())
            IsGreedy = true;

        if (!IsTorchChecksEnabled.has_value())
            IsTorchChecksEnabled = false;

        if (!IsServiceLoggingEnabled.has_value())
            IsServiceLoggingEnabled = false;

        if (!IsThinkingEnabled.has_value())
            IsThinkingEnabled = true;

        if (!isPrintChatTemplateOutput.has_value())
            isPrintChatTemplateOutput = false;

        if (TopK == 0)
            TopK = 40;

        if (Temp == 0)
            Temp = 0.6;

        // Model Config
        std::filesystem::path modelConfigFilePath = ModelPath;
        modelConfigFilePath = modelConfigFilePath / "config.json";

        std::cout << "ModelPath: [" << ModelPath << "]" << std::endl;
        std::cout << "Checking file: [" << modelConfigFilePath.string() << "]" << std::endl;

        if (!std::filesystem::exists(modelConfigFilePath))
            throw std::runtime_error("File (" + modelConfigFilePath.string() + ") doesnt exist");

        std::string configPath = modelConfigFilePath.string();

        std::ifstream modelConfigFile(configPath);

        if (!modelConfigFile.is_open())
            throw std::runtime_error("Failed to open model config: " + configPath);

        std::cout << "Loading Configs from  " << configPath << std::endl;

        modelConfigFile >> jsonDeserializer;

        std::string modelType = TrimToLower(jsonDeserializer.at("model_type").get<std::string>());

        if (modelType == "llama")
            ModelLoadedType = ModelType::LLama;
        else if (modelType == "mistral")
            ModelLoadedType = ModelType::Mistral;
        else if (modelType == "qwen2")
            ModelLoadedType = ModelType::Qwen2_5;
        else if (modelType == "qwen3")
            ModelLoadedType = ModelType::Qwen3;
        else
            throw std::runtime_error("Unsupported model type: " + modelType);

        NumLayers = jsonDeserializer.at("num_hidden_layers").get<int>();
        HiddenSize = jsonDeserializer.at("hidden_size").get<int>();
        NumHeads = jsonDeserializer.at("num_attention_heads").get<int>();
        NumKVHeads = jsonDeserializer.at("num_key_value_heads").get<int>();
        VocabSize = jsonDeserializer.at("vocab_size").get<int>();
        Eps = jsonDeserializer.at("rms_norm_eps").get<float>();
        RopeTheta = jsonDeserializer.at("rope_theta").get<float>();

        auto eosNode = jsonDeserializer["eos_token_id"];

        if (eosNode.is_array())
        {
            EosIds = eosNode.get<std::vector<int64_t>>();
        }
        else
        {
            EosIds = {eosNode.get<int64_t>()};
        }

        MaxSequenceLength = jsonDeserializer.at("max_position_embeddings").get<int>();
        RopeFactor = 1.0f;
        RopeHighFreq = 1.0f;
        RopeLowFreq = 1.0f;
        RopeOrigMaxPos = MaxSequenceLength;

        if (jsonDeserializer.contains("rope_scaling"))
        {
            auto ropeScaling = jsonDeserializer.at("rope_scaling");

            if (ropeScaling.contains("factor"))
                RopeFactor = ropeScaling.at("factor").get<float>();

            if (ropeScaling.contains("high_freq_factor"))
                RopeHighFreq = ropeScaling.at("high_freq_factor").get<float>();

            if (ropeScaling.contains("low_freq_factor"))
                RopeLowFreq = ropeScaling.at("low_freq_factor").get<float>();

            if (ropeScaling.contains("original_max_position_embeddings"))
                RopeOrigMaxPos = ropeScaling.at("original_max_position_embeddings").get<int>();
        }

        if (HiddenSize % NumHeads != 0)
            throw std::runtime_error("HiddenSize must be divisible by NumHeads");

        if (jsonDeserializer.contains("head_dim"))
            HeadDim = jsonDeserializer.at("head_dim").get<int>();
        else
            HeadDim = HiddenSize / NumHeads;

        // Tokenizer config
        std::filesystem::path modelTokenizerConfigFilePath = ModelPath;
        modelTokenizerConfigFilePath = modelTokenizerConfigFilePath / "tokenizer_config.json";

        if (!std::filesystem::exists(modelTokenizerConfigFilePath) || !std::filesystem::is_regular_file(modelTokenizerConfigFilePath))
            throw std::runtime_error("File at Path (" + modelTokenizerConfigFilePath.string() + ") doesnt exist");

        std::cout << "Loading Configs from  " << modelTokenizerConfigFilePath << std::endl;

        std::ifstream tokenizerConfigFileStream(modelTokenizerConfigFilePath);
        tokenizerConfigFileStream >> jsonDeserializer;

        HasChatTemplate = jsonDeserializer.contains("chat_template");

        if(HasChatTemplate)
        {
            std::string chatTemplate = jsonDeserializer["chat_template"].get<std::string>();
            IsModelSupportThinking = chatTemplate.find("<think>") != std::string::npos;
        }

        if (jsonDeserializer.contains("bos_token") && !jsonDeserializer["bos_token"].is_null())
            BosTokenString = jsonDeserializer["bos_token"].get<std::string>();

        EosTokenString = jsonDeserializer.at("eos_token").get<std::string>();
    }

    Validate();

    // yalama config

    _isLoaded = true;
}

void ConfigManager::Validate()
{
    if (NumLayers <= 0)
        throw std::runtime_error("Invalid NumLayers read from model config file");
    if (HiddenSize <= 0)
        throw std::runtime_error("Invalid HiddenSize read from model config file");
    if (NumHeads <= 0)
        throw std::runtime_error("Invalid NumHeads read from model config file");
    if (HeadDim <= 0)
        throw std::runtime_error("Invalid HeadDim read from model config file");
}

void ConfigManager::PrintPropertyLoaded(std::string key, std::string value)
{
    std::cout << "property " << key << "=" << value << " is loaded" << std::endl;
}

void ConfigManager::PrintPropertyNotFound(std::string key, std::string defaultValue)
{
    std::cout << "Couldnt find " << key << " property in file yalam_config.json, settings default value to " << defaultValue << std::endl;
}
