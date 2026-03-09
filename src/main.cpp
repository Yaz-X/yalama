/*
    YALAMA Runtime
    Copyright 2026 Yazeed Hamdan
    Licensed under the Apache License, Version 2.0
    See LICENSE file in project root.
*/
#include "Helpers.h"
#include "ChatSession.h"
#include "ConfigManager.h"
#include "RuntimeEnvironment.h"
#include "Constants.h"
#include <OpenAIService.h>
#include <string>
#include <json.hpp>
#include <csignal>
#include <sstream>
#include "Tokenizer.h"

static std::atomic<bool> _IsInterrupted = false;

std::string ParseArgs(int argsCount, char **args)
{
    std::cout << "Reading Args...." << std::endl;

    std::string configFolderPath;

    for (int i = 1; i < argsCount; i++)
    {
        std::string arg = TrimToLower(args[i]);

        if (arg == "--model" && i + 1 < argsCount)
        {
            std::string modelPath = Trim(args[i + 1]);
            ConfigManager::ModelPath = ResolvePath(modelPath);

            std::cout << "Model Path: " << modelPath << std::endl;
            i++;
        }
        else if (arg == "--config" && i + 1 < argsCount)
        {
            std::string pathValue = ResolvePath(Trim(args[i + 1]));
            std::filesystem::path path = pathValue;

            if (!std::filesystem::exists(path))
                std::cout << "Config Folder Path at (" << path << ") doesnt exist, will be ignored" << std::endl;
            else
            {
                configFolderPath = path.string();
                std::cout << "config Folder supplied: " << configFolderPath << std::endl;
            }

            i++;
        }
        else if (arg == "--logs" && i + 1 < argsCount)
        {
            std::filesystem::path path = ResolvePath((args[i + 1]));

            if (!std::filesystem::exists(path))
                std::cout << "Folder at Path (" << path << ") doesnt exist, will be ignored" << std::endl;
            else
            {
                ConfigManager::LogsPath = path.string();
                std::cout << "Log File Folder Path supplied: " << path << std::endl;
            }

            i++;
        }
        else if (arg == "--mode" && i + 1 < argsCount)
        {
            std::string value = Trim(args[i + 1]);
            std::ostringstream loadedMsg;
            loadedMsg << "Service run mode supplied: " << value << std::endl;

            if (value == "0")
            {
                ConfigManager::IsServicesRunMode = false;
                std::cout << loadedMsg.str();
            }
            else if (value == "1")
            {
                ConfigManager::IsServicesRunMode = true;
                std::cout << loadedMsg.str();
            }
            else
                std::cout << "invalid value supplied for --mode arg, will use default value 0 as services mode" << std::endl;

            i++;
        }
        else if (arg == "--port" && i + 1 < argsCount)
        {
            try
            {
                ConfigManager::ServicePort = std::stoi(Trim(args[i + 1]));
                std::cout << "Services port supplied: " << Trim(args[i + 1]) << std::endl;
            }
            catch (...)
            {
                std::cout << "Invalid Services port supplied for arg --port, will use default port 5067" << std::endl;
            }

            i++;
        }
        else if (arg == "--debug" && i + 1 < argsCount)
        {
            try
            {
                std::string value = Trim(args[i + 1]);
                std::ostringstream loadedMsg;
                loadedMsg << "debug supplied: " << value << std::endl;

                if (value == "1")
                {
                    ConfigManager::IsDebuggingEnabled = true;
                    std::cout << loadedMsg.str();
                }
                else if (value == "0")
                {
                    ConfigManager::IsDebuggingEnabled = false;
                    std::cout << loadedMsg.str();
                }
                else
                    std::cout << "invalid value supplied for --debug arg, will use default as 0 (false)" << std::endl;
            }
            catch (...)
            {
                std::cout << "Invalid value supplied for arg --debug, will use default as 0 (false)" << std::endl;
            }

            i++;
        }
        else if (arg == "--istorchvalidationsenabled" && i + 1 < argsCount)
        {
            try
            {
                std::string value = Trim(args[i + 1]);
                std::ostringstream loadedMsg;
                loadedMsg << "isTorchValidationsEnabled supplied: " << value << std::endl;

                if (value == "1")
                {
                    ConfigManager::IsTorchChecksEnabled = true;
                    std::cout << loadedMsg.str();
                }
                else if (value == "0")
                {
                    ConfigManager::IsTorchChecksEnabled = false;
                    std::cout << loadedMsg.str();
                }
                else
                    std::cout << "invalid value supplied for --isTorchValidationsEnabled arg, will use default as 0 (false)" << std::endl;
            }
            catch (...)
            {
                std::cout << "Invalid value supplied for arg --isTorchValidationsEnabled, will use default as 0 (false)" << std::endl;
            }

            i++;
        }
        else if (arg == "--iskvcacheenabled" && i + 1 < argsCount)
        {
            try
            {
                std::string value = Trim(args[i + 1]);
                std::ostringstream loadedMsg;
                loadedMsg << "isKVCacheEnabled supplied: " << value << std::endl;

                if (value == "1")
                {
                    ConfigManager::IsKVCacheEnabled = true;
                    std::cout << loadedMsg.str();
                }
                else if (value == "0")
                {
                    ConfigManager::IsKVCacheEnabled = false;
                    std::cout << loadedMsg.str();
                }
                else
                    std::cout << "invalid value supplied for --isKVCacheEnabled arg, will use default as 0 (false)" << std::endl;
            }
            catch (...)
            {
                std::cout << "Invalid value supplied for arg --isKVCacheEnabled, will use default as 0 (false)" << std::endl;
            }

            i++;
        }
        else if (arg == "--kvcachesizeingb" && i + 1 < argsCount)
        {
            try
            {
                ConfigManager::KVCacheSizeInGB = std::stoi(Trim(args[i + 1]));
                std::cout << "kvCacheSizeInGB supplied: " << Trim(args[i + 1]) << std::endl;
            }
            catch (...)
            {
                std::cout << "Invalid KV Cache Size supplied for arg --kvCacheSizeInGB, will use default 2GB" << std::endl;
            }

            i++;
        }
        else if (arg == "--isgreedy" && i + 1 < argsCount)
        {
            try
            {
                std::string value = Trim(args[i + 1]);
                std::ostringstream loadedMsg;
                loadedMsg << "isGreedy supplied: " << value << std::endl;

                if (value == "1")
                {
                    ConfigManager::IsGreedy = true;
                    std::cout << loadedMsg.str();
                }
                else if (value == "0")
                {
                    ConfigManager::IsGreedy = false;
                    std::cout << loadedMsg.str();
                }
                else
                    std::cout << "invalid value supplied for --isGreedy arg, will use default as true" << std::endl;
            }
            catch (...)
            {
                std::cout << "Invalid value supplied for arg --isGreedy, will use default as true" << std::endl;
            }

            i++;
        }
        else if (arg == "--topk" && i + 1 < argsCount)
        {
            try
            {
                int value = std::stoi(Trim(args[i + 1]));
                std::cout << "topk supplied: " << Trim(args[i + 1]) << std::endl;

                if (value > 40 || value < 2)
                    std::cout << "property topk must be between 2 and 40, it will be set to 40" << std::endl;
                else
                    ConfigManager::TopK = value;
            }
            catch (...)
            {
                std::cout << "Invalid topk value supplied for arg --topk, will use default 40" << std::endl;
            }

            i++;
        }
        else if (arg == "--temp" && i + 1 < argsCount)
        {
            try
            {
                float value = std::stof(Trim(args[i + 1]));

                std::cout << "temp supplied: " << Trim(args[i + 1]) << std::endl;

                if (value > 0.7 || value < 0.05)
                    std::cout << "property temp must be between 0.05 and 0.7, it will be set to 0.6" << std::endl;
                else
                    ConfigManager::Temp = value;
            }
            catch (...)
            {
                std::cout << "Invalid temp value supplied for arg --temp, will use default 0.6" << std::endl;
            }

            i++;
        }
        else if (arg == "--showloadedweights" && i + 1 < argsCount)
        {
            try
            {
                std::string value = Trim(args[i + 1]);
                std::ostringstream loadedMsg;
                loadedMsg << "Show Loaded Weights supplied: " << value << std::endl;

                if (value == "1")
                {
                    ConfigManager::IsShowLoadedWeights = true;
                    std::cout << loadedMsg.str();
                }
                else if (value == "0")
                {
                    ConfigManager::IsShowLoadedWeights = false;
                    std::cout << loadedMsg.str();
                }
                else
                    std::cout << "invalid value supplied for --showloadedweights arg, will use default as 0 (false)" << std::endl;
            }
            catch (...)
            {
                std::cout << "Invalid value supplied for arg --showloadedweights, will use default as 0 (false)" << std::endl;
            }

            i++;
        }
    }

    return configFolderPath;
}

void TermSignalHandler()
{
    _IsInterrupted = true;
    
    close(STDIN_FILENO);

    if (ConfigManager::IsServicesRunMode.value())
        OpenAIService::Shutdown();
}

int main(int argsCount, char **args)
{
    std::string modelArgMissing = R"(Missing required Model HF SafeTensors Path argument or path for --model <Path> doesnt exist. Use --model <path>, if using default HF download location then it will be 
        ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/[SNAP_SHOT_HASH])";

    auto configFolderPath = ParseArgs(argsCount, args);

    std::cout << "Loading Configs...." << std::endl;
    ConfigManager::Load(configFolderPath);

    if (ConfigManager::ModelPath.empty() || !std::filesystem::exists(ConfigManager::ModelPath))
        throw std::runtime_error(modelArgMissing);

    std::cout << std::endl
              << std::endl
              << "========== Final Resolved Configuration ==========" << std::endl;

    std::cout << "Model Path: " << ConfigManager::ModelPath << std::endl;
    std::cout << "Logs Path: " << (ConfigManager::LogsPath.empty() ? "." : ConfigManager::LogsPath) << std::endl;
    std::cout << "Run Mode: " << (ConfigManager::IsServicesRunMode.value() ? "Services" : "REPL") << std::endl;
    std::cout << "Service Port: " << ConfigManager::ServicePort << std::endl;
    std::cout << "Debug: " << (ConfigManager::IsDebuggingEnabled.value() ? "true" : "false") << std::endl;
    std::cout << "Torch Validations: " << (ConfigManager::IsTorchChecksEnabled.value() ? "true" : "false") << std::endl;
    std::cout << "KV Cache Enabled: " << (ConfigManager::IsKVCacheEnabled.value() ? "true" : "false") << std::endl;
    std::cout << "KV Cache Size (GB): " << ConfigManager::KVCacheSizeInGB << std::endl;
    std::cout << "Sampling Mode: " << (ConfigManager::IsGreedy.value() ? "Greedy" : "TopK + Temp") << std::endl;

    if (!ConfigManager::IsGreedy.value())
    {
        std::cout << "TopK: " << ConfigManager::TopK << std::endl;
        std::cout << "Temperature: " << ConfigManager::Temp << std::endl;
    }

    std::cout << "Show Loaded Weights: " << (ConfigManager::IsShowLoadedWeights.value() ? "true" : "false") << std::endl;
    std::cout << "==================================================" << std::endl
              << std::endl;

    nlohmann::json messages = nlohmann::json::array();
    std::string assistantResponse;

    ChatSession session;

    std::string modeString = ConfigManager::IsServicesRunMode.value() ? "OpenAI Compatible Services" : "REPL";

    std::cout << "Mode is set to " << modeString << ", you can set the mode by passing --mode 0 for services and --mode 1 for REPL, or in yalama_config file, Default is Services if the arg is not supplied" << std::endl;

    std::cout << R"(

    ██╗   ██╗ █████╗ ██╗      █████╗ ███╗   ███╗ █████╗ 
    ╚██╗ ██╔╝██╔══██╗██║     ██╔══██╗████╗ ████║██╔══██╗
     ╚████╔╝ ███████║██║     ███████║██╔████╔██║███████║
      ╚██╔╝  ██╔══██║██║     ██╔══██║██║╚██╔╝██║██╔══██║
       ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██║  ██║
       ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

)" << std::endl;

    std::signal(SIGTERM, [](int)
                { TermSignalHandler(); });

    std::signal(SIGINT, [](int)
                { TermSignalHandler(); });

    if (ConfigManager::IsServicesRunMode.value())
    {

        OpenAIService::Start(ConfigManager::ServicePort);

        std::cout << "Model is running..." << std::endl
                  << std::flush;
    }
    else
    {

        bool end = false;

        if (isatty(fileno(stdin)))
        {
            while (!end)
            {
                if (!assistantResponse.empty())
                    messages.push_back({{"role", "assistant"}, {"content", assistantResponse}});

                assistantResponse.clear();
                std::string input;

                std::cout << "\nPrompt>> ";
                std::getline(std::cin, input);

                if (_IsInterrupted || !std::cin || input == "exit")
                {
                    end = true;                    
                }
                else if (!input.empty())
                {

                    messages.push_back({{"role", "user"}, {"content", input}});

                    nlohmann::json request;
                    request["messages"] = messages;

                    std::string requestStr = request.dump();

                    session.Generate(requestStr, [&](const std::string &token)
                                     { 
                            assistantResponse += token;
                            std::cout << token << std::flush; });

                    std::cout << std::endl;
                }
            }
        }
        else
            std::cout << "REPL requires interactive terminal. Run docker with -it." << std::endl;
    }

    return 0;
}
