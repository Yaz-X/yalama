#pragma once

#include "Constants.h"
#include "ConfigManager.h"
#include <torch/torch.h>
#include <fstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <algorithm>

class TraceLogger
{
private:
    inline static std::ofstream logFileStream;
    inline static bool _isInitialized = false;

    static void InitFileStream()
    {
        if (ConfigManager::IsDebuggingEnabled.value() && !_isInitialized)
        {
            std::filesystem::path path = ConfigManager::LogsPath;
            path = path / Constants::LOG_FILE_NAME;

            if (std::filesystem::exists(path))
                std::filesystem::remove(path);

            logFileStream.open(path, std::ios::out | std::ios::app);
            _isInitialized = true;
        }
    }

public:
    static int _TraceStep;

    // Tensor Dump
    static void Dump(const std::string &name, const torch::Tensor &tensor, int step)
    {
        if (CheckIfFileOpened())
        {
            auto x = tensor.detach().cpu();

            logFileStream << name << " sizes=" << x.sizes() << " strides=" << x.strides() << std::endl;
            logFileStream.flush();
            int numOfTokensToLog = 10;

            if (x.dim() == 4) //[B,H,T,D]
            {
                numOfTokensToLog = x.size(2);
            }
            else if (x.dim() == 3) //[B,T,D]
            {
                numOfTokensToLog = x.size(1);
            }
            else if (x.dim() == 2) //[T,D]
            {
                numOfTokensToLog = x.size(0);
            }

           numOfTokensToLog = std::min(numOfTokensToLog, 10);

            for (int m = 0; m < numOfTokensToLog; m++)
            {
                torch::Tensor t;

                if (x.dim() == 4)
                {
                    t = x.index({0, 0, m}).slice(0, 0, 8);
                }
                else if (x.dim() == 3)
                {
                    t = x.index({0, m}).slice(0, 0, 8);
                }
                else if (x.dim() == 2)
                {
                    t = x.index({m}).slice(0, 0, 8);
                }

                logFileStream << "[DEBUG] " << std::setw(4) << std::setfill('0') << step
                              << " " << name << " Token " << m << ": ";

                for (int i = 0; i < t.size(0); i++)
                {
                    logFileStream << "[DEBUG] "
                                  << std::showpos
                                  << std::scientific
                                  << std::setprecision(6)
                                  << t[i].item<float>() << " ";
                }

                logFileStream << std::endl;
            }
        }
    }

    // String Logger
    static void DumpStr(const std::string &key, const std::string &value)
    {
        if (CheckIfFileOpened())
            logFileStream << "[DEBUG] " << key << ": " << value << "\n";
    }

    static void DumpLine(const std::string &line)
    {
        if (CheckIfFileOpened())
            logFileStream << "[DEBUG] " << line << std::endl;
    }

    static bool CheckIfFileOpened()
    {
        if (!ConfigManager::IsDebuggingEnabled.value())
            return false;

        InitFileStream();

        return logFileStream.is_open();
    }
};
