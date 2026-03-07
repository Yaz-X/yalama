#pragma once

#include "ConfigManager.h"

#define TORCH_CHECKER(cond) \
    do \
    { \
        if (ConfigManager::IsTorchChecksEnabled) \
        { \
            TORCH_CHECK(cond); \
        } \
    } while (0)

    