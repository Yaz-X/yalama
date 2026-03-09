#pragma once

#define CPPHTTPLIB_NO_EXCEPTIONS
#include "ChatSession.h"
#include <httplib.h>
#include "GenerationResult.h"
#include <json.hpp>

class OpenAIService
{

public:
    static void Start(int port);
    static void Shutdown();

private:
    static httplib::Server _httpServer;

    static void HandleHealth(const httplib::Request &req, httplib::Response &res);
    static void HandleModel(const httplib::Request &req, httplib::Response &res);
    static void HandleCompletion(const httplib::Request &req, httplib::Response &res);    
    static nlohmann::json FormatError(GenerationResult result);
    
};