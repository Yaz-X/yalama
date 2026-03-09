#include "ConfigManager.h"
#include <OpenAIService.h>

httplib::Server OpenAIService::_httpServer;

void OpenAIService::Start(int port)
{
    _httpServer.Get("/health", HandleHealth);
    _httpServer.Get("/model", HandleModel);
    _httpServer.Post("/v1/chat/completions", HandleCompletion);

    _httpServer.listen("0.0.0.0", port);
    std::cout << "OpenAI Compatible Service is running at http://localhost:" << std::to_string(port) << std::endl
              << std::flush;
}

void OpenAIService::Shutdown()
{
    _httpServer.stop();
    std::cout << "OpenAI Compatible Service Stopped..." << std::endl
              << std::flush;
}

void OpenAIService::HandleHealth(const httplib::Request &request, httplib::Response &response)
{
    std::cout << "[HTTP] Request received: Get /health" << std::endl
              << std::flush;

    response.set_content("OK", "text/plain");
}

void OpenAIService::HandleModel(const httplib::Request &request, httplib::Response &response)
{
    std::cout << "[HTTP] Request received: Get /model" << std::endl
              << std::flush;

    response.set_content(ConfigManager::ModelPath, "text/plain");
}

void OpenAIService::HandleCompletion(const httplib::Request &request, httplib::Response &response)
{

    std::cout << "[HTTP] Request received: POST /v1/chat/completions" << std::endl
              << std::flush;

    std::string requestBody = request.body;
    nlohmann::json requestJson;

    bool isValidJson = true;

    try
    {
        requestJson = nlohmann::json::parse(requestBody);
    }
    catch (...)
    {
        isValidJson = false;
    }

    if (isValidJson)
    {
        std::string nonStreamingAssistantOutput = "\n";
        bool isStream = false;

        if (requestJson.contains("stream") &&
            requestJson["stream"].is_boolean())
        {
            isStream = requestJson["stream"].get<bool>();
        }

        if (isStream)
        {
            response.status = 200;

            response.set_chunked_content_provider(
                "text/event-stream",
                [request](size_t, httplib::DataSink &sink)
                {
                    std::atomic<bool> clientDisconnected = false;
                    ChatSession session;
                    std::string requestBody = request.body;

                    auto genResults = session.Generate(
                        requestBody,
                        [&](const std::string &token)
                        {
                            if (clientDisconnected)
                                return;

                            if (!token.empty())
                            {
                                nlohmann::json chunk;

                                chunk["choices"] = nlohmann::json::array({{{"delta", {{"content", token}}},
                                                                           {"index", 0}}});

                                std::string payload =
                                    "data: " + chunk.dump() + "\n\n";

                                if (!sink.write(payload.c_str(), payload.size()))
                                {
                                    clientDisconnected = true;
                                    return;
                                }
                            }
                        }, clientDisconnected);

                    if (!genResults.IsSuccess)
                    {
                        auto error = FormatError(genResults);

                        std::string payload =
                            "data: " + error.dump() + "\n\n";

                        sink.write(payload.c_str(), payload.size());
                        sink.done();
                        return true;
                    }

                    nlohmann::json finalChunk;

                    finalChunk["choices"] = nlohmann::json::array({{{"delta", nlohmann::json::object()},
                                                                    {"index", 0},
                                                                    {"finish_reason", "stop"}}});

                    std::string finalPayload =
                        "data: " + finalChunk.dump() + "\n\n";

                    sink.write(finalPayload.c_str(), finalPayload.size());

                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.c_str(), done.size());

                    sink.done();
                    return true;
                });
        }
        else
        {
            ChatSession session;
            auto genResult = session.Generate(requestBody, [&](const std::string &token)
                                              { nonStreamingAssistantOutput += token; });

            nlohmann::json reply;

            if (genResult.IsSuccess)
            {

                reply["id"] = "chatcmpl-yalama";
                reply["object"] = "chat.completion";

                reply["choices"] = nlohmann::json::array({{{"index", 0},
                                                           {"message", {{"role", "assistant"}, {"content", nonStreamingAssistantOutput}}},
                                                           {"finish_reason", "stop"}}});

                response.status = 200;
            }
            else
            {
                reply = FormatError(genResult);
                response.status = 400;
            }

            response.set_content(reply.dump(), "application/json");
        }
    }

    if (!isValidJson)
    {
        nlohmann::json error;

        error["error"] = {
            {"message", "Invalid JSON body"},
            {"type", "invalid_request_error"}};

        response.status = 400;
        response.set_content(error.dump(), "application/json");
    }

    std::cout << "[HTTP] Completed with status: "
              << response.status
              << std::endl
              << std::flush;
}

nlohmann::json OpenAIService::FormatError(GenerationResult genResults)
{
    nlohmann::json error;

    if (genResults.Error == GenerationError::InvalidPrompt)
        error["error"] = {
            {"message", "Invalid request format"},
            {"type", "invalid_prompt_error"}};
    else if (genResults.Error == GenerationError::KVCacheExceeded)
        error["error"] = {
            {"message", "OOM, KV Cache Capacity Exceeded, consider increasing KV Cache Capacity from args or yalam_config.json file"},
            {"type", "OOM_error"}};
    else if (genResults.Error == GenerationError::SequenceLengthExceeded)
        error["error"] = {
            {"message", "Max Sequence Length Exceeded, Make sure your prompt is less or equal to model max sequence length"},
            {"type", "invalid_seq_length_error"}};

    return error;
}