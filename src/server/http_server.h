#pragma once
#include <cstdint>
#include <functional>
#include <string>

// Handler called for each POST request.
// Sets status_out (HTTP status) and response_body_out.
using PostHandler = std::function<void(
    const std::string& request_body,
    int& status_out,
    std::string& response_body_out)>;

// Start the HTTP server and block until stopped.
// Returns false if the server fails to bind.
bool run_http_server(const std::string& host,
                     uint16_t port,
                     const std::string& path,
                     PostHandler handler);
