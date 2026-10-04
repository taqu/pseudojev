#ifndef INC_PJEV_HTTP_SERVER_H_
#define INC_PJEV_HTTP_SERVER_H_
#include "server_config.h"
#include <functional>
#include <memory>
#include <string>
#include <cstdint>

namespace pjev {

using GetHandler  = std::function<void(int32_t& status, std::string& body)>;
using PostHandler = std::function<void(const std::string& req_body,
                                       int32_t& status,
                                       std::string& resp_body)>;

class PjevHttpServer {
public:
    PjevHttpServer();
    ~PjevHttpServer();

    void configure(const ServerConfig& cfg);
    void add_get (const std::string& path, GetHandler  h);
    void add_post(const std::string& path, PostHandler h);

    // Block until stop() is called. Returns false if bind fails.
    bool start();
    // Thread-safe. Safe to call from signal handler (sets stop flag + closes socket).
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pjev
#endif
