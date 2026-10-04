#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace pjev {

// Abstract connection (both client and server-side accepted connections)
class ILocalConn {
public:
    virtual ~ILocalConn() = default;
    // Send framed message (4-byte LE length prefix + data)
    virtual bool send_frame(const std::string& data) = 0;
    // Receive framed message. timeout_ms=-1 means block indefinitely.
    // Returns false on error/timeout/disconnect.
    virtual bool recv_frame(std::string& data, int timeout_ms = 30000) = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;
};

// Abstract server (listens for connections)
class ILocalServer {
public:
    virtual ~ILocalServer() = default;
    // Bind to the endpoint. Returns false on failure.
    virtual bool bind(const std::string& endpoint) = 0;
    // Accept one connection. timeout_ms=-1 blocks.
    virtual std::unique_ptr<ILocalConn> accept(int timeout_ms = -1) = 0;
    virtual void close() = 0;
    virtual bool is_bound() const = 0;
};

// Factory functions (platform-specific implementations)
std::unique_ptr<ILocalConn>   make_local_client(const std::string& endpoint, int timeout_ms = 5000);
std::unique_ptr<ILocalServer> make_local_server();

} // namespace pjev
