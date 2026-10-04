#include "local_transport.h"
#include "ipc_protocol.h"
#include <chrono>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace pjev {

// ---------------------------------------------------------------------------
// Windows Named Pipe implementation
// ---------------------------------------------------------------------------

// Helper: write exactly n bytes to a HANDLE
static bool write_all(HANDLE h, const void* buf, DWORD n)
{
    DWORD written = 0;
    while (written < n) {
        DWORD w = 0;
        if (!WriteFile(h, (const char*)buf + written, n - written, &w, nullptr))
            return false;
        written += w;
    }
    return true;
}

// Helper: read exactly n bytes from a HANDLE with optional timeout (overlapped)
static bool read_all(HANDLE h, void* buf, DWORD n, int timeout_ms)
{
    DWORD read_total = 0;
    while (read_total < n) {
        OVERLAPPED ov = {};
        ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!ov.hEvent) return false;

        DWORD rd = 0;
        BOOL ok = ReadFile(h, (char*)buf + read_total, n - read_total, &rd, &ov);
        if (!ok) {
            DWORD err = GetLastError();
            if (err == ERROR_IO_PENDING) {
                DWORD wait_ms = (timeout_ms < 0) ? INFINITE : (DWORD)timeout_ms;
                DWORD wait_result = WaitForSingleObject(ov.hEvent, wait_ms);
                if (wait_result == WAIT_OBJECT_0) {
                    if (!GetOverlappedResult(h, &ov, &rd, FALSE)) {
                        CloseHandle(ov.hEvent);
                        return false;
                    }
                } else {
                    // Timeout or error — cancel the pending IO
                    CancelIo(h);
                    CloseHandle(ov.hEvent);
                    return false;
                }
            } else {
                CloseHandle(ov.hEvent);
                return false;
            }
        }
        CloseHandle(ov.hEvent);
        read_total += rd;
        // Adjust remaining timeout
        if (timeout_ms > 0) {
            // Rough: just use full timeout per read chunk (acceptable for our usage)
        }
    }
    return true;
}

class WinPipeConn : public ILocalConn {
public:
    explicit WinPipeConn(HANDLE h) : h_(h) {}
    ~WinPipeConn() override { close(); }

    bool send_frame(const std::string& data) override
    {
        if (h_ == INVALID_HANDLE_VALUE) return false;
        uint32_t len = (uint32_t)data.size();
        if (len > IPC_MAX_FRAME_BYTES) return false;
        // Write length LE
        uint8_t hdr[4];
        hdr[0] = (uint8_t)(len & 0xFF);
        hdr[1] = (uint8_t)((len >> 8) & 0xFF);
        hdr[2] = (uint8_t)((len >> 16) & 0xFF);
        hdr[3] = (uint8_t)((len >> 24) & 0xFF);
        if (!write_all(h_, hdr, 4)) return false;
        if (len > 0 && !write_all(h_, data.data(), len)) return false;
        return true;
    }

    bool recv_frame(std::string& data, int timeout_ms = 30000) override
    {
        if (h_ == INVALID_HANDLE_VALUE) return false;
        uint8_t hdr[4] = {};
        if (!read_all(h_, hdr, 4, timeout_ms)) return false;
        uint32_t len = (uint32_t)hdr[0]
                     | ((uint32_t)hdr[1] << 8)
                     | ((uint32_t)hdr[2] << 16)
                     | ((uint32_t)hdr[3] << 24);
        if (len > IPC_MAX_FRAME_BYTES) return false;
        data.resize(len);
        if (len > 0 && !read_all(h_, &data[0], len, timeout_ms)) return false;
        return true;
    }

    void close() override
    {
        if (h_ != INVALID_HANDLE_VALUE) {
            CloseHandle(h_);
            h_ = INVALID_HANDLE_VALUE;
        }
    }

    bool is_open() const override { return h_ != INVALID_HANDLE_VALUE; }

private:
    HANDLE h_ = INVALID_HANDLE_VALUE;
};

class WinPipeServer : public ILocalServer {
public:
    WinPipeServer() = default;
    ~WinPipeServer() override { close(); }

    bool bind(const std::string& endpoint) override
    {
        endpoint_ = endpoint;
        std::wstring wep(endpoint.begin(), endpoint.end());
        pipe_h_ = CreateNamedPipeW(
            wep.c_str(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            1, // max instances
            65536, 65536,
            0, nullptr);
        if (pipe_h_ == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            // ERROR_ACCESS_DENIED means another server has this pipe
            (void)err;
            return false;
        }
        return true;
    }

    std::unique_ptr<ILocalConn> accept(int timeout_ms = -1) override
    {
        if (pipe_h_ == INVALID_HANDLE_VALUE) return nullptr;

        OVERLAPPED ov = {};
        ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!ov.hEvent) return nullptr;

        BOOL connected = ConnectNamedPipe(pipe_h_, &ov);
        if (!connected) {
            DWORD err = GetLastError();
            if (err == ERROR_IO_PENDING) {
                DWORD wait_ms = (timeout_ms < 0) ? INFINITE : (DWORD)timeout_ms;
                DWORD wait_result = WaitForSingleObject(ov.hEvent, wait_ms);
                if (wait_result != WAIT_OBJECT_0) {
                    CancelIo(pipe_h_);
                    CloseHandle(ov.hEvent);
                    return nullptr;
                }
                DWORD dummy = 0;
                if (!GetOverlappedResult(pipe_h_, &ov, &dummy, FALSE)) {
                    DWORD err2 = GetLastError();
                    if (err2 != ERROR_PIPE_CONNECTED) {
                        CloseHandle(ov.hEvent);
                        return nullptr;
                    }
                }
            } else if (err != ERROR_PIPE_CONNECTED) {
                CloseHandle(ov.hEvent);
                return nullptr;
            }
        }
        CloseHandle(ov.hEvent);

        // Transfer ownership of pipe_h_ to connection
        HANDLE conn_h = pipe_h_;
        pipe_h_ = INVALID_HANDLE_VALUE;

        // Re-create the server pipe for next connection
        std::wstring wep(endpoint_.begin(), endpoint_.end());
        pipe_h_ = CreateNamedPipeW(
            wep.c_str(),
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            1, 65536, 65536, 0, nullptr);
        // If this fails, the server can no longer accept new connections (acceptable for our use case)

        return std::make_unique<WinPipeConn>(conn_h);
    }

    void close() override
    {
        if (pipe_h_ != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_h_);
            pipe_h_ = INVALID_HANDLE_VALUE;
        }
    }

    bool is_bound() const override { return pipe_h_ != INVALID_HANDLE_VALUE; }

private:
    HANDLE      pipe_h_ = INVALID_HANDLE_VALUE;
    std::string endpoint_;
};

std::unique_ptr<ILocalConn> make_local_client(const std::string& endpoint, int timeout_ms)
{
    std::wstring wep(endpoint.begin(), endpoint.end());

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (true) {
        HANDLE h = CreateFileW(
            wep.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0, nullptr,
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,
            nullptr);

        if (h != INVALID_HANDLE_VALUE) {
            return std::make_unique<WinPipeConn>(h);
        }

        DWORD err = GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PIPE_BUSY) {
            if (std::chrono::steady_clock::now() >= deadline) return nullptr;
            DWORD wait_ms = (err == ERROR_PIPE_BUSY) ? 50 : 0;
            if (err == ERROR_PIPE_BUSY) {
                WaitNamedPipeW(wep.c_str(), 50);
            } else {
                Sleep(50);
            }
            (void)wait_ms;
        } else {
            return nullptr;
        }

        if (std::chrono::steady_clock::now() >= deadline) return nullptr;
    }
}

std::unique_ptr<ILocalServer> make_local_server()
{
    return std::make_unique<WinPipeServer>();
}

} // namespace pjev

#else // Unix

#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <chrono>

namespace pjev {

// ---------------------------------------------------------------------------
// Unix AF_UNIX socket implementation
// ---------------------------------------------------------------------------

// Helper: write exactly n bytes with loop
static bool write_all_fd(int fd, const void* buf, size_t n)
{
    size_t written = 0;
    while (written < n) {
        ssize_t w = ::write(fd, (const char*)buf + written, n - written);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        written += (size_t)w;
    }
    return true;
}

// Helper: read exactly n bytes with timeout (ms). timeout_ms=-1 = block forever.
static bool read_all_fd(int fd, void* buf, size_t n, int timeout_ms)
{
    size_t read_total = 0;
    while (read_total < n) {
        if (timeout_ms >= 0) {
            struct pollfd pfd;
            pfd.fd = fd;
            pfd.events = POLLIN;
            pfd.revents = 0;
            int r = poll(&pfd, 1, timeout_ms);
            if (r <= 0) return false; // timeout or error
            if (!(pfd.revents & POLLIN)) return false;
        }
        ssize_t rd = ::read(fd, (char*)buf + read_total, n - read_total);
        if (rd < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (rd == 0) return false; // EOF
        read_total += (size_t)rd;
    }
    return true;
}

class UnixConn : public ILocalConn {
public:
    explicit UnixConn(int fd) : fd_(fd) {}
    ~UnixConn() override { close(); }

    bool send_frame(const std::string& data) override
    {
        if (fd_ < 0) return false;
        uint32_t len = (uint32_t)data.size();
        if (len > IPC_MAX_FRAME_BYTES) return false;
        uint8_t hdr[4];
        hdr[0] = (uint8_t)(len & 0xFF);
        hdr[1] = (uint8_t)((len >> 8) & 0xFF);
        hdr[2] = (uint8_t)((len >> 16) & 0xFF);
        hdr[3] = (uint8_t)((len >> 24) & 0xFF);
        if (!write_all_fd(fd_, hdr, 4)) return false;
        if (len > 0 && !write_all_fd(fd_, data.data(), len)) return false;
        return true;
    }

    bool recv_frame(std::string& data, int timeout_ms = 30000) override
    {
        if (fd_ < 0) return false;
        uint8_t hdr[4] = {};
        if (!read_all_fd(fd_, hdr, 4, timeout_ms)) return false;
        uint32_t len = (uint32_t)hdr[0]
                     | ((uint32_t)hdr[1] << 8)
                     | ((uint32_t)hdr[2] << 16)
                     | ((uint32_t)hdr[3] << 24);
        if (len > IPC_MAX_FRAME_BYTES) return false;
        data.resize(len);
        if (len > 0 && !read_all_fd(fd_, &data[0], len, timeout_ms)) return false;
        return true;
    }

    void close() override
    {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

    bool is_open() const override { return fd_ >= 0; }

private:
    int fd_ = -1;
};

class UnixServer : public ILocalServer {
public:
    UnixServer() = default;
    ~UnixServer() override { close(); }

    bool bind(const std::string& endpoint) override
    {
        endpoint_ = endpoint;
        fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) return false;

        struct sockaddr_un addr = {};
        addr.sun_family = AF_UNIX;
        if (endpoint.size() >= sizeof(addr.sun_path) - 1) { ::close(fd_); fd_ = -1; return false; }
        strncpy(addr.sun_path, endpoint.c_str(), sizeof(addr.sun_path) - 1);

        // Remove stale socket
        ::unlink(endpoint.c_str());

        if (::bind(fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            ::close(fd_); fd_ = -1; return false;
        }
        chmod(endpoint.c_str(), 0600);
        if (listen(fd_, 8) < 0) {
            ::close(fd_); fd_ = -1; return false;
        }
        return true;
    }

    std::unique_ptr<ILocalConn> accept(int timeout_ms = -1) override
    {
        if (fd_ < 0) return nullptr;
        if (timeout_ms >= 0) {
            struct pollfd pfd;
            pfd.fd = fd_;
            pfd.events = POLLIN;
            pfd.revents = 0;
            int r = poll(&pfd, 1, timeout_ms);
            if (r <= 0) return nullptr;
            if (!(pfd.revents & POLLIN)) return nullptr;
        }
        int conn_fd = ::accept(fd_, nullptr, nullptr);
        if (conn_fd < 0) return nullptr;
        return std::make_unique<UnixConn>(conn_fd);
    }

    void close() override
    {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
            if (!endpoint_.empty()) ::unlink(endpoint_.c_str());
        }
    }

    bool is_bound() const override { return fd_ >= 0; }

private:
    int         fd_ = -1;
    std::string endpoint_;
};

std::unique_ptr<ILocalConn> make_local_client(const std::string& endpoint, int timeout_ms)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (true) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) return nullptr;

        struct sockaddr_un addr = {};
        addr.sun_family = AF_UNIX;
        if (endpoint.size() >= sizeof(addr.sun_path) - 1) { ::close(fd); return nullptr; }
        strncpy(addr.sun_path, endpoint.c_str(), sizeof(addr.sun_path) - 1);

        if (::connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            return std::make_unique<UnixConn>(fd);
        }

        ::close(fd);

        if (std::chrono::steady_clock::now() >= deadline) return nullptr;

        struct timespec ts = {0, 100 * 1000 * 1000}; // 100ms
        nanosleep(&ts, nullptr);

        if (std::chrono::steady_clock::now() >= deadline) return nullptr;
    }
}

std::unique_ptr<ILocalServer> make_local_server()
{
    return std::make_unique<UnixServer>();
}

} // namespace pjev
#endif
