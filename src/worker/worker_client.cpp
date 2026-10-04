#include "worker_client.h"
#include "runtime_dir.h"
#include "local_transport.h"
#include "../util/platform.h"
#include <chrono>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <cerrno>
#endif

namespace pjev {

WorkerClient::WorkerClient(const ClientConfig& cfg)
    : cfg_(cfg)
{
}

bool WorkerClient::is_compatible(const WorkerInfo& info) const
{
    std::string expected = make_config_hash(cfg_.model_path, cfg_.threads,
                                             cfg_.ctx_size, cfg_.calibration_path);
    return info.config_hash == expected && info.protocol_version == IPC_PROTOCOL_VERSION;
}

std::string WorkerClient::get_exe_path() const
{
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 2] = {};
    DWORD len = GetModuleFileNameW(nullptr, buf, (DWORD)(MAX_PATH * 2));
    if (len == 0) return "";
    // Convert to UTF-8
    int needed = WideCharToMultiByte(CP_UTF8, 0, buf, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 0) return "";
    std::string s(needed - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf, -1, &s[0], needed, nullptr, nullptr);
    return s;
#else
    // Use get_executable_dir and find exe name
    std::string dir = get_executable_dir();
    if (dir.empty()) return "pjev";
    return dir + "/pjev";
#endif
}

std::unique_ptr<ILocalConn> WorkerClient::try_connect_compatible()
{
    std::string endpoint = get_worker_endpoint();
    auto conn = make_local_client(endpoint, 2000);
    if (!conn) return nullptr;

    // Send ping
    nlohmann::json ping = {{"type", "ping"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if (!conn->send_frame(ping.dump())) return nullptr;

    std::string resp_str;
    if (!conn->recv_frame(resp_str, 5000)) return nullptr;

    nlohmann::json resp;
    try { resp = nlohmann::json::parse(resp_str); }
    catch (...) { return nullptr; }

    if (resp.value("type", "") != "pong") return nullptr;
    if (!resp.contains("info")) return nullptr;

    WorkerInfo info = ipc_worker_info_from_json(resp["info"]);
    if (!is_compatible(info)) return nullptr;

    return conn;
}

bool WorkerClient::spawn_worker()
{
    std::string exe = get_exe_path();
    if (exe.empty()) return false;

    // Build argument list
    // pjev internal-worker --model PATH --threads N --ctx-size N [--calibration FILE] [--idle-timeout N]
    std::vector<std::string> args;
    args.push_back(exe);
    args.push_back("internal-worker");
    args.push_back("--model");
    args.push_back(cfg_.model_path);
    args.push_back("--threads");
    args.push_back(std::to_string(cfg_.threads));
    args.push_back("--ctx-size");
    args.push_back(std::to_string(cfg_.ctx_size));
    if (!cfg_.calibration_path.empty()) {
        args.push_back("--calibration");
        args.push_back(cfg_.calibration_path);
    }
    args.push_back("--idle-timeout");
    args.push_back(std::to_string(cfg_.idle_timeout_secs));

#ifdef _WIN32
    // Build a quoted command line string
    // Simplified quoting: wrap any arg containing spaces in double-quotes
    std::string cmdline;
    for (size_t i = 0; i < args.size(); i++) {
        if (i > 0) cmdline += " ";
        const std::string& a = args[i];
        bool needs_quote = a.find(' ') != std::string::npos;
        if (needs_quote) cmdline += "\"";
        cmdline += a;
        if (needs_quote) cmdline += "\"";
    }

    // Convert to wstring
    int wlen = MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, nullptr, 0);
    std::wstring wcmd(wlen - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, &wcmd[0], wlen);

    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    // Redirect stdin/stdout to NUL, inherit stderr
    HANDLE nul_h = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE,
                                nullptr, OPEN_EXISTING, 0, nullptr);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = nul_h;
    si.hStdOutput = nul_h;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessW(
        nullptr,
        &wcmd[0],
        nullptr, nullptr,
        TRUE, // inherit handles (for stderr)
        DETACHED_PROCESS | CREATE_NO_WINDOW,
        nullptr, nullptr,
        &si, &pi);

    if (nul_h != INVALID_HANDLE_VALUE) CloseHandle(nul_h);

    if (!ok) return false;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return true;

#else
    // Unix: fork + exec
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        // Child process
        setsid();

        // Redirect stdin/stdout to /dev/null
        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            close(devnull);
        }
        // Keep stderr (fd 2) as-is

        // Close all fds > 2
        // (simplified: just close a reasonable range)
        for (int fd = 3; fd < 1024; fd++) {
            close(fd);
        }

        // Build argv
        std::vector<const char*> argv_vec;
        for (const auto& a : args) argv_vec.push_back(a.c_str());
        argv_vec.push_back(nullptr);

        execv(exe.c_str(), const_cast<char* const*>(argv_vec.data()));
        _exit(127); // exec failed
    }
    // Parent: don't wait, let it run detached
    return true;
#endif
}

std::unique_ptr<ILocalConn> WorkerClient::wait_for_worker(int timeout_ms)
{
    std::string endpoint = get_worker_endpoint();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        auto conn = make_local_client(endpoint, 100);
        if (conn) return conn;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return nullptr;
}

void WorkerClient::clean_stale_endpoint()
{
#ifndef _WIN32
    std::string endpoint = get_worker_endpoint();
    ::unlink(endpoint.c_str());
#endif
    // On Windows, CreateNamedPipeW with FILE_FLAG_FIRST_PIPE_INSTANCE handles this
}

DecisionOutput WorkerClient::decide(const DecisionInput& input)
{
    // Try connecting to existing worker
    auto conn = try_connect_compatible();

    bool spawned = false;
    if (!conn) {
        // No compatible worker — spawn one
        if (!spawn_worker()) {
            throw std::runtime_error("pjev worker: failed to spawn worker process");
        }
        spawned = true;
        // Wait for it to become available
        auto raw_conn = wait_for_worker(cfg_.startup_timeout_ms);
        if (!raw_conn) {
            throw std::runtime_error("pjev worker: worker did not start in time");
        }
        // Now validate compatibility
        conn = try_connect_compatible();
        if (!conn) {
            throw std::runtime_error("pjev worker: worker is not compatible after spawn");
        }
    }

    // Send decide request
    nlohmann::json req;
    req["type"]             = "decide";
    req["protocol_version"] = IPC_PROTOCOL_VERSION;
    req["input"]            = ipc_input_to_json(input);

    if (!conn->send_frame(req.dump())) {
        // Broken connection — retry once (if we didn't just spawn)
        if (spawned) {
            throw std::runtime_error("pjev worker: broken connection after spawn");
        }
        clean_stale_endpoint();
        // Retry once
        return decide(input);
    }

    std::string resp_str;
    if (!conn->recv_frame(resp_str, cfg_.request_timeout_ms)) {
        if (spawned) {
            throw std::runtime_error("pjev worker: no response from worker");
        }
        clean_stale_endpoint();
        return decide(input);
    }

    nlohmann::json resp;
    try { resp = nlohmann::json::parse(resp_str); }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("pjev worker: response parse error: ") + e.what());
    }

    return ipc_output_from_json(resp);
}

bool WorkerClient::get_status(WorkerInfo& info)
{
    std::string endpoint = get_worker_endpoint();
    auto conn = make_local_client(endpoint, 2000);
    if (!conn) return false;

    nlohmann::json ping = {{"type", "ping"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if (!conn->send_frame(ping.dump())) return false;

    std::string resp_str;
    if (!conn->recv_frame(resp_str, 5000)) return false;

    nlohmann::json resp;
    try { resp = nlohmann::json::parse(resp_str); }
    catch (...) { return false; }

    if (resp.value("type", "") != "pong") return false;
    if (!resp.contains("info")) return false;

    info = ipc_worker_info_from_json(resp["info"]);
    return true;
}

bool WorkerClient::stop_worker()
{
    auto conn = make_local_client(get_worker_endpoint(), 2000);
    if (!conn) return false;

    // Ping first to validate
    nlohmann::json ping = {{"type", "ping"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if (!conn->send_frame(ping.dump())) return false;

    std::string pong_str;
    if (!conn->recv_frame(pong_str, 5000)) return false;

    // Send shutdown
    nlohmann::json shut = {{"type", "shutdown"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if (!conn->send_frame(shut.dump())) return false;

    // Recv ack (optional — ignore failure)
    std::string ack_str;
    conn->recv_frame(ack_str, 5000);

    return true;
}

} // namespace pjev
