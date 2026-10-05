#include "worker_client.h"

#include <chrono>
#include <thread>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/socket.h>
#    include <sys/stat.h>
#    include <sys/un.h>
#    include <unistd.h>
#endif

#include "../util/platform.h"
#include "local_transport.h"
#include "runtime_dir.h"

namespace pjev
{
ClientConfig::ClientConfig()
    : threads(8)
    , ctx_size(4096)
    , idle_timeout_secs(600)
    , startup_timeout_ms(30000)
    , request_timeout_ms(120000)
    , retry_count(3)
{
    threads = get_physical_core_count();
}

WorkerClient::WorkerClient(const ClientConfig& cfg)
    : cfg_(cfg)
{
}

bool WorkerClient::is_compatible(const WorkerInfo& info) const
{
    const std::string& expected = make_config_hash(cfg_.model_path);
    return info.config_hash == expected && info.protocol_version == IPC_PROTOCOL_VERSION;
}

std::string WorkerClient::get_exe_path() const
{
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 2] = {};
    DWORD len = GetModuleFileNameW(nullptr, buf, (DWORD)(MAX_PATH * 2));
    if(len == 0)
        return "";
    // Convert to UTF-8
    int32_t needed = WideCharToMultiByte(CP_UTF8, 0, buf, -1, nullptr, 0, nullptr, nullptr);
    if(needed <= 0)
        return "";
    std::string s(needed - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf, -1, &s[0], needed, nullptr, nullptr);
    return s;
#else
    // Use get_executable_dir and find exe name
    std::string dir = get_executable_dir();
    if(dir.empty())
        return "pjev";
    return dir + "/pjev";
#endif
}

std::unique_ptr<ILocalConn> WorkerClient::try_connect_compatible()
{
    const std::string& endpoint = get_worker_endpoint();
    std::unique_ptr<ILocalConn> conn = make_local_client(endpoint, 2000);
    return conn;
}

bool WorkerClient::spawn_worker()
{
    std::string exe = get_exe_path();
    if(exe.empty())
        return false;

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
    if(!cfg_.calibration_path.empty()) {
        args.push_back("--calibration");
        args.push_back(cfg_.calibration_path);
    }
    args.push_back("--idle-timeout");
    args.push_back(std::to_string(cfg_.idle_timeout_secs));

#ifdef _WIN32
    // Build a quoted command line string
    // Simplified quoting: wrap any arg containing spaces in double-quotes
    std::string cmdline;
    for(size_t i = 0; i < args.size(); i++) {
        if(i > 0)
            cmdline += " ";
        const std::string& a = args[i];
        bool needs_quote = a.find(' ') != std::string::npos;
        if(needs_quote)
            cmdline += "\"";
        cmdline += a;
        if(needs_quote)
            cmdline += "\"";
    }

    // Convert to wstring
    int32_t wlen = MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, nullptr, 0);
    std::wstring wcmd(wlen - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, &wcmd[0], wlen);

    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    // Redirect stdin/stdout to NUL, inherit stderr
    HANDLE nul_h = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, OPEN_EXISTING, 0, nullptr);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul_h;
    si.hStdOutput = nul_h;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessW(
        nullptr,
        &wcmd[0],
        nullptr, nullptr,
        TRUE, // inherit handles (for stderr)
        DETACHED_PROCESS | CREATE_NO_WINDOW,
        nullptr, nullptr,
        &si, &pi);

    if(nul_h != INVALID_HANDLE_VALUE)
        CloseHandle(nul_h);

    if(!ok)
        return false;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return true;

#else
    // Unix: fork + exec
    pid_t pid = fork();
    if(pid < 0)
        return false;
    if(pid == 0) {
        // Child process
        setsid();

        // Redirect stdin/stdout to /dev/null
        int32_t devnull = open("/dev/null", O_RDWR);
        if(devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            close(devnull);
        }
        // Keep stderr (fd 2) as-is

        // Close all fds > 2
        // (simplified: just close a reasonable range)
        for(int32_t fd = 3; fd < 1024; fd++) {
            close(fd);
        }

        // Build argv
        std::vector<const char*> argv_vec;
        for(const auto& a: args) argv_vec.push_back(a.c_str());
        argv_vec.push_back(nullptr);

        execv(exe.c_str(), const_cast<char* const*>(argv_vec.data()));
        _exit(127); // exec failed
    }
    // Parent: don't wait, let it run detached
    return true;
#endif
}

std::unique_ptr<ILocalConn> WorkerClient::wait_for_worker(int32_t timeout_ms)
{
    const std::string& endpoint = get_worker_endpoint();
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while(std::chrono::steady_clock::now() < deadline) {
        std::unique_ptr<ILocalConn> conn = make_local_client(endpoint, 0);
        if(conn) {
            return conn;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return nullptr;
}

bool WorkerClient::validate_compatibility(std::unique_ptr<ILocalConn>& conn)
{
    // Send ping
    nlohmann::json ping = {{"type", "ping"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if(!conn->send_frame(ping.dump()))
        return false;

    std::string resp_str;
    if(!conn->recv_frame(resp_str, 5000))
        return false;

    nlohmann::json resp;
    try {
        resp = nlohmann::json::parse(resp_str);
    } catch(...) {
        return false;
    }

    if(resp.value("type", "") != "pong")
        return false;
    if(!resp.contains("info"))
        return false;

    WorkerInfo info = ipc_worker_info_from_json(resp["info"]);
    if(!is_compatible(info))
        return false;
    return true;
}

void WorkerClient::clean_stale_endpoint()
{
#ifndef _WIN32
    const std::string& endpoint = get_worker_endpoint();
    ::unlink(endpoint.c_str());
#endif
    // On Windows, CreateNamedPipeW with FILE_FLAG_FIRST_PIPE_INSTANCE handles this
}

DecisionOutput WorkerClient::decide(const DecisionInput& input)
{
    nlohmann::json request;
    request["type"] = "decide";
    request["protocol_version"] = IPC_PROTOCOL_VERSION;
    request["input"] = ipc_input_to_json(input);
    std::string requestJson = request.dump();
    std::string response;

    for(int32_t i = 0; i < cfg_.retry_count; ++i) {
        // Try connecting to existing worker
        std::unique_ptr<ILocalConn> conn = try_connect_compatible();
        if(conn){
            conn = nullptr;
        }

        if(!conn) {
            // No compatible worker — spawn one
            if(!spawn_worker()) {
                throw std::runtime_error("pjev worker: failed to spawn worker process");
            }
            // Wait for it to become available
            conn = wait_for_worker(cfg_.startup_timeout_ms);
            if(!conn) {
                throw std::runtime_error("pjev worker: worker did not start in time");
            }
        }

        // Send decide request
        if(!conn->send_frame(requestJson)) {
            // Broken connection — retry
            clean_stale_endpoint();
            continue;
        }

        if(!conn->recv_frame(response, cfg_.request_timeout_ms)) {
            clean_stale_endpoint();
            continue;
        }
        break;
    }
    if(response.empty()){
        throw std::runtime_error("pjev worker: no response from worker");
    }

    nlohmann::json responseJson;
    try {
        responseJson = nlohmann::json::parse(response);
    } catch(const std::exception& e) {
        throw std::runtime_error(std::string("pjev worker: response parse error: ") + e.what());
    }

    return ipc_output_from_json(responseJson);
}

bool WorkerClient::get_status(WorkerInfo& info)
{
    const std::string& endpoint = get_worker_endpoint();
    auto conn = make_local_client(endpoint, 2000);
    if(!conn)
        return false;

    nlohmann::json ping = {{"type", "ping"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if(!conn->send_frame(ping.dump()))
        return false;

    std::string resp_str;
    if(!conn->recv_frame(resp_str, 5000))
        return false;

    nlohmann::json resp;
    try {
        resp = nlohmann::json::parse(resp_str);
    } catch(...) {
        return false;
    }

    if(resp.value("type", "") != "pong")
        return false;
    if(!resp.contains("info"))
        return false;

    info = ipc_worker_info_from_json(resp["info"]);
    return true;
}

bool WorkerClient::stop_worker()
{
    auto conn = make_local_client(get_worker_endpoint(), 2000);
    if(!conn)
        return false;

    // Ping first to validate
    nlohmann::json ping = {{"type", "ping"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if(!conn->send_frame(ping.dump()))
        return false;

    std::string pong_str;
    if(!conn->recv_frame(pong_str, 5000))
        return false;

    // Send shutdown
    nlohmann::json shut = {{"type", "shutdown"}, {"protocol_version", IPC_PROTOCOL_VERSION}};
    if(!conn->send_frame(shut.dump()))
        return false;

    // Recv ack (optional — ignore failure)
    std::string ack_str;
    conn->recv_frame(ack_str, 5000);

    return true;
}

} // namespace pjev
