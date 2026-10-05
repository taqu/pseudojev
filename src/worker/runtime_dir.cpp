#include "runtime_dir.h"
#include <format>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <shlobj.h>
#    include <string>
#    include <windows.h>

#    include <murmur3/murmur3.h>

namespace pjev
{

namespace
{
    std::string worker_endpoint;
    std::string worker_lock_path;
    std::string worker_runtime_dir;

    std::string utf16_to_utf8(const std::wstring& ws)
    {
        if(ws.empty())
            return {};
        int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, nullptr, 0, nullptr, nullptr);
        if(len <= 0)
            return {};
        std::string s(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, &s[0], len, nullptr, nullptr);
        return s;
    }

    std::string get_windows_username()
    {
        wchar_t buf[256] = {};
        DWORD sz = 256;
        if(GetUserNameW(buf, &sz) && sz > 1) {
            uint64_t hash[2];
            MurmurHash3_x64_128(buf, static_cast<int32_t>((sz-1) * 2), 42, hash);
            return std::format("{:X}", hash[0]);
        }
        return "user";
    }
} // namespace

void initialize_strings()
{
    wchar_t tmp[MAX_PATH] = {};
    DWORD len = GetTempPathW(MAX_PATH, tmp);
    std::string base;
    if(len > 0) {
        base = utf16_to_utf8(std::wstring(tmp, len));
    } else {
        base = "C:\\Temp\\";
    }
    std::string username = get_windows_username();
    worker_runtime_dir = base + "pjev-" + username + "\\";
    // Create if needed
    CreateDirectoryA(worker_runtime_dir.c_str(), nullptr);

    worker_endpoint = "\\\\.\\pipe\\pjev-" + username + "-worker-v1";
}

const std::string& get_worker_runtime_dir()
{
    return worker_runtime_dir;
}

const std::string& get_worker_endpoint()
{
    return worker_endpoint;
}

const std::string& get_worker_lock_path()
{
    return worker_lock_path; // Windows pipes are exclusive by design
}

} // namespace pjev

#else // Unix

#    include <cstdlib>
#    include <cstring>
#    include <string>
#    include <sys/stat.h>
#    include <unistd.h>

namespace pjev
{

namespace
{
    std::string worker_endpoint;
    std::string worker_lock_path;
    std::string worker_runtime_dir;
} // namespace

void initialize_strings()
{
    std::string base;
    const char* xdg = std::getenv("XDG_RUNTIME_DIR");
    if(xdg && xdg[0] != '\0') {
        base = xdg;
    } else {
        base = "/tmp";
    }
    uid_t uid = getuid();
    worker_runtime_dir = base + "/pjev-" + std::to_string((unsigned)uid) + "/";
    mkdir(worker_runtime_dir.c_str(), 0700);
    worker_endpoint = worker_runtime_dir + "worker-v1.sock";
    worker_lock_path = worker_runtime_dir + "worker-v1.lock";
}

const std::string& get_worker_runtime_dir()
{
    return worker_runtime_dir;
}

const std::string& get_worker_endpoint()
{
    return worker_endpoint;
}

const std::string& get_worker_lock_path()
{
    return worker_lock_path;
}

} // namespace pjev
#endif
