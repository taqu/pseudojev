#include "runtime_dir.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#include <string>

namespace pjev {

static std::string utf16_to_utf8(const std::wstring& ws)
{
    if (ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string s(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, &s[0], len, nullptr, nullptr);
    return s;
}

static std::string get_windows_username()
{
    wchar_t buf[256] = {};
    DWORD sz = 256;
    if (GetUserNameW(buf, &sz) && sz > 1) {
        return utf16_to_utf8(std::wstring(buf, sz - 1));
    }
    return "user";
}

std::string get_worker_runtime_dir()
{
    wchar_t tmp[MAX_PATH] = {};
    DWORD len = GetTempPathW(MAX_PATH, tmp);
    std::string base;
    if (len > 0) {
        base = utf16_to_utf8(std::wstring(tmp, len));
    } else {
        base = "C:\\Temp\\";
    }
    std::string username = get_windows_username();
    std::string dir = base + "pjev-" + username + "\\";
    // Create if needed
    CreateDirectoryA(dir.c_str(), nullptr);
    return dir;
}

std::string get_worker_endpoint()
{
    std::string username = get_windows_username();
    return "\\\\.\\pipe\\pjev-" + username + "-worker-v1";
}

std::string get_worker_lock_path()
{
    return {}; // Windows pipes are exclusive by design
}

} // namespace pjev

#else // Unix

#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace pjev {

std::string get_worker_runtime_dir()
{
    std::string base;
    const char* xdg = std::getenv("XDG_RUNTIME_DIR");
    if (xdg && xdg[0] != '\0') {
        base = xdg;
    } else {
        base = "/tmp";
    }
    uid_t uid = getuid();
    std::string dir = base + "/pjev-" + std::to_string((unsigned)uid) + "/";
    mkdir(dir.c_str(), 0700);
    return dir;
}

std::string get_worker_endpoint()
{
    return get_worker_runtime_dir() + "worker-v1.sock";
}

std::string get_worker_lock_path()
{
    return get_worker_runtime_dir() + "worker-v1.lock";
}

} // namespace pjev
#endif
