#include "platform.h"

#include <memory>

#ifdef _WIN32
#include <Windows.h>
#elif __linux__
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <unistd.h>
#endif

namespace pjev
{
int32_t get_physical_core_count() {
#ifdef _WIN32
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return 0;
    }

    std::unique_ptr<SYSTEM_LOGICAL_PROCESSOR_INFORMATION[]> buffer = std::make_unique<SYSTEM_LOGICAL_PROCESSOR_INFORMATION[]>(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buffer.get(), &length)) {
        return 0;
    }

    int32_t physicalCores = 0;
    DWORD count = length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (DWORD i = 0; i < count; ++i) {
        if (buffer[i].Relationship == RelationProcessorCore) {
            ++physicalCores;
        }
    }
    return 1<physicalCores ? physicalCores/2 : 1;

#elif __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return 1;
    }

    std::string line;
    std::set<std::string> uniqueCores;
    std::string physicalId = "";
    std::string coreId = "";

    while (std::getline(cpuinfo, line)) {
        if (line.rfind("physical id", 0) == 0) {
            physicalId = line.substr(line.find(':') + 1);
        } else if (line.rfind("core id", 0) == 0) {
            coreId = line.substr(line.find(':') + 1);
            uniqueCores.insert(physicalId + "_" + coreId);
        }
    }

    if (uniqueCores.empty()) {
        int64_t cores = sysconf(_SC_NPROCESSORS_ONLN);
        return (cores > 1) ? static_cast<int32_t>(cores/2) : 1;
    }

    return uniqueCores.size()>1 ? static_cast<int32_t>(uniqueCores.size()/2) : 1;
#else
    uint32_t cores = std::thread::hardware_concurrency();
    return cores > 1 ? static_cast<int32_t>(cores/2) : 1;
#endif
}

}