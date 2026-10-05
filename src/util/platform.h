#ifndef INC_PJEV_PLATFORM_H_
#define INC_PJEV_PLATFORM_H_
#include <cstdint>
#include <string>

namespace pjev
{

// Returns resident set size of this process in bytes, or -1 on failure.
int64_t get_rss_bytes();

// Returns OS name string: "Windows", "Linux", "macOS", or "unknown".
std::string get_os_name();

// Returns the directory containing the current executable (no trailing separator).
// Returns "" on failure.
std::string get_executable_dir();

int32_t get_physical_core_count();
} // namespace pjev
#endif // INC_PJEV_PLATFORM_H_
