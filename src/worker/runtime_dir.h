#pragma once
#include <string>

namespace pjev {

// Returns path for worker's IPC endpoint.
// Unix: socket path like /tmp/pjev-{uid}-worker-v1.sock
// Windows: named pipe like \\.\pipe\pjev-{username}-worker-v1
std::string get_worker_endpoint();

// Returns path to a lock file (Unix only, empty on Windows)
std::string get_worker_lock_path();

// Returns the user-local runtime directory (creates if needed)
std::string get_worker_runtime_dir();

} // namespace pjev
