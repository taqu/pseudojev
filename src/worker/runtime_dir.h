#pragma once
#include <string>

namespace pjev
{

void initialize_strings();

// Returns path for worker's IPC endpoint.
// The name carries IPC_PROTOCOL_VERSION (N below).
// Unix: socket path like /tmp/pjev-{uid}/worker-vN.sock
// Windows: named pipe like \\.\pipe\pjev-{username}-worker-vN
const std::string& get_worker_endpoint();

// Returns path to a lock file (Unix only, empty on Windows)
const std::string& get_worker_lock_path();

// Returns the user-local runtime directory (creates if needed)
const std::string& get_worker_runtime_dir();

} // namespace pjev
