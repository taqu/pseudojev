#pragma once
#include "../decision/decision_engine.h"
#include "ipc_protocol.h"
#include "local_transport.h"
#include <memory>
#include <string>

namespace pjev
{

struct ClientConfig
{
    ClientConfig();
    RuntimePaths paths;   // absolute paths, so workers started from other directories compare equal
    int32_t threads;
    int32_t ctx_size;
    int32_t n_gpu_layers;
    int32_t main_gpu;
    int32_t idle_timeout_secs;  // worker idle timeout
    int32_t startup_timeout_ms; // 30s for worker to start
    int32_t request_timeout_ms; // 2 min for inference
    int32_t retry_count; // 3 times for retry
};

// Connects to background worker (spawning it if needed), sends request, gets result.
// Handles stale worker detection and one-shot recovery.
class WorkerClient
{
public:
    explicit WorkerClient(const ClientConfig& cfg);

    // Run a decision request via the worker.
    // Spawns a worker if none is running; a running worker with a different configuration is stopped
    // and replaced. Retries on a broken connection.
    DecisionOutput decide(const DecisionInput& input);

    // Check worker status without starting one.
    // Returns false if no worker is running.
    bool get_status(WorkerInfo& info);

    // Send shutdown to worker. Returns true if stopped, false if none running.
    bool stop_worker();

private:
    ClientConfig cfg_;

    // Spawn a new worker process.
    bool spawn_worker();

    // Wait for worker to become available (poll endpoint).
    std::unique_ptr<ILocalConn> wait_for_worker(int timeout_ms);

    // Wait until no worker accepts connections on the endpoint.
    void wait_for_endpoint_free(int timeout_ms);

    // Validate WorkerInfo compatibility against our config.
    bool is_compatible(const WorkerInfo& info) const;

    // Get the path to the current executable for spawning.
    std::string get_exe_path() const;

    // Clean stale endpoint artifacts.
    void clean_stale_endpoint();
};

} // namespace pjev
