#pragma once
#include "../decision/decision_engine.h"
#include "../inference/llama_backend.h"
#include "../calibration/calibration.h"
#include "ipc_protocol.h"
#include "local_transport.h"
#include <atomic>
#include <string>
#include <mutex>
#include <memory>

namespace pjev {

struct WorkerConfig {
    std::string model_path;
    std::string calibration_path;
    int threads = 8;
    int ctx_size = 4096;
    int idle_timeout_secs = 300; // 5 minutes default
    int max_queued = 8;
};

// WorkerServer: loads model, runs IPC listener, handles requests.
// Call run() to block until shutdown.
class WorkerServer {
public:
    explicit WorkerServer(const WorkerConfig& cfg);
    ~WorkerServer();

    // Blocks until shutdown (idle timeout, stop request, or signal).
    // Returns 0 on clean shutdown, 1 on error.
    int run();

    // Signal shutdown (thread-safe, can be called from signal handler)
    void request_shutdown();

private:
    WorkerConfig  cfg_;
    WorkerInfo    info_;

    std::unique_ptr<LlamaBackend>   backend_;
    std::unique_ptr<DecisionEngine> engine_;

    std::mutex              engine_mutex_; // serialize inference
    std::atomic<bool>       shutdown_requested_{false};
    std::atomic<int64_t>    last_activity_us_{0};
    std::atomic<int>        active_requests_{0};

    void handle_connection(std::unique_ptr<ILocalConn> conn);
    bool handle_ping(ILocalConn& conn);
    bool handle_decide(ILocalConn& conn, const nlohmann::json& req);
    bool handle_shutdown(ILocalConn& conn);

    void    update_activity();
    bool    is_idle_timeout_exceeded() const;
    int64_t now_us() const;
};

} // namespace pjev
