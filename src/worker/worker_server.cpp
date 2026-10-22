#include "worker_server.h"
#include "runtime_dir.h"
#include <chrono>
#include "spdlog/spdlog.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

#ifndef PJEV_VERSION
#define PJEV_VERSION "unknown"
#endif

namespace pjev {

static int64_t get_now_us()
{
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

WorkerServer::WorkerServer(const WorkerConfig& cfg)
    : cfg_(cfg)
{
}

WorkerServer::~WorkerServer() = default;

int64_t WorkerServer::now_us() const
{
    return get_now_us();
}

void WorkerServer::update_activity()
{
    last_activity_us_.store(now_us());
}

bool WorkerServer::is_idle_timeout_exceeded() const
{
    if (cfg_.idle_timeout_secs <= 0) return false;
    int64_t last = last_activity_us_.load();
    int64_t now  = now_us();
    return (now - last) > (int64_t)cfg_.idle_timeout_secs * 1000000LL;
}

void WorkerServer::request_shutdown()
{
    shutdown_requested_.store(true);
}

int WorkerServer::run()
{
    // 1. Load models and the exemplar database
    RuntimeOptions opts;
    opts.paths        = cfg_.paths;
    opts.calibration_required = cfg_.calibration_required;
    opts.n_threads    = cfg_.threads;
    opts.n_ctx        = cfg_.ctx_size;
    opts.n_gpu_layers = cfg_.n_gpu_layers;
    opts.main_gpu     = cfg_.main_gpu;
    std::string err;
    runtime_ = RouterRuntime::load(opts, err);
    if (!runtime_) {
        spdlog::error("worker: {}", err);
        return 1;
    }

    // 2. Build WorkerInfo
    const ModelIdentity& mid = runtime_->model_identity();
    info_.protocol_version = IPC_PROTOCOL_VERSION;
    info_.pjev_version     = PJEV_VERSION;
#ifdef _WIN32
    info_.pid = (int32_t)GetCurrentProcessId();
#else
    info_.pid = (int32_t)getpid();
#endif
    info_.model_identity = mid.name.empty() ? cfg_.paths.model : mid.name;
    info_.config_hash    = make_config_hash(cfg_.paths, cfg_.threads, cfg_.ctx_size);
    info_.calibration    = to_string(runtime_->calibration().state);
    info_.start_time_us  = get_now_us();

    // 3. Create ILocalServer and bind
    auto server = make_local_server();
    const std::string& endpoint = get_worker_endpoint();
    if (!server->bind(endpoint)) {
        spdlog::error("worker: failed to bind IPC endpoint: {}", endpoint);
        return 1;
    }

    update_activity();

    // 4. Log ready
    spdlog::info("pjev worker ready pid={} endpoint={}", info_.pid, endpoint);

    // 5. Main loop
    while (!shutdown_requested_.load()) {
        auto conn = server->accept(1000); // 1-second timeout
        if (conn) {
            update_activity();
            handle_connection(std::move(conn));
            update_activity();
        }

        // Check idle timeout
        if (active_requests_.load() == 0 && is_idle_timeout_exceeded()) {
            spdlog::info("worker: idle timeout, shutting down");
            break;
        }
    }

    // 6. Cleanup
    server->close();

    spdlog::info("worker shutdown");
    return 0;
}

void WorkerServer::handle_connection(std::unique_ptr<ILocalConn> conn)
{
    std::string frame;
    if (!conn->recv_frame(frame, 30000)) {
        spdlog::warn("worker: connection timeout or disconnect");
        return; // timeout or disconnect
    }

    nlohmann::json req;
    try {
        req = nlohmann::json::parse(frame);
    } catch (std::exception const& e) {
        spdlog::warn("worker: request parse error: {}", e.what());
        return;
    }

    // Check protocol version
    int pver = req.value("protocol_version", IPC_PROTOCOL_VERSION);
    if (pver != IPC_PROTOCOL_VERSION) {
        nlohmann::json resp = {{"error", "protocol version mismatch"}, {"ok", false}};
        std::string s = resp.dump();
        conn->send_frame(s);
        spdlog::error("worker: protocol version mismatch: expected={}, got={}", IPC_PROTOCOL_VERSION, pver);
        return;
    }

    std::string type = req.value("type", "");
    spdlog::info("worker: request type={}", type);
    if (type == "decide") {
        handle_decide(*conn, req);
    } else if (type == "ping") {
        handle_ping(*conn);
    } else if (type == "shutdown") {
        handle_shutdown(*conn);
    } else {
        nlohmann::json resp = {{"error", "unknown request type"}, {"ok", false}};
        conn->send_frame(resp.dump());
    }
}

bool WorkerServer::handle_ping(ILocalConn& conn)
{
    nlohmann::json resp;
    resp["type"] = "pong";
    resp["info"] = ipc_worker_info_to_json(info_);
    return conn.send_frame(resp.dump());
}

bool WorkerServer::handle_decide(ILocalConn& conn, const nlohmann::json& req)
{
    DecisionInput input;
    try {
        input = ipc_input_from_json(req["input"]);
    } catch (const std::exception& e) {
        nlohmann::json resp = {{"ok", false}, {"error", std::string("input parse: ") + e.what()}};
        return conn.send_frame(resp.dump());
    }

    active_requests_.fetch_add(1);
    DecisionOutput out;
    try {
        std::lock_guard<std::mutex> lk(engine_mutex_);
        out = runtime_->service().decide(input);
    } catch (const std::exception& e) {
        active_requests_.fetch_sub(1);
        nlohmann::json resp = {{"ok", false}, {"error", std::string("decide: ") + e.what()}};
        return conn.send_frame(resp.dump());
    }
    active_requests_.fetch_sub(1);

    nlohmann::json resp = ipc_output_to_json(out);
    #ifdef _DEBUG
    spdlog::info(resp.dump());
    #endif
    return conn.send_frame(resp.dump());
}

bool WorkerServer::handle_shutdown(ILocalConn& conn)
{
    nlohmann::json resp = {{"type", "shutdown_ack"}, {"ok", true}};
    bool sent = conn.send_frame(resp.dump());
    request_shutdown();
    return sent;
}

} // namespace pjev
