#ifndef PJEV_VERSION
#define PJEV_VERSION "0.6.0"
#endif
#ifndef PJEV_PLATFORM
#define PJEV_PLATFORM "unknown"
#endif
#ifndef PJEV_ARCH
#define PJEV_ARCH "unknown"
#endif
#ifndef PJEV_GIT_COMMIT
#define PJEV_GIT_COMMIT "unknown"
#endif
#ifndef PJEV_LLAMA_REVISION
#define PJEV_LLAMA_REVISION "unknown"
#endif

#include "calibration/calibration_generator.h"
#include "model/model_identity.h"
#include "retrieval/exemplar_store.h"
#include "util/platform.h"
#include "decision/decision_service.h"
#include "routing/router_runtime.h"
#include "routing/task_policy.h"
#include "server/jev_api.h"
#include "server/server_config.h"
#include "server/http_server.h"
#include "distribution/release_config.h"
#include "cli/cli.h"
#include "worker/ipc_protocol.h"
#include "worker/runtime_dir.h"
#include "worker/local_transport.h"
#include "worker/worker_server.h"
#include "worker/worker_client.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <string>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_sinks.h"

// ---------------------------------------------------------------------------
// Usage helpers
// ---------------------------------------------------------------------------
using namespace pjev;

static const char* kResourceOptionsHelp =
    "  --model PATH               inference model (.gguf)\n"
    "  --embedding-model PATH     query embedding model (.gguf)\n"
    "  --exemplar-db PATH         exemplar database (read-only)\n"
    "  --exemplar-allowlist PATH  eligible exemplar IDs\n"
    "  --calibration PATH         calibration artifact (default: calibration.json; uncalibrated if absent)\n"
    "  --no-calibration           decide uncalibrated (R4 reference behaviour)\n"
    "  --threads N                CPU threads (default: physical cores)\n"
    "  --ctx-size N               context size (default: 4096)\n"
    "  --n-gpu-layers N           number of layers to offload to GPU (default: 0)\n"
    "  --main-gpu N               GPU device index to use as main GPU (default: 0)\n";

static void usage_server(const char* prog) {
    fprintf(stderr,
        "usage: %s serve [options]\n"
        "%s"
        "  --port N                   HTTP port (default: 8080)\n"
        "  --host ADDR                bind address (default: 127.0.0.1)\n"
        "  --threads-batch N          batch CPU threads (default: same as --threads)\n"
        "  --verbose                  enable llama.cpp verbose logging\n"
        "  --model-name NAME          model name in responses (default: pseudojev)\n"
        "  --request-limit BYTES      max request body size in bytes (default: 1048576)\n"
        "  --read-timeout SECS        read timeout in seconds (default: 30)\n"
        "  --max-queued N             max concurrent requests before returning 503 (default: 8)\n",
        prog, kResourceOptionsHelp);
}

static void usage_benchmark(const char* prog) {
    fprintf(stderr,
        "usage: %s benchmark latency [options]\n"
        "  Measures end-to-end decision latency per task kind (models loaded once, one untimed\n"
        "  warm-up pass, sequential decisions).\n"
        "%s"
        "  --input FILE               JSONL tasks: {\"type\",\"state\",\"question\",\"options\":[[key,desc],..],\n"
        "                             \"domain\",\"family\"} or {\"input\": {...}} (default: built-in tasks)\n"
        "  --state TEXT               state of the built-in tasks\n"
        "  --iterations N             measured passes over the tasks (default: 20)\n"
        "  --warmup N                 untimed passes (default: 1)\n"
        "  --output FILE              JSON output file (default: stdout)\n",
        prog, kResourceOptionsHelp);
}

static void usage_diagnostics(const char* prog) {
    fprintf(stderr,
        "usage: %s diagnostics [options]\n"
        "%s"
        "  --load                     also load the models and the exemplar database and check them\n"
        "  --json                     output JSON to stdout\n"
        "\n"
        "Shows pjev version, build, platform, CPU, runtime resources, worker status, and config.\n"
        "Use 'pjev diagnostics --json' for machine-readable output (useful for bug reports).\n",
        prog, kResourceOptionsHelp);
}

static void usage_decision(const char* prog, Type type) {
    const char* cmd = to_string(type);
    fprintf(stderr, "usage: %s %s --question TEXT", prog, cmd);
    if (type == Type::Choice) fprintf(stderr, " --option KEY[:DESC] --option KEY[:DESC] ...");
    if (type == Type::Score)  fprintf(stderr, " --level DESC --level DESC ...");
    fprintf(stderr, " [options]\n"
        "  --question TEXT            question to evaluate (required)\n");
    if (type == Type::Choice)
        fprintf(stderr, "  --option KEY[:DESC]        add a candidate option (at least 2 required); KEY is the label\n");
    if (type == Type::Score)
        fprintf(stderr, "  --level DESC               add a score level description (at least 2, lowest first)\n");
    fprintf(stderr,
        "  --state TEXT               context state (default: empty); a JSON object is also\n"
        "                             matched field by field against the exemplar bank\n"
        "  --state-file FILE          read state from file (- for stdin)\n"
        "  --domain TEXT              task domain, used for exemplar retrieval (optional)\n"
        "  --family TEXT              task family, used for exemplar retrieval (optional)\n"
        "%s"
        "  --idle-timeout SECS        worker idle timeout in seconds (default: 600)\n"
        "  --verbose                  enable verbose logging to stderr\n"
        "  --json                     output JSON instead of plain text\n"
        "  --direct                   bypass worker, load models in-process\n"
        "\n", kResourceOptionsHelp);
    if (type == Type::Noul)
        fprintf(stderr, "output: 'true' or 'false' (plain); JSON with p_true and probabilities (--json)\n");
    if (type == Type::Choice)
        fprintf(stderr, "output: selected KEY (plain); JSON with index, key, probabilities (--json)\n");
    if (type == Type::Score)
        fprintf(stderr, "output: expected level as decimal (plain); JSON with expected_score and probabilities (--json)\n");
}

static void usage(const char* prog) {
    fprintf(stderr,
        "pjev — local Jev-compatible decision engine powered by llama.cpp\n"
        "\n"
        "usage: %s <command> [options]\n"
        "\n"
        "decision commands (background worker reused automatically; use --direct for in-process):\n"
        "  noul         true/false question  -> 'true' | 'false'\n"
        "  choice       multiple-choice      -> selected key\n"
        "  score        quality score        -> expected score value\n"
        "\n"
        "worker commands:\n"
        "  status       show background worker state\n"
        "  stop         stop the background worker and release model memory\n"
        "\n"
        "server:\n"
        "  serve        start the Jev-compatible HTTP server\n"
        "\n"
        "utilities:\n"
        "  diagnostics  runtime info, resource paths, CPU, worker state  (--json for reports)\n"
        "  benchmark    measure decision latency\n"
        "  calibrate    generate calibration.json for the inference model\n"
        "  --version    print version and exit\n"
        "\n"
        "run '%s <command> --help' for command-specific options\n"
        "\n"
        "examples:\n"
        "  %s noul --state \"The sky is blue.\" --question \"Is it daytime?\"\n"
        "  %s choice --question \"Pick one\" --option A:\"First\" --option B:\"Second\"\n"
        "  %s diagnostics --json\n",
        prog, prog, prog, prog, prog);
}

struct SpdLogShutdown {
    SpdLogShutdown() = default;
    ~SpdLogShutdown()
    {
        spdlog::shutdown();
    }
};

// ---------------------------------------------------------------------------
// Runtime options shared by every command that decides
// ---------------------------------------------------------------------------

// Options without effect since task-aware routing fixed the decision configuration. Accepted so
// existing scripts keep working; `takes_value` options consume their argument.
static bool consume_deprecated(const std::string& a, int32_t& i, int32_t argc)
{
    static const std::map<std::string, bool> kDeprecated = {
        {"--layout", true}, {"--scheme", true}, {"--questions", true},
        {"--ensemble", false}, {"--no-kv-reuse", false}, {"--no-batch-questions", false}};
    auto it = kDeprecated.find(a);
    if (it == kDeprecated.end()) return false;
    if (it->second && i + 1 < argc) ++i;
    fprintf(stderr, "warning: %s has no effect (the decision configuration is fixed by the task routing policy)\n",
            a.c_str());
    return true;
}

struct RuntimeArgs {
    RuntimePaths explicit_paths;
    bool    no_calibration = false;
    int32_t threads      = -1;
    int32_t ctx_size     = -1;
    int32_t n_gpu_layers = 0;
    int32_t main_gpu     = 0;
    int32_t threads_batch = -1;
    bool    verbose      = false;

    // Parses one resource / runtime option at argv[i]; false when `a` is not one.
    template <class Next>
    bool parse(const std::string& a, Next&& next) {
        if      (a == "--model")              explicit_paths.model = next();
        else if (a == "--embedding-model")    explicit_paths.embedding_model = next();
        else if (a == "--exemplar-db")        explicit_paths.exemplar_db = next();
        else if (a == "--exemplar-allowlist") explicit_paths.exemplar_allowlist = next();
        else if (a == "--calibration")        explicit_paths.calibration = next();
        else if (a == "--no-calibration")     no_calibration = true;
        else if (a == "--threads")            threads = std::stoi(next());
        else if (a == "--ctx-size")           ctx_size = std::stoi(next());
        else if (a == "--n-gpu-layers")       n_gpu_layers = std::stoi(next());
        else if (a == "--main-gpu")           main_gpu = std::stoi(next());
        else if (a == "--verbose")            verbose = true;
        else return false;
        return true;
    }

    // Resolves paths (flags > pjev.json > next to the executable > models/) and the thread / context
    // defaults (flags > pjev.json > built-in). A calibration path set by flag or pjev.json must exist;
    // the default one may be absent (decisions are then uncalibrated).
    RuntimeOptions resolve() const {
        const std::string exe_dir = get_executable_dir();
        ReleaseConfig rc;
        const bool have_cfg = ReleaseConfig::load_adjacent(exe_dir, rc);
        RuntimeOptions o;
        o.paths        = ReleaseConfig::resolve_runtime_paths(explicit_paths, exe_dir);
        if (no_calibration) o.paths.calibration.clear();
        o.calibration_required = !no_calibration &&
                                 (!explicit_paths.calibration.empty() || (have_cfg && !rc.calibration.empty()));
        o.n_threads    = threads > 0 ? threads : (have_cfg && rc.threads > 0 ? rc.threads : get_physical_core_count());
        o.n_ctx        = ctx_size > 0 ? ctx_size : (have_cfg && rc.context_size > 0 ? rc.context_size : 4096);
        o.n_gpu_layers = n_gpu_layers;
        o.main_gpu     = main_gpu;
        o.n_threads_batch = threads_batch;
        o.verbose      = verbose;
        return o;
    }
};

static std::string absolute_path(const std::string& p)
{
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(p, ec);
    return ec ? p : abs.lexically_normal().string();
}

static std::unique_ptr<RouterRuntime> load_runtime(const RuntimeOptions& opts)
{
    std::string err;
    std::unique_ptr<RouterRuntime> rt = RouterRuntime::load(opts, err);
    if (!rt) {
        spdlog::error("{}", err);
        spdlog::error("check the resource paths with 'pjev diagnostics' (see README: Runtime resources)");
    }
    return rt;
}

// ---------------------------------------------------------------------------
// server subcommand
// ---------------------------------------------------------------------------

// Signal handling for graceful shutdown
static PjevHttpServer* g_server_ptr = nullptr;

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static BOOL WINAPI pjev_ctrl_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
        spdlog::info("shutdown requested (Windows signal)");
        if (g_server_ptr) g_server_ptr->stop();
        return TRUE;
    }
    return FALSE;
}
#else
#include <csignal>
static void pjev_sig_handler(int32_t) {
    if (g_server_ptr) g_server_ptr->stop();
}
#endif

enum class ServerState { INITIALIZING, READY, SHUTTING_DOWN };

static int32_t cmd_server(int32_t argc, char** argv) {
    RuntimeArgs  rargs;
    ServerConfig srv_cfg;
    std::string  model_name = "pseudojev";

    for (int32_t i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_server("pjev"); exit(2); }
            return argv[++i];
        };

        if      (rargs.parse(a, next))   {}
        else if (consume_deprecated(a, i, argc)) {}
        else if (a == "--port")          srv_cfg.port = (uint16_t)std::stoi(next());
        else if (a == "--host")          srv_cfg.host = next();
        else if (a == "--threads-batch") rargs.threads_batch = std::stoi(next());
        else if (a == "--model-name")    model_name = next();
        else if (a == "--request-limit") srv_cfg.request_body_limit = (size_t)std::stoul(next());
        else if (a == "--read-timeout")  srv_cfg.read_timeout_secs = std::stoi(next());
        else if (a == "--max-queued")    srv_cfg.max_queued_requests = std::stoi(next());
        else if (a == "-h" || a == "--help") { usage_server("pjev"); return 0; }
        else {
            spdlog::error("unknown option: {}", a);
            usage_server("pjev");
            return 2;
        }
    }
    const RuntimeOptions opts = rargs.resolve();

    SpdLogShutdown spdlog_shutdown;
    spdlog::stdout_logger_mt("console");
    spdlog::set_level(opts.verbose ? spdlog::level::trace : spdlog::level::info);

    std::atomic<ServerState> server_state{ServerState::INITIALIZING};

    // Load models and the exemplar database
    std::unique_ptr<RouterRuntime> runtime = load_runtime(opts);
    if (!runtime) return 1;

    const ModelIdentity& mid = runtime->model_identity();
    const std::string model_id_for_health = mid.name.empty() ? opts.paths.model : mid.name;

    // Build API handler
    auto api_handler = std::make_unique<JevApiHandler>(runtime->service(), model_name, srv_cfg.max_queued_requests);

    // Build HTTP server
    PjevHttpServer server;
    server.configure(srv_cfg);

    // GET /health
    server.add_get("/health", [&](int32_t& status, std::string& body) {
        ServerState st = server_state.load();
        nlohmann::json h;
        if (st == ServerState::READY) {
            h["status"] = "ok";
            h["state"]  = "ready";
            h["model"]  = model_id_for_health;
            status = 200;
        } else {
            h["status"] = "unavailable";
            h["state"]  = (st == ServerState::INITIALIZING) ? "initializing" : "shutting_down";
            status = 503;
        }
        body = h.dump();
    });

    // GET /version
    server.add_get("/version", [&](int32_t& status, std::string& body) {
        nlohmann::json v;
        v["pjev_version"]      = PJEV_VERSION;
        v["pjev_commit"]       = PJEV_GIT_COMMIT;
        v["llamacpp_revision"] = PJEV_LLAMA_REVISION;
        v["model"]             = model_id_for_health;
        v["routing_policy"]    = routing::kPolicyName;
        v["calibration"]       = to_string(runtime->calibration().state);
        status = 200;
        body   = v.dump();
    });

    // POST /v1/systemone/
    server.add_post("/v1/systemone", [&](const std::string& req, int32_t& st, std::string& resp) {
        api_handler->handle(req, st, resp);
    });

    // Log startup info (only after the runtime is ready)
    spdlog::info("pjev {} (commit: {}, llama.cpp: {})", PJEV_VERSION, PJEV_GIT_COMMIT, PJEV_LLAMA_REVISION);
    spdlog::info("model: {} ({})", model_id_for_health, opts.paths.model);
    spdlog::info("routing policy: {}", routing::kPolicyName);
    spdlog::info("calibration: {}{}", to_string(runtime->calibration().state),
                 opts.paths.calibration.empty() ? "" : " (" + opts.paths.calibration + ")");
    spdlog::info("embedding model: {} (fingerprint {})", opts.paths.embedding_model, runtime->embedding_fingerprint());
    spdlog::info("exemplars: {} ({} rows, {} eligible)", opts.paths.exemplar_db, runtime->exemplar_count(),
                 runtime->eligible_exemplars());
    spdlog::info("bind: {}:{}", srv_cfg.host, (int32_t)srv_cfg.port);
    spdlog::info("request_body_limit: {} bytes", srv_cfg.request_body_limit);
    spdlog::info("read_timeout: {}s", srv_cfg.read_timeout_secs);
    spdlog::info("inference: serialized (single mutex, not safe for concurrent llama_context use)");

    server_state.store(ServerState::READY);

    // Install signal handlers
    g_server_ptr = &server;
#ifdef _WIN32
    SetConsoleCtrlHandler(pjev_ctrl_handler, TRUE);
#else
    signal(SIGINT,  pjev_sig_handler);
    signal(SIGTERM, pjev_sig_handler);
#endif

    bool ok = server.start();  // blocks until stop()

    server_state.store(ServerState::SHUTTING_DOWN);
    g_server_ptr = nullptr;

    spdlog::info("server stopped");
    // RAII: runtime (engine, models, database) released here

    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// benchmark subcommand
// ---------------------------------------------------------------------------

static bool parse_task_json(const nlohmann::json& j, DecisionInput& in, std::string& err)
{
    const nlohmann::json& t = j.contains("input") ? j["input"] : j;
    try {
        in = DecisionInput{};
        in.type     = to_type(t.at("type").get<std::string>());
        in.state    = t.value("state", "");
        in.question = t.at("question").get<std::string>();
        in.domain   = t.value("domain", "");
        in.family   = t.value("family", "");
        for (const auto& o : t.at("options")) in.options.push_back({o.at(0).get<std::string>(), o.at(1).get<std::string>()});
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

static std::vector<DecisionInput> builtin_benchmark_tasks(const std::string& state)
{
    std::vector<DecisionInput> tasks(3);
    tasks[0].type     = Type::Noul;
    tasks[0].state    = state;
    tasks[0].question = "Does the state describe a benchmark run?";
    tasks[0].options  = {{"false", ""}, {"true", ""}};
    tasks[1].type     = Type::Choice;
    tasks[1].state    = state;
    tasks[1].question = "Which category best fits the state?";
    tasks[1].options  = {{"test", "test"}, {"production", "production"}, {"unknown", "unknown"}};
    tasks[2].type     = Type::Score;
    tasks[2].state    = state;
    tasks[2].question = "How clearly does the state describe its purpose?";
    tasks[2].options  = {{"0", "Not at all"}, {"1", "Barely"}, {"2", "Partly"}, {"3", "Mostly"}, {"4", "Completely"}};
    return tasks;
}

static int32_t cmd_benchmark_latency(int32_t argc, char** argv) {
    RuntimeArgs rargs;
    int32_t iterations = 20;
    int32_t warmup     = 1;
    std::string state_text = "This is a test state for benchmarking.";
    std::string input_path;
    std::string output_path;

    for (int32_t i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_benchmark("pjev"); exit(2); }
            return argv[++i];
        };
        if      (rargs.parse(a, next))       {}
        else if (consume_deprecated(a, i, argc)) {}
        else if (a == "--iterations")        iterations = std::stoi(next());
        else if (a == "--warmup")            warmup = std::stoi(next());
        else if (a == "--state")             state_text = next();
        else if (a == "--input")             input_path = next();
        else if (a == "--output")            output_path = next();
        else if (a == "-h" || a == "--help") { usage_benchmark("pjev"); return 0; }
        else {
            spdlog::error("unknown option: {}", a);
            usage_benchmark("pjev");
            return 2;
        }
    }

    std::vector<DecisionInput> tasks;
    if (input_path.empty()) {
        tasks = builtin_benchmark_tasks(state_text);
    } else {
        std::ifstream f(input_path, std::ios::binary);
        if (!f) { spdlog::error("cannot open input: {}", input_path); return 1; }
        std::string line, err;
        size_t line_no = 0;
        while (std::getline(f, line)) {
            ++line_no;
            if (line.empty()) continue;
            DecisionInput in;
            nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_discarded() || !parse_task_json(j, in, err)) {
                spdlog::error("{}:{}: invalid task {}", input_path, line_no, err);
                return 1;
            }
            tasks.push_back(std::move(in));
        }
    }
    if (tasks.empty()) { spdlog::error("no tasks to benchmark"); return 1; }

    const RuntimeOptions opts = rargs.resolve();
    const auto t_load = std::chrono::steady_clock::now();
    std::unique_ptr<RouterRuntime> runtime = load_runtime(opts);
    if (!runtime) return 1;
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load).count();
    DecisionService& service = runtime->service();

    for (int32_t w = 0; w < warmup; w++)
        for (const auto& t : tasks) service.decide(t);

    std::map<std::string, std::vector<double>> latencies;  // kind -> ms
    std::map<std::string, std::map<std::string, int>> strategies;
    int failures = 0;
    for (int32_t it = 0; it < iterations; it++) {
        for (const auto& t : tasks) {
            const auto t0 = std::chrono::steady_clock::now();
            DecisionOutput out = service.decide(t);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const std::string kind = to_string(t.type);
            latencies[kind].push_back(ms);
            latencies["all"].push_back(ms);
            if (!out.ok) { failures++; strategies[kind]["failed:" + out.error_code]++; continue; }
            strategies[kind][out.routing ? out.routing->strategy + (out.routing->fallback ? " (fallback: " + out.routing->fallback_reason + ")" : "") : "?"]++;
        }
    }

    auto stats = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        auto pct = [&](double p) { return v.empty() ? 0.0 : v[(size_t)(p / 100.0 * (v.size() - 1))]; };
        double s = 0;
        for (double x : v) s += x;
        return nlohmann::json{{"n", v.size()}, {"mean", v.empty() ? 0.0 : s / v.size()}, {"p50", pct(50)},
                              {"p90", pct(90)}, {"p95", pct(95)}, {"max", v.empty() ? 0.0 : v.back()}};
    };
    nlohmann::json by_kind = nlohmann::json::object();
    for (const auto& [kind, v] : latencies) by_kind[kind] = stats(v);
    nlohmann::json result = {
        {"config", {
            {"model",           opts.paths.model},
            {"embedding_model", opts.paths.embedding_model},
            {"exemplar_db",     opts.paths.exemplar_db},
            {"routing_policy",  routing::kPolicyName},
            {"tasks",           tasks.size()},
            {"input",           input_path.empty() ? "built-in" : input_path},
            {"iterations",      iterations},
            {"warmup",          warmup},
            {"threads",         opts.n_threads},
            {"ctx_size",        opts.n_ctx}
        }},
        {"load_ms", load_ms},
        {"latency_ms", by_kind},
        {"strategies", strategies},
        {"failures", failures}
    };

    std::string out_str = result.dump(2);
    if (output_path.empty()) {
        printf("%s\n", out_str.c_str());
    } else {
        std::ofstream f(output_path);
        if (!f) { spdlog::error("cannot open output: {}", output_path); return 1; }
        f << out_str << "\n";
    }
    return failures == 0 ? 0 : 1;
}

static int32_t cmd_benchmark(int32_t argc, char** argv) {
    if (argc < 1 || std::string(argv[0]) == "-h" || std::string(argv[0]) == "--help") {
        usage_benchmark("pjev");
        return (argc < 1) ? 2 : 0;
    }
    std::string subcmd = argv[0];
    if (subcmd == "latency") return cmd_benchmark_latency(argc - 1, argv + 1);
    spdlog::error("unknown benchmark subcommand: {}", subcmd);
    usage_benchmark("pjev");
    return 2;
}

// ---------------------------------------------------------------------------
// diagnostics subcommand
// ---------------------------------------------------------------------------

static int32_t cmd_diagnostics(int32_t argc, char** argv) {
    RuntimeArgs rargs;
    bool json_out = false;
    bool load = false;

    for (int32_t i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_diagnostics("pjev"); exit(2); }
            return argv[++i];
        };
        if      (rargs.parse(a, next)) {}
        else if (a == "--json")         json_out = true;
        else if (a == "--load")         load = true;
        else if (a == "-h" || a == "--help") { usage_diagnostics("pjev"); return 0; }
        else {
            spdlog::error("unknown option: {}", a);
            usage_diagnostics("pjev");
            return 2;
        }
    }
    const RuntimeOptions opts = rargs.resolve();

    // CPU detection
    unsigned hw_concurrency = std::thread::hardware_concurrency();

    // Worker status
    bool worker_running = false;
    WorkerInfo worker_info;
    {
        ClientConfig cli_cfg;
        WorkerClient worker_client(cli_cfg);
        worker_running = worker_client.get_status(worker_info);
    }

    struct Resource { const char* key; const char* label; std::string path; bool ok; bool lfs_pointer; };
    auto resource = [](const char* key, const char* label, const std::string& p) {
        const bool found = std::ifstream(p).good();
        const bool lfs = found && is_lfs_pointer(p);
        return Resource{key, label, p, found && !lfs, lfs};
    };
    std::vector<Resource> resources = {
        resource("model",              "model path:      ", opts.paths.model),
        resource("embedding_model",    "embedding model: ", opts.paths.embedding_model),
        resource("exemplar_db",        "exemplar db:     ", opts.paths.exemplar_db),
        resource("exemplar_allowlist", "exemplar allow:  ", opts.paths.exemplar_allowlist),
    };
    bool all_found = true;
    for (const auto& r : resources) all_found = all_found && r.ok;

    // Optional full load
    nlohmann::json load_report;
    bool load_ok = true;
    if (load) {
        // The runtime logs while loading (calibration state, warnings); keep stdout for the JSON.
        if (json_out) spdlog::set_default_logger(spdlog::stderr_logger_mt("diagnostics"));
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<RouterRuntime> rt = all_found ? RouterRuntime::load(opts, err) : nullptr;
        if (!all_found) err = "resources missing";
        load_ok = rt != nullptr;
        load_report["ok"] = load_ok;
        load_report["ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rt) {
            load_report["model_name"]            = rt->model_identity().name;
            load_report["model_sha256"]          = rt->model_identity().sha256;
            load_report["embedding_fingerprint"] = rt->embedding_fingerprint();
            load_report["exemplar_rows"]         = rt->exemplar_count();
            load_report["eligible_exemplars"]    = rt->eligible_exemplars();
            load_report["calibration"]           = to_string(rt->calibration().state);
            if (!rt->calibration().message.empty()) load_report["calibration_message"] = rt->calibration().message;
        } else {
            load_report["error"] = err;
        }
    }

    if (json_out) {
        nlohmann::json d;
        d["pjev_version"]    = PJEV_VERSION;
        d["git_commit"]      = PJEV_GIT_COMMIT;
        d["llama_revision"]  = PJEV_LLAMA_REVISION;
        d["platform"]        = PJEV_PLATFORM;
        d["arch"]            = PJEV_ARCH;
        d["cpu"]["logical_cores"] = (int32_t)hw_concurrency;
        for (const auto& r : resources) {
            d[r.key]["path"]       = r.path;
            d[r.key]["accessible"] = r.ok;
            if (r.lfs_pointer) d[r.key]["lfs_pointer"] = true;
        }
        d["routing"]["policy"]    = routing::kPolicyName;
        d["calibration"]["path"]  = opts.paths.calibration;
        d["calibration"]["found"] = !opts.paths.calibration.empty() && std::ifstream(opts.paths.calibration).good();
        d["runtime"]["n_threads"] = opts.n_threads;
        d["runtime"]["n_ctx"]     = opts.n_ctx;
        d["worker"]["running"]    = worker_running;
        if (worker_running) {
            d["worker"]["pid"]            = (int32_t)worker_info.pid;
            d["worker"]["pjev_version"]   = worker_info.pjev_version;
            d["worker"]["model_identity"] = worker_info.model_identity;
        }
        if (load) d["load"] = load_report;
        printf("%s\n", d.dump(2).c_str());
    } else {
        printf("pjev version:    %s (commit: %s)\n", PJEV_VERSION, PJEV_GIT_COMMIT);
        printf("llama.cpp:       %s\n", PJEV_LLAMA_REVISION);
        printf("platform:        %s-%s\n", PJEV_PLATFORM, PJEV_ARCH);
        printf("cpu logical cores: %u\n", hw_concurrency);
        for (const auto& r : resources)
            printf("%s%s (%s)\n", r.label, r.path.c_str(), r.ok ? "found" : r.lfs_pointer ? "Git LFS pointer" : "NOT FOUND");
        printf("routing policy:  %s\n", routing::kPolicyName);
        if (opts.paths.calibration.empty())
            printf("calibration:     disabled\n");
        else
            printf("calibration:     %s (%s)\n", opts.paths.calibration.c_str(),
                   std::ifstream(opts.paths.calibration).good() ? "found" : "not found: decisions are uncalibrated");
        printf("threads:         %d\n", opts.n_threads);
        printf("context size:    %d\n", opts.n_ctx);
        if (worker_running) {
            printf("worker:          running (pid %d, model: %s)\n",
                   (int32_t)worker_info.pid, worker_info.model_identity.c_str());
        } else {
            printf("worker:          not running\n");
        }
        if (load) {
            if (load_ok) {
                printf("load:            ok (%.0f ms; model %s; embedding fingerprint %s; %lld exemplar rows, %zu eligible)\n",
                       load_report["ms"].get<double>(), load_report["model_name"].get<std::string>().c_str(),
                       load_report["embedding_fingerprint"].get<std::string>().c_str(),
                       (long long)load_report["exemplar_rows"].get<int64_t>(),
                       load_report["eligible_exemplars"].get<size_t>());
                printf("calibration:     %s%s%s\n", load_report["calibration"].get<std::string>().c_str(),
                       load_report.contains("calibration_message") ? ": " : "",
                       load_report.value("calibration_message", "").c_str());
            } else {
                printf("load:            FAILED: %s\n", load_report["error"].get<std::string>().c_str());
            }
        }
        for (const auto& r : resources) {
            if (r.ok) continue;
            if (r.lfs_pointer) {
                printf("\naction: %s '%s' is a Git LFS pointer; fetch the file with: git lfs pull\n", r.key, r.path.c_str());
                continue;
            }
            printf("\naction: %s not found at '%s'\n", r.key, r.path.c_str());
            printf("        pass the path on the command line, set it in pjev.json, or place the file\n");
            printf("        next to the executable (see README: Runtime resources)\n");
        }
    }
    return load_ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// CLI decision subcommands (noul / choice / score)
// ---------------------------------------------------------------------------

// Read state text from --state, --state-file, or nothing.
// Returns false and prints error on failure.
static bool resolve_state(const std::string& state_inline,
                           const std::string& state_file,
                           std::string& out)
{
    if (!state_inline.empty() && !state_file.empty()) {
        spdlog::error("--state and --state-file are mutually exclusive");
        return false;
    }
    if (!state_file.empty()) {
        if (state_file == "-") {
            std::string line, buf;
            while (std::getline(std::cin, line)) { buf += line; buf += '\n'; }
            if (!buf.empty() && buf.back() == '\n') buf.pop_back();
            out = buf;
        } else {
            std::ifstream f(state_file);
            if (!f) {
                spdlog::error("cannot open state file: {}", state_file);
                return false;
            }
            out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            if (!out.empty() && out.back() == '\n') out.pop_back();
        }
    } else {
        out = state_inline;
    }
    return true;
}

static int32_t cmd_decision(Type type, int32_t argc, char** argv) {
    RuntimeArgs rargs;
    DecisionInput input;
    input.type = type;
    std::string state_inline;
    std::string state_file;
    int32_t idle_timeout_secs = 600;
    bool json_out = false;
    bool direct_mode = false;

    for (int32_t i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_decision("pjev", type); exit(2); }
            return argv[++i];
        };
        if      (rargs.parse(a, next))   {}
        else if (consume_deprecated(a, i, argc)) {}
        else if (a == "--question")      input.question = next();
        else if (a == "--state")         state_inline = next();
        else if (a == "--state-file")    state_file = next();
        else if (a == "--domain")        input.domain = next();
        else if (a == "--family")        input.family = next();
        else if (a == "--idle-timeout")  idle_timeout_secs = std::stoi(next());
        else if (a == "--json")          json_out = true;
        else if (a == "--direct")        direct_mode = true;
        else if (a == "--option" && type == Type::Choice) {
            std::string val = next();
            auto colon = val.find(':');
            if (colon != std::string::npos) input.options.push_back({val.substr(0, colon), val.substr(colon + 1)});
            else                            input.options.push_back({val, ""});
        }
        else if (a == "--level" && type == Type::Score) {
            input.options.push_back({std::to_string(input.options.size()), next()});
        }
        else if (a == "-h" || a == "--help") { usage_decision("pjev", type); return 0; }
        else { spdlog::error("unknown option: {}", a); usage_decision("pjev", type); return 2; }
    }

    if (input.question.empty()) {
        spdlog::error("--question is required");
        usage_decision("pjev", type);
        return 2;
    }
    if (type == Type::Noul) input.options = {{"true", ""}, {"false", ""}};
    if (input.options.size() < 2) {
        spdlog::error("{} requires at least 2 {} values", to_string(type), type == Type::Score ? "--level" : "--option");
        usage_decision("pjev", type);
        return 2;
    }
    if (!resolve_state(state_inline, state_file, input.state)) return 2;

    const RuntimeOptions opts = rargs.resolve();
    DecisionOutput out;

    if (!direct_mode) {
        // A worker that cannot load its resources never starts; report missing files right away.
        for (const std::string* p : {&opts.paths.model, &opts.paths.embedding_model, &opts.paths.exemplar_db,
                                     &opts.paths.exemplar_allowlist}) {
            if (!std::filesystem::is_regular_file(*p) || is_lfs_pointer(*p)) {
                spdlog::error("runtime resource missing or not fetched: {} (see 'pjev diagnostics')", *p);
                return 1;
            }
        }
        ClientConfig cli_cfg;
        cli_cfg.paths.model              = absolute_path(opts.paths.model);
        cli_cfg.paths.embedding_model    = absolute_path(opts.paths.embedding_model);
        cli_cfg.paths.exemplar_db        = absolute_path(opts.paths.exemplar_db);
        cli_cfg.paths.exemplar_allowlist = absolute_path(opts.paths.exemplar_allowlist);
        if (!opts.paths.calibration.empty()) cli_cfg.paths.calibration = absolute_path(opts.paths.calibration);
        cli_cfg.calibration_required     = opts.calibration_required;
        cli_cfg.threads           = opts.n_threads;
        cli_cfg.ctx_size          = opts.n_ctx;
        cli_cfg.n_gpu_layers      = opts.n_gpu_layers;
        cli_cfg.main_gpu          = opts.main_gpu;
        cli_cfg.idle_timeout_secs = idle_timeout_secs;
        try {
            WorkerClient client(cli_cfg);
            out = client.decide(input);
        } catch (const std::exception& e) {
            spdlog::error("worker decision failed: {}", e.what());
            return 1;
        }
    } else {
        // The default logger stays alive: the outcome below is reported through it.
        spdlog::set_level(opts.verbose ? spdlog::level::trace : spdlog::level::warn);

        std::unique_ptr<RouterRuntime> runtime = load_runtime(opts);
        if (!runtime) return 1;
        try {
            out = runtime->service().decide(input);
        } catch (const std::exception& e) {
            spdlog::error("decision failed: {}", e.what());
            return 1;
        }
    }

    if (!out.ok) {
        if (out.error_code.empty()) spdlog::error("{}", out.error);
        else                        spdlog::error("{}: {}", out.error_code, out.error);
        return 1;
    }

    switch (type) {
    case Type::Noul:
        printf("%s\n", (json_out ? format_noul_json(out) : format_noul_human(out)).c_str());
        break;
    case Type::Choice:
        printf("%s\n", (json_out ? format_choice_json(out) : format_choice_human(out)).c_str());
        break;
    default:
        printf("%s\n", (json_out ? format_score_json(out, input.options) : format_score_human(out)).c_str());
        break;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// status subcommand
// ---------------------------------------------------------------------------

static int32_t cmd_status(int32_t argc, char** argv) {
    bool json_out = false;
    for (int32_t i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--json") json_out = true;
        else if (a == "-h" || a == "--help") {
            fprintf(stderr, "usage: pjev status [--json]\n");
            return 0;
        }
        // ignore unknown args
    }

    ClientConfig cli_cfg; // default config
    WorkerClient client(cli_cfg);
    WorkerInfo info;
    if (client.get_status(info)) {
        if (json_out) {
            printf("%s\n", ipc_worker_info_to_json(info).dump(2).c_str());
        } else {
            printf("pjev worker: running\n");
            printf("  pid:              %d\n",  (int32_t)info.pid);
            printf("  protocol:         %d\n",  info.protocol_version);
            printf("  version:          %s\n",  info.pjev_version.c_str());
            printf("  model_identity:   %s\n",  info.model_identity.c_str());
            printf("  config_hash:      %s\n",  info.config_hash.c_str());
            printf("  calibration:      %s\n",  info.calibration.c_str());
        }
        return 0;
    } else {
        if (json_out) {
            printf("{\"running\":false}\n");
        } else {
            printf("pjev worker: not running\n");
        }
        return 0;
    }
}

// ---------------------------------------------------------------------------
// stop subcommand
// ---------------------------------------------------------------------------

static int32_t cmd_stop(int32_t argc, char** argv) {
    (void)argc; (void)argv;
    ClientConfig cli_cfg;
    WorkerClient client(cli_cfg);
    if (client.stop_worker()) {
        printf("worker stopped\n");
        return 0;
    } else {
        printf("no worker running\n");
        return 0;
    }
}

// ---------------------------------------------------------------------------
// calibrate subcommand
// ---------------------------------------------------------------------------

static void usage_calibrate(const char* prog) {
    fprintf(stderr,
        "usage: %s calibrate [options]\n"
        "  Generates a model-specific calibration.json from the exemplar database: the production decision\n"
        "  path's uncalibrated candidate logits on held-out training exemplars, temperature and prior\n"
        "  correction fitted per task kind, checked on a validation partition, written atomically.\n"
        "  --model PATH               inference model to calibrate (.gguf)\n"
        "  --database PATH            exemplar database, opened read-only (default: examplar.db)\n"
        "  --output PATH              calibration file to write (default: the file the runtime loads,\n"
        "                             calibration.json; models/calibration.json in a checkout)\n"
        "  --seed N                   sampling and partition seed (default: %llu)\n"
        "  --max-samples N            maximum calibration queries per task kind (default: %d)\n"
        "  --validation-ratio R       fraction of query groups held out for validation (default: %.2f)\n"
        "  --batch-size N             queries per processing batch and progress report (default: %d)\n"
        "  --threads N                CPU threads (default: physical cores)\n"
        "  --verbose                  one line per query\n"
        "  --embedding-model PATH     query embedding model (.gguf)\n"
        "  --exemplar-allowlist PATH  training-split exemplar IDs\n"
        "  --ctx-size N               context size (default: 4096, as at runtime)\n"
        "  --n-gpu-layers N           number of layers to offload to GPU (default: 0)\n"
        "  --main-gpu N               GPU device index to use as main GPU (default: 0)\n"
        "\n"
        "Needs at least %d fitting and %d validation samples per task kind. An existing output file is\n"
        "replaced only by a calibration that passes validation; a report is written next to it\n"
        "(<output>.report.json, or <output>.failed-report.json when validation rejects the result).\n",
        prog, (unsigned long long)calibration::kDefaultSeed, calibration::kDefaultMaxSamples,
        calibration::kDefaultValidationRatio, calibration::kDefaultBatchSize, calibration::kMinFitSamples,
        calibration::kMinValidationSamples);
}

static int32_t cmd_calibrate(int32_t argc, char** argv) {
    RuntimeArgs rargs;
    calibration::GeneratorOptions gopts;
    std::string output;

    for (int32_t i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_calibrate("pjev"); exit(2); }
            return argv[++i];
        };
        try {
            if      (a == "--database")         rargs.explicit_paths.exemplar_db = next();
            else if (a == "--output")           output = next();
            else if (a == "--seed")             gopts.seed = std::stoull(next());
            else if (a == "--max-samples")      gopts.max_samples = std::stoi(next());
            else if (a == "--validation-ratio") gopts.validation_ratio = std::stod(next());
            else if (a == "--batch-size")       gopts.batch_size = std::stoi(next());
            else if (a == "--verbose")          gopts.verbose = true;
            else if (a == "--calibration" || a == "--no-calibration") {
                spdlog::error("{}: calibrate never reads a calibration file (use --output for the file to write)", a);
                return 2;
            }
            else if (a == "-h" || a == "--help") { usage_calibrate("pjev"); return 0; }
            else if (rargs.parse(a, next)) {}
            else {
                spdlog::error("unknown option: {}", a);
                usage_calibrate("pjev");
                return 2;
            }
        } catch (const std::exception&) {
            spdlog::error("invalid value for {}", a);
            return 2;
        }
    }
    std::string err;
    if (!gopts.validate(err)) { spdlog::error("{}", err); return 2; }

    // Model execution exactly as at runtime, but uncalibrated: the raw logits are what is fitted.
    RuntimeOptions opts = rargs.resolve();
    if (output.empty()) output = opts.paths.calibration;
    opts.paths.calibration.clear();
    opts.calibration_required = false;
    opts.verbose = false;

    fprintf(stderr, "[calibrate] model %s\n[calibrate] database %s (read-only)\n[calibrate] output %s\n",
            opts.paths.model.c_str(), opts.paths.exemplar_db.c_str(), output.c_str());
    fprintf(stderr, "[calibrate] stage: load\n");
    const std::string db_sha_before = sha256_file(opts.paths.exemplar_db);
    std::unique_ptr<RouterRuntime> runtime = load_runtime(opts);
    if (!runtime) return 1;
    const ModelIdentity mid = runtime->model_identity();
    retrieval::ExemplarStore& store = runtime->exemplar_store();

    calibration::GenerateRequest req;
    req.output_path  = output;
    req.model_sha256 = mid.sha256;
    req.options      = gopts;
    req.provenance   = {
        {"pjev_version", PJEV_VERSION},
        {"pjev_commit", PJEV_GIT_COMMIT},
        {"llamacpp_revision", PJEV_LLAMA_REVISION},
        {"model", {{"path", absolute_path(opts.paths.model)}, {"name", mid.name}, {"sha256", mid.sha256},
                   {"size_bytes", mid.size_bytes}, {"architecture", mid.architecture},
                   {"quantization", mid.quantization}}},
        {"database", {{"path", absolute_path(opts.paths.exemplar_db)}, {"sha256", db_sha_before},
                      {"schema_version", store.info("schema_version")}, {"rows", runtime->exemplar_count()},
                      {"embedding_fingerprint", store.info("embedding_fingerprint")}}},
        {"allowlist", {{"path", absolute_path(opts.paths.exemplar_allowlist)},
                       {"sha256", sha256_file(opts.paths.exemplar_allowlist)}, {"ids", runtime->eligible_exemplars()}}},
        {"embedding_model", {{"path", absolute_path(opts.paths.embedding_model)},
                             {"fingerprint", runtime->embedding_fingerprint()}}},
        {"runtime", {{"n_ctx", opts.n_ctx}, {"n_threads", opts.n_threads}, {"n_gpu_layers", opts.n_gpu_layers}}},
    };

    const auto t0 = std::chrono::steady_clock::now();
    auto progress = [&](const calibration::Progress& p) {
        if (p.stage != "collect") { fprintf(stderr, "[calibrate] stage: %s\n", p.stage.c_str()); return; }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::string kinds;
        for (const auto& [t, c] : p.by_kind)
            kinds += std::string(kinds.empty() ? "" : ", ") + to_string(t) + " " + std::to_string(c.first) + "/" +
                     std::to_string(c.second);
        fprintf(stderr, "[calibrate] collect %zu/%zu (%s); accepted %zu, rejected %zu; %.0fs elapsed\n", p.processed,
                p.total, kinds.c_str(), p.accepted, p.rejected, s);
    };
    calibration::GenerateOutcome res = calibration::generate_calibration(store, runtime->router(), runtime->engine(), req,
                                                                         progress);
    runtime.reset();

    // The database is opened read-only; confirm it is byte-identical after the run.
    const bool db_unchanged = sha256_file(opts.paths.exemplar_db) == db_sha_before;
    if (!db_unchanged) fprintf(stderr, "[calibrate] error: the exemplar database changed during the run\n");

    if (res.report.is_object() && res.report.contains("kinds")) {
        printf("%-7s %-12s %-11s %6s  %-17s %-17s %-17s %-17s\n", "kind", "temperature", "prior_alpha", "n_val",
               "accuracy", "nll", "brier", "ece");
        for (const char* k : {"choice", "noul", "score"}) {
            const auto& kj = res.report["kinds"][k];
            const auto& v  = kj["validation"];
            auto pair = [&](const char* m) {
                char buf[64];
                snprintf(buf, sizeof buf, "%.4f -> %.4f", v["uncalibrated"][m].get<double>(), v["calibrated"][m].get<double>());
                return std::string(buf);
            };
            printf("%-7s %-12.6g %-11.4g %6d  %-17s %-17s %-17s %-17s\n", k, kj["parameters"]["temperature"].get<double>(),
                   kj["parameters"]["prior_alpha"].get<double>(), v["uncalibrated"]["n"].get<int>(), pair("accuracy").c_str(),
                   pair("nll").c_str(), pair("brier").c_str(), pair("ece").c_str());
        }
    }
    for (const auto& w : res.warnings) fprintf(stderr, "[calibrate] warning: %s\n", w.c_str());
    if (!res.report_path.empty()) printf("report: %s\n", res.report_path.c_str());
    if (!res.ok) {
        spdlog::error("calibration failed: {}", res.error);
        return 1;
    }
    printf("calibration written: %s (model sha256 %s)\n", output.c_str(), mid.sha256.c_str());
    return db_unchanged ? 0 : 1;
}

// ---------------------------------------------------------------------------
// internal-worker subcommand (hidden)
// ---------------------------------------------------------------------------

static int32_t cmd_internal_worker(int32_t argc, char** argv) {
    WorkerConfig cfg;
    for (int32_t i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                spdlog::error("{} requires an argument", a);
                exit(2);
            }
            return argv[++i];
        };
        if      (a == "--model")              cfg.paths.model = next();
        else if (a == "--embedding-model")    cfg.paths.embedding_model = next();
        else if (a == "--exemplar-db")        cfg.paths.exemplar_db = next();
        else if (a == "--exemplar-allowlist") cfg.paths.exemplar_allowlist = next();
        else if (a == "--calibration")        cfg.paths.calibration = next();
        else if (a == "--calibration-required") cfg.calibration_required = true;
        else if (a == "--no-calibration")     cfg.paths.calibration.clear();
        else if (a == "--threads")            cfg.threads = std::stoi(next());
        else if (a == "--ctx-size")           cfg.ctx_size = std::stoi(next());
        else if (a == "--n-gpu-layers")       cfg.n_gpu_layers = std::stoi(next());
        else if (a == "--main-gpu")           cfg.main_gpu = std::stoi(next());
        else if (a == "--idle-timeout")       cfg.idle_timeout_secs = std::stoi(next());
        else if (a == "--max-queued")         cfg.max_queued = std::stoi(next());
        else {
            spdlog::error("unknown option: {}", a);
            return 2;
        }
    }

    if (cfg.paths.model.empty() || cfg.paths.embedding_model.empty() || cfg.paths.exemplar_db.empty() ||
        cfg.paths.exemplar_allowlist.empty()) {
        spdlog::error("internal-worker requires --model, --embedding-model, --exemplar-db and --exemplar-allowlist");
        return 1;
    }

    WorkerServer server(cfg);
    return server.run();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int32_t main(int32_t argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    initialize_strings();

    std::string cmd = argv[1];

    if (cmd == "noul") {
        return cmd_decision(Type::Noul, argc - 2, argv + 2);
    } else if (cmd == "choice") {
        return cmd_decision(Type::Choice, argc - 2, argv + 2);
    } else if (cmd == "score") {
        return cmd_decision(Type::Score, argc - 2, argv + 2);
    } else if (cmd == "status") {
        return cmd_status(argc - 2, argv + 2);
    } else if (cmd == "stop") {
        return cmd_stop(argc - 2, argv + 2);
    } else if (cmd == "internal-worker") {
        return cmd_internal_worker(argc - 2, argv + 2);
    } else if (cmd == "server" || cmd == "serve") {
        return cmd_server(argc - 2, argv + 2);
    } else if (cmd == "--version" || cmd == "-V") {
        printf("pjev %s (commit: %s, llama.cpp: %s, platform: %s-%s)\n",
               PJEV_VERSION, PJEV_GIT_COMMIT, PJEV_LLAMA_REVISION, PJEV_PLATFORM, PJEV_ARCH);
        return 0;
    } else if (cmd == "benchmark") {
        return cmd_benchmark(argc - 2, argv + 2);
    } else if (cmd == "diagnostics") {
        return cmd_diagnostics(argc - 2, argv + 2);
    } else if (cmd == "calibrate") {
        return cmd_calibrate(argc - 2, argv + 2);
    } else if (cmd == "-h" || cmd == "--help") {
        usage(argv[0]);
        return 0;
    } else if (cmd[0] == '-') {
        // Legacy: no subcommand, treat all args as server args
        return cmd_server(argc - 1, argv + 1);
    } else {
        spdlog::error("unknown command: {}", cmd);
        usage(argv[0]);
        return 2;
    }
}
