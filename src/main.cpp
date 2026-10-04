#ifndef PJEV_VERSION
#define PJEV_VERSION "0.6.0"
#endif

#include "calibration/calibration.h"
#include "calibration/calibration_fit.h"
#include "calibration/calibration_metrics.h"
#include "multilingual/multilingual.h"
#include "model/model_identity.h"
#include "model/value_profile.h"
#include "util/platform.h"
#include "util/timer.h"
#include "inference/llama_backend.h"
#include "decision/decision_engine.h"
#include "server/jev_api.h"
#include "server/server_config.h"
#include "server/http_server.h"
#include "experiment/config.h"
#include "experiment/dataset.h"
#include "experiment/runner.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_sinks.h"

// ---------------------------------------------------------------------------
// Usage helpers
// ---------------------------------------------------------------------------
using namespace pjev;
static void usage_server(const char* prog) {
    fprintf(stderr,
        "usage: %s server|serve --model PATH [options]\n"
        "  --model PATH           path to .gguf model file (default: models/Bonsai-1.7B.gguf)\n"
        "  --port N               HTTP port (default: 8080)\n"
        "  --host ADDR            bind address (default: 127.0.0.1)\n"
        "  --calibration FILE     load calibration artifact (optional)\n"
        "  --layout LAYOUT        auto|state-first|state-last|question-first (default: auto)\n"
        "  --scheme SCHEME        natural|letters (default: natural)\n"
        "  --threads N            CPU threads (default: 8)\n"
        "  --threads-batch N      batch CPU threads (default: same as --threads)\n"
        "  --ctx-size N           context size (default: 4096)\n"
        "  --verbose              enable llama.cpp verbose logging\n"
        "  --model-name NAME      model name in responses (default: pseudojev)\n"
        "  --request-limit BYTES  max request body size in bytes (default: 1048576)\n"
        "  --read-timeout SECS    read timeout in seconds (default: 30)\n"
        "  --max-queued N         max concurrent requests before returning 503 (default: 8)\n"
        "  --no-kv-reuse          disable KV prefix reuse for multi-question requests\n"
        "  --no-batch-questions   disable batching questions within a request\n",
        prog);
}

static void usage_run(const char* prog) {
    fprintf(stderr,
        "usage: %s run --model PATH --input FILE [options]\n"
        "  --model PATH               path to .gguf model file (default: models/Bonsai-1.7B.gguf)\n"
        "  --input FILE               JSONL/JSON dataset file (required, or - for stdin)\n"
        "  --output FILE              write JSON result here (default: stdout)\n"
        "  --calibration FILE         load calibration artifact (optional)\n"
        "  --layout LAYOUT            auto|state-first|state-last|question-first (default: auto)\n"
        "  --scheme SCHEME            natural|letters (default: natural)\n"
        "  --prior-correction         enable prior correction\n"
        "  --option-order ORDER       original|reversed|random (default: original)\n"
        "  --threads N                CPU threads (default: 8)\n"
        "  --ctx-size N               context size (default: 4096)\n"
        "  --verbose                  enable llama.cpp verbose logging\n",
        prog);
}

static void usage_calibration(const char* prog) {
    fprintf(stderr,
        "usage: %s calibration <fit|evaluate> [options]\n"
        "\n"
        "  fit     fit per-primitive temperatures from a calibration dataset\n"
        "    --model PATH        path to .gguf model file (required)\n"
        "    --input FILE        JSONL/JSON calibration dataset (required)\n"
        "    --output FILE       write calibration artifact JSON here (required)\n"
        "    --model-id NAME     model identifier stored in artifact (default: model path)\n"
        "    --layout LAYOUT     auto|state-first|state-last|question-first (default: auto)\n"
        "    --scheme SCHEME     natural|letters (default: natural)\n"
        "    --prior-correction  enable prior correction\n"
        "    --threads N         CPU threads (default: 8)\n"
        "    --ctx-size N        context size (default: 4096)\n"
        "    --verbose           enable llama.cpp verbose logging\n"
        "\n"
        "  evaluate  compare raw vs calibrated metrics on a dataset\n"
        "    --model PATH        path to .gguf model file (required)\n"
        "    --input FILE        JSONL/JSON evaluation dataset (required)\n"
        "    --calibration FILE  calibration artifact (required)\n"
        "    --output FILE       write report JSON here (default: stdout)\n"
        "    --layout LAYOUT     auto|state-first|state-last|question-first (default: auto)\n"
        "    --scheme SCHEME     natural|letters (default: natural)\n"
        "    --prior-correction  enable prior correction\n"
        "    --threads N         CPU threads (default: 8)\n"
        "    --ctx-size N        context size (default: 4096)\n"
        "    --verbose           enable llama.cpp verbose logging\n",
        prog);
}

static void usage_experiment(const char* prog) {
    fprintf(stderr,
        "usage: %s experiment NAME --model PATH --input FILE [options]\n"
        "  NAME: candidate-binding | option-order | prompt-layout | prior-correction | score-formulation\n"
        "        multilingual  (requires --input-reference and --input-target instead of --input)\n"
        "  --model PATH              path to .gguf model file (required)\n"
        "  --input FILE              JSONL/JSON dataset file (single-language experiments)\n"
        "  --input-reference FILE    reference/English dataset (multilingual experiment)\n"
        "  --input-target FILE       target/Japanese dataset  (multilingual experiment)\n"
        "  --ref-language LANG       reference language tag (default: en)\n"
        "  --target-language LANG    target language tag (default: ja)\n"
        "  --output DIR/FILE         output file or directory (default: stdout)\n"
        "  --calibration FILE        calibration artifact (optional)\n"
        "  --layout LAYOUT           auto|state-first|state-last|question-first\n"
        "  --scheme SCHEME           natural|letters\n"
        "  --prior-correction        enable prior correction\n"
        "  --threads N               CPU threads (default: 8)\n"
        "  --ctx-size N              context size (default: 4096)\n"
        "  --verbose                 enable llama.cpp verbose logging\n",
        prog);
}

static void usage_model(const char* prog) {
    fprintf(stderr,
        "usage: %s model <inspect|evaluate> [options]\n"
        "\n"
        "  inspect   load a model and report identity, candidate compatibility, RSS, smoke test\n"
        "    --model PATH         path to .gguf model file (required)\n"
        "    --threads N          CPU threads (default: 8)\n"
        "    --ctx-size N         context size (default: 4096)\n"
        "    --output FILE        write JSON report here (default: stdout)\n"
        "    --verbose            enable llama.cpp verbose logging\n"
        "\n"
        "  evaluate   run full benchmark and produce a model value profile\n"
        "    --model PATH         path to .gguf model file (required)\n"
        "    --input FILE         English benchmark dataset (required)\n"
        "    --input-ja FILE      Japanese benchmark dataset (optional)\n"
        "    --calibration FILE   load existing calibration artifact\n"
        "    --fit-calibration    fit model-specific calibration from the run\n"
        "    --output FILE        write ValueProfile JSON here (required)\n"
        "    --layout LAYOUT      auto|state-first|state-last|question-first (default: auto)\n"
        "    --scheme SCHEME      natural|letters (default: natural)\n"
        "    --prior-correction   enable prior correction\n"
        "    --threads N          CPU threads (default: 8)\n"
        "    --ctx-size N         context size (default: 4096)\n"
        "    --verbose            enable llama.cpp verbose logging\n",
        prog);
}

static void usage_benchmark(const char* prog) {
    fprintf(stderr,
        "usage: %s benchmark latency --model PATH [options]\n"
        "  --model PATH       path to .gguf model file (required)\n"
        "  --iterations N     number of measurement iterations (default: 20)\n"
        "  --warmup N         number of warmup iterations (default: 3)\n"
        "  --questions N      questions per request (default: 1)\n"
        "  --state TEXT       shared state text (default: \"This is a test state for benchmarking.\")\n"
        "  --output FILE      JSON output file (optional)\n"
        "  --threads N        CPU threads (default: 8)\n"
        "  --ctx-size N       context size (default: 4096)\n"
        "  --no-kv-reuse      disable KV prefix reuse\n"
        "  --no-batch-questions disable question batching\n",
        prog);
}

static void usage_diagnostics(const char* prog) {
    fprintf(stderr,
        "usage: %s diagnostics --model PATH [options]\n"
        "  --model PATH       path to .gguf model file (required)\n"
        "  --threads N        CPU threads (default: 8)\n"
        "  --ctx-size N       context size (default: 4096)\n"
        "  --no-kv-reuse      disable KV prefix reuse\n"
        "  --no-batch-questions disable question batching\n",
        prog);
}

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s <command> [options]\n"
        "commands:\n"
        "  server|serve start the HTTP server (default if no command given)\n"
        "  run          batch evaluation with a single config\n"
        "  experiment   run a named experiment\n"
        "  calibration  fit or evaluate calibration temperatures\n"
        "  model        model value gate: inspect and evaluate models\n"
        "  benchmark    latency benchmarking\n"
        "  diagnostics  print configuration without loading model\n"
        "  --version    print version and exit\n"
        "run '%s <command> --help' for command-specific options\n",
        prog, prog);
}

// ---------------------------------------------------------------------------
// Layout / Scheme / OptionOrder parse helpers
// ---------------------------------------------------------------------------

static Layout parse_layout(const std::string& s) {
    if (s == "state-first")    return Layout::STATE_FIRST;
    if (s == "state-last")     return Layout::STATE_LAST;
    if (s == "question-first") return Layout::QUESTION_FIRST;
    return Layout::AUTO;
}

static Scheme parse_scheme(const std::string& s) {
    return (s == "letters") ? Scheme::LETTERS : Scheme::NATURAL;
}

static OptionOrder parse_order(const std::string& s) {
    if (s == "reversed") return OptionOrder::REVERSED;
    if (s == "random")   return OptionOrder::RANDOM;
    return OptionOrder::ORIGINAL;
}

struct SpdLogShutdown {
    SpdLogShutdown() = default;
    ~SpdLogShutdown()
    {
        spdlog::shutdown();
    }
};

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
static void pjev_sig_handler(int) {
    if (g_server_ptr) g_server_ptr->stop();
}
#endif

enum class ServerState { INITIALIZING, READY, SHUTTING_DOWN };

static int cmd_server(int argc, char** argv) {
    LlamaConfig  llama_cfg;
    PromptConfig prompt_cfg;
    llama_cfg.model_path = "models/Bonsai-1.7B.gguf";
    ServerConfig srv_cfg;
    std::string  model_name      = "pseudojev";
    std::string  calibration_path;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_server(argv[-1]); exit(2); }
            return argv[++i];
        };

        if      (a == "--model")        llama_cfg.model_path = next();
        else if (a == "--port")         srv_cfg.port = (uint16_t)std::stoi(next());
        else if (a == "--host")         srv_cfg.host = next();
        else if (a == "--threads")      llama_cfg.n_threads = std::stoi(next());
        else if (a == "--threads-batch") llama_cfg.n_threads_batch = std::stoi(next());
        else if (a == "--ctx-size")     llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")      llama_cfg.verbose = true;
        else if (a == "--model-name")   model_name = next();
        else if (a == "--layout")       prompt_cfg.layout = parse_layout(next());
        else if (a == "--scheme")       prompt_cfg.scheme = parse_scheme(next());
        else if (a == "--calibration")  calibration_path = next();
        else if (a == "--request-limit") srv_cfg.request_body_limit = (size_t)std::stoul(next());
        else if (a == "--read-timeout")  srv_cfg.read_timeout_secs = std::stoi(next());
        else if (a == "--max-queued")    srv_cfg.max_queued_requests = std::stoi(next());
        else if (a == "--no-kv-reuse")   srv_cfg.kv_reuse = false;
        else if (a == "--no-batch-questions") srv_cfg.batch_questions = false;
        else if (a == "-h" || a == "--help") { usage_server("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_server("pjev");
            return 2;
        }
    }

    SpdLogShutdown spdlog_shutdown;
    spdlog::stdout_logger_mt("console");
    spdlog::set_level(llama_cfg.verbose ? spdlog::level::trace : spdlog::level::info);

    if (llama_cfg.model_path.empty()) {
        spdlog::error("--model is required");
        usage_server("pjev");
        return 1;
    }

    std::atomic<ServerState> server_state{ServerState::INITIALIZING};
    std::string model_id_for_health;

    // Load model
    std::unique_ptr<LlamaBackend> backend;
    try {
        backend = std::make_unique<LlamaBackend>(llama_cfg);
    } catch (const std::exception& e) {
        spdlog::error("model load error: {}", e.what());
        return 1;
    }

    // Load calibration (optional)
    CalibrationConfig calib_cfg;
    if (!calibration_path.empty()) {
        std::ifstream cf(calibration_path);
        if (!cf) {
            spdlog::error("cannot open calibration file: {}", calibration_path);
            return 1;
        }
        nlohmann::json cj;
        try { cf >> cj; } catch (const std::exception& e) {
            spdlog::error("calibration JSON parse error: {}", e.what());
            return 1;
        }
        CalibrationArtifact art;
        std::string calib_err;
        if (!CalibrationArtifact::from_json(cj, art, calib_err)) {
            spdlog::error("calibration artifact error: {}", calib_err);
            return 1;
        }
        calib_cfg = art.to_config();
        spdlog::info("calibration loaded from {}", calibration_path);
    }

    // Initialize DecisionEngine
    std::unique_ptr<DecisionEngine> engine;
    try {
        engine = std::make_unique<DecisionEngine>(*backend, prompt_cfg, calib_cfg);
    } catch (const std::exception& e) {
        spdlog::error("engine init error: {}", e.what());
        return 1;
    }

    // Get model identity for health/version
    ModelIdentity mid = ModelIdentity::from_path(llama_cfg.model_path, backend.get());
    model_id_for_health = mid.name.empty() ? llama_cfg.model_path : mid.name;

    // Build API handler
    auto api_handler = std::make_unique<JevApiHandler>(
        *engine, model_name, srv_cfg.max_queued_requests,
        srv_cfg.kv_reuse, srv_cfg.batch_questions);

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
        status = 200;
        body   = v.dump();
    });

    // POST /api/v1/systemone/
    server.add_post("/api/v1/systemone/", [&](const std::string& req, int32_t& st, std::string& resp) {
        api_handler->handle(req, st, resp);
    });

    // Log startup info (only after model is ready)
    spdlog::info("pjev {} (commit: {}, llama.cpp: {})", PJEV_VERSION, PJEV_GIT_COMMIT, PJEV_LLAMA_REVISION);
    spdlog::info("model: {}", model_id_for_health);
    spdlog::info("bind: {}:{}", srv_cfg.host, (int)srv_cfg.port);
    spdlog::info("request_body_limit: {} bytes", srv_cfg.request_body_limit);
    spdlog::info("read_timeout: {}s", srv_cfg.read_timeout_secs);
    spdlog::info("inference: serialized (single mutex, not safe for concurrent llama_context use)");
    spdlog::info("kv_reuse: {} batch_questions: {}",
        srv_cfg.kv_reuse, srv_cfg.batch_questions);

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
    // RAII: engine and backend destroyed in reverse order here

    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// run subcommand
// ---------------------------------------------------------------------------

static bool load_calibration_artifact(const std::string& path,
                                      CalibrationArtifact& art,
                                      std::string& err)
{
    std::ifstream cf(path);
    if (!cf) { err = "cannot open file: " + path; return false; }
    nlohmann::json cj;
    try { cf >> cj; } catch (const std::exception& e) {
        err = std::string("JSON parse error: ") + e.what(); return false;
    }
    return CalibrationArtifact::from_json(cj, art, err);
}

static int cmd_run(int argc, char** argv) {
    LlamaConfig    llama_cfg;
    llama_cfg.model_path = "models/Bonsai-1.7B.gguf";
    ExperimentConfig exp_cfg;
    exp_cfg.name = "run";
    std::string input_path;
    std::string output_path;
    std::string calibration_path;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_run("pjev"); exit(2); }
            return argv[++i];
        };

        if      (a == "--model")           llama_cfg.model_path = next();
        else if (a == "--input")           input_path = next();
        else if (a == "--output")          output_path = next();
        else if (a == "--calibration")     calibration_path = next();
        else if (a == "--threads")         llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")        llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")         llama_cfg.verbose = true;
        else if (a == "--layout")          exp_cfg.layout = parse_layout(next());
        else if (a == "--scheme")          exp_cfg.scheme = parse_scheme(next());
        else if (a == "--prior-correction") exp_cfg.prior_correction = true;
        else if (a == "--option-order")    exp_cfg.option_order = parse_order(next());
        else if (a == "-h" || a == "--help") { usage_run("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_run("pjev");
            return 2;
        }
    }

    if (!calibration_path.empty()) {
        CalibrationArtifact art;
        std::string calib_err;
        if (!load_calibration_artifact(calibration_path, art, calib_err)) {
            spdlog::error("calibration error: {}", calib_err.c_str());
            return 1;
        }
        exp_cfg.calibration = art.to_config();
    }

    SpdLogShutdown spdlog_shutdown;
    spdlog::stdout_logger_mt("console");
    if(llama_cfg.verbose){
        spdlog::set_level(spdlog::level::trace);
    }else{
        spdlog::set_level(spdlog::level::warn);
    }

    if (llama_cfg.model_path.empty()) {
        spdlog::error("--model is required");
        usage_run("pjev");
        return 1;
    }
    if (input_path.empty()) {
        spdlog::error("--input is required");
        usage_run("pjev");
        return 1;
    }

    std::vector<DatasetRow> rows;
    std::string err;
    if (!load_dataset(input_path, rows, err)) {
        spdlog::error("dataset error: {}", err.c_str());
        return 1;
    }
    if (rows.empty()) {
        spdlog::warn("dataset is empty");
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        spdlog::error("model load error: {}", e.what());
        return 1;
    }

    RunResult result = run_experiment(*backend, rows, exp_cfg);
    delete backend;

    std::string output_str = result.to_json().dump(2);
    if (output_path.empty()) {
        printf("%s\n", output_str.c_str());
    } else {
        std::ofstream f(output_path);
        if (!f) {
            spdlog::error("cannot open output: {}", output_path.c_str());
            return 1;
        }
        f << output_str << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------
// experiment subcommand
// ---------------------------------------------------------------------------

static int cmd_experiment(int argc, char** argv) {
    if (argc < 1) {
        usage_experiment("pjev");
        return 2;
    }

    std::string exp_name = argv[0];
    LlamaConfig      llama_cfg;
    ExperimentConfig exp_cfg;
    std::string input_path;
    std::string input_reference;
    std::string input_target;
    std::string ref_language   = "en";
    std::string target_language = "ja";
    std::string output_dir;
    std::string calibration_path;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_experiment("pjev"); exit(2); }
            return argv[++i];
        };

        if      (a == "--model")            llama_cfg.model_path = next();
        else if (a == "--input")            input_path = next();
        else if (a == "--input-reference")  input_reference = next();
        else if (a == "--input-target")     input_target = next();
        else if (a == "--ref-language")     ref_language = next();
        else if (a == "--target-language")  target_language = next();
        else if (a == "--output")           output_dir = next();
        else if (a == "--calibration")      calibration_path = next();
        else if (a == "--layout")           exp_cfg.layout = parse_layout(next());
        else if (a == "--scheme")           exp_cfg.scheme = parse_scheme(next());
        else if (a == "--prior-correction") exp_cfg.prior_correction = true;
        else if (a == "--threads")          llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")         llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")          llama_cfg.verbose = true;
        else if (a == "-h" || a == "--help") { usage_experiment("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_experiment("pjev");
            return 2;
        }
    }
    SpdLogShutdown spdlog_shutdown;
    spdlog::stdout_logger_mt("console");
    if(llama_cfg.verbose){
        spdlog::set_level(spdlog::level::trace);
    }else{
        spdlog::set_level(spdlog::level::warn);
    }

    if (llama_cfg.model_path.empty()) {
        spdlog::error("--model is required");
        usage_experiment("pjev");
        return 1;
    }

    // Load calibration if provided
    if (!calibration_path.empty()) {
        CalibrationArtifact art;
        std::string calib_err;
        if (!load_calibration_artifact(calibration_path, art, calib_err)) {
            spdlog::error("calibration error: {}", calib_err.c_str());
            return 1;
        }
        exp_cfg.calibration = art.to_config();
    }

    // ---------------------------------------------------------------------------
    // Multilingual experiment
    // ---------------------------------------------------------------------------
    if (exp_name == "multilingual") {
        if (input_reference.empty() || input_target.empty()) {
            spdlog::error("--input-reference and --input-target are required for multilingual");
            return 1;
        }

        std::vector<DatasetRow> ref_rows, tgt_rows;
        std::string load_err;
        if (!load_dataset(input_reference, ref_rows, load_err)) {
            spdlog::error("reference dataset error: {}", load_err.c_str());
            return 1;
        }
        if (!load_dataset(input_target, tgt_rows, load_err)) {
            spdlog::error("target dataset error: {}", load_err.c_str());
            return 1;
        }

        // Validate pairs
        auto violations = validate_pairs(ref_rows, tgt_rows);
        for (const auto& v : violations)
            spdlog::warn("pair validation: {}", v.c_str());

        LlamaBackend* backend = nullptr;
        try {
            backend = new LlamaBackend(llama_cfg);
        } catch (const std::exception& e) {
            spdlog::error("model load error: {}", e.what());
            return 1;
        }

        ExperimentConfig ref_cfg = exp_cfg;
        ref_cfg.name = "multilingual-reference";
        ref_cfg.collect_corrected_logits = true;

        ExperimentConfig tgt_cfg = exp_cfg;
        tgt_cfg.name = "multilingual-target";
        tgt_cfg.collect_corrected_logits = true;

        RunResult ref_run = run_experiment(*backend, ref_rows, ref_cfg);
        RunResult tgt_run = run_experiment(*backend, tgt_rows, tgt_cfg);
        delete backend;

        MultilingualResult ml = compare_runs(ref_run, tgt_run,
                                              ref_language, target_language,
                                              llama_cfg.model_path);

        std::string output_str = ml.to_json().dump(2);
        if (output_dir.empty()) {
            printf("%s\n", output_str.c_str());
        } else {
            std::ofstream f(output_dir);
            if (!f) {
                spdlog::error("cannot open output: {}", output_dir.c_str());
                return 1;
            }
            f << output_str << "\n";
            printf("wrote: %s\n", output_dir.c_str());
        }
        return 0;
    }

    // ---------------------------------------------------------------------------
    // Standard single-language experiments
    // ---------------------------------------------------------------------------
    if (input_path.empty()) {
        spdlog::error("--input is required");
        usage_experiment("pjev");
        return 1;
    }

    const std::string valid_names[] = {
        "candidate-binding", "option-order", "prompt-layout", "prior-correction", "score-formulation"
    };
    bool valid = false;
    for (const auto& n : valid_names) { if (n == exp_name) { valid = true; break; } }
    if (!valid) {
        spdlog::error("unknown experiment: {}", exp_name.c_str());
        usage_experiment("pjev");
        return 2;
    }

    std::vector<DatasetRow> rows;
    std::string err;
    if (!load_dataset(input_path, rows, err)) {
        spdlog::error("dataset error: {}", err.c_str());
        return 1;
    }
    if (rows.empty()) {
        spdlog::warn("dataset is empty");
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        spdlog::error("model load error: {}", e.what());
        return 1;
    }

    CompareResult cr;
    if      (exp_name == "candidate-binding")  cr = exp_candidate_binding(*backend, rows, llama_cfg.model_path);
    else if (exp_name == "option-order")       cr = exp_option_order(*backend, rows, llama_cfg.model_path);
    else if (exp_name == "prompt-layout")      cr = exp_prompt_layout(*backend, rows, llama_cfg.model_path);
    else if (exp_name == "prior-correction")   cr = exp_prior_correction(*backend, rows, llama_cfg.model_path);
    else                                       cr = exp_score_formulation(*backend, rows, llama_cfg.model_path);

    delete backend;

    std::string output_str = cr.to_json().dump(2);
    if (output_dir.empty()) {
        printf("%s\n", output_str.c_str());
    } else {
        std::string out_path = output_dir + "/" + exp_name + ".json";
        std::ofstream f(out_path);
        if (!f) {
            spdlog::error("cannot open output: {}", out_path.c_str());
            return 1;
        }
        f << output_str << "\n";
        printf("wrote: %s\n", out_path.c_str());
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Helper: build CalibrationSamples from ItemResults
// ---------------------------------------------------------------------------

static std::vector<CalibrationSample>
collect_samples(const std::vector<ItemResult>& items, const std::string& type)
{
    std::vector<CalibrationSample> out;
    for (const auto& ir : items) {
        if (ir.type != type) continue;
        if (!ir.ok || ir.correct_index < 0 || ir.corrected_logits.empty()) continue;
        CalibrationSample s;
        s.logits        = ir.corrected_logits;
        s.correct_index = ir.correct_index;
        s.type          = ir.type;
        s.expected_score = ir.expected_score;
        out.push_back(s);
    }
    return out;
}

// ---------------------------------------------------------------------------
// calibration subcommand
// ---------------------------------------------------------------------------

static int cmd_calibration(int argc, char** argv) {
    if (argc < 1 || std::string(argv[0]) == "-h" || std::string(argv[0]) == "--help") {
        usage_calibration("pjev");
        return (argc < 1) ? 2 : 0;
    }
    std::string subcmd = argv[0];
    argc--; argv++;

    LlamaConfig llama_cfg;
    ExperimentConfig exp_cfg;
    exp_cfg.collect_corrected_logits = true;
    std::string input_path;
    std::string output_path;
    std::string calibration_path;
    std::string model_id;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_calibration("pjev"); exit(2); }
            return argv[++i];
        };
        if      (a == "--model")           llama_cfg.model_path = next();
        else if (a == "--input")           input_path = next();
        else if (a == "--output")          output_path = next();
        else if (a == "--calibration")     calibration_path = next();
        else if (a == "--model-id")        model_id = next();
        else if (a == "--threads")         llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")        llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")         llama_cfg.verbose = true;
        else if (a == "--layout")          exp_cfg.layout = parse_layout(next());
        else if (a == "--scheme")          exp_cfg.scheme = parse_scheme(next());
        else if (a == "--prior-correction") exp_cfg.prior_correction = true;
        else if (a == "-h" || a == "--help") { usage_calibration("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_calibration("pjev");
            return 2;
        }
    }

    SpdLogShutdown spdlog_shutdown;
    spdlog::stdout_logger_mt("console");
    if (llama_cfg.verbose) spdlog::set_level(spdlog::level::trace);
    else                   spdlog::set_level(spdlog::level::warn);

    if (subcmd != "fit" && subcmd != "evaluate") {
        spdlog::error("unknown calibration subcommand: {}", subcmd.c_str());
        usage_calibration("pjev");
        return 2;
    }
    if (llama_cfg.model_path.empty()) {
        spdlog::error("--model is required");
        return 1;
    }
    if (input_path.empty()) {
        spdlog::error("--input is required");
        return 1;
    }

    // For evaluate, also load calibration artifact
    CalibrationArtifact eval_art;
    if (subcmd == "evaluate") {
        if (calibration_path.empty()) {
            spdlog::error("--calibration is required for evaluate");
            return 1;
        }
        std::string calib_err;
        if (!load_calibration_artifact(calibration_path, eval_art, calib_err)) {
            spdlog::error("calibration error: {}", calib_err.c_str());
            return 1;
        }
    }

    std::vector<DatasetRow> rows;
    std::string load_err;
    if (!load_dataset(input_path, rows, load_err)) {
        spdlog::error("dataset error: {}", load_err.c_str());
        return 1;
    }
    if (rows.empty()) {
        spdlog::warn("dataset is empty");
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        spdlog::error("model load error: {}", e.what());
        return 1;
    }

    // Run with calibration disabled and corrected logits collection enabled
    exp_cfg.name = (subcmd == "fit") ? "calibration-fit-collect" : "calibration-eval-collect";
    RunResult collected = run_experiment(*backend, rows, exp_cfg);
    delete backend;

    auto noul_s   = collect_samples(collected.items, "noul");
    auto choice_s = collect_samples(collected.items, "choice");
    auto score_s  = collect_samples(collected.items, "score");

    if (subcmd == "fit") {
        if (output_path.empty()) {
            spdlog::error("--output is required for fit");
            return 1;
        }
        FitResult fr_noul   = fit_temperature(noul_s);
        FitResult fr_choice = fit_temperature(choice_s);
        FitResult fr_score  = fit_temperature(score_s);

        CalibrationArtifact art;
        art.model_identifier  = model_id.empty() ? llama_cfg.model_path : model_id;
        art.formulation.layout           = exp_cfg.layout_str();
        art.formulation.candidate_scheme = exp_cfg.scheme_str();
        art.formulation.prior_correction = exp_cfg.prior_correction;
        art.noul_temperature   = fr_noul.converged   ? fr_noul.temperature   : 1.0;
        art.choice_temperature = fr_choice.converged ? fr_choice.temperature : 1.0;
        art.score_temperature  = fr_score.converged  ? fr_score.temperature  : 1.0;

        nlohmann::json result = {
            {"artifact", art.to_json()},
            {"fit", {
                {"noul",   {{"n", fr_noul.n_samples},   {"temperature", art.noul_temperature},   {"nll", fr_noul.nll},   {"converged", fr_noul.converged}}},
                {"choice", {{"n", fr_choice.n_samples}, {"temperature", art.choice_temperature}, {"nll", fr_choice.nll}, {"converged", fr_choice.converged}}},
                {"score",  {{"n", fr_score.n_samples},  {"temperature", art.score_temperature},  {"nll", fr_score.nll},  {"converged", fr_score.converged}}}
            }}
        };

        // Write artifact to output
        std::ofstream f(output_path);
        if (!f) {
            spdlog::error("cannot open output: {}", output_path.c_str());
            return 1;
        }
        f << art.to_json().dump(2) << "\n";
        printf("%s\n", result.dump(2).c_str());
        return 0;

    } else { // evaluate
        double T_noul   = eval_art.noul_temperature;
        double T_choice = eval_art.choice_temperature;
        double T_score  = eval_art.score_temperature;

        CalibrationReport report = build_calibration_report(
            noul_s, choice_s, score_s, T_noul, T_choice, T_score);

        nlohmann::json result = {
            {"calibration", eval_art.to_json()},
            {"results", report.to_json()}
        };

        std::string output_str = result.dump(2);
        if (output_path.empty()) {
            printf("%s\n", output_str.c_str());
        } else {
            std::ofstream f(output_path);
            if (!f) {
                spdlog::error("cannot open output: {}", output_path.c_str());
                return 1;
            }
            f << output_str << "\n";
        }
        return 0;
    }
}

// ---------------------------------------------------------------------------
// model subcommand
// ---------------------------------------------------------------------------

// Standard candidate tokens for the natural scheme (digits for score, letters for choice)
static const std::vector<std::string> g_default_candidates = {
    "Yes", "No", "A", "B", "C", "D", "E", "1", "2", "3", "4", "5"
};

static int cmd_model_inspect(int argc, char** argv) {
    LlamaConfig llama_cfg;
    std::string output_path;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_model("pjev"); exit(2); }
            return argv[++i];
        };
        if      (a == "--model")           llama_cfg.model_path = next();
        else if (a == "--threads")         llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")        llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")         llama_cfg.verbose = true;
        else if (a == "--output")          output_path = next();
        else if (a == "-h" || a == "--help") { usage_model("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_model("pjev");
            return 2;
        }
    }

    SpdLogShutdown spdlog_shutdown;
    spdlog::stdout_logger_mt("console");
    spdlog::set_level(llama_cfg.verbose ? spdlog::level::trace : spdlog::level::warn);

    if (llama_cfg.model_path.empty()) {
        spdlog::error("--model is required");
        return 1;
    }

    // File-level identity (no backend yet)
    ModelIdentity id = ModelIdentity::from_path(llama_cfg.model_path);
    spdlog::info("sha256: {}", id.sha256.c_str());

    // Load model (timed)
    int64_t rss_before = get_rss_bytes();
    int64_t load_ms    = 0;
    LlamaBackend* backend = nullptr;
    {
        Timer t; t.start();
        try {
            backend = new LlamaBackend(llama_cfg);
        } catch (const std::exception& e) {
            spdlog::error("model load error: {}", e.what());
            return 1;
        }
        t.stop();
        load_ms = t.elapsed().count();
    }
    int64_t rss_after = get_rss_bytes();
    spdlog::info("model load: {} ms", load_ms);

    // Fill backend-derived identity
    id = ModelIdentity::from_path(llama_cfg.model_path, backend);
    id.pjev_commit      = PJEV_GIT_COMMIT;
    id.llamacpp_revision = PJEV_LLAMA_REVISION;

    // Candidate compatibility
    CandidateCompatibility compat = probe_candidates(*backend, g_default_candidates);

    // Smoke test
    nlohmann::json smoke = nlohmann::json::object();
    {
        auto run_smoke = [&](const std::string& type,
                             const std::vector<std::pair<std::string,std::string>>& opts) {
            try {
                DecisionEngine eng(*backend);
                DecisionInput inp;
                inp.type    = type;
                inp.state   = "Test";
                inp.question = "Test question";
                inp.options = opts;
                DecisionOutput out = eng.decide(inp);
                smoke[type] = out.ok ? "pass" : ("fail: " + out.error);
            } catch (const std::exception& e) {
                smoke[type] = std::string("fail: ") + e.what();
            }
        };
        run_smoke("noul",   {{"true","Yes"},{"false","No"}});
        run_smoke("choice", {{"A","Option A"},{"B","Option B"},{"C","Option C"},{"D","Option D"}});
        run_smoke("score",  {{"1","1"},{"2","2"},{"3","3"},{"4","4"},{"5","5"}});
    }

    delete backend;

    nlohmann::json report = {
        {"model",                  id.to_json()},
        {"candidate_compatibility",compat.to_json()},
        {"performance", {
            {"model_load_ms",       load_ms},
            {"rss_before_load",     rss_before},
            {"rss_post_load_bytes", rss_after}
        }},
        {"smoke_test", smoke},
        {"platform", {
            {"os",       get_os_name()},
            {"n_threads", llama_cfg.n_threads}
        }}
    };

    std::string out_str = report.dump(2);
    if (output_path.empty()) {
        printf("%s\n", out_str.c_str());
    } else {
        std::ofstream f(output_path);
        if (!f) { spdlog::error("cannot open output: {}", output_path.c_str()); return 1; }
        f << out_str << "\n";
        fprintf(stderr, "wrote: %s\n", output_path.c_str());
    }
    return 0;
}

static int cmd_model_evaluate(int argc, char** argv) {
    LlamaConfig      llama_cfg;
    ExperimentConfig exp_cfg;
    exp_cfg.collect_corrected_logits = true;
    std::string input_en;
    std::string input_ja;
    std::string calibration_path;
    bool        fit_calibration = false;
    std::string output_path;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_model("pjev"); exit(2); }
            return argv[++i];
        };
        if      (a == "--model")            llama_cfg.model_path = next();
        else if (a == "--input")            input_en = next();
        else if (a == "--input-ja")         input_ja = next();
        else if (a == "--calibration")      calibration_path = next();
        else if (a == "--fit-calibration")  fit_calibration = true;
        else if (a == "--output")           output_path = next();
        else if (a == "--layout")           exp_cfg.layout = parse_layout(next());
        else if (a == "--scheme")           exp_cfg.scheme = parse_scheme(next());
        else if (a == "--prior-correction") exp_cfg.prior_correction = true;
        else if (a == "--threads")          llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")         llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")          llama_cfg.verbose = true;
        else if (a == "-h" || a == "--help") { usage_model("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_model("pjev");
            return 2;
        }
    }

    SpdLogShutdown spdlog_shutdown;
    spdlog::stdout_logger_mt("console");
    spdlog::set_level(llama_cfg.verbose ? spdlog::level::trace : spdlog::level::warn);

    if (llama_cfg.model_path.empty()) { spdlog::error("--model is required"); return 1; }
    if (input_en.empty())             { spdlog::error("--input is required"); return 1; }
    if (output_path.empty())          { spdlog::error("--output is required"); return 1; }

    // Load EN dataset
    std::vector<DatasetRow> rows_en;
    {
        std::string err;
        if (!load_dataset(input_en, rows_en, err)) {
            spdlog::error("dataset error: {}", err.c_str()); return 1;
        }
    }

    // Load JA dataset (optional)
    std::vector<DatasetRow> rows_ja;
    if (!input_ja.empty()) {
        std::string err;
        if (!load_dataset(input_ja, rows_ja, err)) {
            spdlog::error("ja dataset error: {}", err.c_str()); return 1;
        }
    }

    // Load calibration artifact if provided
    CalibrationArtifact provided_art;
    bool has_provided_art = false;
    if (!calibration_path.empty()) {
        std::string calib_err;
        if (!load_calibration_artifact(calibration_path, provided_art, calib_err)) {
            spdlog::error("calibration error: {}", calib_err.c_str()); return 1;
        }
        has_provided_art = true;
    }

    // Load model (timed)
    int64_t load_ms = 0;
    int64_t rss_after = -1;
    LlamaBackend* backend = nullptr;
    {
        Timer t; t.start();
        try { backend = new LlamaBackend(llama_cfg); }
        catch (const std::exception& e) {
            spdlog::error("model load error: {}", e.what()); return 1;
        }
        t.stop();
        load_ms  = t.elapsed().count();
        rss_after = get_rss_bytes();
    }
    spdlog::info("model load: {} ms", load_ms);

    // Build model identity
    ModelIdentity id = ModelIdentity::from_path(llama_cfg.model_path, backend);
    id.pjev_commit       = PJEV_GIT_COMMIT;
    id.llamacpp_revision = PJEV_LLAMA_REVISION;

    // Candidate compatibility
    CandidateCompatibility compat = probe_candidates(*backend, g_default_candidates);

    // Run EN benchmark (collect corrected logits for calibration)
    exp_cfg.name = "model-evaluate-en";
    RunResult en_run = run_experiment(*backend, rows_en, exp_cfg);
    spdlog::info("EN run: {}/{} ok", en_run.metrics.n_total - en_run.metrics.n_errors, en_run.metrics.n_total);

    // Collect EN calibration samples
    auto en_noul_s   = collect_samples(en_run.items, "noul");
    auto en_choice_s = collect_samples(en_run.items, "choice");
    auto en_score_s  = collect_samples(en_run.items, "score");

    // Determine temperatures
    double T_noul = 1.0, T_choice = 1.0, T_score = 1.0;
    CalibrationArtifact fitted_art;
    fitted_art.model_identifier  = id.gguf_path;
    fitted_art.formulation.layout           = exp_cfg.layout_str();
    fitted_art.formulation.candidate_scheme = exp_cfg.scheme_str();
    fitted_art.formulation.prior_correction = exp_cfg.prior_correction;

    if (has_provided_art) {
        T_noul   = provided_art.noul_temperature;
        T_choice = provided_art.choice_temperature;
        T_score  = provided_art.score_temperature;
        fitted_art = provided_art;
    } else if (fit_calibration) {
        FitResult fr_noul   = fit_temperature(en_noul_s);
        FitResult fr_choice = fit_temperature(en_choice_s);
        FitResult fr_score  = fit_temperature(en_score_s);
        T_noul   = fr_noul.converged   ? fr_noul.temperature   : 1.0;
        T_choice = fr_choice.converged ? fr_choice.temperature : 1.0;
        T_score  = fr_score.converged  ? fr_score.temperature  : 1.0;
        fitted_art.noul_temperature   = T_noul;
        fitted_art.choice_temperature = T_choice;
        fitted_art.score_temperature  = T_score;
        spdlog::info("fitted EN calibration: noul={:.4f} choice={:.4f} score={:.4f}",
                     T_noul, T_choice, T_score);
    }

    CalibrationReport en_cal = build_calibration_report(
        en_noul_s, en_choice_s, en_score_s, T_noul, T_choice, T_score);

    // Run JA benchmark (optional)
    RunResult ja_run;
    CalibrationReport ja_cal;
    MultilingualResult ml_result;
    bool has_ja = !rows_ja.empty();

    if (has_ja) {
        ExperimentConfig ja_cfg = exp_cfg;
        ja_cfg.name = "model-evaluate-ja";
        ja_run = run_experiment(*backend, rows_ja, ja_cfg);
        spdlog::info("JA run: {}/{} ok", ja_run.metrics.n_total - ja_run.metrics.n_errors, ja_run.metrics.n_total);

        auto ja_noul_s   = collect_samples(ja_run.items, "noul");
        auto ja_choice_s = collect_samples(ja_run.items, "choice");
        auto ja_score_s  = collect_samples(ja_run.items, "score");

        double jT_noul = T_noul, jT_choice = T_choice, jT_score = T_score;
        if (fit_calibration) {
            FitResult fjn = fit_temperature(ja_noul_s);
            FitResult fjc = fit_temperature(ja_choice_s);
            FitResult fjs = fit_temperature(ja_score_s);
            jT_noul   = fjn.converged   ? fjn.temperature   : 1.0;
            jT_choice = fjc.converged   ? fjc.temperature   : 1.0;
            jT_score  = fjs.converged   ? fjs.temperature   : 1.0;
        }
        ja_cal = build_calibration_report(ja_noul_s, ja_choice_s, ja_score_s,
                                           jT_noul, jT_choice, jT_score);
        ml_result = compare_runs(en_run, ja_run, "en", "ja", llama_cfg.model_path);
    }

    delete backend;

    // Collect warm latency samples from EN run
    std::vector<int64_t> warm_ms;
    double sum_tokens = 0.0;
    for (const auto& ir : en_run.items) {
        if (ir.ok && ir.eval_ms > 0) { warm_ms.push_back(ir.eval_ms); sum_tokens += ir.prompt_token_count; }
    }
    PerformanceStats perf = compute_perf_stats(load_ms, rss_after, warm_ms,
                                                get_os_name(), "", llama_cfg.n_threads);
    if (!warm_ms.empty()) perf.mean_prompt_tokens = sum_tokens / warm_ms.size();

    // Build ValueProfile
    ValueProfile vp;
    vp.version             = 1;
    vp.pjev_commit         = PJEV_GIT_COMMIT;
    vp.llamacpp_revision   = PJEV_LLAMA_REVISION;
    vp.model               = id;
    vp.frozen_config       = exp_cfg;
    vp.quality_en          = compute_quality_stats(en_run);
    vp.calibration_en      = en_cal;
    vp.candidate_compat    = compat;
    vp.performance         = perf;
    if (has_ja) {
        vp.quality_ja      = compute_quality_stats(ja_run);
        vp.calibration_ja  = ja_cal;
        vp.multilingual    = ml_result;
    }
    // Embed calibration artifact in compatibility_issues for traceability
    if (has_provided_art || fit_calibration) {
        vp.compatibility_issues.push_back(
            "calibration: " + fitted_art.to_json().dump());
    }

    // Write output
    std::ofstream f(output_path);
    if (!f) { spdlog::error("cannot open output: {}", output_path.c_str()); return 1; }
    f << vp.to_json().dump(2) << "\n";
    fprintf(stderr, "wrote: %s\n", output_path.c_str());

    // Print summary
    fprintf(stdout, "EN choice accuracy: %.4f  noul: %.4f  score MAE: %.4f\n",
            vp.quality_en.choice.accuracy,
            vp.quality_en.noul.accuracy,
            vp.quality_en.score.mae);
    fprintf(stdout, "EN calibration NLL: %.4f  Brier: %.4f  ECE: %.4f\n",
            en_cal.choice_calibrated.nll,
            en_cal.choice_calibrated.brier,
            en_cal.choice_calibrated.ece);
    fprintf(stdout, "load: %lld ms  warm p50: %.1f ms  RSS: %lld bytes\n",
            (long long)load_ms, perf.warm_p50_ms, (long long)rss_after);

    return 0;
}

static int cmd_model(int argc, char** argv) {
    if (argc < 1 || std::string(argv[0]) == "-h" || std::string(argv[0]) == "--help") {
        usage_model("pjev");
        return (argc < 1) ? 2 : 0;
    }
    std::string subcmd = argv[0];
    argc--; argv++;

    if (subcmd == "inspect")  return cmd_model_inspect(argc, argv);
    if (subcmd == "evaluate") return cmd_model_evaluate(argc, argv);

    fprintf(stderr, "unknown model subcommand: %s\n", subcmd.c_str());
    usage_model("pjev");
    return 2;
}

// ---------------------------------------------------------------------------
// benchmark subcommand
// ---------------------------------------------------------------------------

static int cmd_benchmark_latency(int argc, char** argv) {
    LlamaConfig  llama_cfg;
    ServerConfig srv_cfg;
    int iterations   = 20;
    int warmup       = 3;
    int n_questions  = 1;
    std::string state_text = "This is a test state for benchmarking.";
    std::string output_path;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_benchmark("pjev"); exit(2); }
            return argv[++i];
        };
        if      (a == "--model")           llama_cfg.model_path = next();
        else if (a == "--iterations")      iterations = std::stoi(next());
        else if (a == "--warmup")          warmup = std::stoi(next());
        else if (a == "--questions")       n_questions = std::stoi(next());
        else if (a == "--state")           state_text = next();
        else if (a == "--output")          output_path = next();
        else if (a == "--threads")         llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")        llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--no-kv-reuse")     srv_cfg.kv_reuse = false;
        else if (a == "--no-batch-questions") srv_cfg.batch_questions = false;
        else if (a == "-h" || a == "--help") { usage_benchmark("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_benchmark("pjev");
            return 2;
        }
    }

    if (llama_cfg.model_path.empty()) {
        fprintf(stderr, "--model is required\n");
        return 1;
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "model load error: %s\n", e.what());
        return 1;
    }

    PromptConfig prompt_cfg;
    CalibrationConfig calib_cfg;
    DecisionEngine* engine = nullptr;
    try {
        engine = new DecisionEngine(*backend, prompt_cfg, calib_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "engine init error: %s\n", e.what());
        delete backend;
        return 1;
    }

    // Build a batch of n_questions noul inputs with the same state
    auto make_inputs = [&]() {
        std::vector<DecisionInput> inputs;
        for (int q = 0; q < n_questions; q++) {
            DecisionInput inp;
            inp.type     = "noul";
            inp.state    = state_text;
            inp.question = "Is this a benchmark question " + std::to_string(q) + "?";
            inp.options  = {{"false", ""}, {"true", ""}};
            inputs.push_back(inp);
        }
        return inputs;
    };

    BatchConfig bcfg;
    bcfg.use_kv_reuse = srv_cfg.kv_reuse;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        auto inputs = make_inputs();
        if (srv_cfg.batch_questions && n_questions > 1)
            engine->decide_batch(inputs, bcfg);
        else
            for (const auto& inp : inputs) engine->decide(inp);
    }

    // Measure
    std::vector<int64_t> latencies_us;
    latencies_us.reserve(iterations);
    for (int i = 0; i < iterations; i++) {
        auto inputs = make_inputs();
        auto t0 = std::chrono::steady_clock::now();
        if (srv_cfg.batch_questions && n_questions > 1)
            engine->decide_batch(inputs, bcfg);
        else
            for (const auto& inp : inputs) engine->decide(inp);
        auto t1 = std::chrono::steady_clock::now();
        latencies_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    }

    delete engine;
    delete backend;

    std::sort(latencies_us.begin(), latencies_us.end());
    auto percentile = [&](double p) -> double {
        if (latencies_us.empty()) return 0.0;
        size_t idx = (size_t)(p / 100.0 * (latencies_us.size() - 1));
        return latencies_us[idx] / 1000.0;
    };

    nlohmann::json result = {
        {"config", {
            {"model",         llama_cfg.model_path},
            {"iterations",    iterations},
            {"warmup",        warmup},
            {"questions",     n_questions},
            {"threads",       llama_cfg.n_threads},
            {"ctx_size",      llama_cfg.n_ctx},
            {"kv_reuse",      srv_cfg.kv_reuse},
            {"batch_questions", srv_cfg.batch_questions}
        }},
        {"latency_ms", {
            {"p50",  percentile(50)},
            {"p90",  percentile(90)},
            {"p95",  percentile(95)},
            {"mean", [&]() {
                double s = 0;
                for (auto v : latencies_us) s += v;
                return (latencies_us.empty() ? 0.0 : s / latencies_us.size()) / 1000.0;
            }()}
        }}
    };

    std::string out_str = result.dump(2);
    if (output_path.empty()) {
        printf("%s\n", out_str.c_str());
    } else {
        std::ofstream f(output_path);
        if (!f) { fprintf(stderr, "cannot open output: %s\n", output_path.c_str()); return 1; }
        f << out_str << "\n";
    }
    return 0;
}

static int cmd_benchmark(int argc, char** argv) {
    if (argc < 1 || std::string(argv[0]) == "-h" || std::string(argv[0]) == "--help") {
        usage_benchmark("pjev");
        return (argc < 1) ? 2 : 0;
    }
    std::string subcmd = argv[0];
    if (subcmd == "latency") return cmd_benchmark_latency(argc - 1, argv + 1);
    fprintf(stderr, "unknown benchmark subcommand: %s\n", subcmd.c_str());
    usage_benchmark("pjev");
    return 2;
}

// ---------------------------------------------------------------------------
// diagnostics subcommand
// ---------------------------------------------------------------------------

static int cmd_diagnostics(int argc, char** argv) {
    LlamaConfig  llama_cfg;
    ServerConfig srv_cfg;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_diagnostics("pjev"); exit(2); }
            return argv[++i];
        };
        if      (a == "--model")           llama_cfg.model_path = next();
        else if (a == "--threads")         llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")        llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--no-kv-reuse")     srv_cfg.kv_reuse = false;
        else if (a == "--no-batch-questions") srv_cfg.batch_questions = false;
        else if (a == "-h" || a == "--help") { usage_diagnostics("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_diagnostics("pjev");
            return 2;
        }
    }

    nlohmann::json cfg = {
        {"model",           llama_cfg.model_path},
        {"n_threads",       llama_cfg.n_threads},
        {"n_threads_batch", llama_cfg.n_threads_batch},
        {"n_ctx",           llama_cfg.n_ctx},
        {"n_batch",         llama_cfg.n_batch},
        {"kv_reuse",        srv_cfg.kv_reuse},
        {"batch_questions", srv_cfg.batch_questions},
        {"max_queued",      srv_cfg.max_queued_requests}
    };
    printf("%s\n", cfg.dump(2).c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    std::string cmd = argv[1];

    // If first arg looks like an option (legacy / direct server invocation), treat as "server"
    if (cmd == "server" || cmd == "serve") {
        return cmd_server(argc - 2, argv + 2);
    } else if (cmd == "--version" || cmd == "-V") {
        printf("pjev %s (commit: %s, llama.cpp: %s)\n",
               PJEV_VERSION, PJEV_GIT_COMMIT, PJEV_LLAMA_REVISION);
        return 0;
    } else if (cmd == "run") {
        return cmd_run(argc - 2, argv + 2);
    } else if (cmd == "experiment") {
        return cmd_experiment(argc - 2, argv + 2);
    } else if (cmd == "calibration") {
        return cmd_calibration(argc - 2, argv + 2);
    } else if (cmd == "model") {
        return cmd_model(argc - 2, argv + 2);
    } else if (cmd == "benchmark") {
        return cmd_benchmark(argc - 2, argv + 2);
    } else if (cmd == "diagnostics") {
        return cmd_diagnostics(argc - 2, argv + 2);
    } else if (cmd == "-h" || cmd == "--help") {
        usage(argv[0]);
        return 0;
    } else if (cmd[0] == '-') {
        // Legacy: no subcommand, treat all args as server args
        return cmd_server(argc - 1, argv + 1);
    } else {
        fprintf(stderr, "unknown command: %s\n", cmd.c_str());
        usage(argv[0]);
        return 2;
    }
}
