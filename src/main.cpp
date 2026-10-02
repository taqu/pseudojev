#include "inference/llama_backend.h"
#include "decision/decision_engine.h"
#include "server/jev_api.h"
#include "server/http_server.h"
#include "experiment/config.h"
#include "experiment/dataset.h"
#include "experiment/runner.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_sinks.h"

// ---------------------------------------------------------------------------
// Usage helpers
// ---------------------------------------------------------------------------
using namespace pjev;
static void usage_server(const char* prog) {
    fprintf(stderr,
        "usage: %s server --model PATH [options]\n"
        "  --model PATH       path to .gguf model file (default: models/Bonsai-1.7B.gguf)\n"
        "  --port N           HTTP port (default: 8080)\n"
        "  --host ADDR        bind address (default: 127.0.0.1)\n"
        "  --layout LAYOUT    auto|state-first|state-last|question-first (default: auto)\n"
        "  --scheme SCHEME    natural|letters (default: natural)\n"
        "  --threads N        CPU threads (default: 8)\n"
        "  --ctx-size N       context size (default: 4096)\n"
        "  --verbose          enable llama.cpp verbose logging\n"
        "  --model-name NAME  model name in responses (default: pseudojev)\n",
        prog);
}

static void usage_run(const char* prog) {
    fprintf(stderr,
        "usage: %s run --model PATH --input FILE [options]\n"
        "  --model PATH               path to .gguf model file (required)\n"
        "  --input FILE               JSONL/JSON dataset file (required, or - for stdin)\n"
        "  --output FILE              write JSON result here (default: stdout)\n"
        "  --layout LAYOUT            auto|state-first|state-last|question-first (default: auto)\n"
        "  --scheme SCHEME            natural|letters (default: natural)\n"
        "  --prior-correction         enable prior correction\n"
        "  --option-order ORDER       original|reversed|random (default: original)\n"
        "  --threads N                CPU threads (default: 8)\n"
        "  --ctx-size N               context size (default: 4096)\n"
        "  --verbose                  enable llama.cpp verbose logging\n",
        prog);
}

static void usage_experiment(const char* prog) {
    fprintf(stderr,
        "usage: %s experiment NAME --model PATH --input FILE [options]\n"
        "  NAME: candidate-binding | option-order | prompt-layout | prior-correction | score-formulation\n"
        "  --model PATH       path to .gguf model file (required)\n"
        "  --input FILE       JSONL/JSON dataset file (required, or - for stdin)\n"
        "  --output DIR       directory to write per-run JSON results (default: stdout)\n"
        "  --threads N        CPU threads (default: 8)\n"
        "  --ctx-size N       context size (default: 4096)\n"
        "  --verbose          enable llama.cpp verbose logging\n",
        prog);
}

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s <command> [options]\n"
        "commands:\n"
        "  server      start the HTTP server (default if no command given)\n"
        "  run         batch evaluation with a single config\n"
        "  experiment  run a named experiment\n"
        "run '%s server --help', '%s run --help', or '%s experiment --help' for details\n",
        prog, prog, prog, prog);
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

// ---------------------------------------------------------------------------
// server subcommand
// ---------------------------------------------------------------------------

static int cmd_server(int argc, char** argv) {
    LlamaConfig  llama_cfg;
    PromptConfig prompt_cfg;
    llama_cfg.model_path = "models/Bonsai-1.7B.gguf";
    std::string  host       = "127.0.0.1";
    uint16_t     port       = 8080;
    std::string  model_name = "pseudojev";

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_server(argv[-1]); exit(2); }
            return argv[++i];
        };

        if      (a == "--model")      llama_cfg.model_path = next();
        else if (a == "--port")       port = (uint16_t)std::stoi(next());
        else if (a == "--host")       host = next();
        else if (a == "--threads")    llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")   llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")    llama_cfg.verbose = true;
        else if (a == "--model-name") model_name = next();
        else if (a == "--layout")     prompt_cfg.layout = parse_layout(next());
        else if (a == "--scheme")     prompt_cfg.scheme = parse_scheme(next());
        else if (a == "-h" || a == "--help") { usage_server("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_server("pjev");
            return 2;
        }
    }

    spdlog::stdout_color_mt("console");
    if(llama_cfg.verbose){
        spdlog::set_level(spdlog::level::trace);
    }else{
        spdlog::set_level(spdlog::level::warn);
    }

    if (llama_cfg.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        usage_server("pjev");
        return 1;
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "model load error: %s\n", e.what());
        return 1;
    }

    DecisionEngine* engine = nullptr;
    try {
        engine = new DecisionEngine(*backend, prompt_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "engine init error: %s\n", e.what());
        delete backend;
        return 1;
    }

    JevApiHandler handler(*engine, model_name);

    bool ok = run_http_server(host, port, "/api/v1/systemone/",
        [&handler](const std::string& body, int& status, std::string& resp) {
            handler.handle(body, status, resp);
        });

    delete engine;
    delete backend;
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// run subcommand
// ---------------------------------------------------------------------------

static int cmd_run(int argc, char** argv) {
    LlamaConfig    llama_cfg;
    ExperimentConfig exp_cfg;
    exp_cfg.name = "run";
    std::string input_path;
    std::string output_path;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_run("pjev"); exit(2); }
            return argv[++i];
        };

        if      (a == "--model")           llama_cfg.model_path = next();
        else if (a == "--input")           input_path = next();
        else if (a == "--output")          output_path = next();
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

    if (llama_cfg.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        usage_run("pjev");
        return 1;
    }
    if (input_path.empty()) {
        fprintf(stderr, "error: --input is required\n");
        usage_run("pjev");
        return 1;
    }

    std::vector<DatasetRow> rows;
    std::string err;
    if (!load_dataset(input_path, rows, err)) {
        fprintf(stderr, "dataset error: %s\n", err.c_str());
        return 1;
    }
    if (rows.empty()) {
        fprintf(stderr, "warning: dataset is empty\n");
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "model load error: %s\n", e.what());
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
            fprintf(stderr, "cannot open output: %s\n", output_path.c_str());
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
    LlamaConfig llama_cfg;
    std::string input_path;
    std::string output_dir;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_experiment("pjev"); exit(2); }
            return argv[++i];
        };

        if      (a == "--model")   llama_cfg.model_path = next();
        else if (a == "--input")   input_path = next();
        else if (a == "--output")  output_dir = next();
        else if (a == "--threads") llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size") llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose") llama_cfg.verbose = true;
        else if (a == "-h" || a == "--help") { usage_experiment("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_experiment("pjev");
            return 2;
        }
    }

    if (llama_cfg.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        usage_experiment("pjev");
        return 1;
    }
    if (input_path.empty()) {
        fprintf(stderr, "error: --input is required\n");
        usage_experiment("pjev");
        return 1;
    }

    const std::string valid_names[] = {
        "candidate-binding", "option-order", "prompt-layout", "prior-correction", "score-formulation"
    };
    bool valid = false;
    for (const auto& n : valid_names) { if (n == exp_name) { valid = true; break; } }
    if (!valid) {
        fprintf(stderr, "unknown experiment: %s\n", exp_name.c_str());
        usage_experiment("pjev");
        return 2;
    }

    std::vector<DatasetRow> rows;
    std::string err;
    if (!load_dataset(input_path, rows, err)) {
        fprintf(stderr, "dataset error: %s\n", err.c_str());
        return 1;
    }
    if (rows.empty()) {
        fprintf(stderr, "warning: dataset is empty\n");
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "model load error: %s\n", e.what());
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
            fprintf(stderr, "cannot open output: %s\n", out_path.c_str());
            return 1;
        }
        f << output_str << "\n";
        printf("wrote: %s\n", out_path.c_str());
    }
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
    if (cmd == "server") {
        return cmd_server(argc - 2, argv + 2);
    } else if (cmd == "run") {
        return cmd_run(argc - 2, argv + 2);
    } else if (cmd == "experiment") {
        return cmd_experiment(argc - 2, argv + 2);
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
