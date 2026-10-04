#include "calibration/calibration.h"
#include "calibration/calibration_fit.h"
#include "calibration/calibration_metrics.h"
#include "multilingual/multilingual.h"
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
        "  --model PATH         path to .gguf model file (default: models/Bonsai-1.7B.gguf)\n"
        "  --port N             HTTP port (default: 8080)\n"
        "  --host ADDR          bind address (default: 127.0.0.1)\n"
        "  --calibration FILE   load calibration artifact (optional)\n"
        "  --layout LAYOUT      auto|state-first|state-last|question-first (default: auto)\n"
        "  --scheme SCHEME      natural|letters (default: natural)\n"
        "  --threads N          CPU threads (default: 8)\n"
        "  --ctx-size N         context size (default: 4096)\n"
        "  --verbose            enable llama.cpp verbose logging\n"
        "  --model-name NAME    model name in responses (default: pseudojev)\n",
        prog);
}

static void usage_run(const char* prog) {
    fprintf(stderr,
        "usage: %s run --model PATH --input FILE [options]\n"
        "  --model PATH               path to .gguf model file (required)\n"
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

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s <command> [options]\n"
        "commands:\n"
        "  server       start the HTTP server (default if no command given)\n"
        "  run          batch evaluation with a single config\n"
        "  experiment   run a named experiment\n"
        "  calibration  fit or evaluate calibration temperatures\n"
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

static int cmd_server(int argc, char** argv) {
    LlamaConfig  llama_cfg;
    PromptConfig prompt_cfg;
    llama_cfg.model_path = "models/Bonsai-1.7B.gguf";
    std::string  host            = "127.0.0.1";
    uint16_t     port            = 8080;
    std::string  model_name      = "pseudojev";
    std::string  calibration_path;

    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage_server(argv[-1]); exit(2); }
            return argv[++i];
        };

        if      (a == "--model")        llama_cfg.model_path = next();
        else if (a == "--port")         port = (uint16_t)std::stoi(next());
        else if (a == "--host")         host = next();
        else if (a == "--threads")      llama_cfg.n_threads = std::stoi(next());
        else if (a == "--ctx-size")     llama_cfg.n_ctx = llama_cfg.n_batch = std::stoi(next());
        else if (a == "--verbose")      llama_cfg.verbose = true;
        else if (a == "--model-name")   model_name = next();
        else if (a == "--layout")       prompt_cfg.layout = parse_layout(next());
        else if (a == "--scheme")       prompt_cfg.scheme = parse_scheme(next());
        else if (a == "--calibration")  calibration_path = next();
        else if (a == "-h" || a == "--help") { usage_server("pjev"); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage_server("pjev");
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
        usage_server("pjev");
        return 1;
    }

    LlamaBackend* backend = nullptr;
    try {
        backend = new LlamaBackend(llama_cfg);
    } catch (const std::exception& e) {
        spdlog::error("model load error: {}", e.what());
        return 1;
    }

    CalibrationConfig calib_cfg;
    if (!calibration_path.empty()) {
        std::ifstream cf(calibration_path);
        if (!cf) {
            spdlog::error("cannot open calibration file: {}", calibration_path.c_str());
            delete backend;
            return 1;
        }
        nlohmann::json cj;
        try { cf >> cj; } catch (const std::exception& e) {
            spdlog::error("calibration JSON parse error: {}", e.what());
            delete backend;
            return 1;
        }
        CalibrationArtifact art;
        std::string calib_err;
        if (!CalibrationArtifact::from_json(cj, art, calib_err)) {
            spdlog::error("calibration artifact error: {}", calib_err.c_str());
            delete backend;
            return 1;
        }
        CalibrationFormulation cur_form;
        cur_form.layout           = ExperimentConfig{}.layout_str(); // "auto"
        cur_form.candidate_scheme = (prompt_cfg.scheme == Scheme::LETTERS) ? "letters" : "natural";
        cur_form.prior_correction = false;
        std::string compat_warn;
        if (!art.check_compatible(cur_form, compat_warn)) {
            spdlog::warn("{}", compat_warn.c_str());
        }
        calib_cfg = art.to_config();
        spdlog::info("calibration loaded from {}", calibration_path.c_str());
    }

    DecisionEngine* engine = nullptr;
    try {
        engine = new DecisionEngine(*backend, prompt_cfg, calib_cfg);
    } catch (const std::exception& e) {
        spdlog::error("engine init error: {}", e.what());
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
    } else if (cmd == "calibration") {
        return cmd_calibration(argc - 2, argv + 2);
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
