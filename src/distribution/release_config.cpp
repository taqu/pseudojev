#include "release_config.h"
#include <fstream>

namespace pjev {

static bool file_exists(const std::string& p) {
    std::ifstream f(p); return f.good();
}

static std::string path_join(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
#ifdef _WIN32
    return dir + '\\' + name;
#else
    return dir + '/' + name;
#endif
}

bool ReleaseConfig::from_file(const std::string& path, ReleaseConfig& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open: " + path; return false; }
    nlohmann::json j;
    try { f >> j; } catch (const std::exception& e) { err = e.what(); return false; }
    if (j.contains("schema_version")) out.schema_version = j["schema_version"].get<int>();
    if (j.contains("pjev_version"))   out.pjev_version   = j["pjev_version"].get<std::string>();
    if (j.contains("default_model"))  out.default_model  = j["default_model"].get<std::string>();
    if (j.contains("calibration"))    out.calibration    = j["calibration"].get<std::string>();
    if (j.contains("threads"))        out.threads        = j["threads"].get<int>();
    if (j.contains("context_size"))   out.context_size   = j["context_size"].get<int>();
    if (j.contains("kv_reuse"))       out.kv_reuse       = j["kv_reuse"].get<bool>();
    if (j.contains("batch_questions")) out.batch_questions = j["batch_questions"].get<bool>();
    return true;
}

nlohmann::json ReleaseConfig::to_json() const {
    return {
        {"schema_version",  schema_version},
        {"pjev_version",    pjev_version},
        {"default_model",   default_model},
        {"calibration",     calibration},
        {"threads",         threads},
        {"context_size",    context_size},
        {"kv_reuse",        kv_reuse},
        {"batch_questions", batch_questions}
    };
}

std::string ReleaseConfig::resolve_model(const std::string& explicit_model,
                                          const std::string& exe_dir) {
    // 1. Explicit CLI flag
    if (!explicit_model.empty()) return explicit_model;

    // 2. pjev.json adjacent to executable
    if (!exe_dir.empty()) {
        std::string cfg_path = path_join(exe_dir, "pjev.json");
        if (file_exists(cfg_path)) {
            ReleaseConfig rc;
            std::string err;
            if (from_file(cfg_path, rc, err) && !rc.default_model.empty()) {
                std::string model_path = rc.default_model;
                // Resolve relative to cfg dir (= exe_dir)
                if (model_path.find('/') == std::string::npos &&
                    model_path.find('\\') == std::string::npos) {
                    model_path = path_join(exe_dir, model_path);
                }
                return model_path;
            }
        }

        // 3. model.gguf adjacent to executable
        std::string adjacent = path_join(exe_dir, "model.gguf");
        if (file_exists(adjacent)) return adjacent;
    }

    // 4. Not found via executable-relative search
    return "";
}

} // namespace pjev
