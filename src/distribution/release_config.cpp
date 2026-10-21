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

// A bare file name in pjev.json is relative to the directory holding pjev.json (= exe_dir).
static std::string config_relative(const std::string& value, const std::string& exe_dir) {
    if (value.find('/') == std::string::npos && value.find('\\') == std::string::npos)
        return path_join(exe_dir, value);
    return value;
}

bool ReleaseConfig::from_file(const std::string& path, ReleaseConfig& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open: " + path; return false; }
    nlohmann::json j;
    try { f >> j; } catch (const std::exception& e) { err = e.what(); return false; }
    try {
        if (j.contains("schema_version"))     out.schema_version     = j["schema_version"].get<int>();
        if (j.contains("pjev_version"))       out.pjev_version       = j["pjev_version"].get<std::string>();
        if (j.contains("default_model"))      out.default_model      = j["default_model"].get<std::string>();
        if (j.contains("embedding_model"))    out.embedding_model    = j["embedding_model"].get<std::string>();
        if (j.contains("exemplar_db"))        out.exemplar_db        = j["exemplar_db"].get<std::string>();
        if (j.contains("exemplar_allowlist")) out.exemplar_allowlist = j["exemplar_allowlist"].get<std::string>();
        if (j.contains("threads"))            out.threads            = j["threads"].get<int>();
        if (j.contains("context_size"))       out.context_size       = j["context_size"].get<int>();
    } catch (const std::exception& e) {
        err = path + ": " + e.what();
        return false;
    }
    return true;
}

bool ReleaseConfig::load_adjacent(const std::string& exe_dir, ReleaseConfig& out) {
    if (exe_dir.empty()) return false;
    const std::string cfg_path = path_join(exe_dir, "pjev.json");
    if (!file_exists(cfg_path)) return false;
    std::string err;
    return from_file(cfg_path, out, err);
}

nlohmann::json ReleaseConfig::to_json() const {
    return {
        {"schema_version",     schema_version},
        {"pjev_version",       pjev_version},
        {"default_model",      default_model},
        {"embedding_model",    embedding_model},
        {"exemplar_db",        exemplar_db},
        {"exemplar_allowlist", exemplar_allowlist},
        {"threads",            threads},
        {"context_size",       context_size}
    };
}

std::string ReleaseConfig::resolve_model(const std::string& explicit_model,
                                          const std::string& exe_dir) {
    // 1. Explicit CLI flag
    if (!explicit_model.empty()) return explicit_model;

    // 2. pjev.json adjacent to executable
    if (!exe_dir.empty()) {
        ReleaseConfig rc;
        if (load_adjacent(exe_dir, rc) && !rc.default_model.empty()) {
            return config_relative(rc.default_model, exe_dir);
        }

        // 3. model.gguf adjacent to executable
        std::string adjacent = path_join(exe_dir, kDefaultModelFile);
        if (file_exists(adjacent)) return adjacent;
    }

    // 4. Not found via executable-relative search
    return "";
}

RuntimePaths ReleaseConfig::resolve_runtime_paths(const RuntimePaths& explicit_paths,
                                                  const std::string& exe_dir) {
    ReleaseConfig rc;
    const bool have_cfg = load_adjacent(exe_dir, rc);
    auto resolve = [&](const std::string& flag, const std::string& configured, const char* file) {
        if (!flag.empty()) return flag;
        if (have_cfg && !configured.empty()) return config_relative(configured, exe_dir);
        if (!exe_dir.empty()) {
            const std::string adjacent = path_join(exe_dir, file);
            if (file_exists(adjacent)) return adjacent;
        }
        return std::string("models/") + file;
    };
    RuntimePaths p;
    p.model = resolve_model(explicit_paths.model, exe_dir);
    if (p.model.empty()) p.model = std::string("models/") + kDefaultDevModelFile;
    p.embedding_model    = resolve(explicit_paths.embedding_model, rc.embedding_model, kDefaultEmbeddingModelFile);
    p.exemplar_db        = resolve(explicit_paths.exemplar_db, rc.exemplar_db, kDefaultExemplarDbFile);
    p.exemplar_allowlist = resolve(explicit_paths.exemplar_allowlist, rc.exemplar_allowlist, kDefaultExemplarAllowlistFile);
    return p;
}

} // namespace pjev
