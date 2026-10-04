#ifndef INC_PJEV_RELEASE_CONFIG_H_
#define INC_PJEV_RELEASE_CONFIG_H_
#include <string>
#include <nlohmann/json.hpp>

namespace pjev {

struct ReleaseConfig {
    int         schema_version  = 1;
    std::string pjev_version;          // informational
    std::string default_model  = "model.gguf";
    std::string calibration;           // empty = none
    int         threads        = -1;   // -1 = not set (use CLI default)
    int         context_size   = -1;   // -1 = not set
    bool        kv_reuse       = true;
    bool        batch_questions = true;

    // Load from a JSON file. Returns false and sets err on failure.
    static bool from_file(const std::string& path, ReleaseConfig& out, std::string& err);
    nlohmann::json to_json() const;

    // Model path resolution.
    // Precedence:
    //   1. explicit_model (non-empty CLI flag) — returned unchanged
    //   2. pjev.json found in exe_dir — use default_model resolved relative to exe_dir
    //   3. model.gguf adjacent to executable
    //   4. "" (empty — caller should fall back to CWD-relative default)
    //
    // exe_dir: result of get_executable_dir()
    // explicit_model: value of --model flag, or "" if not given
    static std::string resolve_model(const std::string& explicit_model,
                                      const std::string& exe_dir);
};

} // namespace pjev
#endif
