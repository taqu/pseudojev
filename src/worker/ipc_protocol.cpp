#include "ipc_protocol.h"

namespace pjev {

nlohmann::json ipc_input_to_json(const DecisionInput& in)
{
    nlohmann::json j;
    j["type"]     = in.type;
    j["state"]    = in.state;
    j["question"] = in.question;
    nlohmann::json opts = nlohmann::json::array();
    for (const auto& p : in.options) {
        opts.push_back({p.first, p.second});
    }
    j["options"] = opts;
    j["prior_correction"] = in.prior_correction;
    if (!in.prior_logits.empty()) {
        j["prior_logits"] = in.prior_logits;
    }
    j["collect_corrected_logits"] = in.collect_corrected_logits;
    return j;
}

DecisionInput ipc_input_from_json(const nlohmann::json& j)
{
    DecisionInput in;
    in.type     = j.value("type", "");
    in.state    = j.value("state", "");
    in.question = j.value("question", "");
    if (j.contains("options") && j["options"].is_array()) {
        for (const auto& opt : j["options"]) {
            if (opt.is_array() && opt.size() >= 2) {
                in.options.push_back({opt[0].get<std::string>(), opt[1].get<std::string>()});
            }
        }
    }
    in.prior_correction = j.value("prior_correction", false);
    if (j.contains("prior_logits") && j["prior_logits"].is_array()) {
        in.prior_logits = j["prior_logits"].get<std::vector<float>>();
    }
    in.collect_corrected_logits = j.value("collect_corrected_logits", false);
    return in;
}

nlohmann::json ipc_output_to_json(const DecisionOutput& out)
{
    nlohmann::json j;
    j["ok"]             = out.ok;
    j["error"]          = out.error;
    j["selected"]       = out.selected;
    j["p_true"]         = out.p_true;
    j["expected_score"] = out.expected_score;
    j["probs"]          = out.probs;
    j["raw_probs"]      = out.raw_probs;
    j["keys"]           = out.keys;
    j["prompt_token_count"] = out.prompt_token_count;
    j["tokenize_us"]    = out.tokenize_us;
    j["eval_us"]        = out.eval_us;
    j["decision_us"]    = out.decision_us;
    if (!out.corrected_logits.empty()) {
        j["corrected_logits"] = out.corrected_logits;
    }
    return j;
}

DecisionOutput ipc_output_from_json(const nlohmann::json& j)
{
    DecisionOutput out;
    out.ok             = j.value("ok", false);
    out.error          = j.value("error", "");
    out.selected       = j.value("selected", -1);
    out.p_true         = j.value("p_true", 0.0);
    out.expected_score = j.value("expected_score", 0.0);
    if (j.contains("probs") && j["probs"].is_array()) {
        out.probs = j["probs"].get<std::vector<double>>();
    }
    if (j.contains("raw_probs") && j["raw_probs"].is_array()) {
        out.raw_probs = j["raw_probs"].get<std::vector<double>>();
    }
    if (j.contains("keys") && j["keys"].is_array()) {
        out.keys = j["keys"].get<std::vector<std::string>>();
    }
    out.prompt_token_count = j.value("prompt_token_count", 0);
    out.tokenize_us        = j.value("tokenize_us", (int64_t)0);
    out.eval_us            = j.value("eval_us", (int64_t)0);
    out.decision_us        = j.value("decision_us", (int64_t)0);
    if (j.contains("corrected_logits") && j["corrected_logits"].is_array()) {
        out.corrected_logits = j["corrected_logits"].get<std::vector<float>>();
    }
    return out;
}

nlohmann::json ipc_worker_info_to_json(const WorkerInfo& info)
{
    nlohmann::json j;
    j["protocol_version"] = info.protocol_version;
    j["pjev_version"]     = info.pjev_version;
    j["pid"]              = info.pid;
    j["model_identity"]   = info.model_identity;
    j["config_hash"]      = info.config_hash;
    j["start_time_us"]    = info.start_time_us;
    return j;
}

WorkerInfo ipc_worker_info_from_json(const nlohmann::json& j)
{
    WorkerInfo info;
    info.protocol_version = j.value("protocol_version", IPC_PROTOCOL_VERSION);
    info.pjev_version     = j.value("pjev_version", "");
    info.pid              = j.value("pid", 0);
    info.model_identity   = j.value("model_identity", "");
    info.config_hash      = j.value("config_hash", "");
    info.start_time_us    = j.value("start_time_us", (int64_t)0);
    return info;
}

std::string make_config_hash(const std::string& model_path,
                              int threads, int ctx_size,
                              const std::string& calibration_path)
{
    return model_path + "|" + std::to_string(threads) + "|"
         + std::to_string(ctx_size) + "|" + calibration_path;
}

} // namespace pjev
