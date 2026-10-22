#include "calibration.h"
#include <algorithm>
#include <cmath>
#include <fstream>

namespace pjev
{

double CalibrationConfig::temperature_for(Type type) const
{
    if (!enabled) return 1.0;
    if (type == Type::Noul)  return noul_temperature;
    if (type == Type::Score) return score_temperature;
    return choice_temperature;
}

double CalibrationConfig::prior_alpha_for(Type type) const
{
    if (type == Type::Noul)  return noul_prior_alpha;
    if (type == Type::Score) return score_prior_alpha;
    return choice_prior_alpha;
}

std::string temperature_softmax(const std::vector<float>& logits, double T,
                                std::vector<double>& probs)
{
    probs.clear();
    if (logits.empty()) return "empty logit vector";
    if (T <= 0.0)       return "temperature must be > 0";

    double max_val = (double)logits[0] / T;
    for (float l : logits) {
        double s = (double)l / T;
        if (!std::isfinite(s)) return "non-finite scaled logit";
        if (s > max_val) max_val = s;
    }
    probs.reserve(logits.size());
    double sum = 0.0;
    for (float l : logits) {
        double e = std::exp((double)l / T - max_val);
        probs.push_back(e);
        sum += e;
    }
    if (sum <= 0.0 || !std::isfinite(sum)) return "invalid softmax denominator";
    double check = 0.0;
    for (double& p : probs) {
        p /= sum;
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) return "probability out of range";
        check += p;
    }
    if (std::fabs(check - 1.0) > 1e-9) return "probabilities do not sum to 1";
    return "";
}

std::vector<double> log_softmax(const std::vector<float>& logits)
{
    if (logits.empty()) return {};
    double max_val = (double)logits[0];
    for (float l : logits) if ((double)l > max_val) max_val = (double)l;
    double sum = 0.0;
    for (float l : logits) sum += std::exp((double)l - max_val);
    if (sum <= 0.0 || !std::isfinite(sum)) return {};
    double log_sum = std::log(sum) + max_val;
    std::vector<double> result;
    result.reserve(logits.size());
    for (float l : logits) result.push_back((double)l - log_sum);
    return result;
}

CalibrationConfig CalibrationArtifact::to_config() const
{
    CalibrationConfig cfg;
    cfg.enabled            = true;
    cfg.method             = method;
    cfg.noul_temperature   = noul_temperature;
    cfg.choice_temperature = choice_temperature;
    cfg.score_temperature  = score_temperature;
    cfg.noul_prior_alpha   = noul_prior_alpha;
    cfg.choice_prior_alpha = choice_prior_alpha;
    cfg.score_prior_alpha  = score_prior_alpha;
    return cfg;
}

nlohmann::json CalibrationArtifact::to_json() const
{
    return nlohmann::json{
        {"version", version},
        {"method",  method},
        {"model",   model_identifier},
        {"formulation", {
            {"layout",           formulation.layout},
            {"candidate_scheme", formulation.candidate_scheme},
            {"prior_correction", formulation.prior_correction}
        }},
        {"parameters", {
            {"noul_temperature",   noul_temperature},
            {"choice_temperature", choice_temperature},
            {"score_temperature",  score_temperature},
            {"noul_prior_alpha",   noul_prior_alpha},
            {"choice_prior_alpha", choice_prior_alpha},
            {"score_prior_alpha",  score_prior_alpha}
        }}
    };
}

bool CalibrationArtifact::from_json(const nlohmann::json& j,
                                    CalibrationArtifact& art,
                                    std::string& err)
{
    try {
        if (!j.contains("version") || !j["version"].is_number_integer()) {
            err = "missing or invalid 'version'"; return false;
        }
        art.version = j["version"].get<int>();
        if (art.version != 1) {
            err = "unsupported artifact version: " + std::to_string(art.version);
            return false;
        }
        if (!j.contains("method") || !j["method"].is_string()) {
            err = "missing or invalid 'method'"; return false;
        }
        art.method = j["method"].get<std::string>();
        if (art.method != "temperature") {
            err = "unsupported calibration method: " + art.method; return false;
        }
        if (j.contains("model") && !j["model"].is_string()) {
            err = "invalid 'model'"; return false;
        }
        art.model_identifier = j.value("model", "");
        if (!j.contains("formulation") || !j["formulation"].is_object()) {
            err = "missing 'formulation'"; return false;
        }
        const auto& f = j["formulation"];
        for (const char* key : {"layout", "candidate_scheme"}) {
            if (!f.contains(key) || !f[key].is_string()) {
                err = std::string("missing or invalid 'formulation.") + key + "'"; return false;
            }
        }
        if (!f.contains("prior_correction") || !f["prior_correction"].is_boolean()) {
            err = "missing or invalid 'formulation.prior_correction'"; return false;
        }
        art.formulation.layout           = f["layout"].get<std::string>();
        art.formulation.candidate_scheme = f["candidate_scheme"].get<std::string>();
        art.formulation.prior_correction = f["prior_correction"].get<bool>();
        if (!j.contains("parameters") || !j["parameters"].is_object()) {
            err = "missing 'parameters'"; return false;
        }
        const auto& p = j["parameters"];
        // Every parameter is required and kept exactly as stored (no defaults, no rounding).
        auto param = [&](const char* key, double& v) {
            if (!p.contains(key) || !p[key].is_number()) {
                err = std::string("missing or non-numeric 'parameters.") + key + "'"; return false;
            }
            v = p[key].get<double>();
            if (!std::isfinite(v)) {
                err = std::string("'parameters.") + key + "' is not finite"; return false;
            }
            return true;
        };
        if (!param("noul_temperature",   art.noul_temperature) ||
            !param("choice_temperature", art.choice_temperature) ||
            !param("score_temperature",  art.score_temperature) ||
            !param("noul_prior_alpha",   art.noul_prior_alpha) ||
            !param("choice_prior_alpha", art.choice_prior_alpha) ||
            !param("score_prior_alpha",  art.score_prior_alpha)) {
            return false;
        }
        if (art.noul_temperature <= 0.0 || art.choice_temperature <= 0.0 || art.score_temperature <= 0.0) {
            err = "temperature values must be > 0"; return false;
        }
        if (art.noul_prior_alpha < 0.0 || art.choice_prior_alpha < 0.0 || art.score_prior_alpha < 0.0) {
            err = "prior_alpha values must be >= 0"; return false;
        }
    } catch (const std::exception& e) {
        err = std::string("JSON parse error: ") + e.what();
        return false;
    }
    return true;
}

bool CalibrationArtifact::check_compatible(const CalibrationFormulation& current,
                                            std::string& warn) const
{
    std::string issues;
    if (!formulation.layout.empty() && formulation.layout != current.layout)
        issues += "layout mismatch (artifact=" + formulation.layout +
                  " current=" + current.layout + "); ";
    if (!formulation.candidate_scheme.empty() &&
        formulation.candidate_scheme != current.candidate_scheme)
        issues += "candidate_scheme mismatch (artifact=" + formulation.candidate_scheme +
                  " current=" + current.candidate_scheme + "); ";
    if (formulation.prior_correction != current.prior_correction)
        issues += "prior_correction mismatch; ";
    if (!issues.empty()) {
        warn = "calibration artifact formulation mismatch: " + issues;
        return false;
    }
    return true;
}

const char* to_string(CalibrationState s)
{
    switch (s) {
    case CalibrationState::Disabled:     return "disabled";
    case CalibrationState::Missing:      return "missing";
    case CalibrationState::Incompatible: return "incompatible";
    case CalibrationState::Enabled:      return "enabled";
    }
    return "unknown";
}

CalibrationConfig CalibrationLoad::config() const
{
    return state == CalibrationState::Enabled ? artifact.to_config() : CalibrationConfig{};
}

bool CalibrationLoad::prior_correction() const
{
    return state == CalibrationState::Enabled && artifact.formulation.prior_correction;
}

bool load_calibration(const std::string& path, bool required, const CalibrationFormulation& active,
                      CalibrationLoad& out, std::string& err)
{
    out = CalibrationLoad{};
    out.path = path;
    if (path.empty()) return true;

    std::ifstream f(path);
    if (!f) {
        if (required) {
            err = "calibration file not found: " + path;
            return false;
        }
        out.state   = CalibrationState::Missing;
        out.message = "calibration file not found: " + path + "; decisions are uncalibrated";
        return true;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        err = "calibration " + path + ": JSON parse error: " + e.what();
        return false;
    }
    std::string why;
    if (!CalibrationArtifact::from_json(j, out.artifact, why)) {
        err = "calibration " + path + ": " + why;
        return false;
    }
    // Prior correction is part of the calibration, applied as the artifact specifies.
    CalibrationFormulation current = active;
    current.prior_correction = out.artifact.formulation.prior_correction;
    if (!out.artifact.check_compatible(current, why)) {
        out.state   = CalibrationState::Incompatible;
        out.message = "calibration " + path + " cannot be applied: " + why +
                      "its parameters were fitted for another candidate-logit formulation; decisions are uncalibrated";
        return true;
    }
    out.state = CalibrationState::Enabled;
    return true;
}

bool check_calibration_model(CalibrationLoad& c, const std::string& model_sha256)
{
    if (c.state != CalibrationState::Enabled || c.artifact.model_identifier.empty()) return false;
    std::string id = c.artifact.model_identifier;
    std::transform(id.begin(), id.end(), id.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
    const bool is_sha256 = id.size() == 64 && std::all_of(id.begin(), id.end(), [](char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
    });
    std::string current = model_sha256;
    std::transform(current.begin(), current.end(), current.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
    if (is_sha256 && id == current) return true;
    c.state   = CalibrationState::Incompatible;
    c.message = "calibration " + c.path + " cannot be applied: " +
                (is_sha256 ? "it was fitted for another inference model (artifact model sha256 " + id +
                                 ", current " + (current.empty() ? std::string("unknown") : current) + ")"
                           : "its model identity '" + c.artifact.model_identifier +
                                 "' is not a SHA-256 and cannot be verified") +
                "; regenerate it with 'pjev calibrate'; decisions are uncalibrated";
    return false;
}

} // namespace pjev
