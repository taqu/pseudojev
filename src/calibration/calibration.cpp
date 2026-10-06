#include "calibration.h"
#include <algorithm>
#include <cmath>

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
        art.model_identifier = j.value("model", "");
        if (j.contains("formulation") && j["formulation"].is_object()) {
            const auto& f = j["formulation"];
            art.formulation.layout           = f.value("layout", "");
            art.formulation.candidate_scheme = f.value("candidate_scheme", "");
            art.formulation.prior_correction = f.value("prior_correction", false);
        }
        if (!j.contains("parameters") || !j["parameters"].is_object()) {
            err = "missing 'parameters'"; return false;
        }
        const auto& p = j["parameters"];
        art.noul_temperature   = p.value("noul_temperature",   1.0);
        art.choice_temperature = p.value("choice_temperature", 1.0);
        art.score_temperature  = p.value("score_temperature",  1.0);
        if (art.noul_temperature <= 0.0 || art.choice_temperature <= 0.0 || art.score_temperature <= 0.0) {
            err = "temperature values must be > 0"; return false;
        }
        art.noul_prior_alpha   = p.value("noul_prior_alpha",   1.0);
        art.choice_prior_alpha = p.value("choice_prior_alpha", 1.0);
        art.score_prior_alpha  = p.value("score_prior_alpha",  1.0);
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
        warn = "calibration artifact compatibility warning: " + issues;
        return false;
    }
    return true;
}

} // namespace pjev
