#ifndef INC_PJEV_CALIBRATION_H_
#define INC_PJEV_CALIBRATION_H_
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pjev
{

struct CalibrationConfig
{
    bool enabled = false;
    std::string method = "temperature";
    double noul_temperature   = 1.0;
    double choice_temperature = 1.0;
    double score_temperature  = 1.0;

    double temperature_for(const std::string& type) const;
};

struct CalibrationFormulation
{
    std::string layout;
    std::string candidate_scheme;
    bool prior_correction = false;
};

struct CalibrationArtifact
{
    int version = 1;
    std::string method = "temperature";
    std::string model_identifier;
    CalibrationFormulation formulation;
    double noul_temperature   = 1.0;
    double choice_temperature = 1.0;
    double score_temperature  = 1.0;

    CalibrationConfig to_config() const;
    nlohmann::json to_json() const;
    static bool from_json(const nlohmann::json& j, CalibrationArtifact& art, std::string& err);
    // Returns true if fully compatible; writes mismatch description to warn.
    bool check_compatible(const CalibrationFormulation& current, std::string& warn) const;
};

// Numerically stable temperature-scaled softmax. T must be > 0.
// Returns empty string on success, error description on failure.
std::string temperature_softmax(const std::vector<float>& logits, double T,
                                std::vector<double>& probs);

// Numerically stable log-softmax (T=1). Returns empty vector on failure.
std::vector<double> log_softmax(const std::vector<float>& logits);

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_H_
