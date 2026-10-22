#ifndef INC_PJEV_CALIBRATION_H_
#define INC_PJEV_CALIBRATION_H_
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <server/jev_api.h>

namespace pjev
{

struct CalibrationConfig
{
    bool enabled = false;
    std::string method = "temperature";
    double noul_temperature   = 1.0;
    double choice_temperature = 1.0;
    double score_temperature  = 1.0;
    double noul_prior_alpha   = 1.0;  // alpha for prior correction (0 = no correction)
    double choice_prior_alpha = 1.0;
    double score_prior_alpha  = 1.0;

    double temperature_for(Type type) const;
    double prior_alpha_for(Type type) const;  // returns alpha regardless of enabled
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
    double noul_prior_alpha   = 1.0;
    double choice_prior_alpha = 1.0;
    double score_prior_alpha  = 1.0;

    CalibrationConfig to_config() const;
    nlohmann::json to_json() const;
    // Requires version 1, method "temperature", the formulation (layout, candidate_scheme,
    // prior_correction) and all six parameters; values are kept exactly as stored. Temperatures must be
    // finite and > 0, prior alphas finite and >= 0.
    static bool from_json(const nlohmann::json& j, CalibrationArtifact& art, std::string& err);
    bool check_compatible(const CalibrationFormulation& current, std::string& warn) const;
};

// Production calibration state (see load_calibration).
enum class CalibrationState { Disabled, Missing, Incompatible, Enabled };
const char* to_string(CalibrationState s);

struct CalibrationLoad
{
    CalibrationState    state = CalibrationState::Disabled;
    std::string         path;
    std::string         message;   // Missing: the warning; Incompatible: the compatibility error
    CalibrationArtifact artifact;  // as stored (Incompatible, Enabled)

    // Decision engine configuration: the artifact's parameters when Enabled, otherwise uncalibrated.
    CalibrationConfig config() const;
    // Whether decisions apply prior correction (Enabled and the artifact's formulation has it).
    bool prior_correction() const;
};

// Reads and validates the calibration artifact at `path` against the formulation the decision engine
// uses (`active`: layout and candidate_scheme; prior correction is applied as the artifact specifies).
//   path empty                       -> Disabled
//   file absent and not `required`   -> Missing (message is the warning); decisions are uncalibrated
//   artifact for another formulation -> Incompatible (message names the mismatch); never applied
//   otherwise                        -> Enabled
// Returns false with err on a configuration error: unreadable, malformed or unsupported file, invalid
// parameters, or an absent file when `required`.
bool load_calibration(const std::string& path, bool required, const CalibrationFormulation& active,
                      CalibrationLoad& out, std::string& err);

// Model identity check, once the inference model's SHA-256 is known. An artifact's "model" is the
// SHA-256 (hex) of the GGUF file it was fitted for (`pjev calibrate`); empty means not recorded (older
// artifacts) and is not checked. An Enabled calibration fitted for another model, or naming its model
// in a form that cannot be verified, becomes Incompatible (never applied). Returns c.state ==
// Enabled-and-verified for a recorded identity, false otherwise.
bool check_calibration_model(CalibrationLoad& c, const std::string& model_sha256);

// Numerically stable temperature-scaled softmax. T must be > 0.
// Returns empty string on success, error description on failure.
std::string temperature_softmax(const std::vector<float>& logits, double T,
                                std::vector<double>& probs);

// Numerically stable log-softmax (T=1). Returns empty vector on failure.
std::vector<double> log_softmax(const std::vector<float>& logits);

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_H_
