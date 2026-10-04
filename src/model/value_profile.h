#ifndef INC_PJEV_VALUE_PROFILE_H_
#define INC_PJEV_VALUE_PROFILE_H_
#include "model_identity.h"
#include "../calibration/calibration_metrics.h"
#include "../experiment/config.h"
#include "../experiment/runner.h"
#include "../multilingual/multilingual.h"
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace pjev
{

struct PerformanceStats
{
    int64_t model_load_ms       = 0;
    double  warm_p50_ms         = -1.0;
    double  warm_p90_ms         = -1.0;
    double  warm_p95_ms         = -1.0;
    int     n_warm_samples      = 0;
    double  mean_prompt_tokens  = 0.0;
    int64_t rss_post_load_bytes = -1;

    std::string os;
    std::string cpu_info;
    int         n_threads         = 0;
    std::string llama_backend_tag; // "cpu" / "cuda" / etc.

    nlohmann::json to_json() const;
};

struct PrimitiveQuality
{
    int    n          = 0;
    double accuracy   = -1.0;
    double mae        = -1.0;
    double qwk        = -1.0;
    int    n_levels   = 0;
    std::map<std::string, double> accuracy_by_difficulty;
    std::map<std::string, int>    n_by_difficulty;

    nlohmann::json to_json() const;
};

struct QualityStats
{
    PrimitiveQuality noul;
    PrimitiveQuality choice;
    PrimitiveQuality score;

    nlohmann::json to_json() const;
};

// Build QualityStats from a RunResult (uses metrics + per-item difficulty data).
QualityStats compute_quality_stats(const RunResult& result);

// Build PerformanceStats from timing data collected during a run.
PerformanceStats compute_perf_stats(int64_t load_ms,
                                     int64_t rss_bytes,
                                     const std::vector<int64_t>& warm_ms,
                                     const std::string& os = "",
                                     const std::string& cpu_info = "",
                                     int n_threads = 0);

// Items valid for all compared models (no single-token incompatibility).
struct CommonValidSubset
{
    std::vector<std::string> model_ids;
    int total_items = 0;
    int valid_items = 0;                              // items valid for every model

    std::map<std::string, double> full_accuracy;      // model_id → accuracy, full set
    std::map<std::string, double> subset_accuracy;    // model_id → accuracy, valid subset
    std::map<std::string, int>    n_compat_issues;    // model_id → incompatible item count

    nlohmann::json to_json() const;
};

// Build a common-valid-subset given per-model RunResults and their incompatible item IDs.
// incompatible_ids maps model_id → set of item IDs that were incompatible.
CommonValidSubset build_common_valid_subset(
    const std::map<std::string, RunResult>& results,
    const std::map<std::string, std::vector<std::string>>& incompatible_ids);

struct ValueProfile
{
    int         version = 1;
    std::string timestamp;        // ISO 8601
    std::string pjev_commit;
    std::string llamacpp_revision;

    ModelIdentity    model;
    ExperimentConfig frozen_config;

    QualityStats quality_en;
    QualityStats quality_ja;

    CalibrationReport calibration_en;
    CalibrationReport calibration_ja;

    MultilingualResult multilingual;

    PerformanceStats   performance;
    CandidateCompatibility candidate_compat;

    std::vector<std::string> compatibility_issues;

    nlohmann::json to_json() const;
};

struct Phase5Decision
{
    int phase = 5;
    std::string timestamp;
    std::string baseline_model;
    std::vector<std::string> evaluated_models;
    std::string selected_default_model;
    std::vector<std::string> evidence_artifacts;
    std::vector<std::string> notes;

    nlohmann::json to_json() const;
    static bool from_json(const nlohmann::json& j, Phase5Decision& out, std::string& err);
};

} // namespace pjev
#endif // INC_PJEV_VALUE_PROFILE_H_
