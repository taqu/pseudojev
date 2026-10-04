#ifndef INC_PJEV_RUNNER_H_
#define INC_PJEV_RUNNER_H_
#include "config.h"
#include "dataset.h"
#include "metrics.h"
#include "permute.h"
#include "../decision/decision_engine.h"
#include "../inference/backend.h"
#include <nlohmann/json.hpp>
namespace pjev
{
struct ItemResult {
    std::string id;
    std::string type;
    bool        ok       = false;
    std::string error;
    int         selected = -1;
    int         correct_index = -1;      // ground-truth candidate index (for calibration)
    double      expected_score = 0.0;    // ground-truth score level (score items)
    std::vector<double> probs;           // calibrated (or raw if disabled)
    std::vector<double> raw_probs;       // before temperature scaling
    std::vector<float>  corrected_logits; // post prior-correction, pre temperature
    std::vector<std::string> keys;
    bool        correct  = false;
    // Option ordering experiment
    std::string stable_key; // the key selected in this run (for stability check)
};

struct RunResult {
    ExperimentConfig    config;
    std::string         dataset_path;
    RunMetrics          metrics;
    StabilityMetrics    stability;
    std::vector<ItemResult> items;
    nlohmann::json to_json() const;
};

// Runs a single configuration over all rows; returns RunResult.
RunResult run_experiment(ILlamaBackend& backend,
                         const std::vector<DatasetRow>& rows,
                         const ExperimentConfig& cfg);

// Comparison across multiple configs (for named experiments).
struct CompareResult {
    std::string experiment_name;
    std::string model_path;
    std::vector<RunResult> runs;
    nlohmann::json to_json() const;
};

// Run candidate-binding experiment: NATURAL vs LETTERS schemes.
CompareResult exp_candidate_binding(ILlamaBackend& backend,
                                    const std::vector<DatasetRow>& rows,
                                    const std::string& model_path);

// Run option-order experiment: original vs reversed, measure stability.
CompareResult exp_option_order(ILlamaBackend& backend,
                               const std::vector<DatasetRow>& rows,
                               const std::string& model_path);

// Run prompt-layout experiment: all 4 layouts.
CompareResult exp_prompt_layout(ILlamaBackend& backend,
                                const std::vector<DatasetRow>& rows,
                                const std::string& model_path);

// Run prior-correction experiment: with and without.
CompareResult exp_prior_correction(ILlamaBackend& backend,
                                   const std::vector<DatasetRow>& rows,
                                   const std::string& model_path);

// Run score-formulation experiment: NATURAL vs LETTERS for score questions.
CompareResult exp_score_formulation(ILlamaBackend& backend,
                                    const std::vector<DatasetRow>& rows,
                                    const std::string& model_path);
}
#endif //INC_PJEV_RUNNER_H_

