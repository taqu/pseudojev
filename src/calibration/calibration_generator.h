#ifndef INC_PJEV_CALIBRATION_GENERATOR_H_
#define INC_PJEV_CALIBRATION_GENERATOR_H_
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "calibration.h"
#include "calibration_metrics.h"
#include "decision/decision_engine.h"
#include "nlohmann/json.hpp"
#include "retrieval/exemplar_store.h"
#include "routing/task_router.h"

// `pjev calibrate`: model-specific calibration generated from the exemplar database.
//
//   1. Partition. Calibration queries are training-split exemplars (the allowlist) with a one-hot
//      answer. Selection is by protected group (examples.group_id): groups in seeded XXH64 order, one
//      query per task kind per selected group, until each kind has max_samples queries. A selected
//      group is assigned wholly to fitting or validation (seeded XXH64, validation_ratio), and all of its
//      rows are held out of the retrieval pool, so no query is answered with its own or a sibling
//      variant as a demonstration. Validation, test and external rows (not in the allowlist) are never
//      used.
//   2. Collection. Each query runs through the production decision path, uncalibrated: choice and noul
//      through the TaskRouter (nearest 2-shot order A / nearest 1-shot, with the production zero-shot
//      fallback when the pool lacks exemplars), score through S1 zero-shot (the LLM path its calibration
//      parameters apply to; kNN-5 votes are never calibrated). The raw candidate logits, the correct
//      candidate index and the executed strategy are recorded.
//   3. Fit. Per kind on the fitting queries: prior alpha (grid) and temperature (golden section),
//      minimizing NLL (calibration_fit.h), applied as the decision engine applies them.
//   4. Validation. Uncalibrated and calibrated metrics on the validation queries, per kind and per
//      executed strategy. The calibration is rejected when it does not improve validation NLL of a kind.
//   5. Output. calibration.json (version 1) written atomically after it loads through the production
//      loader for this model; a companion report holds provenance, counts, metrics and samples.

namespace pjev
{
namespace calibration
{

constexpr uint64_t kDefaultSeed            = 1;
constexpr int      kDefaultMaxSamples      = 1000;  // per task kind (fitting + validation)
constexpr double   kDefaultValidationRatio = 0.2;
constexpr int      kDefaultBatchSize       = 32;
// Minimum accepted samples per task kind; with fewer the generation fails (no reduced-data mode).
constexpr int      kMinFitSamples          = 50;
constexpr int      kMinValidationSamples   = 20;

struct GeneratorOptions {
    uint64_t seed             = kDefaultSeed;
    int      max_samples      = kDefaultMaxSamples;       // per task kind
    double   validation_ratio = kDefaultValidationRatio;  // fraction of selected groups held for validation
    int      batch_size       = kDefaultBatchSize;        // queries per processing batch (progress, memory)
    bool     verbose          = false;                    // one log line per query

    bool validate(std::string& err) const;
};

enum class Split { Fit, Validation };
const char* to_string(Split s);

// One calibration query.
struct QueryRef {
    int64_t     id       = 0;
    int64_t     group_id = 0;
    Type        type     = Type::Unknown;
    Split       split    = Split::Fit;
    std::string domain;
    std::string family;
};

struct KindCounts {
    size_t candidate_rows      = 0;  // training split, one-hot, this kind
    size_t candidate_groups    = 0;
    size_t examined            = 0;  // candidate rows validated during selection
    size_t selected_fit        = 0;
    size_t selected_validation = 0;
    size_t processed           = 0;  // queries evaluated
    size_t accepted_fit        = 0;
    size_t accepted_validation = 0;
    std::map<std::string, size_t> rejected;  // reason -> count (selection and evaluation)

    size_t rejected_total() const;
};

struct Partition {
    std::vector<QueryRef>       queries;   // processing order (selected groups in seeded order)
    std::unordered_set<int64_t> held_out;  // training-split rows of the selected groups
    size_t held_out_groups       = 0;
    size_t rows_total            = 0;
    size_t rows_not_training     = 0;  // not in the allowlist: validation / test / external rows
    std::map<Type, KindCounts> kinds;   // choice, noul, score
    std::map<std::string, size_t> rejected_other;  // rows of unsupported kinds
};

// Validates one candidate row as a calibration query: Ok, Invalid (reason recorded), or Failed (database
// failure; reason is the error).
using RowValidator = std::function<retrieval::LoadResult(const retrieval::ExemplarRow&, std::string& reason)>;

// Step 1 (see above). Deterministic for identical rows, allowlist and options.
bool partition_rows(const std::vector<retrieval::ExemplarRow>& rows,
                    const std::function<bool(int64_t)>&       in_training_split,
                    const RowValidator&                       validate,
                    const GeneratorOptions&                   opts,
                    Partition&                                out,
                    std::string&                              err);

// Fails (insufficient data) unless every kind has kMinFitSamples fitting and kMinValidationSamples
// validation queries; `accepted` counts accepted samples instead of selected queries.
bool check_sufficient(const Partition& p, bool accepted, std::string& err);

// The query a pjev client would send for a stored exemplar, in the router's decision order
// (choice: options as keys; score: levels "0".."N-1" lowest first; noul: false, true), and the index
// of its ground-truth answer. Fails with a reason when the exemplar is not a valid calibration example.
bool make_query(const retrieval::ExemplarPayload& payload, const QueryRef& q, DecisionInput& in,
                int& correct, std::string& reason);

// One collected sample: uncalibrated candidate logits of the executed production path.
struct CollectedSample {
    int64_t            id       = 0;
    Type               type     = Type::Unknown;
    Split              split    = Split::Fit;
    std::string        strategy;         // executed: nearest_2shot | nearest_1shot | zero_shot
    std::string        fallback_reason;  // empty on the requested path
    int                correct  = -1;
    std::vector<float> raw_logits;       // decision order
    int32_t            prompt_tokens = 0;
};

// Step 2 for one query at a time.
class SampleCollector
{
public:
    SampleCollector(routing::TaskRouter& router, DecisionEngine& engine, const retrieval::IExemplarIndex& index);

    // Ok (sample filled), Invalid (rejected; reason), Failed (database / inference failure; reason).
    retrieval::LoadResult collect(const QueryRef& q, CollectedSample& out, std::string& reason);

    // Blank-prompt candidate logits (DecisionEngine::compute_blank_logits, as the router's prior
    // correction uses them), cached per kind and candidate count. Empty on failure.
    const std::vector<float>& prior(Type type, size_t n);

private:
    routing::TaskRouter&              router_;
    DecisionEngine&                   engine_;
    const retrieval::IExemplarIndex&  index_;
    std::map<std::pair<Type, size_t>, std::vector<float>> priors_;
};

struct Progress {
    std::string stage;   // scan | partition | collect | fit | validate | write
    size_t processed = 0;
    size_t total     = 0;
    size_t accepted  = 0;
    size_t rejected  = 0;
    std::map<Type, std::pair<size_t, size_t>> by_kind;  // processed / selected
};
using ProgressFn = std::function<void(const Progress&)>;

struct GenerateRequest {
    std::string      output_path;     // calibration.json
    std::string      model_sha256;    // inference model identity recorded in the artifact
    GeneratorOptions options;
    nlohmann::json   provenance = nlohmann::json::object();  // model / database / runtime, for the report
};

struct GenerateOutcome {
    bool                     ok = false;       // calibration written
    std::string              error;
    CalibrationArtifact      artifact;         // when fitted
    std::string              report_path;      // written report ("" when none)
    nlohmann::json           report;
    std::vector<std::string> warnings;
};

// <output>.report.json (accepted) / <output>.failed-report.json (rejected by validation), next to the
// output ("calibration.json" -> "calibration.report.json").
std::string report_path_for(const std::string& output_path, bool accepted);

// Runs steps 1-5 on the store (opened read-only; its held-out set is replaced) with the given
// uncalibrated router and engine. A failure before validation writes nothing; a calibration rejected by
// validation writes only the failed report. The existing output file is replaced only by a validated
// calibration.
GenerateOutcome generate_calibration(retrieval::ExemplarStore& store, routing::TaskRouter& router,
                                     DecisionEngine& engine, const GenerateRequest& req,
                                     const ProgressFn& progress = {});

// Writes `content` to `dest` atomically: a temporary file in dest's directory, `verify(tmp, err)`, then
// a rename over dest. On any failure the temporary file is removed and dest is untouched.
bool write_file_atomically(const std::string& dest, const std::string& content,
                           const std::function<bool(const std::string&, std::string&)>& verify,
                           std::string& err);

// Writes a calibration artifact atomically after verifying the serialized file through the production
// loader: it must load as Enabled for the production formulation, be accepted for `model_sha256`, and
// (when `expect` is set) hold exactly expect's parameters.
bool write_calibration_file(const std::string& dest, const std::string& serialized,
                            const std::string& model_sha256, const CalibrationArtifact* expect,
                            std::string& err);

// The production candidate-logit formulation (task_policy.h) with prior correction.
CalibrationFormulation production_formulation();

} // namespace calibration
} // namespace pjev
#endif // INC_PJEV_CALIBRATION_GENERATOR_H_
