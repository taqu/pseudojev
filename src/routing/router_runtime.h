#ifndef INC_PJEV_ROUTER_RUNTIME_H_
#define INC_PJEV_ROUTER_RUNTIME_H_
#include <cstdint>
#include <memory>
#include <string>

#include "calibration/calibration.h"
#include "decision/decision_service.h"
#include "distribution/release_config.h"
#include "model/model_identity.h"

namespace pjev
{

class DecisionEngine;
namespace retrieval { class ExemplarStore; }
namespace routing { class TaskRouter; }

struct RuntimeOptions {
    RuntimePaths paths;                  // paths.calibration empty = calibration disabled
    bool    calibration_required = false;  // configured explicitly: a missing file is an error
    int32_t n_threads    = -1;  // inference threads; -1 = physical core count
    int32_t n_threads_batch = -1;  // -1 = same as n_threads
    int32_t n_ctx        = 4096;
    int32_t n_gpu_layers = 0;
    int32_t main_gpu     = 0;
    bool    verbose      = false;
};

// True when `path` holds a Git LFS pointer instead of the file content (repository checked out
// without Git LFS).
bool is_lfs_pointer(const std::string& path);

// Owns the models, the exemplar database and the decision path built on them. Loaded once per process
// (CLI --direct invocation, background worker, HTTP server) and shared by all of its decisions.
class RouterRuntime
{
public:
    // Loads everything up front; on any failure returns nullptr and a message naming the resource.
    // The calibration is read once (see load_calibration) and checked against the loaded model
    // (check_calibration_model): a missing default file, or an artifact for another formulation or
    // another model, is logged (warning / error) and decisions stay uncalibrated; a malformed or
    // unsupported file is a load failure.
    static std::unique_ptr<RouterRuntime> load(const RuntimeOptions& opts, std::string& err);
    ~RouterRuntime();

    RouterRuntime(const RouterRuntime&)            = delete;
    RouterRuntime& operator=(const RouterRuntime&) = delete;

    DecisionService&     service();
    // The production components behind service(), for `pjev calibrate` (which loads the runtime
    // uncalibrated and holds its query groups out of the exemplar store).
    routing::TaskRouter&      router();
    DecisionEngine&           engine();
    retrieval::ExemplarStore& exemplar_store();
    const ModelIdentity& model_identity() const { return model_identity_; }
    const std::string&   embedding_fingerprint() const { return embedding_fp_; }
    int64_t              exemplar_count() const { return exemplar_count_; }
    size_t               eligible_exemplars() const { return eligible_exemplars_; }
    const CalibrationLoad& calibration() const { return calibration_; }

private:
    RouterRuntime();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ModelIdentity         model_identity_;
    std::string           embedding_fp_;
    int64_t               exemplar_count_     = 0;
    size_t                eligible_exemplars_ = 0;
    CalibrationLoad       calibration_;
};

} // namespace pjev
#endif // INC_PJEV_ROUTER_RUNTIME_H_
