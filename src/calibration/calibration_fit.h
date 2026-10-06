#ifndef INC_PJEV_CALIBRATION_FIT_H_
#define INC_PJEV_CALIBRATION_FIT_H_
#include "calibration_metrics.h"

namespace pjev
{

struct FitResult
{
    double temperature    = 1.0;
    double nll            = -1.0; // final objective value (NLL at fitted T)
    int    n_samples      = 0;
    bool   converged      = false;
};

// Fit a single temperature by minimizing mean NLL over calibration samples.
// Uses golden-section search over log(T) in [log(T_min), log(T_max)].
// Parameterizing as log(T) guarantees T > 0 throughout.
FitResult fit_temperature(const std::vector<CalibrationSample>& samples,
                           double T_min    = 0.01,
                           double T_max    = 100.0,
                           int    max_iter = 200,
                           double tol      = 1e-8);

struct AlphaTResult
{
    double alpha        = 0.0;
    double temperature  = 1.0;
    double tuning_nll   = 1e30;
    double val_nll      = 1e30;
    int    n_tuning     = 0;
    int    n_val        = 0;
    bool   converged    = false;
};

// Joint alpha+temperature search.
// Grid-searches alpha in [alpha_min, alpha_max] with step alpha_step.
// For each alpha, fits temperature on tuning_samples, evaluates on val_samples.
// If val_samples is empty, uses tuning NLL for selection.
AlphaTResult fit_alpha_temperature(
    const std::vector<CalibrationSample>& tuning_samples,
    const std::vector<CalibrationSample>& val_samples,
    double alpha_min  = 0.0,
    double alpha_max  = 2.0,
    double alpha_step = 0.1);

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_FIT_H_
