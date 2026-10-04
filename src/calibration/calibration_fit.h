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

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_FIT_H_
