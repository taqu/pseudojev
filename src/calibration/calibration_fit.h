#ifndef INC_PJEV_CALIBRATION_FIT_H_
#define INC_PJEV_CALIBRATION_FIT_H_
#include "calibration_metrics.h"

namespace pjev
{

// Search ranges of the fit. Temperatures outside [kFitTemperatureMin, kFitTemperatureMax] and prior
// alphas outside [kFitAlphaMin, kFitAlphaMax] are never produced.
constexpr double kFitTemperatureMin = 0.01;
constexpr double kFitTemperatureMax = 100.0;
constexpr double kFitAlphaMin       = 0.0;
constexpr double kFitAlphaMax       = 2.0;
constexpr double kFitAlphaStep      = 0.1;

struct FitResult
{
    double temperature    = 1.0;
    double nll            = -1.0; // final objective value (mean NLL at fitted T)
    int    n_samples      = 0;
    bool   converged      = false;
    bool   at_lower_bound = false; // T at T_min (within the search tolerance)
    bool   at_upper_bound = false; // T at T_max
};

// Mean NLL of the samples' `logits` at temperature T; +inf when any sample cannot be scored.
double mean_nll(const std::vector<CalibrationSample>& samples, double T);

// Fit a single temperature by minimizing mean NLL over calibration samples.
// Uses golden-section search over log(T) in [log(T_min), log(T_max)] (the NLL is unimodal in log T).
// Parameterizing as log(T) guarantees T > 0 throughout. Deterministic.
FitResult fit_temperature(const std::vector<CalibrationSample>& samples,
                          double T_min    = kFitTemperatureMin,
                          double T_max    = kFitTemperatureMax,
                          int    max_iter = 200,
                          double tol      = 1e-8);

struct AlphaTResult
{
    double alpha        = 0.0;
    double temperature  = 1.0;
    double nll          = 1e30;  // mean NLL on the fitting samples at (alpha, temperature)
    int    n_samples    = 0;
    bool   converged    = false;
    bool   alpha_at_lower_bound = false;
    bool   alpha_at_upper_bound = false;
    bool   temperature_at_lower_bound = false;
    bool   temperature_at_upper_bound = false;
};

// Joint prior-alpha + temperature fit on one set of samples (raw_logits and prior_logits populated).
// Grid-searches alpha = alpha_min + i * alpha_step (rounded to 1e-9) up to alpha_max; for each alpha
// fits the temperature on the prior-corrected logits; keeps the lowest mean NLL (ties: the smaller
// alpha). Uses only the given samples, so validation samples never influence the parameters.
// Deterministic.
AlphaTResult fit_alpha_temperature(const std::vector<CalibrationSample>& samples,
                                   double alpha_min  = kFitAlphaMin,
                                   double alpha_max  = kFitAlphaMax,
                                   double alpha_step = kFitAlphaStep);

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_FIT_H_
