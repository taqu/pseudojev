#pragma once

#include <string>

namespace pjev
{
namespace r1
{

// Options a caller may inject into the R1 evaluation loop. The defaults reproduce pjev-r1.
struct HarnessHooks {
    // R2 similarity gate: when enabled, the 1-shot pass runs only for queries whose Top-1 exemplar
    // estimated cosine is >= gate_threshold; every other query keeps its S1 prediction.
    bool        gate_enabled   = false;
    double      gate_threshold = 0.0;
    std::string experiment     = "R1 retrieved 1-shot";
    std::string default_output = "results/r1";
};

// Full pjev-r1 command line (argv[0] is the program name). Returns the process exit code.
int run_harness(int argc, char** argv, const HarnessHooks& hooks);

} // namespace r1
} // namespace pjev
