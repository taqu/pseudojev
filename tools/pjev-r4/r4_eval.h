#pragma once

#include <cstdint>
#include <string>

namespace pjev
{
namespace r4
{

struct EvalOptions {
    std::string dir       = "results/r4";
    int         bootstrap = 2000;
    uint64_t    seed      = 20261020;
};

// Stage 1: tuning partition only. Selects the 2-shot order, kNN voting and routing finalists and writes
// <dir>/frozen_policy.json + frozen_policy.sha256. Refuses once an evaluation run exists.
int tune_main(const EvalOptions& o);

// Stage 2: applies the frozen configuration to <dir>/evaluation and writes the report.
int evaluate_main(const EvalOptions& o);

} // namespace r4
} // namespace pjev
