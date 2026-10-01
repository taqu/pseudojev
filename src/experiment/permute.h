#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include <random>
#include "config.h"

inline std::vector<std::pair<std::string, std::string>> permute_options(
    const std::vector<std::pair<std::string, std::string>>& opts,
    OptionOrder order,
    int seed = 42)
{
    auto result = opts;
    if (order == OptionOrder::REVERSED) {
        std::reverse(result.begin(), result.end());
    } else if (order == OptionOrder::RANDOM) {
        std::mt19937 rng(static_cast<unsigned>(seed));
        std::shuffle(result.begin(), result.end(), rng);
    }
    return result;
}

// Generate all permutations of opts (up to max_perms to avoid factorial blowup)
inline std::vector<std::vector<std::pair<std::string, std::string>>> all_permutations(
    const std::vector<std::pair<std::string, std::string>>& opts,
    int max_perms = 24)
{
    std::vector<std::vector<std::pair<std::string, std::string>>> result;
    auto perm = opts;
    std::sort(perm.begin(), perm.end()); // canonical start
    do {
        result.push_back(perm);
        if ((int)result.size() >= max_perms) break;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return result;
}
