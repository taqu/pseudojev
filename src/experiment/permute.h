#ifndef INV_PJEV_PERMUTE_H_
#define INV_PJEV_PERMUTE_H_
#include "config.h"
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace pjev
{
inline std::vector<std::pair<std::string, std::string>> permute_options(
    const std::vector<std::pair<std::string, std::string>>& opts,
    OptionOrder order,
    int32_t seed = 42)
{
    std::vector<std::pair<std::string, std::string>> result = opts;
    if(order == OptionOrder::REVERSED) {
        std::reverse(result.begin(), result.end());
    } else if(order == OptionOrder::RANDOM) {
        std::mt19937 rng(static_cast<unsigned>(seed));
        std::shuffle(result.begin(), result.end(), rng);
    }
    return result;
}

// Generate all permutations of opts (up to max_perms to avoid factorial blowup)
inline std::vector<std::vector<std::pair<std::string, std::string>>> all_permutations(
    const std::vector<std::pair<std::string, std::string>>& opts,
    int32_t max_perms = 24)
{
    std::vector<std::vector<std::pair<std::string, std::string>>> result;
    std::vector<std::pair<std::string, std::string>> perm = opts;
    std::sort(perm.begin(), perm.end()); // canonical start
    do {
        result.push_back(perm);
        if(static_cast<int32_t>(result.size()) >= max_perms){
            break;
        }
    } while(std::next_permutation(perm.begin(), perm.end()));
    return result;
}
} // namespace pjev
#endif // INV_PJEV_PERMUTE_H_
