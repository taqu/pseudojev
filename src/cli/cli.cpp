#include "cli.h"
#include <nlohmann/json.hpp>
#include <cstdio>

namespace pjev {

std::string format_noul_human(const DecisionOutput& out)
{
    return (out.selected == 0) ? "true" : "false";
}

std::string format_noul_json(const DecisionOutput& out)
{
    nlohmann::json j;
    j["primitive"]        = "noul";
    j["result"]["value"]  = (out.selected == 0);
    j["result"]["p_true"] = out.p_true;
    j["probabilities"]    = out.probs;
    return j.dump();
}

std::string format_choice_human(const DecisionOutput& out)
{
    if (out.selected >= 0 && out.selected < (int)out.keys.size())
        return out.keys[out.selected];
    return "";
}

std::string format_choice_json(const DecisionOutput& out)
{
    nlohmann::json j;
    j["primitive"]       = "choice";
    j["result"]["index"] = out.selected;
    if (out.selected >= 0 && out.selected < (int)out.keys.size())
        j["result"]["key"] = out.keys[out.selected];
    j["probabilities"] = out.probs;
    j["keys"]          = out.keys;
    return j.dump();
}

std::string format_score_human(const DecisionOutput& out)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%.4f", out.expected_score);
    return buf;
}

std::string format_score_json(const DecisionOutput& out,
                               const std::vector<std::pair<std::string, std::string>>& options)
{
    nlohmann::json j;
    j["primitive"]                = "score";
    j["result"]["expected_score"] = out.expected_score;
    j["probabilities"]            = out.probs;
    j["keys"]                     = out.keys;
    nlohmann::json levels = nlohmann::json::array();
    for (auto& opt : options) {
        nlohmann::json lev;
        lev["key"]         = opt.first;
        lev["description"] = opt.second;
        levels.push_back(lev);
    }
    j["levels"] = levels;
    return j.dump();
}

} // namespace pjev
