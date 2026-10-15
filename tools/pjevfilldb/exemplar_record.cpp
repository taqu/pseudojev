#include "exemplar_record.h"

#include <cmath>

#include <decision/prompt_strategy.h>
#include "id_hash.h"

namespace pjev
{
namespace filldb
{

static std::string require_string(
    const nlohmann::ordered_json& j, const char* key, std::string& out)
{
    auto it = j.find(key);
    if(it == j.end() || !it->is_string()) {
        return std::string("missing or non-string field: ") + key;
    }
    out = it->get<std::string>();
    return {};
}

std::string parse_exemplar(const nlohmann::ordered_json& j, ExemplarRecord& rec)
{
    std::string err;

    std::string id_str, group_id_str;
    if((err = require_string(j, "id",       id_str))       != "") return err;
    if((err = require_string(j, "group_id", group_id_str)) != "") return err;
    rec.id       = hash_exemplar_id(id_str);
    rec.group_id = hash_exemplar_id(group_id_str);
    if((err = require_string(j, "kind", rec.kind)) != "") return err;
    if((err = require_string(j, "question", rec.question)) != "") return err;
    if((err = require_string(j, "source", rec.source)) != "") return err;
    if((err = require_string(j, "split", rec.split)) != "") return err;

    // Validate kind
    if(rec.kind != "choice" && rec.kind != "noul" && rec.kind != "score") {
        return "unsupported kind: " + rec.kind;
    }

    // options
    {
        auto it = j.find("options");
        if(it == j.end() || !it->is_array()) return "missing or non-array field: options";
        rec.options.clear();
        for(const auto& v : *it) {
            if(!v.is_string()) return "options contains non-string element";
            rec.options.push_back(v.get<std::string>());
        }
        if(static_cast<size_t>(PromptStrategy::MAX_CANDIDATES)<rec.options.size()){
            return "options length is too long";
        }
    }

    // target
    {
        auto it = j.find("target");
        if(it == j.end() || !it->is_array()) return "missing or non-array field: target";
        rec.target.clear();
        for(const auto& v : *it) {
            if(!v.is_number()) return "target contains non-numeric element";
            rec.target.push_back(v.get<float>());
        }
        if(rec.target.size() != rec.options.size()) {
            return "target length does not match options length";
        }
    }

    // state
    {
        auto it = j.find("state");
        if(it == j.end()) return "missing field: state";
        rec.state = *it;
    }

    // Derive one-hot answer
    {
        int ones = 0, idx = -1;
        for(int i = 0; i < (int)rec.target.size(); ++i) {
            if(rec.target[i] == 1.0f) { ++ones; idx = i; }
            else if(rec.target[i] != 0.0f) { ones = -1; break; }
        }
        if(ones == 1) {
            rec.answer_index = idx;
            rec.answer_text  = rec.options[(size_t)idx];
        } else {
            rec.answer_index = -1;
            rec.answer_text.clear();
        }
    }

    // Score-specific
    if(rec.kind == "score") {
        rec.number_of_levels = (int)rec.options.size();
    } else {
        rec.number_of_levels = 0;
    }

    // metadata (all optional)
    {
        auto it = j.find("metadata");
        if(it != j.end() && it->is_object()) {
            rec.metadata = *it;
            auto get_str = [&](const char* k, std::string& out) {
                auto f = it->find(k);
                if(f != it->end() && f->is_string()) out = f->get<std::string>();
            };
            get_str("language",               rec.language);
            get_str("domain",                 rec.domain);
            get_str("family",                 rec.family);
            get_str("semantic_context_sha256",rec.semantic_context_sha256);

            auto sv = it->find("score_values");
            if(sv != it->end() && sv->is_array()) {
                rec.score_values.clear();
                for(const auto& v : *sv) {
                    if(v.is_number_integer()) {
                        rec.score_values.push_back(v.get<int>());
                    }
                }
            }

            auto cf = it->find("counterfactual");
            if(cf != it->end()) rec.counterfactual = *cf;
        }
    }

    return {};
}

bool is_eligible(const ExemplarRecord& rec)
{
    //return rec.split == "train" && rec.language == "en";
    return rec.language == "en";
}

} // namespace filldb
} // namespace pjev
