#include "embedding_text.h"

#include <sstream>
#include <unordered_set>
#include <vector>

namespace pjev
{
namespace filldb
{

// Generic value formatter — handles strings, numbers, booleans, arrays, objects.
static void format_value(std::ostringstream& out, const nlohmann::ordered_json& v, int depth = 0)
{
    if(v.is_string()) {
        out << v.get<std::string>();
    } else if(v.is_boolean()) {
        out << (v.get<bool>() ? "true" : "false");
    } else if(v.is_number_integer()) {
        out << v.get<int64_t>();
    } else if(v.is_number_float()) {
        out << v.get<double>();
    } else if(v.is_null()) {
        out << "null";
    } else if(v.is_array()) {
        bool first = true;
        for(const auto& elem : v) {
            if(!first) out << ", ";
            format_value(out, elem, depth + 1);
            first = false;
        }
    } else if(v.is_object()) {
        std::string indent(depth * 2, ' ');
        for(auto it = v.begin(); it != v.end(); ++it) {
            out << "\n" << indent << it.key() << ": ";
            format_value(out, it.value(), depth + 1);
        }
    }
}

// Trim trailing whitespace.
static std::string trim(std::string s)
{
    auto pos = s.find_last_not_of(" \t\r\n");
    if(pos != std::string::npos) s.erase(pos + 1);
    return s;
}

std::string build_embedding_text(
    const std::string&              kind,
    const std::string&              domain,
    const std::string&              family,
    const std::string&              question,
    const nlohmann::ordered_json&   state,
    const std::vector<std::string>& options)
{
    std::ostringstream out;

    // Header
    out << "Task: " << kind << "\n";
    if(!domain.empty()) out << "Domain: " << domain << "\n";
    if(!family.empty()) out << "Family: " << family << "\n";

    // Known semantic state fields
    bool has_sections = false;

    auto emit_section = [&](const char* heading, const std::string& text) {
        if(text.empty()) return;
        out << "\n" << heading << "\n" << text;
        has_sections = true;
    };

    // policy_name → Scenario
    if(state.contains("policy_name") && state["policy_name"].is_string()) {
        emit_section("Scenario:", state["policy_name"].get<std::string>());
    }

    // question
    out << "\n\nQuestion:\n" << question;

    // evidence_policy
    if(state.contains("evidence_policy") && state["evidence_policy"].is_string()) {
        emit_section("\n\nEvidence policy:", state["evidence_policy"].get<std::string>());
    }

    // numbered_requirements
    if(state.contains("numbered_requirements") && state["numbered_requirements"].is_array()) {
        out << "\n\nRequirements:";
        int n = 1;
        for(const auto& req : state["numbered_requirements"]) {
            out << "\n" << n++ << ". ";
            if(req.is_string()) out << req.get<std::string>();
            else { format_value(out, req); }
        }
        has_sections = true;
    }

    // selection_rule
    if(state.contains("selection_rule") && state["selection_rule"].is_string()) {
        emit_section("\n\nSelection rule:", state["selection_rule"].get<std::string>());
    }

    // request_parameters (generic key-value block)
    if(state.contains("request_parameters") && state["request_parameters"].is_object()) {
        out << "\n\nRequest parameters:";
        const auto& rp = state["request_parameters"];
        for(auto it = rp.begin(); it != rp.end(); ++it) {
            out << "\n" << it.key() << ": ";
            format_value(out, it.value());
        }
        has_sections = true;
    }

    // When state is a plain string (e.g. a conversation transcript), emit it as Context.
    if(state.is_string()) {
        emit_section("\n\nContext:", state.get<std::string>());
    }

    // Any other state fields not handled above (generic formatter, skip candidates)
    static const std::unordered_set<std::string> kKnown{
        "policy_name", "evidence_policy", "numbered_requirements",
        "selection_rule", "request_parameters",
        // candidates: deliberately omitted — structure-oriented policy
        "candidates", "candidate_facts"};
    if(state.is_object()) {
        for(auto it = state.begin(); it != state.end(); ++it) {
            if(kKnown.count(it.key())) continue;
            out << "\n\n" << it.key() << ":";
            format_value(out, it.value());
            has_sections = true;
        }
    }

    // Score meanings (score kind only)
    if(kind == "score" && !options.empty()) {
        out << "\n\nScore meanings:";
        for(int i = 0; i < (int)options.size(); ++i) {
            out << "\n" << i << ": " << options[i];
        }
    }

    return trim(out.str());
}

std::string build_document_embedding_text(
    const std::string&              kind,
    const std::string&              domain,
    const std::string&              family,
    const std::string&              question,
    const nlohmann::ordered_json&   state,
    const std::vector<std::string>& options)
{
    return std::string(kDocumentPrefix) +
           build_embedding_text(kind, domain, family, question, state, options);
}

} // namespace filldb
} // namespace pjev
