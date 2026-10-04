#include "prompt_strategy.h"
#include <cassert>

namespace pjev
{
PromptStrategy::PromptStrategy(const PromptConfig& cfg) : cfg_(cfg) {}

bool PromptStrategy::use_state_last(const std::string& type) const {
    return cfg_.layout == Layout::STATE_LAST ||
           (cfg_.layout == Layout::AUTO && type == "score");
    // Note: QUESTION_FIRST is handled separately in build_user_content.
}

void PromptStrategy::assign_labels(const std::string& type,
                                   std::vector<Candidate>& candidates) const {
    for (size_t i = 0; i < candidates.size(); i++) {
        std::string s = std::string(1, (char)('A' + i));
        if (cfg_.scheme == Scheme::NATURAL && type == "noul") {
            s = (i == 0) ? "No" : "Yes";
        } else if (cfg_.scheme == Scheme::NATURAL && type == "score") {
            s = std::to_string(i);
        }
        candidates[i].internal = s;
    }
}

std::vector<std::string> PromptStrategy::all_internal_texts() const {
    std::vector<std::string> v;
    for (int32_t i = 0; i < MAX_CANDIDATES; i++) v.push_back(std::string(1, (char)('A' + i)));
    if (cfg_.scheme == Scheme::NATURAL) {
        for (int32_t i = 0; i < MAX_CANDIDATES; i++) v.push_back(std::to_string(i));
        v.push_back("No");
        v.push_back("Yes");
    }
    return v;
}

std::string PromptStrategy::build_user_content(
    const std::string& type,
    const std::string& state,
    const std::string& question,
    const std::vector<Candidate>& candidates) const
{
    // Build the answer token list
    std::string answers;
    for (size_t i = 0; i < candidates.size(); i++) {
        answers += (i ? ", " : "") + candidates[i].internal;
    }

    // Build options block
    std::string opts = (type == "score")
        ? "Possible answers (ordered from lowest to highest):\n"
        : "Possible answers:\n";
    for (size_t i = 0; i < candidates.size(); i++) {
        const auto& c = candidates[i];
        std::string desc = c.description;
        if (type == "noul") {
            const std::string word = (i == 0) ? "No" : "Yes";
            if (c.internal != word) {
                desc = desc.empty() ? word : word + ". " + desc;
            }
        }
        opts += c.internal + (desc.empty() ? "" : ": " + desc) + "\n";
    }

    std::string p = "You are performing a classification task.\n\n";
    // QUESTION_FIRST: question, options, state
    if (cfg_.layout == Layout::QUESTION_FIRST) {
        p += "Question:\n" + question + "\n\n" + opts + "\nState:\n" + state + "\n";
    } else if (use_state_last(type)) {
        p += opts + "\nQuestion:\n" + question + "\n\nState:\n" + state + "\n";
    } else {
        p += "State:\n" + state + "\n\nQuestion:\n" + question + "\n\n" + opts;
    }
    p += "\nReply with exactly one of " + answers + " and nothing else.";
    return p;
}

PromptStrategy::PrefixInfo PromptStrategy::build_prefix_info(
    const std::string& type, const std::string& state) const
{
    PrefixInfo info;
    if (use_state_last(type) || cfg_.layout == Layout::QUESTION_FIRST) {
        return info;
    }
    info.seg0_text      = "<|im_start|>user\n";
    info.partial_content = "You are performing a classification task.\n\nState:\n" + state + "\n\n";
    info.valid = true;
    return info;
}

std::vector<PromptSegment> PromptStrategy::build_prompt_segments(
    const std::string& type,
    const std::string& state,
    const std::string& question,
    const std::vector<Candidate>& candidates) const
{
    std::string user_content = build_user_content(type, state, question, candidates);
    std::vector<PromptSegment> segs;
    // Chat template prefix — trusted, contains special tokens
    PromptSegment seg0;
    seg0.text    = "<|im_start|>user\n";
    seg0.trusted = true;
    segs.push_back(seg0);
    // User content — untrusted, never parse special tokens here
    PromptSegment seg1;
    seg1.text    = std::move(user_content);
    seg1.trusted = false;
    segs.push_back(seg1);
    // Chat template suffix — trusted, contains special tokens
    PromptSegment seg2;
    seg2.text    = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    seg2.trusted = true;
    segs.push_back(seg2);
    return segs;
}
}

