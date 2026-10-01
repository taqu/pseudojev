#pragma once
#include "../decision/prompt_strategy.h"
#include <string>

enum class OptionOrder { ORIGINAL, REVERSED, RANDOM };

struct ExperimentConfig {
    std::string name;
    Layout      layout           = Layout::AUTO;
    Scheme      scheme           = Scheme::NATURAL;
    OptionOrder option_order     = OptionOrder::ORIGINAL;
    int         random_seed      = 42;
    bool        prior_correction = false;

    std::string layout_str() const {
        switch (layout) {
            case Layout::STATE_FIRST:    return "state-first";
            case Layout::STATE_LAST:     return "state-last";
            case Layout::QUESTION_FIRST: return "question-first";
            default:                     return "auto";
        }
    }
    std::string scheme_str() const { return scheme == Scheme::LETTERS ? "letters" : "natural"; }
    std::string order_str() const {
        switch (option_order) {
            case OptionOrder::REVERSED: return "reversed";
            case OptionOrder::RANDOM:   return "random";
            default:                    return "original";
        }
    }
};
