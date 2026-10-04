#ifndef INC_PJEV_CONFIG_H_
#define INC_PJEV_CONFIG_H_
#include "../calibration/calibration.h"
#include "../decision/prompt_strategy.h"
#include <string>

namespace pjev
{
enum class OptionOrder
{
    ORIGINAL,
    REVERSED,
    RANDOM
};

struct ExperimentConfig
{
    std::string name;
    Layout layout = Layout::AUTO;
    Scheme scheme = Scheme::NATURAL;
    OptionOrder option_order = OptionOrder::ORIGINAL;
    int32_t random_seed = 42;
    bool prior_correction = false;
    CalibrationConfig calibration;
    bool collect_corrected_logits = false; // populate corrected_logits in ItemResult

    std::string layout_str() const
    {
        switch(layout) {
        case Layout::STATE_FIRST:
            return "state-first";
        case Layout::STATE_LAST:
            return "state-last";
        case Layout::QUESTION_FIRST:
            return "question-first";
        default:
            return "auto";
        }
    }
    std::string scheme_str() const
    {
        return scheme == Scheme::LETTERS ? "letters" : "natural";
    }
    std::string order_str() const
    {
        switch(option_order) {
        case OptionOrder::REVERSED:
            return "reversed";
        case OptionOrder::RANDOM:
            return "random";
        default:
            return "original";
        }
    }
};
} // namespace pjev
#endif // INC_PJEV_CONFIG_H_
