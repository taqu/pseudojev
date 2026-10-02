#ifndef INC_PJEV_PROMPT_STRATEGY_H_
#define INC_PJEV_PROMPT_STRATEGY_H_
#include <cstdint>
#include <string>
#include <vector>

namespace pjev
{
enum class Layout { AUTO, STATE_FIRST, STATE_LAST, QUESTION_FIRST };
enum class Scheme { NATURAL, LETTERS };

struct PromptConfig {
    Layout layout = Layout::AUTO;
    Scheme scheme = Scheme::NATURAL;
};

struct Candidate {
    std::string key;          // semantic key ("billing", "true", "2", ...)
    std::string description;  // text shown to model
    std::string internal;     // internal token text ("A", "Yes", "0", ...)
};

// A segment of the final prompt.
// trusted=true: parse special tokens (template delimiters).
// trusted=false: treat as plain text (user-controlled content).
struct PromptSegment {
    std::string text;
    bool trusted;
};

class PromptStrategy {
public:
    static constexpr int32_t MAX_CANDIDATES = 10;

    explicit PromptStrategy(const PromptConfig& cfg = {});

    // Assign internal labels to candidates in-place.
    void assign_labels(const std::string& type,
                       std::vector<Candidate>& candidates) const;

    // All possible internal texts for the scheme (for token verification at startup).
    std::vector<std::string> all_internal_texts() const;

    // Build the prompt as ordered segments. Caller tokenizes each with the
    // appropriate parse_special flag to prevent special-token injection from
    // user-controlled text.
    //
    // Segments are:
    //   [0] trusted  = "<|im_start|>user\n"
    //   [1] untrusted = full user content (our framing + user state/question/descriptions)
    //   [2] trusted  = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    std::vector<PromptSegment> build_prompt_segments(
        const std::string& type,
        const std::string& state,
        const std::string& question,
        const std::vector<Candidate>& candidates) const;

private:
    PromptConfig cfg_;

    bool use_state_last(const std::string& type) const;

    // Build the user-visible content string (no chat template wrapping).
    std::string build_user_content(
        const std::string& type,
        const std::string& state,
        const std::string& question,
        const std::vector<Candidate>& candidates) const;
};
}
#endif //INC_PJEV_PROMPT_STRATEGY_H_
