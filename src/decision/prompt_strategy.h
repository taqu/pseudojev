#ifndef INC_PJEV_PROMPT_STRATEGY_H_
#define INC_PJEV_PROMPT_STRATEGY_H_
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "server/jev_api.h"

namespace pjev
{
enum class Layout { AUTO, STATE_FIRST, STATE_LAST, QUESTION_FIRST };
enum class Scheme { NATURAL, LETTERS };

// Semantic truth value for a noul candidate — set by assign_labels(), used by
// finish_from_logits() to locate p_true without relying on candidate index.
enum class NoulValue { False, True };

struct PromptConfig {
    Layout layout = Layout::AUTO;
    Scheme scheme = Scheme::NATURAL;
    // Optional fixed filler inserted at the end of user content, immediately before ANSWER_ANCHOR.
    // Empty string = no filler (F0 behaviour).
    std::string filler_text;
};

struct Candidate {
    std::string key;          // semantic key ("billing", "true", "2", ...)
    std::string description;  // text shown to model
    std::string internal;     // internal token text ("A", "Yes", "0", ...)
    std::optional<NoulValue> noul_value; // set only for noul type
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
    void assign_labels(Type type, std::vector<Candidate>& candidates) const;

    // All possible internal texts for the scheme (for token verification at startup).
    std::vector<std::string> all_internal_texts() const;

    // Build the prompt as ordered segments. Caller tokenizes each with the
    // appropriate parse_special flag to prevent special-token injection from
    // user-controlled text.
    //
    // Segments are:
    //   [0] trusted  = "<|im_start|>user\n"
    //   [1] untrusted = full user content (our framing + user state/question/descriptions)
    //   [2] trusted  = ANSWER_ANCHOR
    // Candidate logits are read at the position right after ANSWER_ANCHOR.
    static constexpr const char* ANSWER_ANCHOR = "<|im_end|>\n<|im_start|>assistant\n\n";

    std::vector<PromptSegment> build_prompt_segments(
        Type type,
        const std::string& state,
        const std::string& question,
        const std::vector<Candidate>& candidates) const;

    // Info for shared-prefix KV reuse across same-state questions.
    struct PrefixInfo {
        bool        valid = false;
        std::string seg0_text;       // tokenize with add_special=true, parse_special=true
        std::string partial_content; // tokenize with add_special=false, parse_special=false
    };
    // Returns a PrefixInfo for this (type, state) pair.
    // valid=true only when layout puts state before per-question content (STATE_FIRST for noul/choice).
    PrefixInfo build_prefix_info(Type type,
                                  const std::string& state) const;

private:
    PromptConfig cfg_;

    bool use_state_last(Type type) const;

    // Build the user-visible content string (no chat template wrapping).
    std::string build_user_content(
        Type type,
        const std::string& state,
        const std::string& question,
        const std::vector<Candidate>& candidates) const;
};
}
#endif //INC_PJEV_PROMPT_STRATEGY_H_
