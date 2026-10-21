#ifndef INC_PJEV_BACKEND_H_
#define INC_PJEV_BACKEND_H_
#include <string>
#include <vector>

namespace pjev
{
class ILlamaBackend
{
public:
    virtual ~ILlamaBackend() = default;

    // add_special: prepend BOS if applicable.
    // parse_special: treat special-token strings (e.g. <|im_end|>) as tokens.
    //                Use false for user-controlled text to prevent injection.
    virtual std::vector<int> tokenize(const std::string& text,
                                      bool add_special,
                                      bool parse_special) const = 0;
    virtual int vocab_size() const = 0;
    virtual std::string token_to_piece(int token_id) const = 0;

    // Evaluate token sequence; returns logits for last token (size = vocab_size()).
    // Pointer is valid until the next call. Throws on failure.
    virtual const float* eval_tokens(const std::vector<int>& tokens) = 0;

    // Evaluate tokens reusing the first prefix_len positions of the KV cache
    // from the most recent eval_tokens() or eval_tokens_reuse_prefix() call.
    // tokens is the FULL sequence; only tokens[prefix_len..] are newly evaluated.
    // Default: falls back to eval_tokens (safe for mocks/tests).
    virtual const float* eval_tokens_reuse_prefix(
        const std::vector<int>& tokens, int32_t prefix_len) {
        (void)prefix_len;
        return eval_tokens(tokens);
    }

    // Optional model metadata — default implementations return empty / zero.
    // Real backends override these; mock backends in tests use the defaults.
    virtual std::string model_meta_val(const std::string& /*key*/) const { return ""; }
    virtual std::string model_desc()    const { return ""; }
    virtual int64_t     model_n_params() const { return 0; }
};
} // namespace pjev
#endif //INC_PJEV_BACKEND_H_
