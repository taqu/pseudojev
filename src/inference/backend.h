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
};
} // namespace pjev
#endif //INC_PJEV_BACKEND_H_
