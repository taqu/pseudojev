#ifndef INC_PJEV_LLAMA_BACKEND_H_
#define INC_PJEV_LLAMA_BACKEND_H_
#include "backend.h"
#include <string>

namespace pjev
{
struct LlamaConfig {
    std::string model_path;
    int n_ctx     = 4096;
    int n_batch   = 4096;
    int n_threads = 8;
    bool verbose  = false;
};

class LlamaBackend : public ILlamaBackend {
public:
    explicit LlamaBackend(const LlamaConfig& cfg);
    ~LlamaBackend() override;

    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    std::vector<int> tokenize(const std::string& text,
                              bool add_special,
                              bool parse_special) const override;
    int vocab_size() const override;
    std::string token_to_piece(int token_id) const override;
    const float* eval_tokens(const std::vector<int>& tokens) override;

private:
    struct Impl;
    Impl* impl_;
};
}
#endif //INC_PJEV_LLAMA_BACKEND_H_

