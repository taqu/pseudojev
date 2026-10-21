#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"

namespace pjev
{

struct EmbeddingConfig {
    std::string model_path;
    int32_t n_ctx     = 2048;
    int32_t n_threads = 0;
    bool    verbose   = false;
    EmbeddingConfig();
};

class Embedding
{
public:
    explicit Embedding(const EmbeddingConfig& config);
    ~Embedding();

    Embedding(const Embedding&)            = delete;
    Embedding& operator=(const Embedding&) = delete;

    bool    encode(std::vector<float>& embedding, const std::string& text);
    int32_t dimension() const;

private:
    static constexpr int32_t MaxTokens = 8192;

    llama_model*       model_   = nullptr;
    llama_context*     context_ = nullptr;
    const llama_vocab* vocab_   = nullptr;
    int32_t            n_vocab_ = 0;
    llama_token        tokens_[MaxTokens]{};
    llama_batch        batch_{};
};

} // namespace pjev
