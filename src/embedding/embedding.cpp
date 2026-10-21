#include "embedding.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "spdlog/spdlog.h"
#include "util/platform.h"

namespace pjev
{

EmbeddingConfig::EmbeddingConfig()
    : n_ctx(2048)
    , n_threads(0)
    , verbose(false)
{
    n_threads = get_physical_core_count();
}

Embedding::Embedding(const EmbeddingConfig& config)
{
    // Do not override llama_log_set — the default callback is required
    // for correct llama.cpp operation; custom callbacks can corrupt internal state.

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = 99;
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if(!model_) {
        throw std::runtime_error("failed to load model: " + config.model_path);
    }

    llama_context_params ctx_params   = llama_context_default_params();
    ctx_params.n_ctx                  = (uint32_t)config.n_ctx;
    ctx_params.n_batch                = (uint32_t)config.n_ctx;
    ctx_params.n_ubatch               = (uint32_t)config.n_ctx;
    ctx_params.n_threads              = config.n_threads;
    ctx_params.n_threads_batch        = (ctx_params.n_threads_batch < 0)
                                          ? config.n_threads
                                          : ctx_params.n_threads_batch;
    ctx_params.n_outputs_max          = 0; // 0 == n_batch; all positions needed for mean pooling
    ctx_params.n_outputs_max_per_seq  = 0;
    ctx_params.no_perf                = true;
    ctx_params.embeddings             = true;
    ctx_params.pooling_type           = LLAMA_POOLING_TYPE_MEAN;

    context_ = llama_init_from_model(model_, ctx_params);
    if(!context_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("failed to create embedding context");
    }

    vocab_   = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);
    batch_   = llama_batch_init(MaxTokens, 0, 1);
}

Embedding::~Embedding()
{
    llama_batch_free(batch_);
    llama_free(context_);
    context_ = nullptr;
    llama_model_free(model_);
    model_   = nullptr;
    vocab_   = nullptr;
    n_vocab_ = 0;
}

bool Embedding::encode(std::vector<float>& embedding, const std::string& text)
{
    assert(nullptr != model_);
    assert(nullptr != context_);
    assert(nullptr != vocab_);

    int32_t n_tokens = llama_tokenize(
        vocab_, text.c_str(), (int32_t)text.size(), tokens_, MaxTokens, false, false);
    if(n_tokens <= 0) {
        return false;
    }

    if((uint32_t)n_tokens > llama_n_ctx(context_) ||
       (uint32_t)n_tokens > llama_n_batch(context_)) {
        return false;
    }

    if(llama_pooling_type(context_) == LLAMA_POOLING_TYPE_NONE) {
        return false;
    }

    batch_.n_tokens = n_tokens;
    for(int32_t i = 0; i < n_tokens; ++i) {
        batch_.token[i]    = tokens_[i];
        batch_.pos[i]      = i;
        batch_.n_seq_id[i] = 1;
        batch_.seq_id[i][0] = 0;
        batch_.logits[i]   = 1;
    }

    llama_memory_clear(llama_get_memory(context_), true);

    const llama_model* model = llama_get_model(context_);
    int32_t result;
    // This context is always configured with embeddings=true and mean pooling,
    // so llama_encode is always correct here regardless of architecture flag.
    result = llama_encode(context_, batch_);
    if(result != 0) {
        return false;
    }

    const float* data = llama_get_embeddings_seq(context_, 0);
    if(!data) {
        return false;
    }

    const int32_t n_embd = llama_model_n_embd_out(model);
    if(n_embd <= 0) {
        return false;
    }

    embedding.assign(data, data + n_embd);

    // L2 normalisation — nomic-embed-text uses cosine similarity
    double norm2 = 0.0;
    for(float v : embedding) {
        norm2 += (double)v * v;
    }
    if(norm2 <= 0.0) {
        embedding.clear();
        return false;
    }
    const float scale = 1.0f / (float)std::sqrt(norm2);
    for(float& v : embedding) {
        v *= scale;
    }

    return true;
}

int32_t Embedding::dimension() const
{
    return model_ ? llama_model_n_embd_out(model_) : 0;
}

} // namespace pjev
