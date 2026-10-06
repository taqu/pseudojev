#include "llama_backend.h"

#include <stdexcept>
#include <cstdio>
#include <cstring>

#include "llama.h"
#include "util/platform.h"

namespace pjev
{
LlamaConfig::LlamaConfig()
    : n_ctx(4096)
    , n_batch(4096)
    , n_threads(8)
    , n_threads_batch(-1)
    , verbose(false)
{
    n_threads = get_physical_core_count();
}

struct LlamaBackend::Impl {
    llama_model*       model  = nullptr;
    llama_context*     ctx    = nullptr;
    const llama_vocab* vocab  = nullptr;
    int32_t                n_vocab = 0;
};

LlamaBackend::LlamaBackend(const LlamaConfig& cfg) : impl_(new Impl) {
    if (!cfg.verbose) {
        llama_log_set([](ggml_log_level lvl, const char* text, void*) {
            if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
        }, nullptr);
    }
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    impl_->model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
    if (!impl_->model) {
        llama_backend_free();
        delete impl_;
        impl_ = nullptr;
        throw std::runtime_error("failed to load model: " + cfg.model_path);
    }

    auto cp            = llama_context_default_params();
    cp.n_ctx           = (uint32_t)cfg.n_ctx;
    cp.n_batch         = (uint32_t)cfg.n_batch;
    cp.n_threads       = cfg.n_threads;
    cp.n_threads_batch = (cfg.n_threads_batch < 0) ? cfg.n_threads : cfg.n_threads_batch;
    cp.n_outputs_max = 4;
    cp.n_outputs_max_per_seq = 4;
    cp.no_perf         = true;
    impl_->ctx = llama_init_from_model(impl_->model, cp);
    if (!impl_->ctx) {
        llama_model_free(impl_->model);
        llama_backend_free();
        delete impl_;
        impl_ = nullptr;
        throw std::runtime_error("failed to create llama context");
    }
    impl_->vocab   = llama_model_get_vocab(impl_->model);
    impl_->n_vocab = llama_vocab_n_tokens(impl_->vocab);
}

LlamaBackend::~LlamaBackend() {
    if (!impl_) return;
    if (impl_->ctx)   llama_free(impl_->ctx);
    if (impl_->model) llama_model_free(impl_->model);
    llama_backend_free();
    delete impl_;
}

std::vector<int32_t> LlamaBackend::tokenize(const std::string& text,
                                         bool add_special,
                                         bool parse_special) const {
    const llama_vocab* v = impl_->vocab;
    int32_t n = -llama_tokenize(v, text.c_str(), (int32_t)text.size(),
                            nullptr, 0, add_special, parse_special);
    if (n <= 0) return {};
    std::vector<llama_token> buf(n);
    llama_tokenize(v, text.c_str(), (int32_t)text.size(),
                   buf.data(), n, add_special, parse_special);
    return std::vector<int32_t>(buf.begin(), buf.end());
}

int32_t LlamaBackend::vocab_size() const { return impl_->n_vocab; }

std::string LlamaBackend::token_to_piece(int32_t token_id) const {
    char buf[256];
    int32_t n = llama_token_to_piece(impl_->vocab,
                                  static_cast<llama_token>(token_id),
                                  buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : "";
}

std::string LlamaBackend::model_meta_val(const std::string& key) const {
    if (!impl_ || !impl_->model) return "";
    char buf[512];
    int32_t n = llama_model_meta_val_str(impl_->model, key.c_str(), buf, sizeof(buf));
    if (n > 0) return std::string(buf, (size_t)n);
    return "";
}

std::string LlamaBackend::model_desc() const {
    if (!impl_ || !impl_->model) return "";
    char buf[256];
    int32_t n = llama_model_desc(impl_->model, buf, sizeof(buf));
    if (n > 0) return std::string(buf);
    return "";
}

int64_t LlamaBackend::model_n_params() const {
    if (!impl_ || !impl_->model) return 0;
    char buf[64];
    int32_t n = llama_model_meta_val_str(impl_->model, "general.parameter_count", buf, sizeof(buf));
    if (n > 0) return (int64_t)std::atoll(buf);
    return 0;
}

const float* LlamaBackend::eval_tokens(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::runtime_error("empty token sequence");
    if ((int32_t)tokens.size() >= llama_n_ctx(impl_->ctx)) {
        throw std::runtime_error("prompt too long (" + std::to_string(tokens.size()) + " tokens)");
    }
    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i]     = static_cast<llama_token>(tokens[i]);
        batch.pos[i]       = (llama_pos)i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i + 1 == tokens.size()) ? 1 : 0;
    }
    batch.n_tokens = (int32_t)tokens.size();
    int32_t rc = llama_decode(impl_->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
    return llama_get_logits_ith(impl_->ctx, -1);
}

const float* LlamaBackend::eval_tokens_reuse_prefix(
    const std::vector<int32_t>& tokens, int32_t prefix_len)
{
    if (tokens.empty()) throw std::runtime_error("empty token sequence");
    if ((int32_t)tokens.size() >= llama_n_ctx(impl_->ctx)) {
        throw std::runtime_error("prompt too long (" + std::to_string(tokens.size()) + " tokens)");
    }
    if (prefix_len <= 0 || prefix_len >= (int32_t)tokens.size()) {
        return eval_tokens(tokens);
    }
    llama_memory_t mem = llama_get_memory(impl_->ctx);
    if (!llama_memory_seq_rm(mem, 0, (llama_pos)prefix_len, -1)) {
        return eval_tokens(tokens);
    }
    int32_t suffix_len = (int32_t)tokens.size() - prefix_len;
    llama_batch batch = llama_batch_init(suffix_len, 0, 1);
    for (int32_t i = 0; i < suffix_len; i++) {
        batch.token[i]     = static_cast<llama_token>(tokens[prefix_len + i]);
        batch.pos[i]       = (llama_pos)(prefix_len + i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i + 1 == suffix_len) ? 1 : 0;
    }
    batch.n_tokens = suffix_len;
    int32_t rc = llama_decode(impl_->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed (suffix): " + std::to_string(rc));
    return llama_get_logits_ith(impl_->ctx, -1);
}
}

