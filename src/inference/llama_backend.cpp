#include "llama_backend.h"
#include "llama.h"
#include <stdexcept>
#include <cstdio>
#include <cstring>

struct LlamaBackend::Impl {
    llama_model*       model  = nullptr;
    llama_context*     ctx    = nullptr;
    const llama_vocab* vocab  = nullptr;
    int                n_vocab = 0;
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
    cp.n_threads_batch = cfg.n_threads;
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

std::vector<int> LlamaBackend::tokenize(const std::string& text,
                                         bool add_special,
                                         bool parse_special) const {
    const llama_vocab* v = impl_->vocab;
    int n = -llama_tokenize(v, text.c_str(), (int)text.size(),
                            nullptr, 0, add_special, parse_special);
    if (n <= 0) return {};
    std::vector<llama_token> buf(n);
    llama_tokenize(v, text.c_str(), (int)text.size(),
                   buf.data(), n, add_special, parse_special);
    return std::vector<int>(buf.begin(), buf.end());
}

int LlamaBackend::vocab_size() const { return impl_->n_vocab; }

std::string LlamaBackend::token_to_piece(int token_id) const {
    char buf[256];
    int n = llama_token_to_piece(impl_->vocab,
                                  static_cast<llama_token>(token_id),
                                  buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : "";
}

const float* LlamaBackend::eval_tokens(const std::vector<int>& tokens) {
    if (tokens.empty()) throw std::runtime_error("empty token sequence");
    if ((int)tokens.size() >= llama_n_ctx(impl_->ctx)) {
        throw std::runtime_error("prompt too long (" + std::to_string(tokens.size()) + " tokens)");
    }
    llama_memory_clear(llama_get_memory(impl_->ctx), true);
    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i]     = static_cast<llama_token>(tokens[i]);
        batch.pos[i]       = (llama_pos)i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i + 1 == tokens.size()) ? 1 : 0;
    }
    batch.n_tokens = (int)tokens.size();
    int rc = llama_decode(impl_->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
    return llama_get_logits_ith(impl_->ctx, -1);
}
