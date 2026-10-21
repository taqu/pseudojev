#include "router_runtime.h"

#include <filesystem>

#include "embedding/embedding.h"
#include "inference/llama_backend.h"
#include "inference/llama_instance.h"
#include "retrieval/exemplar_store.h"
#include "task_router.h"

namespace pjev
{

namespace
{
class EmbeddingEncoder : public routing::ITextEmbedder
{
public:
    explicit EmbeddingEncoder(Embedding& e) : e_(e) {}
    bool encode(const std::string& text, std::vector<float>& out) override { return e_.encode(out, text); }
    int  dimension() const override { return e_.dimension(); }

private:
    Embedding& e_;
};

bool require_file(const std::string& what, const std::string& path, std::string& err)
{
    std::error_code ec;
    if (path.empty()) {
        err = what + " path is not set";
        return false;
    }
    if (!std::filesystem::is_regular_file(path, ec)) {
        err = what + " not found: " + path;
        return false;
    }
    return true;
}
} // namespace

// Member order is the initialisation order: the llama backend is initialised before any model is
// loaded and freed after all of them; the embedding model is loaded before the inference model, as in
// the validated R4 runs.
struct RouterRuntime::Impl {
    LlamaInstance                          llama;
    retrieval::ExemplarStore               store;
    std::unique_ptr<Embedding>             embedding;
    std::unique_ptr<EmbeddingEncoder>      encoder;
    std::unique_ptr<LlamaBackend>          backend;
    std::unique_ptr<DecisionEngine>        engine;
    std::unique_ptr<routing::TaskRouter>   router;
};

RouterRuntime::RouterRuntime() : impl_(std::make_unique<Impl>()) {}
RouterRuntime::~RouterRuntime() = default;

DecisionService& RouterRuntime::service()
{
    return *impl_->router;
}

std::unique_ptr<RouterRuntime> RouterRuntime::load(const RuntimeOptions& opts, std::string& err)
{
    const RuntimePaths& p = opts.paths;
    if (!require_file("inference model", p.model, err) ||
        !require_file("embedding model", p.embedding_model, err) ||
        !require_file("exemplar database", p.exemplar_db, err) ||
        !require_file("exemplar allowlist", p.exemplar_allowlist, err)) {
        return nullptr;
    }

    std::unique_ptr<RouterRuntime> rt(new RouterRuntime());
    Impl& m = *rt->impl_;

    if (!m.store.open(p.exemplar_db, p.exemplar_allowlist, err)) return nullptr;
    rt->exemplar_count_     = m.store.example_count();
    rt->eligible_exemplars_ = m.store.allowlist_size();

    try {
        EmbeddingConfig emb_cfg;
        emb_cfg.model_path = p.embedding_model;
        emb_cfg.verbose    = opts.verbose;
        m.embedding = std::make_unique<Embedding>(emb_cfg);
    } catch (const std::exception& e) {
        err = std::string("embedding model load failed: ") + e.what();
        return nullptr;
    }
    m.encoder = std::make_unique<EmbeddingEncoder>(*m.embedding);
    const std::string emb_sha = sha256_file(p.embedding_model);
    rt->embedding_fp_ = emb_sha.size() >= 16 ? emb_sha.substr(0, 16) : "";
    if (!m.store.check_compatible(m.embedding->dimension(), rt->embedding_fp_, err)) return nullptr;

    LlamaConfig llama_cfg;
    llama_cfg.model_path = p.model;
    llama_cfg.n_ctx = llama_cfg.n_batch = opts.n_ctx;
    if (opts.n_threads > 0) llama_cfg.n_threads = opts.n_threads;
    llama_cfg.n_threads_batch = opts.n_threads_batch;
    llama_cfg.n_gpu_layers = opts.n_gpu_layers;
    llama_cfg.main_gpu     = opts.main_gpu;
    llama_cfg.verbose      = opts.verbose;
    try {
        m.backend = std::make_unique<LlamaBackend>(llama_cfg);
    } catch (const std::exception& e) {
        err = std::string("inference model load failed: ") + e.what();
        return nullptr;
    }
    rt->model_identity_ = ModelIdentity::from_path(p.model, m.backend.get());

    // Validated decision configuration: letter candidates, automatic layout, one candidate order,
    // no calibration, no prior correction.
    PromptConfig prompt_cfg;
    prompt_cfg.layout = Layout::AUTO;
    prompt_cfg.scheme = Scheme::LETTERS;
    EnsembleConfig ensemble;
    ensemble.noul_mode   = NoulEnsembleMode::NONE;
    ensemble.choice_mode = ChoiceEnsembleMode::NONE;
    ensemble.score_mode  = ScoreEnsembleMode::NONE;
    try {
        m.engine = std::make_unique<DecisionEngine>(*m.backend, prompt_cfg, CalibrationConfig{}, ensemble);
    } catch (const std::exception& e) {
        err = std::string("decision engine init failed: ") + e.what();
        return nullptr;
    }
    m.router = std::make_unique<routing::TaskRouter>(*m.engine, *m.encoder, m.store);
    return rt;
}

} // namespace pjev
