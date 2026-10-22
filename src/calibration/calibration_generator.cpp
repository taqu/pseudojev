#include "calibration_generator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_map>

#include "calibration_fit.h"
#include "lz4/xxhash.h"
#include "retrieval/exemplar_allowlist.h"
#include "routing/task_mapping.h"
#include "routing/task_policy.h"
#include "server/jev_api.h"
#include "spdlog/spdlog.h"

namespace pjev
{
namespace calibration
{

namespace
{
constexpr Type kKinds[] = {Type::Choice, Type::Noul, Type::Score};

// Salts separating the three seeded orders (groups, split, row within a group).
constexpr uint64_t kSplitSalt = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t kRowSalt   = 0xc2b2ae3d27d4eb4fULL;

uint64_t seeded(int64_t v, uint64_t seed)
{
    return XXH64(&v, sizeof v, seed);
}

bool is_validation_group(int64_t group_id, const GeneratorOptions& o)
{
    const uint64_t h = seeded(group_id, o.seed ^ kSplitSalt);
    const double   u = static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
    return u < o.validation_ratio;
}

bool is_supported(Type t)
{
    return t == Type::Choice || t == Type::Noul || t == Type::Score;
}

// The strategy whose logits a kind's parameters are fitted on.
const char* calibration_strategy(Type t)
{
    switch (t) {
    case Type::Choice: return routing::kNearest2Shot;
    case Type::Noul:   return routing::kNearest1Shot;
    default:           return routing::kZeroShot;
    }
}

std::string utc_now()
{
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

nlohmann::json counts_json(const KindCounts& k)
{
    return {
        {"candidate_rows",      k.candidate_rows},
        {"candidate_groups",    k.candidate_groups},
        {"examined",            k.examined},
        {"selected_fit",        k.selected_fit},
        {"selected_validation", k.selected_validation},
        {"processed",           k.processed},
        {"accepted_fit",        k.accepted_fit},
        {"accepted_validation", k.accepted_validation},
        {"rejected",            k.rejected_total()},
        {"rejected_by_reason",  k.rejected},
    };
}

nlohmann::json comparison_json(const PrimitiveCalibrationMetrics& raw, const PrimitiveCalibrationMetrics& cal)
{
    return {
        {"uncalibrated", raw.to_json()},
        {"calibrated",   cal.to_json()},
        {"delta", {
            {"accuracy", cal.accuracy - raw.accuracy},
            {"nll",      cal.nll - raw.nll},
            {"brier",    cal.brier - raw.brier},
            {"ece",      cal.ece - raw.ece},
        }},
    };
}

PrimitiveCalibrationMetrics evaluate(const std::vector<CalibrationSample>& s, double alpha, double T)
{
    return compute_primitive_metrics(apply_prior_correction(s, alpha), T);
}

// Whether a file can be created in dest's directory (created when missing).
bool check_writable(const std::string& dest, std::string& err)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(dest).parent_path();
    if (!dir.empty()) {
        fs::create_directories(dir, ec);
        if (ec) { err = "cannot create output directory " + dir.string() + ": " + ec.message(); return false; }
    }
    const fs::path probe = fs::path(dest).concat(".probe");
    {
        std::ofstream f(probe, std::ios::binary | std::ios::trunc);
        if (!f) { err = "output " + dest + " is not writable"; return false; }
    }
    fs::remove(probe, ec);
    return true;
}

// Removes the store's held-out set when generation ends, whichever way.
struct HeldOutReset {
    retrieval::ExemplarStore& store;
    ~HeldOutReset() { store.set_held_out({}); }
};
} // namespace

bool GeneratorOptions::validate(std::string& err) const
{
    if (max_samples < 1) { err = "--max-samples must be at least 1"; return false; }
    if (!(validation_ratio > 0.0 && validation_ratio < 1.0)) {
        err = "--validation-ratio must be greater than 0 and less than 1";
        return false;
    }
    if (batch_size < 1) { err = "--batch-size must be at least 1"; return false; }
    return true;
}

const char* to_string(Split s)
{
    return s == Split::Fit ? "fit" : "validation";
}

size_t KindCounts::rejected_total() const
{
    size_t n = 0;
    for (const auto& kv : rejected) n += kv.second;
    return n;
}

CalibrationFormulation production_formulation()
{
    CalibrationFormulation f;
    f.layout           = routing::kFormulationLayout;
    f.candidate_scheme = routing::kFormulationCandidateScheme;
    f.prior_correction = true;
    return f;
}

// ---------------------------------------------------------------------------
// Partition
// ---------------------------------------------------------------------------

bool partition_rows(const std::vector<retrieval::ExemplarRow>& rows,
                    const std::function<bool(int64_t)>&       in_training_split,
                    const RowValidator&                       validate,
                    const GeneratorOptions&                   opts,
                    Partition&                                out,
                    std::string&                              err)
{
    out = Partition{};
    for (Type t : kKinds) out.kinds[t];
    out.rows_total = rows.size();

    struct Group {
        std::vector<int64_t> training_ids;
        std::map<Type, std::vector<const retrieval::ExemplarRow*>> candidates;
    };
    std::unordered_map<int64_t, Group> groups;
    for (const auto& r : rows) {
        if (!in_training_split(r.id)) { out.rows_not_training++; continue; }
        Group& g = groups[r.group_id];
        g.training_ids.push_back(r.id);
        const Type t = to_type(r.kind);
        if (!is_supported(t)) { out.rejected_other["unsupported_kind"]++; continue; }
        KindCounts& kc = out.kinds[t];
        if (!r.one_hot) { kc.rejected["no_one_hot_answer"]++; continue; }
        kc.candidate_rows++;
        g.candidates[t].push_back(&r);
    }

    std::vector<std::pair<uint64_t, int64_t>> order;
    order.reserve(groups.size());
    for (const auto& [gid, g] : groups) {
        for (const auto& kv : g.candidates) out.kinds[kv.first].candidate_groups++;
        if (!g.candidates.empty()) order.push_back({seeded(gid, opts.seed), gid});
    }
    std::sort(order.begin(), order.end());

    const size_t quota = static_cast<size_t>(opts.max_samples);
    auto full = [&](Type t) {
        const KindCounts& k = out.kinds[t];
        return k.selected_fit + k.selected_validation >= quota;
    };
    for (const auto& [h, gid] : order) {
        if (std::all_of(std::begin(kKinds), std::end(kKinds), full)) break;
        Group& g = groups[gid];
        const Split split = is_validation_group(gid, opts) ? Split::Validation : Split::Fit;
        std::vector<QueryRef> picked;
        for (Type t : kKinds) {
            auto it = g.candidates.find(t);
            if (it == g.candidates.end() || full(t)) continue;
            std::vector<const retrieval::ExemplarRow*> cands = it->second;
            std::sort(cands.begin(), cands.end(), [&](const retrieval::ExemplarRow* a, const retrieval::ExemplarRow* b) {
                const uint64_t ha = seeded(a->id, opts.seed ^ kRowSalt), hb = seeded(b->id, opts.seed ^ kRowSalt);
                return ha != hb ? ha < hb : a->id < b->id;
            });
            KindCounts& kc = out.kinds[t];
            for (const retrieval::ExemplarRow* r : cands) {
                kc.examined++;
                std::string reason;
                const retrieval::LoadResult v = validate(*r, reason);
                if (v == retrieval::LoadResult::Failed) { err = reason; return false; }
                if (v == retrieval::LoadResult::Invalid) { kc.rejected[reason]++; continue; }
                picked.push_back(QueryRef{r->id, gid, t, split, r->domain, r->family});
                break;
            }
        }
        if (picked.empty()) continue;
        // The whole protected group leaves the retrieval pool, whichever of its rows are queries.
        out.held_out_groups++;
        out.held_out.insert(g.training_ids.begin(), g.training_ids.end());
        for (auto& q : picked) {
            KindCounts& kc = out.kinds[q.type];
            (q.split == Split::Fit ? kc.selected_fit : kc.selected_validation)++;
            out.queries.push_back(std::move(q));
        }
    }
    return true;
}

bool check_sufficient(const Partition& p, bool accepted, std::string& err)
{
    std::string issues;
    for (Type t : kKinds) {
        const KindCounts& k = p.kinds.at(t);
        const size_t fit = accepted ? k.accepted_fit : k.selected_fit;
        const size_t val = accepted ? k.accepted_validation : k.selected_validation;
        if (fit < static_cast<size_t>(kMinFitSamples) || val < static_cast<size_t>(kMinValidationSamples)) {
            issues += std::string(issues.empty() ? "" : "; ") + to_string(t) + ": " + std::to_string(fit) +
                      " fitting / " + std::to_string(val) + " validation " + (accepted ? "samples" : "queries");
        }
    }
    if (issues.empty()) return true;
    err = issues + " (each task kind needs at least " + std::to_string(kMinFitSamples) + " fitting and " +
          std::to_string(kMinValidationSamples) +
          " validation samples; raise --max-samples, adjust --validation-ratio, or use a larger database)";
    return false;
}

// ---------------------------------------------------------------------------
// Queries and collection
// ---------------------------------------------------------------------------

bool make_query(const retrieval::ExemplarPayload& p, const QueryRef& q, DecisionInput& in, int& correct,
                std::string& reason)
{
    const Type t = to_type(p.kind);
    if (!is_supported(t))   { reason = "unsupported_kind"; return false; }
    if (t != q.type)        { reason = "kind_mismatch"; return false; }
    const size_t n = p.options.size();
    if (p.answer_index < 0 || p.answer_index >= static_cast<int>(n)) { reason = "no_one_hot_answer"; return false; }
    if (p.question.empty()) { reason = "missing_question"; return false; }
    if (n < 2 || n > static_cast<size_t>(PromptStrategy::MAX_CANDIDATES)) { reason = "invalid_option_count"; return false; }
    for (const auto& o : p.options)
        if (o.empty()) { reason = "empty_option"; return false; }

    in          = DecisionInput{};
    in.type     = t;
    in.state    = routing::state_text(p.state);
    in.question = p.question;
    in.domain   = q.domain;
    in.family   = q.family;
    switch (t) {
    case Type::Choice:
        // Options are the candidate keys: a repeated option would make the answer ambiguous.
        if (std::set<std::string>(p.options.begin(), p.options.end()).size() != n) {
            reason = "ambiguous_answer";
            return false;
        }
        for (const auto& o : p.options) in.options.push_back({o, o});
        correct = p.answer_index;
        break;
    case Type::Score:
        if (p.number_of_levels != static_cast<int>(n)) { reason = "level_count_mismatch"; return false; }
        for (size_t i = 0; i < n; ++i) in.options.push_back({std::to_string(i), p.options[i]});
        correct = p.answer_index;
        break;
    default: {  // noul
        const std::set<std::string> opts(p.options.begin(), p.options.end());
        if (n != 2 || opts != std::set<std::string>{"no", "yes"}) { reason = "invalid_noul_options"; return false; }
        in.options = {{"false", ""}, {"true", ""}};
        correct    = p.options[static_cast<size_t>(p.answer_index)] == "yes" ? 1 : 0;
        break;
    }
    }
    return true;
}

SampleCollector::SampleCollector(routing::TaskRouter& router, DecisionEngine& engine,
                                 const retrieval::IExemplarIndex& index)
    : router_(router), engine_(engine), index_(index)
{}

const std::vector<float>& SampleCollector::prior(Type type, size_t n)
{
    const auto key = std::make_pair(type, n);
    auto it = priors_.find(key);
    if (it == priors_.end()) it = priors_.emplace(key, engine_.compute_blank_logits(type, static_cast<int>(n))).first;
    return it->second;
}

retrieval::LoadResult SampleCollector::collect(const QueryRef& q, CollectedSample& out, std::string& reason)
{
    using retrieval::LoadResult;
    DecisionInput in;
    int correct = -1;
    {
        retrieval::ExemplarPayload p;
        std::string why;
        const LoadResult lr = index_.load(q.id, p, why);
        if (lr == LoadResult::Failed) { reason = why; return lr; }
        if (lr == LoadResult::Invalid) { reason = "undecodable_payload"; return lr; }
        if (!make_query(p, q, in, correct, reason)) return LoadResult::Invalid;
    }

    out          = CollectedSample{};
    out.id       = q.id;
    out.type     = q.type;
    out.split    = q.split;
    out.correct  = correct;

    DecisionOutput d;
    if (q.type == Type::Score) {
        // The LLM path score calibration applies to: S1 zero-shot (the router's fallback for score).
        routing::RoutedTask task;
        std::string why;
        if (!routing::route_task(in, task, why)) { reason = "not_routable"; return LoadResult::Invalid; }
        d = engine_.decide_zero_shot(task.decision);
        out.strategy = routing::kZeroShot;
        if (!d.ok) {
            if (d.error.rfind("prompt too long", 0) == 0) { reason = "context_too_long"; return LoadResult::Invalid; }
            reason = "inference failed: " + d.error;
            return LoadResult::Failed;
        }
    } else {
        d = router_.decide(in);
        if (!d.ok) {
            if (d.error_code == ErrorCode::CONTEXT_TOO_LONG) {
                reason = d.error.rfind("query embedding failed", 0) == 0 ? "embedding_context_too_long" : "context_too_long";
                return LoadResult::Invalid;
            }
            if (d.error_code == ErrorCode::INVALID_REQUEST || d.error_code == ErrorCode::UNSUPPORTED_PRIMITIVE) {
                reason = "not_routable";
                return LoadResult::Invalid;
            }
            reason = (d.error_code.empty() ? std::string("inference failed") : d.error_code) + ": " + d.error;
            return LoadResult::Failed;
        }
        if (!d.routing) { reason = "routing diagnostics missing"; return LoadResult::Failed; }
        out.strategy = d.routing->strategy;
        if (d.routing->fallback) out.fallback_reason = d.routing->fallback_reason;
    }
    if (d.raw_logits.size() != in.options.size()) {
        reason = "inference returned no candidate logits";
        return LoadResult::Failed;
    }
    for (float l : d.raw_logits)
        if (!std::isfinite(l)) { reason = "non_finite_logits"; return LoadResult::Invalid; }
    out.raw_logits    = std::move(d.raw_logits);
    out.prompt_tokens = d.prompt_token_count;
    if (prior(q.type, in.options.size()).size() != in.options.size()) {
        reason = "blank-prompt prior logits could not be computed";
        return LoadResult::Failed;
    }
    return LoadResult::Ok;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

std::string report_path_for(const std::string& output_path, bool accepted)
{
    std::filesystem::path p(output_path);
    const std::string stem = p.extension() == ".json" ? p.stem().string() : p.filename().string();
    return p.replace_filename(stem + (accepted ? ".report.json" : ".failed-report.json")).string();
}

bool write_file_atomically(const std::string& dest, const std::string& content,
                           const std::function<bool(const std::string&, std::string&)>& verify,
                           std::string& err)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path target(dest);
    fs::path dir = target.parent_path();
    if (!dir.empty()) {
        fs::create_directories(dir, ec);
        if (ec) { err = "cannot create directory " + dir.string() + ": " + ec.message(); return false; }
    }
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path tmp = (dir.empty() ? fs::path(".") : dir) /
                         (target.filename().string() + "." + std::to_string(unique) + ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { err = "cannot write " + tmp.string(); return false; }
        f << content;
        f.flush();
        if (!f) {
            f.close();
            fs::remove(tmp, ec);
            err = "cannot write " + tmp.string();
            return false;
        }
    }
    if (verify && !verify(tmp.string(), err)) {
        fs::remove(tmp, ec);
        return false;
    }
    fs::rename(tmp, target, ec);
    if (ec) {
        err = "cannot replace " + dest + ": " + ec.message();
        std::error_code ec2;
        fs::remove(tmp, ec2);
        return false;
    }
    return true;
}

bool write_calibration_file(const std::string& dest, const std::string& serialized,
                            const std::string& model_sha256, const CalibrationArtifact* expect,
                            std::string& err)
{
    auto verify = [&](const std::string& tmp, std::string& e) {
        CalibrationLoad c;
        std::string why;
        if (!load_calibration(tmp, true, production_formulation(), c, why)) {
            e = "generated calibration failed validation: " + why;
            return false;
        }
        if (c.state != CalibrationState::Enabled) {
            e = "generated calibration does not match the production formulation: " + c.message;
            return false;
        }
        if (!check_calibration_model(c, model_sha256)) {
            e = "generated calibration is not accepted for the inference model: " +
                (c.message.empty() ? std::string("no model identity recorded") : c.message);
            return false;
        }
        if (expect) {
            const CalibrationArtifact& a = c.artifact;
            const bool same =
                a.choice_temperature == expect->choice_temperature && a.choice_prior_alpha == expect->choice_prior_alpha &&
                a.noul_temperature == expect->noul_temperature && a.noul_prior_alpha == expect->noul_prior_alpha &&
                a.score_temperature == expect->score_temperature && a.score_prior_alpha == expect->score_prior_alpha &&
                a.formulation.layout == expect->formulation.layout &&
                a.formulation.candidate_scheme == expect->formulation.candidate_scheme &&
                a.formulation.prior_correction == expect->formulation.prior_correction;
            if (!same) {
                e = "generated calibration does not read back with the fitted parameters";
                return false;
            }
        }
        return true;
    };
    return write_file_atomically(dest, serialized, verify, err);
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

GenerateOutcome generate_calibration(retrieval::ExemplarStore& store, routing::TaskRouter& router,
                                     DecisionEngine& engine, const GenerateRequest& req,
                                     const ProgressFn& progress)
{
    GenerateOutcome res;
    const GeneratorOptions& opts = req.options;
    std::string err;
    auto fail = [&](std::string msg) {
        res.ok    = false;
        res.error = std::move(msg);
        return res;
    };
    auto stage = [&](const char* name) {
        if (progress) {
            Progress p;
            p.stage = name;
            progress(p);
        }
    };

    if (!opts.validate(err)) return fail(err);
    if (req.model_sha256.size() != 64) return fail("the inference model's SHA-256 is unknown (model file unreadable?)");
    if (req.output_path.empty()) return fail("no output path");
    // The output location must be writable before any inference runs.
    if (!check_writable(req.output_path, err)) return fail(err);

    HeldOutReset reset{store};
    store.set_held_out({});

    // 1. Partition.
    stage("scan");
    Partition part;
    {
        std::vector<retrieval::ExemplarRow> rows;
        if (!store.scan(rows, err)) return fail(err);
        stage("partition");
        RowValidator validate = [&](const retrieval::ExemplarRow& r, std::string& reason) {
            QueryRef q;
            q.id     = r.id;
            q.type   = to_type(r.kind);
            q.domain = r.domain;
            q.family = r.family;
            retrieval::ExemplarPayload p;
            std::string why;
            const retrieval::LoadResult lr = store.load(r.id, p, why);
            if (lr == retrieval::LoadResult::Failed) { reason = why; return lr; }
            if (lr == retrieval::LoadResult::Invalid) { reason = "undecodable_payload"; return lr; }
            DecisionInput in;
            int correct = -1;
            if (!make_query(p, q, in, correct, reason)) return retrieval::LoadResult::Invalid;
            routing::RoutedTask task;
            if (!routing::route_task(in, task, why)) { reason = "not_routable"; return retrieval::LoadResult::Invalid; }
            for (size_t i = 0; i < task.to_caller.size(); ++i)
                if (task.to_caller[i] != static_cast<int>(i)) { reason = "candidate_order"; return retrieval::LoadResult::Invalid; }
            return retrieval::LoadResult::Ok;
        };
        if (!partition_rows(rows, [&](int64_t id) { return store.is_allowed(id); }, validate, opts, part, err))
            return fail(err);
    }
    for (Type t : kKinds) {
        const KindCounts& k = part.kinds[t];
        spdlog::info("calibrate: {}: {} candidate rows in {} groups; selected {} fitting + {} validation queries",
                     to_string(t), k.candidate_rows, k.candidate_groups, k.selected_fit, k.selected_validation);
    }
    if (!check_sufficient(part, false, err)) return fail("insufficient calibration data: " + err);
    const size_t pool_rows = store.allowlist_size() - part.held_out.size();
    spdlog::info("calibrate: {} groups ({} rows) held out of the retrieval pool; {} rows remain", part.held_out_groups,
                 part.held_out.size(), pool_rows);
    store.set_held_out(part.held_out);

    // 2. Collection.
    SampleCollector collector(router, engine, store);
    std::vector<CollectedSample> samples;
    samples.reserve(part.queries.size());
    Progress pr;
    pr.stage = "collect";
    pr.total = part.queries.size();
    for (Type t : kKinds) {
        const KindCounts& k = part.kinds[t];
        pr.by_kind[t] = {0, k.selected_fit + k.selected_validation};
    }
    if (progress) progress(pr);
    const size_t batch = static_cast<size_t>(opts.batch_size);
    for (size_t b = 0; b < part.queries.size(); b += batch) {
        const size_t end = std::min(part.queries.size(), b + batch);
        for (size_t i = b; i < end; ++i) {
            const QueryRef& q = part.queries[i];
            KindCounts& kc = part.kinds[q.type];
            CollectedSample s;
            std::string reason;
            const retrieval::LoadResult r = collector.collect(q, s, reason);
            kc.processed++;
            pr.processed++;
            pr.by_kind[q.type].first++;
            if (r == retrieval::LoadResult::Failed)
                return fail("calibration query " + retrieval::format_exemplar_id(q.id) + ": " + reason);
            if (r == retrieval::LoadResult::Invalid) {
                kc.rejected[reason]++;
                pr.rejected++;
                if (opts.verbose)
                    spdlog::info("calibrate: {} {} rejected: {}", to_string(q.type), retrieval::format_exemplar_id(q.id), reason);
                continue;
            }
            (q.split == Split::Fit ? kc.accepted_fit : kc.accepted_validation)++;
            pr.accepted++;
            if (opts.verbose) {
                const auto top = std::max_element(s.raw_logits.begin(), s.raw_logits.end()) - s.raw_logits.begin();
                spdlog::info("calibrate: {} {} {} {}{} tokens={} correct={} argmax={}", to_string(q.type),
                             retrieval::format_exemplar_id(q.id), to_string(q.split), s.strategy,
                             s.fallback_reason.empty() ? "" : " (fallback: " + s.fallback_reason + ")",
                             s.prompt_tokens, s.correct, top);
            }
            samples.push_back(std::move(s));
        }
        if (progress) progress(pr);
    }
    if (!check_sufficient(part, true, err)) return fail("insufficient calibration data after evaluation: " + err);

    // 3. Fit and 4. validation, per kind.
    stage("fit");
    CalibrationArtifact art;
    art.version          = 1;
    art.method           = "temperature";
    art.model_identifier = req.model_sha256;
    art.formulation      = production_formulation();

    auto to_samples = [&](const std::vector<const CollectedSample*>& v) {
        std::vector<CalibrationSample> out;
        out.reserve(v.size());
        for (const CollectedSample* c : v) {
            CalibrationSample s;
            s.raw_logits     = c->raw_logits;
            s.logits         = c->raw_logits;
            s.prior_logits   = collector.prior(c->type, c->raw_logits.size());
            s.correct_index  = c->correct;
            s.type           = to_string(c->type);
            s.expected_score = c->correct;
            out.push_back(std::move(s));
        }
        return out;
    };

    nlohmann::json kinds_json = nlohmann::json::object();
    std::vector<std::string> rejections;
    for (Type t : kKinds) {
        const std::string kind = to_string(t);
        std::vector<const CollectedSample*> fit_c, val_c;
        std::set<std::string> strategies;
        for (const auto& s : samples) {
            if (s.type != t) continue;
            (s.split == Split::Fit ? fit_c : val_c).push_back(&s);
            strategies.insert(s.strategy);
        }
        const std::vector<CalibrationSample> fit_s = to_samples(fit_c);
        const std::vector<CalibrationSample> val_s = to_samples(val_c);

        std::set<int> labels;
        bool varies = false;
        for (const auto& s : fit_s) {
            labels.insert(s.correct_index);
            const auto [lo, hi] = std::minmax_element(s.raw_logits.begin(), s.raw_logits.end());
            if (*hi > *lo) varies = true;
        }
        if (labels.size() < 2)
            return fail("degenerate " + kind + " data: every fitting sample has the same correct candidate (" +
                        std::to_string(*labels.begin()) + "); temperature and prior correction cannot be fitted");
        if (!varies)
            return fail("degenerate " + kind + " logits: the model gives every candidate the same logit in every "
                        "fitting sample");

        const AlphaTResult fit = fit_alpha_temperature(fit_s);
        if (!fit.converged || !std::isfinite(fit.temperature) || fit.temperature <= 0.0 ||
            !std::isfinite(fit.alpha) || fit.alpha < kFitAlphaMin || fit.alpha > kFitAlphaMax || !std::isfinite(fit.nll))
            return fail("invalid " + kind + " optimization result (temperature " + std::to_string(fit.temperature) +
                        ", prior alpha " + std::to_string(fit.alpha) + ")");

        nlohmann::json boundary = nlohmann::json::array();
        if (fit.temperature_at_lower_bound) boundary.push_back("temperature at the lower search bound " + std::to_string(kFitTemperatureMin));
        if (fit.temperature_at_upper_bound) boundary.push_back("temperature at the upper search bound " + std::to_string(kFitTemperatureMax));
        if (fit.alpha_at_lower_bound)       boundary.push_back("prior alpha at the lower bound 0 (no prior correction)");
        if (fit.alpha_at_upper_bound)       boundary.push_back("prior alpha at the upper bound " + std::to_string(kFitAlphaMax));
        for (const auto& b : boundary) res.warnings.push_back(kind + ": boundary solution: " + b.get<std::string>());

        const PrimitiveCalibrationMetrics raw = evaluate(val_s, 0.0, 1.0);
        const PrimitiveCalibrationMetrics cal = evaluate(val_s, fit.alpha, fit.temperature);
        const bool improves = cal.nll <= raw.nll + 1e-12;
        if (!improves)
            rejections.push_back(kind + ": validation NLL " + std::to_string(raw.nll) + " -> " + std::to_string(cal.nll));
        if (cal.accuracy < raw.accuracy) res.warnings.push_back(kind + ": validation accuracy decreases with calibration");
        if (cal.brier > raw.brier)       res.warnings.push_back(kind + ": validation Brier score increases with calibration");
        if (cal.ece > raw.ece)           res.warnings.push_back(kind + ": validation ECE increases with calibration");

        // Per executed path: primary (the kind's calibration strategy) and production fallbacks.
        nlohmann::json paths = nlohmann::json::object();
        for (const std::string& strat : strategies) {
            std::vector<const CollectedSample*> fp, vp;
            std::string reason;
            for (const CollectedSample* c : fit_c) if (c->strategy == strat) { fp.push_back(c); reason = c->fallback_reason; }
            for (const CollectedSample* c : val_c) if (c->strategy == strat) { vp.push_back(c); reason = c->fallback_reason; }
            const bool primary = strat == calibration_strategy(t);
            nlohmann::json pj = {
                {"role", primary ? "primary" : "fallback"},
                {"fit_samples", fp.size()},
                {"validation_samples", vp.size()},
            };
            if (!primary && !reason.empty()) pj["fallback_reason"] = reason;
            const std::vector<CalibrationSample> pv = to_samples(vp);
            PrimitiveCalibrationMetrics p_raw, p_cal;
            if (!pv.empty()) {
                p_raw = evaluate(pv, 0.0, 1.0);
                p_cal = evaluate(pv, fit.alpha, fit.temperature);
                pj["validation"] = comparison_json(p_raw, p_cal);
            }
            if (strategies.size() > 1) {
                if (fp.size() >= static_cast<size_t>(kMinFitSamples)) {
                    const AlphaTResult pf = fit_alpha_temperature(to_samples(fp));
                    pj["path_specific_fit"] = {{"temperature", pf.temperature}, {"prior_alpha", pf.alpha},
                                               {"fit_nll", pf.nll}, {"converged", pf.converged}};
                    if (!pv.empty() && pf.converged)
                        pj["validation"]["path_specific"] = evaluate(pv, pf.alpha, pf.temperature).to_json();
                } else {
                    pj["path_specific_fit"] = "not fitted: " + std::to_string(fp.size()) +
                                              " fitting samples (minimum " + std::to_string(kMinFitSamples) + ")";
                }
            }
            if (pv.size() >= static_cast<size_t>(kMinValidationSamples)) {
                if (p_cal.nll > p_raw.nll + 1e-12)
                    res.warnings.push_back(kind + " " + strat + ": the task-level parameters increase validation NLL on this path");
            } else if (!primary) {
                pj["note"] = "too few validation samples to evaluate this path separately";
            }
            paths[strat] = pj;
        }

        const KindCounts& kc = part.kinds[t];
        kinds_json[kind] = {
            {"calibration_strategy", calibration_strategy(t)},
            {"counts", counts_json(kc)},
            {"parameters", {{"temperature", fit.temperature}, {"prior_alpha", fit.alpha}}},
            {"fit", {{"samples", fit_s.size()}, {"nll", fit.nll},
                     {"uncalibrated_nll", mean_nll(fit_s, 1.0)}, {"distinct_labels", labels.size()},
                     {"boundary", boundary}}},
            {"validation", comparison_json(raw, cal)},
            {"validation_accepted", improves},
            {"paths", paths},
        };
        switch (t) {
        case Type::Choice: art.choice_temperature = fit.temperature; art.choice_prior_alpha = fit.alpha; break;
        case Type::Noul:   art.noul_temperature   = fit.temperature; art.noul_prior_alpha   = fit.alpha; break;
        default:           art.score_temperature  = fit.temperature; art.score_prior_alpha  = fit.alpha; break;
        }
    }
    res.artifact = art;
    stage("validate");
    const bool accepted = rejections.empty();

    // Report.
    nlohmann::json sample_rows = nlohmann::json::array();
    for (const auto& s : samples) {
        nlohmann::json j = {
            {"id", retrieval::format_exemplar_id(s.id)},
            {"kind", to_string(s.type)},
            {"split", to_string(s.split)},
            {"strategy", s.strategy},
            {"correct", s.correct},
            {"prompt_tokens", s.prompt_tokens},
            {"raw_logits", s.raw_logits},
        };
        if (!s.fallback_reason.empty()) j["fallback_reason"] = s.fallback_reason;
        sample_rows.push_back(std::move(j));
    }
    nlohmann::json partition_json = {
        {"unit", "group_id"},
        {"seed", opts.seed},
        {"max_samples_per_kind", opts.max_samples},
        {"validation_ratio", opts.validation_ratio},
        {"batch_size", opts.batch_size},
        {"rows_total", part.rows_total},
        {"rows_not_in_training_split", part.rows_not_training},
        {"rows_unsupported_kind", part.rejected_other.count("unsupported_kind") ? part.rejected_other.at("unsupported_kind") : 0},
        {"held_out_groups", part.held_out_groups},
        {"held_out_rows", part.held_out.size()},
        {"retrieval_pool_rows", pool_rows},
        {"guarantees", {
            "calibration queries are training-split rows (the exemplar allowlist); validation, test and external rows are not used",
            "every row of a query's group_id is held out of the retrieval pool, so no query sees itself or a variant of its group as a demonstration",
            "a group_id belongs wholly to the fitting or the validation partition",
            "exemplars whose content equals the query's are never demonstrations (the production content-key rule)",
            "the database records no scenario-family metadata beyond group_id; groups.source is the source dataset, which is not used as a partition unit",
        }},
    };
    res.report = {
        {"report_version", 1},
        {"status", accepted ? "accepted" : "rejected"},
        {"generated_at", utc_now()},
        {"output", req.output_path},
        {"calibration", art.to_json()},
        {"provenance", req.provenance},
        {"routing_policy", routing::kPolicyName},
        {"prompt_format", routing::kPromptFormatVersion},
        {"inference", {
            {"choice", "nearest_2shot, order A (second-nearest, nearest, query); production zero-shot fallback"},
            {"noul", "nearest_1shot; production zero-shot fallback"},
            {"score", "zero_shot (S1): the LLM path the score parameters apply to; kNN-5 votes are not calibrated"},
            {"logits", "uncalibrated candidate logits of the production candidate scoring"},
        }},
        {"fit_method", {
            {"objective", "mean negative log-likelihood of the correct candidate on the fitting partition"},
            {"transform", "softmax((logits - alpha * blank_prompt_logits) / temperature); blank-prompt logits per kind and candidate count"},
            {"temperature_search", {{"method", "golden-section on log(temperature)"}, {"min", kFitTemperatureMin}, {"max", kFitTemperatureMax}}},
            {"prior_alpha_grid", {{"min", kFitAlphaMin}, {"max", kFitAlphaMax}, {"step", kFitAlphaStep}}},
            {"selection", "fitting partition only; validation is used to accept or reject the result"},
            {"acceptance", "for every kind, calibrated validation NLL <= uncalibrated validation NLL"},
        }},
        {"minimum_samples", {{"fit", kMinFitSamples}, {"validation", kMinValidationSamples}}},
        {"partition", partition_json},
        {"kinds", kinds_json},
        {"warnings", res.warnings},
        {"rejection_reasons", rejections},
        {"samples", sample_rows},
    };

    stage("write");
    if (!accepted) {
        const std::string rp = report_path_for(req.output_path, false);
        if (write_file_atomically(rp, res.report.dump(2) + "\n", {}, err)) res.report_path = rp;
        std::string why;
        for (const auto& r : rejections) why += (why.empty() ? "" : "; ") + r;
        return fail("calibration rejected: it does not improve validation NLL (" + why + "); " + req.output_path +
                    " was not changed");
    }
    if (!write_calibration_file(req.output_path, art.to_json().dump(2) + "\n", req.model_sha256, &art, err))
        return fail(err + "; " + req.output_path + " was not changed");
    res.ok = true;
    const std::string rp = report_path_for(req.output_path, true);
    if (write_file_atomically(rp, res.report.dump(2) + "\n", {}, err)) res.report_path = rp;
    else res.warnings.push_back("report not written: " + err);
    return res;
}

} // namespace calibration
} // namespace pjev
