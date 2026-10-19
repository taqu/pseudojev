#include "r1_output.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace pjev
{
namespace r1
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string fmt(double v, int prec = 4)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

static std::string pct(double v, int prec = 2) { return fmt(v * 100.0, prec); }

static std::string signed_pp(double v)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%+.2f", v * 100.0);
    return buf;
}

static std::string hex_id(int64_t id)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64, static_cast<uint64_t>(id));
    return buf;
}

static bool open_out(const std::string& dir, const char* name, std::ofstream& f)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    f.open(dir + "/" + name, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot write %s/%s\n", dir.c_str(), name);
        return false;
    }
    return true;
}

static std::string truncate(const std::string& s, size_t n)
{
    if (s.size() <= n) return s;
    return s.substr(0, n) + " ...[truncated " + std::to_string(s.size() - n) + " bytes]";
}

using ItemPtrs = std::vector<const R1Item*>;

static ItemPtrs paired_items(const std::vector<R1Item>& items)
{
    ItemPtrs v;
    for (const auto& it : items) if (it.paired) v.push_back(&it);
    return v;
}

static TransitionCounts count(const ItemPtrs& v)
{
    TransitionCounts c;
    for (const auto* it : v) c.add(it->transition);
    return c;
}

static ItemPtrs filter(const ItemPtrs& v, const std::function<bool(const R1Item&)>& pred)
{
    ItemPtrs out;
    for (const auto* it : v) if (pred(*it)) out.push_back(it);
    return out;
}

// Named paired subsets: overall, by kind, by score level count, by choice option count.
static std::vector<std::pair<std::string, ItemPtrs>> paired_groups(const ItemPtrs& paired)
{
    std::vector<std::pair<std::string, ItemPtrs>> g;
    g.push_back({"overall", paired});
    for (const char* k : {"choice", "noul", "score"}) {
        auto sub = filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (!sub.empty()) g.push_back({std::string("kind:") + k, sub});
    }
    std::set<int> levels, nopts;
    for (const auto* it : paired) {
        if (it->kind == "score")  levels.insert(it->number_of_levels);
        if (it->kind == "choice") nopts.insert(it->options_count);
    }
    for (int l : levels) {
        g.push_back({"score_levels:" + std::to_string(l),
            filter(paired, [&](const R1Item& it) { return it.kind == "score" && it.number_of_levels == l; })});
    }
    for (int n : nopts) {
        g.push_back({"choice_options:" + std::to_string(n),
            filter(paired, [&](const R1Item& it) { return it.kind == "choice" && it.options_count == n; })});
    }
    return g;
}

std::vector<SimilarityBin> similarity_bins(const ItemPtrs& paired, int n_bins)
{
    std::vector<SimilarityBin> bins;
    if (paired.empty() || n_bins < 1) return bins;
    ItemPtrs v = paired;
    std::stable_sort(v.begin(), v.end(), [](const R1Item* a, const R1Item* b) {
        return a->retrieval.hamming_distance < b->retrieval.hamming_distance;
    });
    const size_t n = v.size();
    size_t start = 0;
    for (int b = 1; b <= n_bins && start < n; ++b) {
        size_t end = (b == n_bins) ? n : std::max(start + 1, n * static_cast<size_t>(b) / n_bins);
        // Keep equal distances in one bin so bin edges are well defined.
        while (end < n && v[end]->retrieval.hamming_distance == v[end - 1]->retrieval.hamming_distance) end++;
        SimilarityBin bin;
        bin.hamming_lo = v[start]->retrieval.hamming_distance;
        bin.hamming_hi = v[end - 1]->retrieval.hamming_distance;
        for (size_t i = start; i < end; ++i) bin.counts.add(v[i]->transition);
        bins.push_back(bin);
        start = end;
    }
    return bins;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

static nlohmann::ordered_json condition_json(const ConditionResult& c)
{
    nlohmann::ordered_json j;
    j["status"] = c.status;
    if (!c.error.empty()) j["error"] = c.error;
    if (c.ok()) {
        j["prediction_index"] = c.prediction;
        j["prediction_label"] = c.prediction_label;
        j["prediction_text"]  = c.prediction_text;
        j["correct"]          = c.correct;
        j["logits"]           = c.logits;
        j["probs"]            = c.probs;
        j["margin"]           = c.margin;
        j["signed_margin"]    = c.signed_margin;
    }
    j["prompt_tokens"] = c.prompt_tokens;
    j["prompt_ms"]     = c.prompt_ms;
    j["eval_ms"]       = c.eval_ms;
    j["scoring_ms"]    = c.scoring_ms;
    j["latency_ms"]    = c.total_ms;
    return j;
}

nlohmann::ordered_json item_to_json(const R1Item& it)
{
    nlohmann::ordered_json j;
    j["query_id"]   = it.query_id;
    j["query_hash"] = hex_id(it.query_hash);
    j["kind"]       = it.kind;
    if (it.kind == "score") j["number_of_levels"] = it.number_of_levels;
    j["options_count"]      = it.options_count;
    j["option_labels"]      = it.option_labels;
    j["ground_truth_index"] = it.ground_truth;
    j["ground_truth_label"] = (it.ground_truth >= 0) ? it.option_labels[static_cast<size_t>(it.ground_truth)] : "";
    j["s1"] = condition_json(it.s1);
    j["r1"] = condition_json(it.r1);

    const auto& r = it.retrieval;
    nlohmann::ordered_json rj;
    rj["status"] = r.status;
    if (!r.error.empty()) rj["error"] = r.error;
    if (r.ok()) {
        rj["example_id"]                = hex_id(r.example_id);
        rj["kind"]                      = r.kind;
        if (r.kind == "score") rj["number_of_levels"] = r.number_of_levels;
        rj["hamming_distance"]          = r.hamming_distance;
        rj["hamming_similarity"]        = r.hamming_similarity;
        rj["estimated_cosine"]          = r.estimated_cosine;
        rj["estimated_cosine_distance"] = r.estimated_cosine_distance;
        rj["same_group"]                = r.same_group;
        rj["same_domain"]               = r.same_domain;
        rj["same_family"]               = r.same_family;
        if (r.kind == "score") rj["same_option_texts"] = r.same_option_texts;
        rj["domain"]                    = r.domain;
        rj["family"]                    = r.family;
        rj["answer_index"]              = r.answer_index;
        rj["answer_label"]              = r.answer_label;
        rj["options_count"]             = r.options_count;
        rj["fetched_k"]                 = r.fetched_k;
    }
    rj["embed_ms"]      = r.embed_ms;
    rj["search_ms"]     = r.search_ms;
    rj["fetch_ms"]      = r.fetch_ms;
    rj["decompress_ms"] = r.decompress_ms;
    j["retrieval"] = rj;

    if (it.gate.enabled) {
        j["gate"] = {{"threshold", it.gate.threshold},
                     {"similarity", it.gate.similarity},
                     {"decision", it.gate.decision}};
    }
    j["paired"] = it.paired;
    // Ternary on const char* and nullptr would yield a null char pointer, not JSON null.
    if (it.paired) j["transition"] = to_string(it.transition);
    else           j["transition"] = nullptr;
    if (it.paired) j["r1_total_latency_ms"] = it.r1_total_ms();
    return j;
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

bool write_config(const R1RunInfo& info)
{
    std::ofstream f;
    if (!open_out(info.output_dir, "config.json", f)) return false;
    f << info.config.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
    return true;
}

bool write_predictions_jsonl(const std::string& dir, const std::vector<R1Item>& items)
{
    std::ofstream f;
    if (!open_out(dir, "predictions.jsonl", f)) return false;
    for (const auto& it : items) f << item_to_json(it).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
    return true;
}

bool write_transitions_csv(const std::string& dir, const std::vector<R1Item>& items)
{
    std::ofstream f;
    if (!open_out(dir, "transitions.csv", f)) return false;
    f << "group,n,both_correct,rescue,harm,both_incorrect,s1_correct,r1_correct,"
         "s1_accuracy,r1_accuracy,diff_pp,net,ci95_low_pp,ci95_high_pp,mcnemar_p,exact_p\n";
    for (const auto& [name, sub] : paired_groups(paired_items(items))) {
        TransitionCounts c = count(sub);
        PairedTest t = paired_test(c);
        f << name << "," << c.n() << "," << c.both_correct << "," << c.rescue << "," << c.harm << ","
          << c.both_incorrect << "," << c.s1_correct() << "," << c.r1_correct() << ","
          << fmt(c.s1_accuracy()) << "," << fmt(c.r1_accuracy()) << "," << fmt(t.diff * 100, 3) << ","
          << c.net() << "," << fmt(t.ci_low * 100, 3) << "," << fmt(t.ci_high * 100, 3) << ","
          << fmt(t.mcnemar_p, 6) << "," << fmt(t.exact_p, 6) << "\n";
    }
    return true;
}

static void effect_row(std::ofstream& f, const std::string& scope, const std::string& bin,
                       const std::string& lo, const std::string& hi, const TransitionCounts& c)
{
    f << scope << "," << bin << "," << lo << "," << hi << "," << c.n() << ","
      << fmt(c.s1_accuracy()) << "," << fmt(c.r1_accuracy()) << ","
      << c.rescue << "," << c.harm << "," << c.net() << "," << fmt(c.accuracy_diff() * 100, 3) << "\n";
}

bool write_retrieval_effect_csv(const std::string& dir, const std::vector<R1Item>& items)
{
    std::ofstream f;
    if (!open_out(dir, "retrieval_effect.csv", f)) return false;
    f << "scope,bin,hamming_lo,hamming_hi,n,s1_accuracy,r1_accuracy,rescue,harm,net,diff_pp\n";
    ItemPtrs paired = paired_items(items);
    std::vector<std::pair<std::string, ItemPtrs>> scopes = {{"overall", paired}};
    for (const char* k : {"choice", "noul", "score"}) {
        auto sub = filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (!sub.empty()) scopes.push_back({k, sub});
    }
    for (const auto& [scope, sub] : scopes) {
        auto bins = similarity_bins(sub, 5);
        for (size_t i = 0; i < bins.size(); ++i) {
            effect_row(f, scope, "q" + std::to_string(i + 1), fmt(bins[i].hamming_lo, 0),
                       fmt(bins[i].hamming_hi, 0), bins[i].counts);
        }
    }
    // Relationship flags
    struct Flag { const char* name; bool RetrievalInfo::*m; };
    for (const Flag& fl : {Flag{"same_domain", &RetrievalInfo::same_domain},
                           Flag{"same_family", &RetrievalInfo::same_family}}) {
        for (bool v : {true, false}) {
            auto sub = filter(paired, [&](const R1Item& it) { return it.retrieval.*(fl.m) == v; });
            effect_row(f, fl.name, v ? "true" : "false", "", "", count(sub));
        }
    }
    return true;
}

struct StageStats { size_t n = 0; double mean = 0, median = 0, p95 = 0, total = 0; };

static StageStats stage_stats(const std::vector<double>& v)
{
    StageStats s;
    s.n      = v.size();
    s.mean   = mean(v);
    s.median = percentile(v, 50);
    s.p95    = percentile(v, 95);
    for (double x : v) s.total += x;
    return s;
}

using Stage = std::pair<std::string, std::function<double(const R1Item&)>>;

static std::vector<std::pair<std::string, std::vector<Stage>>> latency_stages()
{
    return {
        {"s1", {
            {"prompt_construction", [](const R1Item& i) { return i.s1.prompt_ms; }},
            {"prompt_evaluation",   [](const R1Item& i) { return i.s1.eval_ms; }},
            {"answer_scoring",      [](const R1Item& i) { return i.s1.scoring_ms; }},
            {"total",               [](const R1Item& i) { return i.s1.total_ms; }},
        }},
        {"r1", {
            {"query_embedding",     [](const R1Item& i) { return i.retrieval.embed_ms; }},
            {"vector_search",       [](const R1Item& i) { return i.retrieval.search_ms; }},
            {"exemplar_retrieval",  [](const R1Item& i) { return i.retrieval.fetch_ms; }},
            {"lz4_decompression",   [](const R1Item& i) { return i.retrieval.decompress_ms; }},
            {"prompt_construction", [](const R1Item& i) { return i.r1.prompt_ms; }},
            {"prompt_evaluation",   [](const R1Item& i) { return i.r1.eval_ms; }},
            {"answer_scoring",      [](const R1Item& i) { return i.r1.scoring_ms; }},
            {"decision_total",      [](const R1Item& i) { return i.r1.total_ms; }},
            {"total",               [](const R1Item& i) { return i.r1_total_ms(); }},
        }},
    };
}

bool write_latency_csv(const std::string& dir, const std::vector<R1Item>& items)
{
    std::ofstream f;
    if (!open_out(dir, "latency.csv", f)) return false;
    f << "condition,stage,n,mean_ms,median_ms,p95_ms,total_ms\n";
    ItemPtrs paired = paired_items(items);
    for (const auto& [cond, stages] : latency_stages()) {
        for (const auto& [name, get] : stages) {
            std::vector<double> v;
            for (const auto* it : paired) v.push_back(get(*it));
            StageStats s = stage_stats(v);
            f << cond << "," << name << "," << s.n << "," << fmt(s.mean, 3) << "," << fmt(s.median, 3)
              << "," << fmt(s.p95, 3) << "," << fmt(s.total, 1) << "\n";
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// summary.md
// ---------------------------------------------------------------------------

static void transition_table(std::ostream& o, const TransitionCounts& c)
{
    o << "| | R1 correct | R1 incorrect |\n|---|---:|---:|\n";
    o << "| S1 correct | " << c.both_correct << " (both correct) | " << c.harm << " (harm) |\n";
    o << "| S1 incorrect | " << c.rescue << " (rescue) | " << c.both_incorrect << " (both incorrect) |\n\n";
}

static void accuracy_table(std::ostream& o, const std::vector<std::pair<std::string, ItemPtrs>>& groups)
{
    o << "| Group | n | S1 acc | R1 acc | Diff (pp) | 95% CI (pp) | Rescue | Harm | Net | McNemar p |\n"
         "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|\n";
    for (const auto& [name, sub] : groups) {
        TransitionCounts c = count(sub);
        PairedTest t = paired_test(c);
        o << "| " << name << " | " << c.n() << " | " << pct(c.s1_accuracy()) << "% (" << c.s1_correct()
          << ") | " << pct(c.r1_accuracy()) << "% (" << c.r1_correct() << ") | " << signed_pp(t.diff)
          << " | [" << signed_pp(t.ci_low) << ", " << signed_pp(t.ci_high) << "] | " << c.rescue << " | "
          << c.harm << " | " << c.net() << " | " << fmt(t.mcnemar_p, 4) << " |\n";
    }
    o << "\n";
}

static void bins_table(std::ostream& o, const ItemPtrs& sub, int dim)
{
    auto bins = similarity_bins(sub, 5);
    o << "| Bin | Hamming | Est. cosine | n | S1 acc | R1 acc | Rescue | Harm | Net change (pp) |\n"
         "|---|---|---|---:|---:|---:|---:|---:|---:|\n";
    for (size_t i = 0; i < bins.size(); ++i) {
        const auto& b = bins[i];
        o << "| q" << (i + 1) << " | " << fmt(b.hamming_lo, 0) << "–" << fmt(b.hamming_hi, 0) << " | "
          << fmt(estimated_cosine(b.hamming_hi, dim), 3) << "–" << fmt(estimated_cosine(b.hamming_lo, dim), 3)
          << " | " << b.counts.n() << " | " << pct(b.counts.s1_accuracy()) << "% | "
          << pct(b.counts.r1_accuracy()) << "% | " << b.counts.rescue << " | " << b.counts.harm << " | "
          << signed_pp(b.counts.accuracy_diff()) << " |\n";
    }
    o << "\n";
}

static std::string rate(int k, int n)
{
    if (n == 0) return "n/a";
    return pct(static_cast<double>(k) / n) + "% (" + std::to_string(k) + "/" + std::to_string(n) + ")";
}

static void label_bias_section(std::ostream& o, const ItemPtrs& paired)
{
    o << "## Retrieved answer-label bias\n\n";
    o << "Label agreement compares candidate labels (A, B, ...). Semantic agreement compares the "
         "option text of the prediction with the demonstration's answer text; for noul and score the "
         "label and the semantic value coincide by construction (fixed false/true order, ordered levels), "
         "for choice they do not. Label agreement alone is not evidence of reasoning.\n\n";
    o << "| Kind | n | Demo label = GT label | S1 = demo label | R1 = demo label | Moved toward demo label | of which rescue / harm / other | Moved away | R1 = demo answer text | S1 = demo answer text |\n"
         "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|\n";
    for (const char* k : {"all", "choice", "noul", "score"}) {
        ItemPtrs sub = (std::string(k) == "all") ? paired
                     : filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (sub.empty()) continue;
        int demo_gt = 0, s1_demo = 0, r1_demo = 0, toward = 0, away = 0, tr = 0, th = 0, to = 0;
        int r1_text = 0, s1_text = 0;
        for (const auto* it : sub) {
            int d = it->retrieval.answer_index;
            bool s1d = it->s1.prediction == d, r1d = it->r1.prediction == d;
            demo_gt += (d == it->ground_truth);
            s1_demo += s1d;
            r1_demo += r1d;
            if (!s1d && r1d) {
                toward++;
                if (it->transition == Transition::Rescue) tr++;
                else if (it->transition == Transition::Harm) th++;
                else to++;
            }
            if (s1d && !r1d) away++;
            r1_text += (it->r1.prediction_text == it->retrieval.answer_text);
            s1_text += (it->s1.prediction_text == it->retrieval.answer_text);
        }
        int n = static_cast<int>(sub.size());
        o << "| " << k << " | " << n << " | " << rate(demo_gt, n) << " | " << rate(s1_demo, n) << " | "
          << rate(r1_demo, n) << " | " << toward << " | " << tr << " / " << th << " / " << to << " | " << away
          << " | " << rate(r1_text, n) << " | " << rate(s1_text, n) << " |\n";
    }
    o << "\n";

    // Label distributions
    o << "### Label distributions (paired queries)\n\n";
    o << "| Kind | Label | Demo answers | S1 predictions | R1 predictions | Ground truth |\n|---|---|---:|---:|---:|---:|\n";
    for (const char* k : {"choice", "noul", "score"}) {
        auto sub = filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (sub.empty()) continue;
        std::map<std::string, std::array<int, 4>> m;
        for (const auto* it : sub) {
            auto lab = [&](const std::vector<std::string>& labels, int idx) {
                return (idx >= 0 && idx < static_cast<int>(labels.size())) ? labels[static_cast<size_t>(idx)] : "?";
            };
            m[it->retrieval.answer_label][0]++;
            m[lab(it->option_labels, it->s1.prediction)][1]++;
            m[lab(it->option_labels, it->r1.prediction)][2]++;
            m[lab(it->option_labels, it->ground_truth)][3]++;
        }
        for (const auto& [label, c] : m) {
            o << "| " << k << " | " << label << " | " << c[0] << " | " << c[1] << " | " << c[2] << " | " << c[3] << " |\n";
        }
    }
    o << "\n";
}

bool write_summary_md(const R1RunInfo& info, const std::vector<R1Item>& items)
{
    std::ofstream f;
    if (!open_out(info.output_dir, "summary.md", f)) return false;
    std::ostringstream o;
    const auto& cfg = info.config;
    const int dim = info.embedding_dim;

    o << "# R1 — Retrieved 1-Shot Inference Evaluation\n\n";
    if (info.preliminary) {
        o << "> **PRELIMINARY** — this run used a subset (--limit / --kind / --seed). "
             "Do not treat these numbers as the full validation result.\n\n";
    }
    o << "Corpus: **" << info.corpus << "**  \n";
    o << "Run date: " << cfg.value("run_timestamp", "") << "\n\n";

    o << "## Configuration\n\n| Setting | Value |\n|---|---|\n";
    for (const char* k : {"input_path", "db_path", "train_path", "inference_model", "embedding_model",
                          "prompt_format_version", "scheme", "layout", "ensemble", "calibration",
                          "n_ctx", "top_k", "kind_filter", "limit", "seed", "leakage_policy"}) {
        if (!cfg.contains(k)) continue;
        const auto& v = cfg[k];
        o << "| " << k << " | " << (v.is_string() ? v.get<std::string>() : v.dump()) << " |\n";
    }
    o << "\n";

    // Coverage
    std::map<std::string, int> s1_status, r1_status, ret_status;
    for (const auto& it : items) {
        s1_status[it.s1.status]++;
        r1_status[it.r1.status]++;
        ret_status[it.retrieval.status]++;
    }
    ItemPtrs paired = paired_items(items);
    o << "## Coverage\n\n";
    o << "- Evaluated queries: " << items.size() << "\n";
    o << "- Paired (S1 ok and R1 1-shot ok): **" << paired.size() << "** ("
      << (items.empty() ? "n/a" : pct(static_cast<double>(paired.size()) / items.size()) + "%") << ")\n";
    auto dump_status = [&](const char* name, const std::map<std::string, int>& m) {
        o << "- " << name << ": ";
        bool first = true;
        for (const auto& [k, v] : m) { o << (first ? "" : ", ") << k << "=" << v; first = false; }
        o << "\n";
    };
    dump_status("S1 status", s1_status);
    dump_status("Retrieval status", ret_status);
    dump_status("R1 status", r1_status);
    o << "\nUnpaired queries are excluded from every paired metric below; no zero-shot fallback is "
         "counted as a 1-shot result.\n\n";

    // Headline
    TransitionCounts all = count(paired);
    PairedTest t = paired_test(all);
    o << "## Headline (paired)\n\n";
    o << "| Metric | Value |\n|---|---|\n";
    o << "| Paired examples | " << all.n() << " |\n";
    o << "| S1 accuracy | " << pct(all.s1_accuracy()) << "% (" << all.s1_correct() << ") |\n";
    o << "| R1 accuracy | " << pct(all.r1_accuracy()) << "% (" << all.r1_correct() << ") |\n";
    o << "| Difference | " << signed_pp(t.diff) << " pp |\n";
    o << "| 95% CI (paired Wald) | [" << signed_pp(t.ci_low) << ", " << signed_pp(t.ci_high) << "] pp |\n";
    o << "| Rescue / Harm / Net | " << all.rescue << " / " << all.harm << " / " << all.net() << " |\n";
    o << "| McNemar χ² (cc) p | " << fmt(t.mcnemar_p, 6) << " |\n";
    o << "| Exact binomial p | " << fmt(t.exact_p, 6) << " |\n\n";
    o << "Check: (rescue − harm) / n = " << signed_pp(all.n() ? static_cast<double>(all.net()) / all.n() : 0)
      << " pp = R1 acc − S1 acc.\n\n";

    o << "## Accuracy by group\n\n";
    accuracy_table(o, paired_groups(paired));

    o << "## Rescue / harm transitions\n\n### Overall\n\n";
    transition_table(o, all);
    for (const char* k : {"choice", "noul", "score"}) {
        auto sub = filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (sub.empty()) continue;
        o << "### " << k << "\n\n";
        transition_table(o, count(sub));
    }
    std::set<int> levels;
    for (const auto* it : paired) if (it->kind == "score") levels.insert(it->number_of_levels);
    for (int l : levels) {
        o << "### score, " << l << " levels\n\n";
        transition_table(o, count(filter(paired, [&](const R1Item& it) {
            return it.kind == "score" && it.number_of_levels == l; })));
    }

    // Similarity
    o << "## Similarity versus outcome\n\n";
    o << "The index stores sign-quantized (bit) vectors, so sqlite-vec returns a **Hamming distance** "
         "over " << dim << " bits, not a float cosine distance. Estimated cosine = cos(π·h/" << dim
      << ") (sign-quantization angle estimate). Bins are observed quantiles of the Top-1 Hamming "
         "distance (closest first); equal distances share a bin. They are descriptive only — no gate.\n\n";
    o << "### All kinds\n\n";
    bins_table(o, paired, dim);
    for (const char* k : {"choice", "noul", "score"}) {
        auto sub = filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (sub.empty()) continue;
        o << "### " << k << "\n\n";
        bins_table(o, sub, dim);
    }

    // Relations
    o << "## Retrieval relationships\n\n";
    // Schema v4 dropped template / source-instance / scenario-family columns; they are not reported.
    o << "| Kind | n | Same group | Same domain | Same family | Same score option texts |\n"
         "|---|---:|---:|---:|---:|---:|\n";
    for (const char* k : {"all", "choice", "noul", "score"}) {
        ItemPtrs sub = (std::string(k) == "all") ? paired
                     : filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (sub.empty()) continue;
        int g = 0, sd = 0, sf = 0, so = 0, ns = 0;
        for (const auto* it : sub) {
            g  += it->retrieval.same_group;
            sd += it->retrieval.same_domain;
            sf += it->retrieval.same_family;
            if (it->kind == "score") { ns++; so += it->retrieval.same_option_texts; }
        }
        int n = static_cast<int>(sub.size());
        o << "| " << k << " | " << n << " | " << rate(g, n) << " | " << rate(sd, n) << " | " << rate(sf, n)
          << " | " << (ns ? rate(so, ns) : "n/a") << " |\n";
    }
    o << "\nOutcome by domain / family relationship:\n\n";
    accuracy_table(o, {
        {"same_domain", filter(paired, [](const R1Item& it) { return it.retrieval.same_domain; })},
        {"different_domain", filter(paired, [](const R1Item& it) { return !it.retrieval.same_domain; })},
        {"same_family", filter(paired, [](const R1Item& it) { return it.retrieval.same_family; })},
        {"different_family", filter(paired, [](const R1Item& it) { return !it.retrieval.same_family; })},
    });

    label_bias_section(o, paired);

    // Margins
    o << "## Margins\n\n| Kind | S1 mean winner margin | R1 mean winner margin | S1 mean signed margin | R1 mean signed margin |\n"
         "|---|---:|---:|---:|---:|\n";
    for (const char* k : {"all", "choice", "noul", "score"}) {
        ItemPtrs sub = (std::string(k) == "all") ? paired
                     : filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (sub.empty()) continue;
        std::vector<double> a, b, c, d;
        for (const auto* it : sub) {
            a.push_back(it->s1.margin);        b.push_back(it->r1.margin);
            c.push_back(it->s1.signed_margin); d.push_back(it->r1.signed_margin);
        }
        o << "| " << k << " | " << fmt(mean(a), 3) << " | " << fmt(mean(b), 3) << " | " << fmt(mean(c), 3)
          << " | " << fmt(mean(d), 3) << " |\n";
    }
    o << "\n";

    // Tokens
    o << "## Prompt tokens\n\n";
    o << "Maximum context (n_ctx): " << info.n_ctx << ". Query tokens = S1 prompt tokens; "
         "exemplar tokens = R1 − S1 prompt tokens (demonstration turn).\n\n";
    o << "| Kind | S1 mean | R1 mean | Extra mean | Extra median | Extra p95 | R1 max |\n|---|---:|---:|---:|---:|---:|---:|\n";
    for (const char* k : {"all", "choice", "noul", "score"}) {
        ItemPtrs sub = (std::string(k) == "all") ? paired
                     : filter(paired, [&](const R1Item& it) { return it.kind == k; });
        if (sub.empty()) continue;
        std::vector<double> s1t, r1t, ex;
        for (const auto* it : sub) {
            s1t.push_back(it->s1.prompt_tokens);
            r1t.push_back(it->r1.prompt_tokens);
            ex.push_back(it->r1.prompt_tokens - it->s1.prompt_tokens);
        }
        o << "| " << k << " | " << fmt(mean(s1t), 1) << " | " << fmt(mean(r1t), 1) << " | " << fmt(mean(ex), 1)
          << " | " << fmt(percentile(ex, 50), 0) << " | " << fmt(percentile(ex, 95), 0) << " | "
          << fmt(*std::max_element(r1t.begin(), r1t.end()), 0) << " |\n";
    }
    o << "\n";

    // Latency
    o << "## Latency\n\n";
    o << "S1 and R1 run sequentially per query in one process; both models are loaded once and "
         "one untimed warm-up embedding and decision precede the timed loop. Every decision evaluates its full prompt from an empty KV cache (no prefix reuse "
         "between queries or between S1 and R1). The R1 total includes query embedding, vector search, "
         "exemplar read, LZ4 decompression and the 1-shot decision; it is the per-request cost of R1 "
         "on its own, not the combined experiment time.\n\n";
    o << "| Condition | Stage | Mean ms | Median ms | P95 ms |\n|---|---|---:|---:|---:|\n";
    double s1_mean = 0, r1_mean = 0, s1_med = 0, r1_med = 0;
    for (const auto& [cond, stages] : latency_stages()) {
        for (const auto& [name, get] : stages) {
            std::vector<double> v;
            for (const auto* it : paired) v.push_back(get(*it));
            StageStats s = stage_stats(v);
            o << "| " << cond << " | " << name << " | " << fmt(s.mean, 2) << " | " << fmt(s.median, 2)
              << " | " << fmt(s.p95, 2) << " |\n";
            if (name == "total") {
                (cond == "s1" ? s1_mean : r1_mean) = s.mean;
                (cond == "s1" ? s1_med : r1_med)   = s.median;
            }
        }
    }
    o << "\n- R1 / S1 latency multiplier (mean): **" << (s1_mean > 0 ? fmt(r1_mean / s1_mean, 2) : "n/a") << "×**\n";
    o << "- R1 / S1 latency multiplier (median): " << (s1_med > 0 ? fmt(r1_med / s1_med, 2) : "n/a") << "×\n";
    o << "- Total evaluation wall time: " << fmt(info.wall_seconds, 1) << " s (S1 decisions "
      << fmt(info.s1_seconds, 1) << " s, R1 incl. retrieval " << fmt(info.r1_seconds, 1) << " s)\n\n";

    // Context failures
    o << "## Context-limit failures\n\n";
    int ctx_fail = 0;
    for (const auto& it : items) {
        if (it.s1.status == "context_limit" || it.r1.status == "context_limit") {
            if (ctx_fail == 0) o << "| Query | Kind | S1 | R1 | S1 tokens | R1 tokens |\n|---|---|---|---|---:|---:|\n";
            ctx_fail++;
            o << "| " << it.query_id << " | " << it.kind << " | " << it.s1.status << " | " << it.r1.status
              << " | " << it.s1.prompt_tokens << " | " << it.r1.prompt_tokens << " |\n";
        }
    }
    if (ctx_fail == 0) o << "None.\n";
    o << "\n";

    // Observations
    o << "## Main observations\n\n";
    o << "- R1 − S1 = " << signed_pp(t.diff) << " pp over " << all.n() << " paired queries (95% CI ["
      << signed_pp(t.ci_low) << ", " << signed_pp(t.ci_high) << "]); "
      << ((t.ci_low > 0 || t.ci_high < 0) ? "the interval excludes zero." : "the interval includes zero — not conclusive.")
      << "\n";
    o << "- " << all.rescue << " rescues vs " << all.harm << " harms ("
      << rate(all.rescue + all.harm, all.n()) << " of paired queries change correctness).\n";
    for (const auto& note : info.notes) o << "- " << note << "\n";
    o << "\n";

    f << o.str();
    return true;
}

// ---------------------------------------------------------------------------
// examples.md
// ---------------------------------------------------------------------------

static void option_block(std::ostream& o, const std::vector<std::string>& labels,
                        const std::vector<std::string>& texts)
{
    for (size_t i = 0; i < labels.size() && i < texts.size(); ++i) {
        o << "    " << labels[i] << ": " << texts[i] << "\n";
    }
}

static std::string label_of(const R1Item& it, int idx)
{
    if (idx < 0 || idx >= static_cast<int>(it.option_labels.size())) return "?";
    return it.option_labels[static_cast<size_t>(idx)] + " (" + it.option_texts[static_cast<size_t>(idx)] + ")";
}

static void example_entry(std::ostream& o, const R1Item& it, const char* why)
{
    const auto& r = it.retrieval;
    o << "### " << it.query_id << "\n\n";
    o << "*Selected as: " << why << "*\n\n";
    o << "| | |\n|---|---|\n";
    o << "| Kind | " << it.kind << (it.kind == "score" ? " (" + std::to_string(it.number_of_levels) + " levels)" : "") << " |\n";
    o << "| Hamming / est. cosine | " << fmt(r.hamming_distance, 0) << " / " << fmt(r.estimated_cosine, 3) << " |\n";
    o << "| Ground truth | " << label_of(it, it.ground_truth) << " |\n";
    o << "| S1 prediction | " << label_of(it, it.s1.prediction) << " (margin " << fmt(it.s1.margin, 2) << ") |\n";
    o << "| R1 prediction | " << label_of(it, it.r1.prediction) << " (margin " << fmt(it.r1.margin, 2) << ") |\n";
    o << "| Demo answer | " << r.answer_label << " (" << r.answer_text << ") |\n";
    o << "| Same domain / family / group | " << (r.same_domain ? "yes" : "no") << " / "
      << (r.same_family ? "yes" : "no") << " / " << (r.same_group ? "yes" : "no") << " |\n";
    o << "| Demo domain / family | " << r.domain << " / " << r.family << " |\n\n";

    o << "**Query**\n\n";
    o << "    Question: " << it.question << "\n";
    o << "    State: " << truncate(it.state, 700) << "\n";
    option_block(o, it.option_labels, it.option_texts);
    o << "\n**Retrieved exemplar** (" << hex_id(r.example_id) << ")\n\n";
    o << "    Question: " << r.question << "\n";
    o << "    State: " << truncate(r.state, 700) << "\n";
    option_block(o, r.option_labels, r.option_texts);
    o << "    Answer: " << r.answer_label << "\n\n";

    std::vector<std::string> diffs;
    if (it.s1.prediction != it.r1.prediction) {
        diffs.push_back(it.r1.prediction == r.answer_index ? "R1 moved to the demonstration's answer label"
                                                          : "R1 changed the prediction, not to the demo label");
    }
    if (it.options_count != r.options_count) {
        diffs.push_back("option counts differ (" + std::to_string(it.options_count) + " vs " +
                        std::to_string(r.options_count) + ")");
    }
    if (!r.same_family) diffs.push_back("different family");
    if (it.kind == "score" && !r.same_option_texts) diffs.push_back("score level descriptions differ");
    if (!diffs.empty()) {
        o << "Relevant differences: ";
        for (size_t i = 0; i < diffs.size(); ++i) o << (i ? "; " : "") << diffs[i];
        o << ".\n\n";
    }
}

bool write_examples_md(const std::string& dir, const std::vector<R1Item>& items)
{
    std::ofstream f;
    if (!open_out(dir, "examples.md", f)) return false;
    std::ostringstream o;
    o << "# R1 — Representative examples\n\n";
    o << "Per transition and kind: the highest-similarity (smallest Hamming distance) and the "
         "lowest-similarity case, chosen deterministically (ties broken by file order). Successes "
         "and failures are shown alike.\n\n";

    ItemPtrs paired = paired_items(items);
    for (Transition tr : {Transition::Rescue, Transition::Harm, Transition::BothCorrect, Transition::BothIncorrect}) {
        o << "## " << to_string(tr) << "\n\n";
        bool any = false;
        for (const char* k : {"choice", "noul", "score"}) {
            auto sub = filter(paired, [&](const R1Item& it) { return it.transition == tr && it.kind == k; });
            if (sub.empty()) continue;
            auto by_dist = [](const R1Item* a, const R1Item* b) {
                return a->retrieval.hamming_distance < b->retrieval.hamming_distance;
            };
            const R1Item* hi = *std::min_element(sub.begin(), sub.end(), by_dist);
            const R1Item* lo = *std::max_element(sub.begin(), sub.end(), by_dist);
            example_entry(o, *hi, (std::string(k) + ", highest similarity").c_str());
            if (lo != hi) example_entry(o, *lo, (std::string(k) + ", lowest similarity").c_str());
            any = true;
        }
        if (!any) o << "No cases.\n\n";
    }
    f << o.str();
    return true;
}

} // namespace r1
} // namespace pjev
