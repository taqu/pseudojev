#include "r0_output.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#define PJEV_MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define PJEV_MKDIR(p) mkdir(p, 0755)
#endif

#include "nlohmann/json.hpp"

namespace pjev
{
namespace r0
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool ensure_dir(const std::string& path)
{
    PJEV_MKDIR(path.c_str());
    return true;
}

static FILE* open_output(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) fprintf(stderr, "error: cannot open output file: %s\n", path.c_str());
    return f;
}

static double percentile(std::vector<double> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (static_cast<double>(v.size()) - 1.0);
    int lo = static_cast<int>(idx);
    int hi = lo + 1;
    if (hi >= static_cast<int>(v.size())) return v.back();
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double stddev(const std::vector<double>& v, double m)
{
    if (v.size() < 2) return 0.0;
    double acc = 0.0;
    for (double x : v) acc += (x - m) * (x - m);
    return std::sqrt(acc / static_cast<double>(v.size()));
}

// ---------------------------------------------------------------------------
// config.json
// ---------------------------------------------------------------------------

bool write_config(const R0RunConfig& cfg)
{
    ensure_dir(cfg.output_dir);
    std::string path = cfg.output_dir + "/config.json";
    FILE* f = open_output(path);
    if (!f) return false;

    nlohmann::ordered_json j;
    j["db_path"]            = cfg.db_path;
    j["input_path"]         = cfg.input_path;
    j["split_filter"]       = cfg.split_filter.empty() ? nullptr : nlohmann::json(cfg.split_filter);
    j["embedding_model"]    = cfg.model_path;
    j["db_embedding_model"] = cfg.db_embedding_model;
    j["embedding_dim"]      = cfg.db_embedding_dim;
    j["distance_metric"]    = cfg.db_distance_metric;
    j["normalization"]      = cfg.db_normalization;
    j["query_prefix"]       = kQueryPrefix;
    j["document_prefix"]    = "search_document: ";
    j["top_k"]              = cfg.top_k;
    j["kind_filter"]        = cfg.kind_filter.empty() ? "all" : cfg.kind_filter;
    j["score_level_filter"] = true;
    j["exclude_same_group"] = cfg.exclude_same_group;
    j["skip_db_records"]    = cfg.skip_db_records;
    if (cfg.limit >= 0) j["limit"] = cfg.limit;
    j["db_example_count"]   = cfg.db_example_count;
    j["query_count"]        = cfg.query_count;
    j["run_timestamp"]      = cfg.run_timestamp;
    if (!cfg.note.empty()) j["note"] = cfg.note;

    std::string s = j.dump(2);
    fwrite(s.c_str(), 1, s.size(), f);
    fputc('\n', f);
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// retrieval.jsonl
// ---------------------------------------------------------------------------

bool write_retrieval_jsonl(const std::string& out_dir,
                            const std::vector<QueryResult>& results)
{
    ensure_dir(out_dir);
    std::string path = out_dir + "/retrieval.jsonl";
    FILE* f = open_output(path);
    if (!f) return false;

    for (const auto& r : results) {
        nlohmann::ordered_json j;
        j["query_id"]                    = r.query_id;
        j["kind"]                        = r.kind;
        if (r.kind == "score")
            j["number_of_levels"]        = r.number_of_levels;
        j["query_group_id"]              = r.query_group_id;
        j["query_ground_truth"]          = r.query_ground_truth;
        j["embed_ms"]                    = r.embed_ms;
        j["search_ms"]                   = r.search_ms;
        j["retrieved_count"]             = r.retrieved_count;

        nlohmann::json retrieved = nlohmann::json::array();
        for (const auto& ex : r.retrieved) {
            nlohmann::ordered_json e;
            e["rank"]               = ex.rank;
            e["example_id"]         = ex.example_id;
            e["distance"]           = ex.distance;
            e["similarity"]         = ex.similarity;
            e["kind"]               = ex.kind;
            if (ex.kind == "score")
                e["number_of_levels"] = ex.number_of_levels;
            e["ground_truth"]       = ex.ground_truth;
            e["answer_index"]       = ex.answer_index;
            e["group_id"]           = ex.group_id;
            e["same_group"]         = ex.same_group;
            retrieved.push_back(e);
        }
        j["retrieved"] = retrieved;

        std::string line = j.dump();
        fwrite(line.c_str(), 1, line.size(), f);
        fputc('\n', f);
    }

    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct KindStats {
    std::string kind;
    std::vector<double> top1_sim;
    std::vector<double> topk_mean_sim;
    std::vector<double> top12_gap;
    std::vector<double> all_sim;    // every retrieved similarity

    int same_group_top1    = 0;
    int same_group_topk    = 0;     // any match in top-k
    int short_results      = 0;     // retrieved_count < top_k

    std::unordered_map<int64_t, int>     retrieved_id_freq;
    std::unordered_map<std::string, int> query_gt_dist;
    std::unordered_map<std::string, int> top1_gt_dist;
    int gt_match_top1 = 0;
    int n_queries     = 0;
    int top_k         = 5;
};

static void accumulate_stats(KindStats& ks,
                              const QueryResult& r,
                              int top_k)
{
    ks.n_queries++;
    ks.top_k = top_k;

    // Query ground truth distribution
    ks.query_gt_dist[r.query_ground_truth]++;

    if (r.retrieved_count < top_k) ks.short_results++;

    if (r.retrieved.empty()) return;

    const auto& t1 = r.retrieved[0];
    ks.top1_sim.push_back(t1.similarity);
    ks.top1_gt_dist[t1.ground_truth]++;
    if (t1.ground_truth == r.query_ground_truth) ks.gt_match_top1++;
    if (t1.same_group)    ks.same_group_top1++;

    ks.retrieved_id_freq[t1.example_id]++;

    double sum_sim = 0.0;
    for (const auto& ex : r.retrieved) {
        sum_sim += ex.similarity;
        ks.all_sim.push_back(ex.similarity);
        if (ex.same_group)    ks.same_group_topk++;
        ks.retrieved_id_freq[ex.example_id]++;
    }
    ks.topk_mean_sim.push_back(sum_sim / static_cast<double>(r.retrieved.size()));

    if (r.retrieved.size() >= 2) {
        ks.top12_gap.push_back(r.retrieved[0].similarity - r.retrieved[1].similarity);
    }
}

static void write_kind_section(FILE* f, const KindStats& ks, int top_k)
{
    if (ks.n_queries == 0) return;

    double m1 = mean(ks.top1_sim);
    double sd1 = stddev(ks.top1_sim, m1);

    fprintf(f, "### %s (%d queries)\n\n", ks.kind.c_str(), ks.n_queries);
    fprintf(f, "#### Top-1 Similarity\n\n");
    fprintf(f, "| Metric | Value |\n");
    fprintf(f, "|--------|-------|\n");
    fprintf(f, "| Mean   | %.4f |\n", m1);
    fprintf(f, "| Median | %.4f |\n", percentile(ks.top1_sim, 50));
    fprintf(f, "| Std    | %.4f |\n", sd1);
    fprintf(f, "| P10    | %.4f |\n", percentile(ks.top1_sim, 10));
    fprintf(f, "| P25    | %.4f |\n", percentile(ks.top1_sim, 25));
    fprintf(f, "| P75    | %.4f |\n", percentile(ks.top1_sim, 75));
    fprintf(f, "| P90    | %.4f |\n", percentile(ks.top1_sim, 90));
    fprintf(f, "| Mean Top-%d sim | %.4f |\n", top_k, mean(ks.topk_mean_sim));
    fprintf(f, "| Mean Top1-Top2 gap | %.4f |\n\n", mean(ks.top12_gap));

    fprintf(f, "#### Retrieval Structure\n\n");
    fprintf(f, "| Metric | Rate |\n");
    fprintf(f, "|--------|------|\n");
    fprintf(f, "| Same-group @ Top-1 | %.3f |\n",
            static_cast<double>(ks.same_group_top1) / ks.n_queries);
    fprintf(f, "| Same-group (any in Top-%d) | %.3f |\n", top_k,
            static_cast<double>(ks.same_group_topk > 0) / ks.n_queries);
    fprintf(f, "| Same-group fraction in Top-%d | %.3f |\n", top_k,
            ks.all_sim.empty() ? 0.0 :
            static_cast<double>(ks.same_group_topk) /
            static_cast<double>(ks.n_queries * top_k));
    fprintf(f, "| Queries with < %d results | %d |\n\n", top_k, ks.short_results);

    // Top-5 most retrieved examples
    std::vector<std::pair<int,int64_t>> freq;
    freq.reserve(ks.retrieved_id_freq.size());
    for (const auto& p : ks.retrieved_id_freq) freq.push_back({p.second, p.first});
    std::sort(freq.begin(), freq.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });

    fprintf(f, "#### Most Retrieved Examples (top-5)\n\n");
    fprintf(f, "| ID | Count |\n|-------|-------|\n");
    int shown = 0;
    for (const auto& p : freq) {
        if (shown++ >= 5) break;
        fprintf(f, "| %" PRIx64 " | %d |\n", p.second, p.first);
    }
    fprintf(f, "\n");

    // Ground-truth distributions
    fprintf(f, "#### Ground-Truth Distribution\n\n");
    fprintf(f, "| Label | Query | Top-1 Retrieved |\n|-------|-------|----------------|\n");
    std::set<std::string> all_labels;
    for (const auto& p : ks.query_gt_dist)  all_labels.insert(p.first);
    for (const auto& p : ks.top1_gt_dist)   all_labels.insert(p.first);
    for (const auto& lbl : all_labels) {
        int qc = 0, rc = 0;
        auto qi = ks.query_gt_dist.find(lbl);
        if (qi != ks.query_gt_dist.end()) qc = qi->second;
        auto ri = ks.top1_gt_dist.find(lbl);
        if (ri != ks.top1_gt_dist.end()) rc = ri->second;
        std::string l = lbl.empty() ? "(empty)" : lbl;
        if (l.size() > 30) l = l.substr(0, 27) + "...";
        fprintf(f, "| %s | %d | %d |\n", l.c_str(), qc, rc);
    }
    fprintf(f, "\nQuery-Top1 answer agreement: %d / %d (%.1f%%)\n\n",
            ks.gt_match_top1, ks.n_queries,
            100.0 * ks.gt_match_top1 / ks.n_queries);
}

// ---------------------------------------------------------------------------
// summary.md
// ---------------------------------------------------------------------------

bool write_summary_md(const std::string& out_dir,
                       const R0RunConfig& cfg,
                       const std::vector<QueryResult>& results)
{
    ensure_dir(out_dir);
    std::string path = out_dir + "/summary.md";
    FILE* f = open_output(path);
    if (!f) return false;

    fprintf(f, "# R0 Retrieval Sanity Evaluation — Summary\n\n");
    fprintf(f, "Run date: %s  \n", cfg.run_timestamp.c_str());
    fprintf(f, "Database: %s (%d examples)  \n", cfg.db_path.c_str(), cfg.db_example_count);
    fprintf(f, "Input: %s  \n", cfg.input_path.c_str());
    fprintf(f, "Model: %s  \n", cfg.model_path.c_str());
    fprintf(f, "Top-K: %d  \n", cfg.top_k);
    fprintf(f, "Distance metric: %s  \n", cfg.db_distance_metric.c_str());
    if (!cfg.note.empty()) fprintf(f, "Note: %s  \n", cfg.note.c_str());
    fprintf(f, "\n");

    // Per-kind stats
    std::map<std::string, KindStats> by_kind;
    KindStats overall;
    overall.kind = "overall";
    for (const std::string& k : {"choice", "noul", "score"}) {
        by_kind[k].kind = k;
    }

    for (const auto& r : results) {
        accumulate_stats(by_kind[r.kind], r, cfg.top_k);
        accumulate_stats(overall, r, cfg.top_k);
    }

    // Overall stats table
    fprintf(f, "## Overall Statistics\n\n");
    fprintf(f, "| Kind | Queries | Mean Top-1 Sim | Median Top-1 Sim | Same-Group@1 |\n");
    fprintf(f, "|------|---------|----------------|-----------------||--------------|\n");

    for (const auto& kv : by_kind) {
        const KindStats& ks = kv.second;
        if (ks.n_queries == 0) continue;
        fprintf(f, "| %s | %d | %.4f | %.4f | %.3f |\n",
                ks.kind.c_str(), ks.n_queries,
                mean(ks.top1_sim),
                percentile(ks.top1_sim, 50),
                static_cast<double>(ks.same_group_top1) / ks.n_queries);
    }

    {
        const KindStats& ks = overall;
        if (ks.n_queries > 0) {
            fprintf(f, "| **all** | **%d** | **%.4f** | **%.4f** | **%.3f** |\n",
                    ks.n_queries,
                    mean(ks.top1_sim),
                    percentile(ks.top1_sim, 50),
                    static_cast<double>(ks.same_group_top1) / ks.n_queries);
        }
    }
    fprintf(f, "\n");

    fprintf(f, "## Per-Kind Details\n\n");
    for (const auto& kv : by_kind) {
        write_kind_section(f, kv.second, cfg.top_k);
    }

    // Performance
    std::vector<double> embed_ms, search_ms, total_ms;
    for (const auto& r : results) {
        embed_ms.push_back(r.embed_ms);
        search_ms.push_back(r.search_ms);
        total_ms.push_back(r.embed_ms + r.search_ms);
    }
    fprintf(f, "## Performance\n\n");
    fprintf(f, "| Metric | Mean | P50 | P90 |\n");
    fprintf(f, "|--------|------|-----|-----|\n");
    fprintf(f, "| Embed (ms)  | %.1f | %.1f | %.1f |\n",
            mean(embed_ms), percentile(embed_ms, 50), percentile(embed_ms, 90));
    fprintf(f, "| Search (ms) | %.1f | %.1f | %.1f |\n",
            mean(search_ms), percentile(search_ms, 50), percentile(search_ms, 90));
    fprintf(f, "| Total (ms)  | %.1f | %.1f | %.1f |\n",
            mean(total_ms), percentile(total_ms, 50), percentile(total_ms, 90));
    if (!total_ms.empty()) {
        double qps = results.size() / (std::accumulate(total_ms.begin(), total_ms.end(), 0.0) / 1000.0);
        fprintf(f, "\nQueries/sec: %.1f\n", qps);
    }
    fprintf(f, "\n");

    fprintf(f, "## Kind/Level Filtering Verification\n\n");
    int wrong_kind = 0, level_mismatch = 0;
    for (const auto& r : results) {
        for (const auto& ex : r.retrieved) {
            if (ex.kind != r.kind) wrong_kind++;
            if (r.kind == "score" && ex.number_of_levels != r.number_of_levels) level_mismatch++;
        }
    }
    fprintf(f, "Wrong-kind retrieval count: **%d**  \n", wrong_kind);
    fprintf(f, "Score-level mismatch count: **%d**  \n", level_mismatch);
    fprintf(f, "\n");
    if (wrong_kind == 0 && level_mismatch == 0) {
        fprintf(f, "_Kind and score-level filters are working correctly._\n");
    } else {
        fprintf(f, "**WARNING: filter violations detected — investigate immediately.**\n");
    }

    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// examples.md
// ---------------------------------------------------------------------------

static void render_query_block(FILE* f, const QueryResult& r)
{
    fprintf(f, "**ID:** `%" PRIx64 "`  \n", r.query_id);
    fprintf(f, "**Kind:** %s", r.kind.c_str());
    if (r.kind == "score") fprintf(f, " (levels=%d)", r.number_of_levels);
    fprintf(f, "  \n");
    fprintf(f, "**Group:** `%" PRIx64 "`  \n", r.query_group_id);
    fprintf(f, "**Answer:** `%s`  \n", r.query_ground_truth.c_str());
    fprintf(f, "**Retrieved:** %d results  \n\n", r.retrieved_count);

    for (const auto& ex : r.retrieved) {
        fprintf(f, "  **Rank %d** — sim=%.4f dist=%.4f", ex.rank, ex.similarity, ex.distance);
        if (ex.same_group) fprintf(f, " [SAME-GROUP]");
        fprintf(f, "  \n");
        fprintf(f, "  ID: `%" PRIx64 "`  \n", ex.example_id);
        fprintf(f, "  Answer: `%s` (index=%d)  \n", ex.ground_truth.c_str(), ex.answer_index);
    }
}

// Select up to N examples from results matching a condition, sorted by query_id.
static std::vector<size_t> select_examples(
    const std::vector<QueryResult>& results,
    const std::string& kind,
    std::function<bool(const QueryResult&)> predicate,
    int max_count)
{
    std::vector<std::pair<int64_t, size_t>> candidates;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].kind == kind && predicate(results[i])) {
            candidates.push_back({results[i].query_id, i});
        }
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<size_t> out;
    for (const auto& p : candidates) {
        if (static_cast<int>(out.size()) >= max_count) break;
        out.push_back(p.second);
    }
    return out;
}

bool write_examples_md(const std::string& out_dir,
                        const std::vector<QueryResult>& results)
{
    ensure_dir(out_dir);
    std::string path = out_dir + "/examples.md";
    FILE* f = open_output(path);
    if (!f) return false;

    fprintf(f, "# R0 Retrieval Sanity Evaluation — Examples\n\n");
    fprintf(f,
        "This document shows representative query/retrieval pairs for manual inspection.\n"
        "Similarity = 1 - cosine_distance.\n\n");

    for (const std::string& kind : {"choice", "noul", "score"}) {
        fprintf(f, "---\n\n## Kind: %s\n\n", kind.c_str());

        // High similarity (sim >= 0.9)
        auto high = select_examples(results, kind,
            [](const QueryResult& r){
                return !r.retrieved.empty() && r.retrieved[0].similarity >= 0.9;
            }, 2);
        if (!high.empty()) {
            fprintf(f, "### High Similarity (Top-1 ≥ 0.90)\n\n");
            for (size_t idx : high) render_query_block(f, results[idx]);
        }

        // Medium similarity (0.6 <= sim < 0.80)
        auto med = select_examples(results, kind,
            [](const QueryResult& r){
                return !r.retrieved.empty()
                    && r.retrieved[0].similarity >= 0.6
                    && r.retrieved[0].similarity < 0.80;
            }, 2);
        if (!med.empty()) {
            fprintf(f, "### Medium Similarity (0.60 ≤ Top-1 < 0.80)\n\n");
            for (size_t idx : med) render_query_block(f, results[idx]);
        }

        // Low similarity (sim < 0.5)
        auto low = select_examples(results, kind,
            [](const QueryResult& r){
                return !r.retrieved.empty() && r.retrieved[0].similarity < 0.5;
            }, 2);
        if (!low.empty()) {
            fprintf(f, "### Low Similarity (Top-1 < 0.50)\n\n");
            for (size_t idx : low) render_query_block(f, results[idx]);
        }

        // Large top1-top2 gap (> 0.15)
        auto gap = select_examples(results, kind,
            [](const QueryResult& r){
                return r.retrieved.size() >= 2
                    && (r.retrieved[0].similarity - r.retrieved[1].similarity) > 0.15;
            }, 2);
        if (!gap.empty()) {
            fprintf(f, "### Large Top-1 vs Top-2 Gap (> 0.15)\n\n");
            for (size_t idx : gap) render_query_block(f, results[idx]);
        }

        // Same template (potential boilerplate bias)
        auto tmpl = select_examples(results, kind,
            [](const QueryResult& r){
                return !r.retrieved.empty() && r.retrieved[0].same_template;
            }, 2);
        if (!tmpl.empty()) {
            fprintf(f, "### Same-Template Top-1 (potential boilerplate bias)\n\n");
            for (size_t idx : tmpl) render_query_block(f, results[idx]);
        }

        // Different-template top-1 (genuine semantic match across templates)
        auto diff_tmpl = select_examples(results, kind,
            [](const QueryResult& r){
                return !r.retrieved.empty() && !r.retrieved[0].same_template;
            }, 2);
        if (!diff_tmpl.empty()) {
            fprintf(f, "### Different-Template Top-1 (cross-template match)\n\n");
            for (size_t idx : diff_tmpl) render_query_block(f, results[idx]);
        }

        // Counterfactual / same source instance but different answer
        auto cf = select_examples(results, kind,
            [](const QueryResult& r){
                if (r.retrieved.empty()) return false;
                const auto& t1 = r.retrieved[0];
                return t1.same_source_instance
                    && !t1.ground_truth.empty()
                    && t1.ground_truth != r.query_ground_truth;
            }, 2);
        if (!cf.empty()) {
            fprintf(f, "### Same Source Instance, Different Answer (counterfactual candidates)\n\n");
            for (size_t idx : cf) render_query_block(f, results[idx]);
        }
    }

    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// similarity_histogram.csv
// ---------------------------------------------------------------------------

bool write_histogram_csv(const std::string& out_dir,
                          const std::vector<QueryResult>& results)
{
    ensure_dir(out_dir);
    std::string path = out_dir + "/similarity_histogram.csv";
    FILE* f = open_output(path);
    if (!f) return false;

    // 20 bins: [0.0, 0.05), [0.05, 0.10), ..., [0.95, 1.00]
    static const int N_BINS = 20;
    std::map<std::string, std::array<int, N_BINS>> top1_hist;
    std::map<std::string, std::array<int, N_BINS>> topk_hist;

    for (const std::string& k : {"choice", "noul", "score"}) {
        top1_hist[k].fill(0);
        topk_hist[k].fill(0);
    }
    top1_hist["all"].fill(0);
    topk_hist["all"].fill(0);

    for (const auto& r : results) {
        if (r.retrieved.empty()) continue;
        double sim1 = r.retrieved[0].similarity;
        int bin1 = static_cast<int>(sim1 * N_BINS);
        if (bin1 >= N_BINS) bin1 = N_BINS - 1;
        if (bin1 < 0)       bin1 = 0;
        top1_hist[r.kind][bin1]++;
        top1_hist["all"][bin1]++;

        for (const auto& ex : r.retrieved) {
            int bn = static_cast<int>(ex.similarity * N_BINS);
            if (bn >= N_BINS) bn = N_BINS - 1;
            if (bn < 0)       bn = 0;
            topk_hist[r.kind][bn]++;
            topk_hist["all"][bn]++;
        }
    }

    fprintf(f, "type,kind,bucket_lo,bucket_hi,count\n");
    for (const auto& kv : top1_hist) {
        for (int i = 0; i < N_BINS; ++i) {
            double lo = i * 1.0 / N_BINS;
            double hi = (i + 1) * 1.0 / N_BINS;
            fprintf(f, "top1,%s,%.2f,%.2f,%d\n", kv.first.c_str(), lo, hi, kv.second[i]);
        }
    }
    for (const auto& kv : topk_hist) {
        for (int i = 0; i < N_BINS; ++i) {
            double lo = i * 1.0 / N_BINS;
            double hi = (i + 1) * 1.0 / N_BINS;
            fprintf(f, "topk,%s,%.2f,%.2f,%d\n", kv.first.c_str(), lo, hi, kv.second[i]);
        }
    }

    fclose(f);
    return true;
}

} // namespace r0
} // namespace pjev
