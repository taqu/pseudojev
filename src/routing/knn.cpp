#include "knn.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace pjev
{
namespace routing
{

double estimated_cosine(double hamming, int dim)
{
    static const double kPi = 3.14159265358979323846;
    return dim > 0 ? std::cos(kPi * hamming / dim) : 0.0;
}

LabelMap map_label(const std::string& query_kind, const std::vector<std::string>& query_options,
                   const std::vector<int>& query_option_to_candidate, const std::string& ex_kind,
                   const std::vector<std::string>& ex_options, int ex_answer_index)
{
    LabelMap m;
    if (query_kind != "noul" && query_kind != "score") return m;  // choice: unsupported
    if (ex_kind != query_kind || ex_answer_index < 0 || ex_answer_index >= static_cast<int>(ex_options.size()) ||
        query_option_to_candidate.size() != query_options.size()) {
        m.status = MapStatus::NoMatch;
        return m;
    }
    if (query_kind == "noul") {
        const std::string& ans = ex_options[static_cast<size_t>(ex_answer_index)];
        if (ans != "no" && ans != "yes") {
            m.status = MapStatus::NoMatch;
            return m;
        }
        for (size_t i = 0; i < query_options.size(); ++i) {
            if (query_options[i] == ans) {
                m.status    = MapStatus::Ok;
                m.candidate = query_option_to_candidate[i];
                return m;
            }
        }
        m.status = MapStatus::NoMatch;
        return m;
    }
    // score: ordinal level index on a scale with the same number of levels.
    if (ex_options.size() != query_options.size()) {
        m.status = MapStatus::NoMatch;
        return m;
    }
    m.status    = MapStatus::Ok;
    m.candidate = query_option_to_candidate[static_cast<size_t>(ex_answer_index)];
    return m;
}

VoteResult vote(const std::vector<VoteInput>& neighbours, int k, double tau)
{
    VoteResult r;
    const size_t n = std::min(neighbours.size(), static_cast<size_t>(std::max(k, 0)));
    double s_max = -1e300;
    for (size_t i = 0; i < n; ++i)
        if (neighbours[i].label >= 0) s_max = std::max(s_max, neighbours[i].similarity);
    std::map<int, double> weight;     // label -> total weight
    std::map<int, size_t> first_rank; // label -> best rank carrying it
    for (size_t i = 0; i < n; ++i) {
        const auto& nb = neighbours[i];
        if (nb.label < 0) continue;
        double w = 1.0;
        if (tau > 0.0) w = std::exp((nb.similarity - s_max) / tau);
        weight[nb.label] += w;
        first_rank.emplace(nb.label, i);
        r.used++;
        r.total_weight += w;
    }
    if (r.used == 0) return r;
    double best_w = 0.0;
    for (const auto& [label, w] : weight) {
        if (r.label < 0) {
            r.label = label;
            best_w  = w;
            continue;
        }
        const double tol = 1e-12 * std::max(w, best_w);
        if (w > best_w + tol || (std::fabs(w - best_w) <= tol && first_rank[label] < first_rank[r.label])) {
            r.label = label;
            best_w  = w;
        }
    }
    r.ok = true;
    r.weight.assign(static_cast<size_t>(weight.rbegin()->first) + 1, 0.0);
    for (const auto& [label, w] : weight) r.weight[static_cast<size_t>(label)] = w;
    return r;
}

} // namespace routing
} // namespace pjev
