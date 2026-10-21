#ifndef INC_PJEV_KNN_H_
#define INC_PJEV_KNN_H_
#include <string>
#include <vector>

namespace pjev
{
namespace routing
{

// Estimated cosine similarity of two unit vectors from the Hamming distance of their sign-quantized
// forms: cos(pi * hamming / dim).
double estimated_cosine(double hamming, int dim);

enum class MapStatus { Ok, Unsupported, NoMatch };

struct LabelMap {
    MapStatus status    = MapStatus::Unsupported;
    int       candidate = -1;  // query candidate index
};

// Translates an exemplar's stored answer into the query's candidate through the answer's semantic
// identity, never through its letter:
//   noul   - the answer option text ("no" / "yes") selects the query option with the same text;
//   score  - the answer level index selects the same level of the query (level counts must match);
//   choice - unsupported (option identifiers are per-task item names).
// query_options are in Open Jev order; query_option_to_candidate maps them to candidate indices.
LabelMap map_label(const std::string& query_kind, const std::vector<std::string>& query_options,
                   const std::vector<int>& query_option_to_candidate, const std::string& ex_kind,
                   const std::vector<std::string>& ex_options, int ex_answer_index);

struct VoteInput {
    int    label      = -1;   // -1 = unmapped (abstains)
    double similarity = 0.0;  // estimated cosine
};

struct VoteResult {
    bool   ok           = false;  // false: no mapped neighbour among the first k
    int    label        = -1;
    int    used         = 0;      // mapped neighbours that voted
    double total_weight = 0.0;
    std::vector<double> weight;   // per label (index = candidate), sized to the largest voting label + 1
};

// Similarity-weighted vote over the first min(k, size) neighbours (rank order):
// w_i = exp((s_i - s_max) / tau), s_max = the largest similarity among the voters, so every weight is
// in (0, 1]. tau <= 0 gives uniform weights. Ties in total weight (within 1e-12 relative) go to the
// label of the best-ranked neighbour carrying one of the tied labels.
VoteResult vote(const std::vector<VoteInput>& neighbours, int k, double tau);

} // namespace routing
} // namespace pjev
#endif // INC_PJEV_KNN_H_
