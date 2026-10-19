#pragma once

#include <cmath>

namespace pjev
{
namespace r2
{

// R2 activation rule, shared by the online harness and the offline replay.
// Similarity is the estimated cosine of the already-selected Top-1 exemplar, in [-1, 1].
enum class GateDecision { Activated, Rejected, NoExemplar, InvalidSimilarity };

inline const char* to_string(GateDecision d)
{
    switch (d) {
    case GateDecision::Activated:         return "activated";
    case GateDecision::Rejected:          return "rejected";
    case GateDecision::NoExemplar:        return "no_exemplar";
    case GateDecision::InvalidSimilarity: return "invalid_similarity";
    }
    return "?";
}

inline bool valid_similarity(double s) { return std::isfinite(s) && s >= -1.0 && s <= 1.0; }

// Thresholds live in the same range as the metric; anything else is a configuration error.
inline bool valid_threshold(double t) { return std::isfinite(t) && t >= -1.0 && t <= 1.0; }

// Equality activates: similarity >= threshold.
inline GateDecision gate_decide(bool exemplar_available, double similarity, double threshold)
{
    if (!exemplar_available) return GateDecision::NoExemplar;
    if (!valid_similarity(similarity)) return GateDecision::InvalidSimilarity;
    return similarity >= threshold ? GateDecision::Activated : GateDecision::Rejected;
}

} // namespace r2
} // namespace pjev
