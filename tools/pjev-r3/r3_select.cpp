#include "r3_select.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>

#include "lz4/xxhash.h"

namespace pjev
{
namespace r3
{

const char* to_string(ExampleSelection s)
{
    switch (s) {
    case ExampleSelection::Nearest:       return "nearest";
    case ExampleSelection::LengthMatched: return "length_matched";
    default:                              return "random";
    }
}

std::string pool_key(const std::string& kind, int number_of_levels)
{
    if (kind == "score") return "score/" + std::to_string(number_of_levels);
    return kind;
}

ExemplarPool ExemplarPool::build(const std::vector<PoolRow>& rows, const std::unordered_set<int64_t>& train_ids,
                                 PoolStats* stats)
{
    PoolStats s;
    ExemplarPool pool;
    for (const auto& r : rows) {
        s.rows++;
        if (!train_ids.count(r.id)) { s.not_in_train++; continue; }
        if (r.answer_index < 0)     { s.no_one_hot++;   continue; }
        pool.by_key_[pool_key(r.kind, r.number_of_levels)].push_back(r.id);
    }
    for (auto& [key, ids] : pool.by_key_) {
        std::sort(ids.begin(), ids.end());
        const auto end = std::unique(ids.begin(), ids.end());
        s.duplicate_ids += static_cast<int64_t>(ids.end() - end);
        ids.erase(end, ids.end());
        s.eligible += static_cast<int64_t>(ids.size());
    }
    if (stats) *stats = s;
    return pool;
}

const std::vector<int64_t>& ExemplarPool::candidates(const std::string& key) const
{
    static const std::vector<int64_t> empty;
    auto it = by_key_.find(key);
    return it == by_key_.end() ? empty : it->second;
}

bool ExemplarPool::contains(const std::string& key, int64_t id) const
{
    const auto& c = candidates(key);
    return std::binary_search(c.begin(), c.end(), id);
}

uint64_t random_index(uint64_t seed, const std::string& query_id, uint64_t n, const char* version)
{
    if (n == 0) return 0;
    const std::string text = std::string(version) + "|" + std::to_string(seed) + "|" + query_id;
    // n is at most ~1e5, so the modulo bias of a 64-bit hash is below 1e-14.
    return static_cast<uint64_t>(XXH64(text.data(), text.size(), 0)) % n;
}

RandomPick select_random(const std::vector<int64_t>& candidates, int64_t self_id, uint64_t seed,
                         const std::string& query_id)
{
    RandomPick p;
    const auto self = std::lower_bound(candidates.begin(), candidates.end(), self_id);
    const bool has_self = self != candidates.end() && *self == self_id;
    const size_t self_pos = static_cast<size_t>(self - candidates.begin());
    p.pool_size = static_cast<int64_t>(candidates.size()) - (has_self ? 1 : 0);
    if (p.pool_size <= 0) return p;
    size_t idx = static_cast<size_t>(random_index(seed, query_id, static_cast<uint64_t>(p.pool_size)));
    p.index = static_cast<int64_t>(idx);
    if (has_self && idx >= self_pos) idx++;  // skip the query itself
    p.id = candidates[idx];
    p.ok = true;
    return p;
}

LengthPool LengthPool::build(const std::vector<std::pair<int64_t, int>>& id_tokens)
{
    LengthPool p;
    for (const auto& [id, tok] : id_tokens) p.by_length.emplace_back(tok, id);
    std::sort(p.by_length.begin(), p.by_length.end());
    p.by_length.erase(std::unique(p.by_length.begin(), p.by_length.end()), p.by_length.end());
    return p;
}

LengthPick select_length_matched(const LengthPool& pool, int64_t self_id, int target, int window, uint64_t seed,
                                 const std::string& query_id)
{
    LengthPick p;
    const auto& v = pool.by_length;
    if (window <= 0 || v.empty()) return p;
    // Expand outwards from the target until `window` candidates are held and every remaining candidate
    // is strictly farther than the farthest held one; then rank by (distance, id) and cut.
    const auto mid = std::lower_bound(v.begin(), v.end(), std::make_pair(target, INT64_MIN));
    ptrdiff_t lo = (mid - v.begin()) - 1, hi = mid - v.begin();
    std::vector<std::pair<int, int64_t>> held;  // (distance, id) plus tokens below
    std::vector<int> held_tokens;
    auto dist = [&](ptrdiff_t i) { return std::abs(v[static_cast<size_t>(i)].first - target); };
    const ptrdiff_t n = static_cast<ptrdiff_t>(v.size());
    int max_held = -1;
    for (;;) {
        const bool can_lo = lo >= 0, can_hi = hi < n;
        if (!can_lo && !can_hi) break;
        const int dlo = can_lo ? dist(lo) : INT32_MAX, dhi = can_hi ? dist(hi) : INT32_MAX;
        const int d = std::min(dlo, dhi);
        if (static_cast<int>(held.size()) >= window && d > max_held) break;
        const ptrdiff_t i = (dlo <= dhi) ? lo-- : hi++;
        if (v[static_cast<size_t>(i)].second == self_id) continue;
        held.emplace_back(d, v[static_cast<size_t>(i)].second);
        held_tokens.push_back(v[static_cast<size_t>(i)].first);
        max_held = std::max(max_held, d);
    }
    std::vector<size_t> order(held.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return held[a] < held[b]; });
    if (order.size() > static_cast<size_t>(window)) order.resize(static_cast<size_t>(window));
    if (order.empty()) return p;
    // Draw order inside the window is by ID, so the pick depends only on the window's membership.
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return held[a].second < held[b].second; });
    p.window_size = static_cast<int64_t>(order.size());
    p.window_lo = INT32_MAX;
    p.window_hi = INT32_MIN;
    for (size_t k : order) { p.window_lo = std::min(p.window_lo, held_tokens[k]); p.window_hi = std::max(p.window_hi, held_tokens[k]); }
    const size_t idx = static_cast<size_t>(random_index(seed, query_id, order.size(), kLengthMatchedVersion));
    p.index  = static_cast<int64_t>(idx);
    p.id     = held[order[idx]].second;
    p.tokens = held_tokens[order[idx]];
    p.ok     = true;
    return p;
}

} // namespace r3
} // namespace pjev
