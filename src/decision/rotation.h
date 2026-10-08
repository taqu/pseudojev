#ifndef INC_PJEV_ROTATION_H_
#define INC_PJEV_ROTATION_H_
// Cyclic option rotation (E2) for choice.
//
// Rotation r over N semantic options binds candidate position j (A, B, C, ...) to
//   semantic index (j + r) mod N
// so over r = 0..N-1 every semantic option appears exactly once at every position.
#include <cstdint>
#include <vector>

namespace pjev
{
struct RotationMapping
{
    int32_t rotation = 0;
    std::vector<int32_t> cand_to_sem; // candidate position -> semantic option index
    std::vector<int32_t> sem_to_cand; // semantic option index -> candidate position
};

inline RotationMapping make_cyclic_rotation(int32_t n, int32_t r)
{
    RotationMapping m;
    m.rotation = r;
    m.cand_to_sem.resize(n);
    m.sem_to_cand.resize(n);
    for(int32_t j = 0; j < n; j++) {
        int32_t s = (j + r) % n;
        m.cand_to_sem[j] = s;
        m.sem_to_cand[s] = j;
    }
    return m;
}

// Reorders a per-semantic-option list into candidate order for this rotation.
template<class T>
std::vector<T> to_candidate_order(const std::vector<T>& semantic, const RotationMapping& m)
{
    std::vector<T> out;
    out.reserve(m.cand_to_sem.size());
    for(int32_t s: m.cand_to_sem) out.push_back(semantic[s]);
    return out;
}

// Maps candidate-position scores back to semantic option order.
template<class T>
std::vector<T> to_semantic_order(const std::vector<T>& candidate, const RotationMapping& m)
{
    std::vector<T> out(candidate.size());
    for(size_t j = 0; j < candidate.size() && j < m.cand_to_sem.size(); j++)
        out[m.cand_to_sem[j]] = candidate[j];
    return out;
}
} // namespace pjev
#endif // INC_PJEV_ROTATION_H_
