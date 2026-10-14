#pragma once
#include <cstdint>
#include <cstring>
#include <string_view>
#include "lz4/xxhash.h"

namespace pjev { namespace filldb {

// Hash a textual exemplar identifier to a signed 64-bit integer for SQLite storage.
// Uses XXH64 with seed=0 for deterministic, platform-independent results.
inline int64_t hash_exemplar_id(std::string_view text)
{
    static_assert(sizeof(XXH64_hash_t) == sizeof(int64_t), "xxhash64 size mismatch");
    XXH64_hash_t h = XXH64(text.data(), text.size(), 0);
    int64_t result;
    std::memcpy(&result, &h, sizeof(result));
    return result;
}

}} // namespace pjev::filldb
