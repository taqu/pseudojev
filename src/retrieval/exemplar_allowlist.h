#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace pjev
{
namespace retrieval
{

// Exemplars that may be shown as demonstrations or vote in kNN: the training split of the exemplar
// database, restricted to rows with a one-hot answer. The database itself also holds validation, test
// and external rows and records no split, so eligibility comes from this list.
//
// File format (UTF-8 text): one exemplar ID per line as 16 lowercase hex digits (the XXH64 of the
// dataset ID, as stored in examples.id); lines starting with '#' are comments; blank lines are ignored.
using ExemplarAllowlist = std::unordered_set<int64_t>;

// "%016x" of the ID's unsigned bit pattern, e.g. "95ba7a22fb1c447d".
std::string format_exemplar_id(int64_t id);
// Inverse of format_exemplar_id(); false unless `hex` is exactly 16 hex digits.
bool parse_exemplar_id(const std::string& hex, int64_t& id);

// Fails on an unreadable file, a malformed line, or a file without any ID.
bool load_allowlist(const std::string& path, ExemplarAllowlist& out, std::string& err);

// Writes the IDs sorted by their hex form, after the header lines (each written as "# <line>").
bool write_allowlist(const std::string&              path,
                     std::vector<int64_t>            ids,
                     const std::vector<std::string>& header,
                     std::string&                    err);

} // namespace retrieval
} // namespace pjev
