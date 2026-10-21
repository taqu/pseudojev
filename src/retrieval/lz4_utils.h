#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace pjev
{
namespace lz4
{

// Maximum decompressed size enforced during decompression.
static constexpr size_t kMaxDecompressedSize = 64ULL * 1024 * 1024;  // 64 MB

// Compress UTF-8 input using the LZ4 Frame format (LZ4F_compressFrame).
// Returns compressed bytes. Throws std::runtime_error on LZ4 failure.
std::vector<uint8_t> compress(std::string_view input);

// Decompress a single LZ4 Frame.
// Returns the original UTF-8 string.
// On any error (truncated frame, corrupt data, size limit exceeded),
// sets err_out to a non-empty message and returns "".
std::string decompress(const void* data, size_t size, std::string& err_out);

} // namespace lz4
} // namespace pjev
