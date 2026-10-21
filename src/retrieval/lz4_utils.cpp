#include "lz4_utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "lz4/lz4frame.h"

namespace pjev
{
namespace lz4
{

std::vector<uint8_t> compress(std::string_view input)
{
    size_t bound = LZ4F_compressFrameBound(input.size(), nullptr);
    if (LZ4F_isError(bound))
        throw std::runtime_error(std::string("lz4 compress bound: ") + LZ4F_getErrorName(bound));

    std::vector<uint8_t> out(bound);

    // Set contentSize in preferences so decompressor can preallocate exactly.
    LZ4F_preferences_t prefs = LZ4F_INIT_PREFERENCES;
    prefs.frameInfo.contentSize = input.size();

    size_t n = LZ4F_compressFrame(
        out.data(), out.size(),
        input.data(), input.size(),
        &prefs);
    if (LZ4F_isError(n))
        throw std::runtime_error(std::string("lz4 compress: ") + LZ4F_getErrorName(n));

    out.resize(n);
    return out;
}

std::string decompress(const void* data, size_t size, std::string& err_out)
{
    err_out.clear();

    if (size == 0) return {};

    LZ4F_dctx* dctx = nullptr;
    LZ4F_errorCode_t ec = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
    if (LZ4F_isError(ec)) {
        err_out = std::string("lz4 decompress ctx: ") + LZ4F_getErrorName(ec);
        return {};
    }

    // Consume the frame header to learn the content size.
    LZ4F_frameInfo_t info;
    memset(&info, 0, sizeof(info));
    size_t header_consumed = size;
    ec = LZ4F_getFrameInfo(dctx, &info, data, &header_consumed);
    if (LZ4F_isError(ec)) {
        err_out = std::string("lz4 decompress frame info: ") + LZ4F_getErrorName(ec);
        LZ4F_freeDecompressionContext(dctx);
        return {};
    }

    // Determine output capacity.  contentSize == 0 means the compressor did not
    // store the content size; fall back to an estimate based on compressed size.
    size_t capacity;
    if (info.contentSize > 0) {
        if (info.contentSize > kMaxDecompressedSize) {
            err_out = "lz4 decompress: decompressed size exceeds limit";
            LZ4F_freeDecompressionContext(dctx);
            return {};
        }
        capacity = static_cast<size_t>(info.contentSize);
    } else {
        capacity = std::min(size * 4 + 65536, kMaxDecompressedSize);
    }

    std::string out(capacity, '\0');
    size_t total_written = 0;
    bool   frame_done    = false;

    const uint8_t* src = static_cast<const uint8_t*>(data) + header_consumed;
    size_t src_remaining = size - header_consumed;

    while (src_remaining > 0) {
        size_t dst_avail = capacity - total_written;
        if (dst_avail == 0) {
            // Output buffer is full; grow it.
            size_t new_cap = capacity * 2;
            if (new_cap > kMaxDecompressedSize || new_cap < capacity) {
                err_out = "lz4 decompress: decompressed size exceeds limit";
                LZ4F_freeDecompressionContext(dctx);
                return {};
            }
            out.resize(new_cap);
            capacity = new_cap;
            dst_avail = capacity - total_written;
        }

        size_t chunk_out = dst_avail;
        size_t chunk_in  = src_remaining;

        size_t ret = LZ4F_decompress(
            dctx,
            &out[total_written], &chunk_out,
            src, &chunk_in,
            nullptr);
        if (LZ4F_isError(ret)) {
            err_out = std::string("lz4 decompress: ") + LZ4F_getErrorName(ret);
            LZ4F_freeDecompressionContext(dctx);
            return {};
        }

        total_written  += chunk_out;
        src            += chunk_in;
        src_remaining  -= chunk_in;

        if (ret == 0) {
            frame_done = true;
            break;  // frame complete
        }
    }

    LZ4F_freeDecompressionContext(dctx);

    // If we consumed all input but never saw the end-mark, the frame is truncated.
    if (!frame_done) {
        err_out = "lz4 decompress: truncated frame (end-mark not reached)";
        return {};
    }

    out.resize(total_written);
    return out;
}

} // namespace lz4
} // namespace pjev
