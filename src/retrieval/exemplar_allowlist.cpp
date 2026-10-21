#include "exemplar_allowlist.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <fstream>

namespace pjev
{
namespace retrieval
{

std::string format_exemplar_id(int64_t id)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64, static_cast<uint64_t>(id));
    return buf;
}

bool parse_exemplar_id(const std::string& hex, int64_t& id)
{
    if (hex.size() != 16) return false;
    uint64_t v = 0;
    for (char c : hex) {
        v <<= 4;
        if (c >= '0' && c <= '9') v |= static_cast<uint64_t>(c - '0');
        else if (c >= 'a' && c <= 'f') v |= static_cast<uint64_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= static_cast<uint64_t>(c - 'A' + 10);
        else return false;
    }
    id = static_cast<int64_t>(v);
    return true;
}

bool load_allowlist(const std::string& path, ExemplarAllowlist& out, std::string& err)
{
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        err = "cannot open exemplar allowlist: " + path;
        return false;
    }
    std::string line;
    size_t line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        int64_t id = 0;
        if (!parse_exemplar_id(line, id)) {
            err = "exemplar allowlist " + path + ": malformed ID on line " + std::to_string(line_no);
            out.clear();
            return false;
        }
        out.insert(id);
    }
    if (out.empty()) {
        err = "exemplar allowlist " + path + " contains no IDs";
        return false;
    }
    return true;
}

bool write_allowlist(const std::string&              path,
                     std::vector<int64_t>            ids,
                     const std::vector<std::string>& header,
                     std::string&                    err)
{
    std::sort(ids.begin(), ids.end(),
              [](int64_t a, int64_t b) { return static_cast<uint64_t>(a) < static_cast<uint64_t>(b); });
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        err = "cannot write " + path;
        return false;
    }
    for (const auto& h : header) f << "# " << h << "\n";
    for (int64_t id : ids) f << format_exemplar_id(id) << "\n";
    if (!f.good()) {
        err = "write failed: " + path;
        return false;
    }
    return true;
}

} // namespace retrieval
} // namespace pjev
