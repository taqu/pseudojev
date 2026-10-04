#include "model_identity.h"
#include "../util/platform.h"
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <cstdio>
#endif

namespace pjev
{

// ---------------------------------------------------------------------------
// SHA-256 (public domain, macro-free implementation)
// ---------------------------------------------------------------------------
namespace {

static const uint32_t SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static const uint32_t SHA256_H0[8] = {
    0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
};

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

struct Sha256Ctx {
    uint32_t h[8];
    uint8_t  buf[64];
    uint32_t buflen;
    uint64_t total;  // total bytes processed
};

static void sha256_compress(uint32_t h[8], const uint8_t blk[64])
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)blk[4*i+0]<<24)|((uint32_t)blk[4*i+1]<<16)
              |((uint32_t)blk[4*i+2]<<8) | (uint32_t)blk[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15],18) ^ (w[i-15]>>3);
        uint32_t s1 = rotr32(w[i- 2],17) ^ rotr32(w[i- 2],19) ^ (w[i- 2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1  = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        uint32_t ch  = (e&f)^(~e&g);
        uint32_t T1  = hh + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0  = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        uint32_t maj = (a&b)^(a&c)^(b&c);
        uint32_t T2  = S0 + maj;
        hh=g; g=f; f=e; e=d+T1;
        d=c;  c=b; b=a; a=T1+T2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
    h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

static void sha256_init(Sha256Ctx& ctx)
{
    for (int i=0;i<8;i++) ctx.h[i]=SHA256_H0[i];
    ctx.buflen=0; ctx.total=0;
}

static void sha256_update(Sha256Ctx& ctx, const uint8_t* data, size_t len)
{
    ctx.total += len;
    for (size_t i=0; i<len; i++) {
        ctx.buf[ctx.buflen++] = data[i];
        if (ctx.buflen == 64) {
            sha256_compress(ctx.h, ctx.buf);
            ctx.buflen = 0;
        }
    }
}

static void sha256_final(Sha256Ctx& ctx, uint8_t out[32])
{
    // Append 0x80, then zeros, then 64-bit big-endian bit count
    uint64_t bitlen = ctx.total * 8;
    ctx.buf[ctx.buflen++] = 0x80;
    if (ctx.buflen > 56) {
        while (ctx.buflen < 64) ctx.buf[ctx.buflen++] = 0;
        sha256_compress(ctx.h, ctx.buf);
        ctx.buflen = 0;
    }
    while (ctx.buflen < 56) ctx.buf[ctx.buflen++] = 0;
    // big-endian bit length
    for (int i = 7; i >= 0; i--) {
        ctx.buf[56 + (7-i)] = (uint8_t)(bitlen >> (i*8));
    }
    sha256_compress(ctx.h, ctx.buf);
    // encode result big-endian
    for (int i = 0; i < 8; i++) {
        out[4*i+0] = (uint8_t)(ctx.h[i]>>24);
        out[4*i+1] = (uint8_t)(ctx.h[i]>>16);
        out[4*i+2] = (uint8_t)(ctx.h[i]>> 8);
        out[4*i+3] = (uint8_t)(ctx.h[i]>> 0);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------

std::string sha256_data(const std::string& data)
{
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, reinterpret_cast<const uint8_t*>(data.data()), data.size());
    uint8_t hash[32];
    sha256_final(ctx, hash);
    char hex[65];
    for (int i=0; i<32; ++i) snprintf(hex+2*i, 3, "%02x", hash[i]);
    hex[64] = '\0';
    return std::string(hex);
}

std::string sha256_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    Sha256Ctx ctx;
    sha256_init(ctx);
    uint8_t buf[8192];
    while (f) {
        f.read(reinterpret_cast<char*>(buf), sizeof(buf));
        std::streamsize n = f.gcount();
        if (n > 0) sha256_update(ctx, buf, (size_t)n);
    }
    uint8_t hash[32];
    sha256_final(ctx, hash);
    char hex[65];
    for (int i=0; i<32; ++i) snprintf(hex+2*i, 3, "%02x", hash[i]);
    hex[64] = '\0';
    return std::string(hex);
}

// ---------------------------------------------------------------------------

// Extract quantization tag from model_desc string e.g. "llama 1.7B Q4_K_M" -> "Q4_K_M"
static std::string extract_quant(const std::string& desc)
{
    // The quantization token is typically the last word and starts with Q or IQ or F
    std::istringstream ss(desc);
    std::string token, last;
    while (ss >> token) last = token;
    if (!last.empty() && (last[0]=='Q' || last[0]=='I' || last[0]=='F'))
        return last;
    return "";
}

ModelIdentity ModelIdentity::from_path(const std::string& path, ILlamaBackend* backend)
{
    ModelIdentity id;
    id.gguf_path = path;

    // File-level metadata
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (f) id.size_bytes = (int64_t)f.tellg();
    }
    id.sha256 = sha256_file(path);

    // Derive display name from filename stem
    std::string base = path;
    auto slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base = base.substr(slash + 1);
    auto dot = base.rfind('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    id.name = base;

    if (backend) {
        id.model_desc   = backend->model_desc();
        id.architecture = backend->model_meta_val("general.architecture");
        id.quantization = extract_quant(id.model_desc);
        id.n_params     = backend->model_n_params();
        // Prefer metadata name over filename stem
        std::string meta_name = backend->model_meta_val("general.name");
        if (!meta_name.empty()) id.name = meta_name;
    }

    return id;
}

nlohmann::json ModelIdentity::to_json() const
{
    return {
        {"name",             name},
        {"source",           source},
        {"gguf_path",        gguf_path},
        {"sha256",           sha256},
        {"size_bytes",       size_bytes},
        {"quantization",     quantization},
        {"architecture",     architecture},
        {"model_desc",       model_desc},
        {"n_params",         n_params},
        {"license",          license},
        {"pjev_commit",      pjev_commit},
        {"llamacpp_revision",llamacpp_revision}
    };
}

// ---------------------------------------------------------------------------

CandidateCompatibility probe_candidates(ILlamaBackend& backend,
                                         const std::vector<std::string>& candidates)
{
    CandidateCompatibility cc;
    bool all_ok = true;
    for (const auto& text : candidates) {
        CandidateTokenRecord rec;
        rec.text = text;
        auto toks = backend.tokenize(text, false, false);
        rec.token_count  = (int)toks.size();
        rec.single_token = (rec.token_count == 1);
        rec.token_id     = rec.single_token ? toks[0] : -1;
        if (!rec.single_token) {
            cc.issues.push_back("\"" + text + "\" tokenises to " +
                                std::to_string(rec.token_count) + " tokens (not single-token)");
            all_ok = false;
        }
        cc.records.push_back(rec);
    }
    cc.all_single_token = all_ok;
    return cc;
}

nlohmann::json CandidateCompatibility::to_json() const
{
    nlohmann::json recs = nlohmann::json::array();
    for (const auto& r : records) {
        recs.push_back({
            {"text",         r.text},
            {"token_id",     r.token_id},
            {"token_count",  r.token_count},
            {"single_token", r.single_token}
        });
    }
    return {
        {"all_single_token", all_single_token},
        {"records",          recs},
        {"issues",           issues}
    };
}

// ---------------------------------------------------------------------------
// Platform utilities (declared in util/platform.h)
// ---------------------------------------------------------------------------

int64_t get_rss_bytes()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (int64_t)pmc.WorkingSetSize;
    return -1;
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[128];
    int64_t rss = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            long kb = 0;
            std::sscanf(line + 6, " %ld", &kb);
            rss = (int64_t)kb * 1024;
            break;
        }
    }
    std::fclose(f);
    return rss;
#else
    return -1;
#endif
}

std::string get_os_name()
{
#ifdef _WIN32
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "unknown";
#endif
}

} // namespace pjev
