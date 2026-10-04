#ifndef INC_PJEV_SERVER_CONFIG_H_
#define INC_PJEV_SERVER_CONFIG_H_
#include <cstddef>
#include <cstdint>
#include <string>

namespace pjev {

struct ServerConfig {
    std::string  host                  = "127.0.0.1";
    uint16_t     port                  = 8080;
    size_t       request_body_limit    = 1u * 1024u * 1024u; // 1 MiB
    int          read_timeout_secs     = 30;
    // Soft limit: log warning if exceeded. Hard cancellation of llama.cpp is unsafe.
    int          execution_timeout_secs = 120;
    int          shutdown_grace_secs   = 10;
    int          max_queued_requests   = 8;
    bool kv_reuse        = true;   // reuse KV prefix for multi-question same-state requests
    bool batch_questions = true;   // collect all questions then run decide_batch()
};

} // namespace pjev
#endif
