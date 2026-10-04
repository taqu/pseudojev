#ifndef INC_PJEV_JEV_API_H_
#define INC_PJEV_JEV_API_H_
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace pjev {

class DecisionEngine;

// Stable machine-readable error codes returned in {"error":{"code":"...","message":"..."}}
namespace ErrorCode {
    constexpr const char* INVALID_JSON          = "invalid_json";
    constexpr const char* INVALID_REQUEST       = "invalid_request";
    constexpr const char* UNSUPPORTED_PRIMITIVE = "unsupported_primitive";
    constexpr const char* REQUEST_TOO_LARGE     = "request_too_large";
    constexpr const char* TIMEOUT               = "timeout";
    constexpr const char* MODEL_ERROR           = "model_error";
    constexpr const char* INFERENCE_ERROR       = "inference_error";
    constexpr const char* INTERNAL_ERROR        = "internal_error";
    constexpr const char* SERVER_BUSY           = "server_busy";
    constexpr const char* CONTEXT_TOO_LONG      = "context_too_long";
}

class JevApiHandler {
public:
    // max_queued: reject with server_busy if this many requests are already active.
    JevApiHandler(DecisionEngine& engine,
                  const std::string& model_name = "pseudojev",
                  int max_queued = 8,
                  bool kv_reuse = true,
                  bool batch_questions = true);

    // Parse Jev request, run inference, return Jev response.
    // Sets status_out to HTTP status and response_out to JSON body.
    void handle(const std::string& request_body,
                int32_t& status_out,
                std::string& response_out);

private:
    DecisionEngine&        engine_;
    std::string            model_name_;
    std::mutex             mutex_;
    std::atomic<int>       active_{0};
    int                    max_queued_;
    std::atomic<uint64_t>  req_counter_{0};
    bool                   kv_reuse_;
    bool                   batch_questions_;
};

} // namespace pjev
#endif //INC_PJEV_JEV_API_H_
