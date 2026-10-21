#pragma once
#include "../decision/decision_engine.h"
#include "../distribution/release_config.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>

namespace pjev {

// Version 2: task-aware routing (retrieval context in requests, error codes in responses).
// The worker endpoint name carries the version, so workers of other versions are never contacted.
static constexpr int IPC_PROTOCOL_VERSION = 2;
static constexpr uint32_t IPC_MAX_FRAME_BYTES = 1u * 1024u * 1024u;

// Framing: [4-byte LE uint32 length][JSON bytes]

struct WorkerInfo {
    int      protocol_version = IPC_PROTOCOL_VERSION;
    std::string pjev_version;
    int32_t  pid = 0;
    std::string model_identity;
    std::string config_hash; // canonical string of effective config
    int64_t  start_time_us = 0;
};

// Serialize DecisionInput/Output to/from JSON for IPC
nlohmann::json ipc_input_to_json(const DecisionInput& in);
DecisionInput  ipc_input_from_json(const nlohmann::json& j);
nlohmann::json ipc_output_to_json(const DecisionOutput& out);
DecisionOutput ipc_output_from_json(const nlohmann::json& j);

// WorkerInfo JSON
nlohmann::json ipc_worker_info_to_json(const WorkerInfo& info);
WorkerInfo     ipc_worker_info_from_json(const nlohmann::json& j);

// Canonical string of the effective worker configuration. A client uses a running worker only when
// the worker reports the same string.
std::string make_config_hash(const RuntimePaths& paths, int threads, int ctx_size);

} // namespace pjev
