#pragma once
#include "../decision/decision_engine.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>

namespace pjev {

static constexpr int IPC_PROTOCOL_VERSION = 1;
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

// Config hash: canonical string from model_path + threads + ctx_size + calibration_path
std::string make_config_hash(const std::string& model_path,
                              int threads, int ctx_size,
                              const std::string& calibration_path);

} // namespace pjev
