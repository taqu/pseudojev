// Phase 10 worker unit tests: IPC framing, protocol serialization, runtime dir, config hash, WorkerInfo JSON
#include "worker/ipc_protocol.h"
#include "worker/runtime_dir.h"
#include "worker/local_transport.h"
#include "decision/decision_engine.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>

using namespace pjev;
using json = nlohmann::json;

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fails++; } \
    else         { printf("pass: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// MemoryConn: in-memory ILocalConn pair for framing tests
// ---------------------------------------------------------------------------

// A simple in-memory connection backed by a shared byte buffer.
// Two MemoryConn objects share bidirectional buffers.
struct SharedBuffers {
    std::string a_to_b; // data sent by A, received by B
    std::string b_to_a; // data sent by B, received by A
};

class MemoryConn : public ILocalConn {
public:
    // side: 0 = A side (writes to a_to_b, reads from b_to_a)
    //       1 = B side (writes to b_to_a, reads from a_to_b)
    MemoryConn(std::shared_ptr<SharedBuffers> bufs, int side)
        : bufs_(bufs), side_(side), open_(true) {}

    bool send_frame(const std::string& data) override {
        if (!open_) return false;
        uint32_t len = (uint32_t)data.size();
        if (len > IPC_MAX_FRAME_BYTES) return false;
        uint8_t hdr[4];
        hdr[0] = (uint8_t)(len & 0xFF);
        hdr[1] = (uint8_t)((len >> 8) & 0xFF);
        hdr[2] = (uint8_t)((len >> 16) & 0xFF);
        hdr[3] = (uint8_t)((len >> 24) & 0xFF);
        std::string& out_buf = (side_ == 0) ? bufs_->a_to_b : bufs_->b_to_a;
        out_buf.append((char*)hdr, 4);
        out_buf.append(data);
        return true;
    }

    bool recv_frame(std::string& data, int /*timeout_ms*/ = 30000) override {
        if (!open_) return false;
        std::string& in_buf = (side_ == 0) ? bufs_->b_to_a : bufs_->a_to_b;
        if (in_buf.size() < 4) return false;
        uint32_t len = (uint32_t)(uint8_t)in_buf[0]
                     | ((uint32_t)(uint8_t)in_buf[1] << 8)
                     | ((uint32_t)(uint8_t)in_buf[2] << 16)
                     | ((uint32_t)(uint8_t)in_buf[3] << 24);
        if (len > IPC_MAX_FRAME_BYTES) return false;
        if (in_buf.size() < 4 + len) return false;
        data = in_buf.substr(4, len);
        in_buf.erase(0, 4 + len);
        return true;
    }

    void close() override { open_ = false; }
    bool is_open() const override { return open_; }

private:
    std::shared_ptr<SharedBuffers> bufs_;
    int  side_;
    bool open_;
};

// Create a pair of connected MemoryConn objects
static std::pair<std::unique_ptr<ILocalConn>, std::unique_ptr<ILocalConn>>
make_memory_pair()
{
    auto bufs = std::make_shared<SharedBuffers>();
    auto a = std::make_unique<MemoryConn>(bufs, 0);
    auto b = std::make_unique<MemoryConn>(bufs, 1);
    return {std::move(a), std::move(b)};
}

// ---------------------------------------------------------------------------
// Test 1: IPC framing - send/recv roundtrip
// ---------------------------------------------------------------------------
static void test_ipc_framing()
{
    auto [conn_a, conn_b] = make_memory_pair();

    // A sends, B receives
    std::string msg1 = "hello world";
    CHECK(conn_a->send_frame(msg1), "framing: send_frame returns true");

    std::string got1;
    CHECK(conn_b->recv_frame(got1), "framing: recv_frame returns true");
    CHECK(got1 == msg1, "framing: content matches");

    // B sends JSON, A receives
    json j = {{"type", "ping"}, {"value", 42}};
    std::string msg2 = j.dump();
    CHECK(conn_b->send_frame(msg2), "framing: B sends JSON");

    std::string got2;
    CHECK(conn_a->recv_frame(got2), "framing: A receives JSON");
    CHECK(got2 == msg2, "framing: JSON content matches");

    // Empty frame
    CHECK(conn_a->send_frame(""), "framing: empty send");
    std::string got_empty;
    CHECK(conn_b->recv_frame(got_empty), "framing: empty recv");
    CHECK(got_empty.empty(), "framing: empty content");

    // Multiple frames in sequence
    for (int i = 0; i < 5; i++) {
        std::string msg = "frame-" + std::to_string(i);
        CHECK(conn_a->send_frame(msg), ("framing: multi send " + std::to_string(i)).c_str());
    }
    for (int i = 0; i < 5; i++) {
        std::string got;
        CHECK(conn_b->recv_frame(got), ("framing: multi recv " + std::to_string(i)).c_str());
        CHECK(got == "frame-" + std::to_string(i), ("framing: multi content " + std::to_string(i)).c_str());
    }
}

// ---------------------------------------------------------------------------
// Test 2: IPC protocol serialization - DecisionInput roundtrip
// ---------------------------------------------------------------------------
static void test_ipc_protocol_serialization()
{
    // Test DecisionInput serialization
    DecisionInput in;
    in.type     = "choice";
    in.state    = "The sky is blue.";
    in.question = "What color is the sky?";
    in.options  = {{"A", "Red"}, {"B", "Blue"}, {"C", "Green"}};
    in.prior_correction = false;

    json j = ipc_input_to_json(in);
    CHECK(j["type"] == "choice",           "input_json: type field");
    CHECK(j["state"] == "The sky is blue.", "input_json: state field");
    CHECK(j["question"] == "What color is the sky?", "input_json: question field");
    CHECK(j["options"].is_array(),         "input_json: options is array");
    CHECK(j["options"].size() == 3,        "input_json: options size");

    DecisionInput in2 = ipc_input_from_json(j);
    CHECK(in2.type == in.type,             "input_rt: type matches");
    CHECK(in2.state == in.state,           "input_rt: state matches");
    CHECK(in2.question == in.question,     "input_rt: question matches");
    CHECK(in2.options.size() == in.options.size(), "input_rt: options size matches");
    CHECK(in2.options[0].first == "A",     "input_rt: option[0].key = A");
    CHECK(in2.options[1].second == "Blue", "input_rt: option[1].desc = Blue");

    // Test noul input
    DecisionInput noul_in;
    noul_in.type     = "noul";
    noul_in.question = "Is this a test?";
    noul_in.options  = {{"false", ""}, {"true", ""}};

    json noul_j = ipc_input_to_json(noul_in);
    DecisionInput noul_rt = ipc_input_from_json(noul_j);
    CHECK(noul_rt.type == "noul",                   "noul_input_rt: type");
    CHECK(noul_rt.options.size() == 2,              "noul_input_rt: options count");
    CHECK(noul_rt.options[1].first == "true",       "noul_input_rt: true key");

    // Test DecisionOutput serialization
    DecisionOutput out;
    out.ok             = true;
    out.error          = "";
    out.selected       = 1;
    out.p_true         = 0.75;
    out.expected_score = 2.5;
    out.probs          = {0.1, 0.75, 0.15};
    out.raw_probs      = {0.12, 0.73, 0.15};
    out.keys           = {"A", "B", "C"};
    out.prompt_token_count = 42;

    json out_j = ipc_output_to_json(out);
    CHECK(out_j["ok"] == true,             "output_json: ok");
    CHECK(out_j["selected"] == 1,          "output_json: selected");
    CHECK(out_j["p_true"].get<double>() > 0.7, "output_json: p_true");
    CHECK(out_j["probs"].size() == 3,      "output_json: probs size");
    CHECK(out_j["keys"].size() == 3,       "output_json: keys size");

    DecisionOutput out2 = ipc_output_from_json(out_j);
    CHECK(out2.ok == true,                 "output_rt: ok");
    CHECK(out2.selected == 1,              "output_rt: selected");
    CHECK(out2.p_true > 0.7,              "output_rt: p_true");
    CHECK(out2.probs.size() == 3,          "output_rt: probs size");
    CHECK(out2.keys[0] == "A",            "output_rt: key[0]");
    CHECK(out2.prompt_token_count == 42,   "output_rt: token_count");

    // Error case
    DecisionOutput err_out;
    err_out.ok    = false;
    err_out.error = "something went wrong";
    json err_j = ipc_output_to_json(err_out);
    DecisionOutput err_rt = ipc_output_from_json(err_j);
    CHECK(err_rt.ok == false,              "error_output_rt: ok=false");
    CHECK(err_rt.error == "something went wrong", "error_output_rt: error message");
}

// ---------------------------------------------------------------------------
// Test 3: runtime dir - get_worker_endpoint returns non-empty with correct prefix
// ---------------------------------------------------------------------------
static void test_runtime_dir()
{
    std::string ep = get_worker_endpoint();
    CHECK(!ep.empty(), "runtime_dir: endpoint is non-empty");

#ifdef _WIN32
    // Should start with \\.\pipe\pjev-
    CHECK(ep.substr(0, 9) == "\\\\.\\pipe\\", "runtime_dir: Windows pipe prefix");
    CHECK(ep.find("pjev-") != std::string::npos, "runtime_dir: contains pjev-");
    CHECK(ep.find("worker-v1") != std::string::npos, "runtime_dir: contains worker-v1");
#else
    // Should end with .sock
    CHECK(ep.find("pjev-") != std::string::npos, "runtime_dir: Unix contains pjev-");
    CHECK(ep.find("worker-v1.sock") != std::string::npos, "runtime_dir: Unix socket suffix");
    // Lock path should be non-empty on Unix
    std::string lp = get_worker_lock_path();
    CHECK(!lp.empty(), "runtime_dir: Unix lock path non-empty");
    CHECK(lp.find("worker-v1.lock") != std::string::npos, "runtime_dir: lock path suffix");
#endif

    // Runtime dir
    std::string rd = get_worker_runtime_dir();
    CHECK(!rd.empty(), "runtime_dir: runtime_dir non-empty");
}

// ---------------------------------------------------------------------------
// Test 4: make_config_hash is deterministic
// ---------------------------------------------------------------------------
static void test_config_hash()
{
    std::string h1 = make_config_hash("models/bonsai.gguf", 8, 4096, "");
    std::string h2 = make_config_hash("models/bonsai.gguf", 8, 4096, "");
    CHECK(h1 == h2, "config_hash: deterministic");

    std::string h3 = make_config_hash("models/other.gguf", 8, 4096, "");
    CHECK(h1 != h3, "config_hash: different model → different hash");

    std::string h4 = make_config_hash("models/bonsai.gguf", 4, 4096, "");
    CHECK(h1 != h4, "config_hash: different threads → different hash");

    std::string h5 = make_config_hash("models/bonsai.gguf", 8, 2048, "");
    CHECK(h1 != h5, "config_hash: different ctx_size → different hash");

    std::string h6 = make_config_hash("models/bonsai.gguf", 8, 4096, "calib.json");
    CHECK(h1 != h6, "config_hash: different calibration → different hash");

    // Contains expected fields
    CHECK(h1.find("models/bonsai.gguf") != std::string::npos, "config_hash: contains model_path");
    CHECK(h1.find("|8|") != std::string::npos, "config_hash: contains threads");
    CHECK(h1.find("|4096|") != std::string::npos, "config_hash: contains ctx_size");
}

// ---------------------------------------------------------------------------
// Test 5: WorkerInfo JSON roundtrip
// ---------------------------------------------------------------------------
static void test_worker_info_json()
{
    WorkerInfo info;
    info.protocol_version = IPC_PROTOCOL_VERSION;
    info.pjev_version     = "0.10.0";
    info.pid              = 12345;
    info.model_identity   = "bonsai-q4";
    info.config_hash      = make_config_hash("models/bonsai.gguf", 8, 4096, "");
    info.start_time_us    = 1700000000000000LL;

    json j = ipc_worker_info_to_json(info);
    CHECK(j["protocol_version"] == IPC_PROTOCOL_VERSION, "worker_info_json: protocol_version");
    CHECK(j["pjev_version"] == "0.10.0",     "worker_info_json: pjev_version");
    CHECK(j["pid"] == 12345,                  "worker_info_json: pid");
    CHECK(j["model_identity"] == "bonsai-q4", "worker_info_json: model_identity");
    CHECK(!j["config_hash"].get<std::string>().empty(), "worker_info_json: config_hash non-empty");
    CHECK(j["start_time_us"] == 1700000000000000LL, "worker_info_json: start_time_us");

    WorkerInfo info2 = ipc_worker_info_from_json(j);
    CHECK(info2.protocol_version == info.protocol_version, "worker_info_rt: protocol_version");
    CHECK(info2.pjev_version == info.pjev_version,     "worker_info_rt: pjev_version");
    CHECK(info2.pid == info.pid,                        "worker_info_rt: pid");
    CHECK(info2.model_identity == info.model_identity,  "worker_info_rt: model_identity");
    CHECK(info2.config_hash == info.config_hash,        "worker_info_rt: config_hash");
    CHECK(info2.start_time_us == info.start_time_us,    "worker_info_rt: start_time_us");

    // Default WorkerInfo
    WorkerInfo def_info;
    json def_j = ipc_worker_info_to_json(def_info);
    WorkerInfo def_rt = ipc_worker_info_from_json(def_j);
    CHECK(def_rt.protocol_version == IPC_PROTOCOL_VERSION, "worker_info_default: protocol_version");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    printf("=== test_worker ===\n");

    test_ipc_framing();
    test_ipc_protocol_serialization();
    test_runtime_dir();
    test_config_hash();
    test_worker_info_json();

    printf("\n");
    if (fails == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("%d TEST(S) FAILED\n", fails);
        return 1;
    }
}
