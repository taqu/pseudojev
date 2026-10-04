#include "jev_api.h"
#include <chrono>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include "decision/decision_engine.h"
#include "spdlog/spdlog.h"

namespace pjev {
using json = nlohmann::ordered_json;

// Input size limits
namespace {
static constexpr size_t MAX_STATE_LEN    = 32 * 1024;
static constexpr size_t MAX_QUESTION_LEN =  8 * 1024;
static constexpr size_t MAX_OPTION_LEN   =  1 * 1024;
static constexpr size_t MAX_OPTION_COUNT = 20;

static std::string make_error(const std::string& code, const std::string& msg,
                               const std::string& req_id = "") {
    json e;
    e["error"]["code"]    = code;
    e["error"]["message"] = msg;
    if (!req_id.empty()) e["request_id"] = req_id;
    return e.dump();
}

static bool build_input(DecisionInput& input,
                        int32_t& status_out,
                        std::string& response_out,
                        const std::string& state,
                        const json& question,
                        const std::string& req_id)
{
    if (!question.contains("type") || !question["type"].is_string() ||
        !question.contains("instructions") || !question["instructions"].is_string()) {
        status_out   = 400;
        response_out = make_error(ErrorCode::INVALID_REQUEST,
                                  "decision question missing type or instructions", req_id);
        return false;
    }

    input.state    = state;
    input.question = question["instructions"].get<std::string>();
    input.type     = question["type"].get<std::string>();

    if (input.question.size() > MAX_QUESTION_LEN) {
        status_out   = 400;
        response_out = make_error(ErrorCode::INVALID_REQUEST,
                                  "instructions field exceeds maximum length", req_id);
        return false;
    }

    const json* criteria = question.contains("criteria") ? &question["criteria"] : nullptr;

    if (input.type == "noul") {
        std::string false_desc, true_desc;
        if (criteria && criteria->is_object()) {
            if (criteria->contains("false") && (*criteria)["false"].is_string())
                false_desc = (*criteria)["false"].get<std::string>();
            if (criteria->contains("true") && (*criteria)["true"].is_string())
                true_desc = (*criteria)["true"].get<std::string>();
        }
        input.options.push_back({"false", false_desc});
        input.options.push_back({"true",  true_desc});
    } else if (input.type == "choice") {
        if (!criteria || !criteria->is_object()) {
            status_out   = 400;
            response_out = make_error(ErrorCode::INVALID_REQUEST,
                                      "choice question requires criteria object", req_id);
            return false;
        }
        for (const auto& item : criteria->items()) {
            std::string desc = item.value().is_string() ? item.value().get<std::string>() : "";
            if (desc.size() > MAX_OPTION_LEN) {
                status_out   = 400;
                response_out = make_error(ErrorCode::INVALID_REQUEST,
                                          "option description exceeds maximum length", req_id);
                return false;
            }
            input.options.push_back({item.key(), desc});
        }
        if (input.options.size() > MAX_OPTION_COUNT) {
            status_out   = 400;
            response_out = make_error(ErrorCode::INVALID_REQUEST,
                                      "too many options (max " + std::to_string(MAX_OPTION_COUNT) + ")",
                                      req_id);
            return false;
        }
    } else if (input.type == "score") {
        if (!criteria || !criteria->is_array()) {
            status_out   = 400;
            response_out = make_error(ErrorCode::INVALID_REQUEST,
                                      "score question requires criteria array", req_id);
            return false;
        }
        if (criteria->size() > MAX_OPTION_COUNT) {
            status_out   = 400;
            response_out = make_error(ErrorCode::INVALID_REQUEST,
                                      "too many score levels (max " + std::to_string(MAX_OPTION_COUNT) + ")",
                                      req_id);
            return false;
        }
        for (size_t i = 0; i < criteria->size(); i++) {
            std::string desc = (*criteria)[i].is_string() ? (*criteria)[i].get<std::string>() : "";
            input.options.push_back({std::to_string(i), desc});
        }
    } else {
        status_out   = 422;
        response_out = make_error(ErrorCode::UNSUPPORTED_PRIMITIVE,
                                  "unknown question type: " + input.type, req_id);
        return false;
    }

    if (input.options.size() < 2) {
        status_out   = 422;
        response_out = make_error(ErrorCode::INVALID_REQUEST,
                                  "need at least 2 candidates", req_id);
        return false;
    }
    return true;
}
} // anonymous namespace

// ---------------------------------------------------------------------------

JevApiHandler::JevApiHandler(DecisionEngine& engine,
                              const std::string& model_name,
                              int max_queued,
                              bool kv_reuse,
                              bool batch_questions)
    : engine_(engine), model_name_(model_name), max_queued_(max_queued),
      kv_reuse_(kv_reuse), batch_questions_(batch_questions)
{}

void JevApiHandler::handle(const std::string& request_body,
                            int32_t& status_out,
                            std::string& response_out)
{
    // Overload check — before allocating any request state
    int prior = active_.fetch_add(1);
    if (prior >= max_queued_) {
        active_.fetch_sub(1);
        status_out   = 503;
        response_out = make_error(ErrorCode::SERVER_BUSY, "Server is busy; try again later");
        return;
    }
    // Ensure decrement on all return paths
    struct Guard { std::atomic<int>& c; ~Guard() { c.fetch_sub(1); } } g{active_};

    // Assign request ID for correlation
    uint64_t req_num = req_counter_.fetch_add(1);
    std::string req_id = "req-" + std::to_string(req_num);

    auto t_start = std::chrono::steady_clock::now();

    try {
        // Parse JSON
        json req;
        try {
            req = json::parse(request_body);
        } catch (...) {
            status_out   = 400;
            response_out = make_error(ErrorCode::INVALID_JSON, "Request body is not valid JSON", req_id);
            spdlog::warn("[{}] invalid_json status=400", req_id);
            return;
        }

        // Validate top-level structure
        if (!req.contains("state") || !req["state"].is_string() ||
            !req.contains("questions") || !req["questions"].is_object()) {
            status_out   = 400;
            response_out = make_error(ErrorCode::INVALID_REQUEST,
                                      "missing or invalid required fields: state (string), questions (object)",
                                      req_id);
            spdlog::warn("[{}] invalid_request status=400", req_id);
            return;
        }

        std::string state = req["state"].get<std::string>();
        if (state.size() > MAX_STATE_LEN) {
            status_out   = 400;
            response_out = make_error(ErrorCode::INVALID_REQUEST,
                                      "state field exceeds maximum length", req_id);
            spdlog::warn("[{}] state_too_long status=400", req_id);
            return;
        }

        const json& questions = req["questions"];
        json answers;

        // Phase 1: validate and collect all inputs
        std::vector<std::string> question_keys;
        std::vector<DecisionInput> decision_inputs;
        for (auto itr = questions.begin(); itr != questions.end(); ++itr) {
            DecisionInput input;
            if (!build_input(input, status_out, response_out, state, *itr, req_id)) {
                auto t_end = std::chrono::steady_clock::now();
                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
                spdlog::warn("[{}] validation_error status={} latency={}ms", req_id, status_out, ms);
                return;
            }
            question_keys.push_back(itr.key());
            decision_inputs.push_back(std::move(input));
        }

        // Phase 2: run inference
        std::vector<DecisionOutput> outputs;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            if (batch_questions_ && decision_inputs.size() > 1) {
                BatchConfig bcfg;
                bcfg.use_kv_reuse = kv_reuse_;
                outputs = engine_.decide_batch(decision_inputs, bcfg);
            } else {
                for (const auto& inp : decision_inputs) {
                    outputs.push_back(engine_.decide(inp));
                }
            }
        }

        // Phase 3: build responses
        for (size_t qi = 0; qi < question_keys.size(); qi++) {
            const auto& key   = question_keys[qi];
            const auto& input = decision_inputs[qi];
            const auto& out   = outputs[qi];

            if (!out.ok) {
                status_out = 500;
                std::string err_code = ErrorCode::INFERENCE_ERROR;
                if (out.error.find("context") != std::string::npos ||
                    out.error.find("token") != std::string::npos) {
                    err_code = ErrorCode::CONTEXT_TOO_LONG;
                    status_out = 400;
                }
                response_out = make_error(err_code, out.error, req_id);
                auto t_end = std::chrono::steady_clock::now();
                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
                spdlog::error("[{}] inference_error status={} latency={}ms", req_id, status_out, ms);
                return;
            }

            json answer;
            answer["type"] = input.type;
            if (input.type == "noul") {
                answer["noul"] = out.p_true;
            } else {
                json probs_obj = json::object();
                for (size_t i = 0; i < out.keys.size(); i++) probs_obj[out.keys[i]] = out.probs[i];
                answer["probabilities"] = probs_obj;
                if (input.type == "choice") answer["choice"] = out.keys[out.selected];
            }
            int64_t q_total_us = out.tokenize_us + out.eval_us + out.decision_us;
            answer["duration"] = std::to_string(q_total_us / 1000);
            answers[key] = answer;
        }

        json resp;
        resp["answers"]    = answers;
        resp["model"]      = model_name_;
        resp["usage"]      = json::object();
        resp["request_id"] = req_id;

        status_out   = 200;
        response_out = resp.dump();

        auto t_end = std::chrono::steady_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        spdlog::info("[{}] ok status=200 latency={}ms", req_id, ms);

    } catch (const std::exception& e) {
        spdlog::error("[{}] unexpected exception: {}", req_id, e.what());
        status_out   = 500;
        response_out = make_error(ErrorCode::INTERNAL_ERROR, "Internal server error", req_id);
    } catch (...) {
        spdlog::error("[{}] unknown exception", req_id);
        status_out   = 500;
        response_out = make_error(ErrorCode::INTERNAL_ERROR, "Internal server error", req_id);
    }
}

} // namespace pjev
