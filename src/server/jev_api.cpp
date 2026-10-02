#include "jev_api.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "../decision/decision_engine.h"

namespace pjev
{
using json = nlohmann::ordered_json;

JevApiHandler::JevApiHandler(DecisionEngine& engine, const std::string& model_name)
    : engine_(engine)
    , model_name_(model_name)
{
}
namespace
{
    std::string make_error(int32_t /*status*/, const std::string& msg)
    {
        json e;
        e["error"] = msg;
        return e.dump();
    }

    bool build_input(DecisionInput& input, int32_t& status_out, std::string& response_out, const std::string& state, const json& question)
    {
        if(!question.contains("type") || !question["type"].is_string() || !question.contains("instructions") || !question["instructions"].is_string()) {
            status_out = 400;
            response_out = make_error(400, "decision question missing type or instructions");
            return false;
        }

        input.state = state;
        input.question = question["instructions"].get<std::string>();
        input.type = question["type"].get<std::string>();

        const json* criteria = question.contains("criteria") ? &question["criteria"] : nullptr;

        // Build options list
        if(input.type == "noul") {
            // Fixed order: false first, true second (matches phase0 convention)
            std::string false_desc, true_desc;
            if(criteria && criteria->is_object()) {
                if(criteria->contains("false") && (*criteria)["false"].is_string())
                    false_desc = (*criteria)["false"].get<std::string>();
                if(criteria->contains("true") && (*criteria)["true"].is_string())
                    true_desc = (*criteria)["true"].get<std::string>();
            }
            input.options.push_back({"false", false_desc});
            input.options.push_back({"true", true_desc});
        } else if(input.type == "choice") {
            if(!criteria || !criteria->is_object()) {
                status_out = 400;
                response_out = make_error(400, "choice question requires criteria object");
                return false;
            }
            for(const auto& item: criteria->items()) {
                std::string desc = item.value().is_string() ? item.value().get<std::string>() : "";
                input.options.push_back({item.key(), desc});
            }
        } else if(input.type == "score") {
            if(!criteria || !criteria->is_array()) {
                status_out = 400;
                response_out = make_error(400, "score question requires criteria array");
                return false;
            }
            for(size_t i = 0; i < criteria->size(); i++) {
                std::string desc = (*criteria)[i].is_string() ? (*criteria)[i].get<std::string>() : "";
                input.options.push_back({std::to_string(i), desc});
            }
        } else {
            status_out = 422;
            response_out = make_error(422, "unknown question type: " + input.type);
            return false;
        }

        if(input.options.size() < 2) {
            status_out = 422;
            response_out = make_error(422, "need at least 2 candidates");
            return false;
        }
        return true;
    }
} // namespace

void JevApiHandler::handle(const std::string& request_body,
                           int32_t& status_out,
                           std::string& response_out)
{
    // Parse JSON
    json req;
    try {
        req = json::parse(request_body);
    } catch(...) {
        status_out = 400;
        response_out = make_error(400, "invalid JSON");
        return;
    }

    // Validate structure
    if(!req.contains("state") || !req["state"].is_string() || !req.contains("questions") || !req["questions"].is_object()) {
        status_out = 400;
        response_out = make_error(400, "missing required fields: state, questions");
        return;
    }

    std::string state = req["state"].get<std::string>();

    const json& questions = req["questions"];

    json answers;
    for(auto&& itr = questions.begin(); itr != questions.end(); ++itr) {
        const std::string& key = itr.key();
        const json& question = *itr;
        DecisionInput input;
        if(!build_input(input, status_out, response_out, state, question)) {
            return;
        }
        // Run decision (serialized by mutex)
        DecisionOutput out;
        {
            std::lock_guard<std::mutex> lk(mutex_);
            out = engine_.decide(input);
        }

        if(!out.ok) {
            status_out = 500;
            response_out = make_error(500, out.error);
            return;
        }
        json answer;
        answer["type"] = input.type;
        if(input.type == "noul") {
            answer["noul"] = out.p_true;
        } else {
            json probs_obj = json::object();
            for(size_t i = 0; i < out.keys.size(); i++) {
                probs_obj[out.keys[i]] = out.probs[i];
            }
            answer["probabilities"] = probs_obj;
            if(input.type == "choice") {
                answer["choice"] = out.keys[out.selected];
            }
        }
        answers[key] = answer;
    }
    // Build response
    json resp;
    resp["answers"] = answers;
    resp["model"] = model_name_;
    resp["usage"] = json::object();

    status_out = 200;
    response_out = resp.dump();
}
} // namespace pjev
