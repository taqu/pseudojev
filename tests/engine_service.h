#pragma once
#include "decision/decision_engine.h"
#include "decision/decision_service.h"

// Serves decisions straight from a DecisionEngine (no retrieval), so API-layer tests can exercise
// request parsing, response building and error handling with a mock backend.
class EngineService : public pjev::DecisionService {
public:
    explicit EngineService(pjev::DecisionEngine& engine) : engine_(engine) {}
    pjev::DecisionOutput decide(const pjev::DecisionInput& input) override { return engine_.decide(input); }

private:
    pjev::DecisionEngine& engine_;
};
