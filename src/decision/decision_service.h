#ifndef INC_PJEV_DECISION_SERVICE_H_
#define INC_PJEV_DECISION_SERVICE_H_
#include "decision_engine.h"

namespace pjev
{

// The production inference entry point shared by the CLI, the background worker and the HTTP server.
// Implemented by TaskRouter; tests substitute their own.
class DecisionService
{
public:
    virtual ~DecisionService() = default;

    // Not thread-safe. Callers serialize access.
    virtual DecisionOutput decide(const DecisionInput& input) = 0;
};

} // namespace pjev
#endif // INC_PJEV_DECISION_SERVICE_H_
