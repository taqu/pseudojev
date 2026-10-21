#ifndef INC_PJEV_CLI_H_
#define INC_PJEV_CLI_H_
#include "../decision/decision_engine.h"
#include <string>
#include <utility>
#include <vector>

namespace pjev {

std::string format_noul_human(const DecisionOutput& out);
std::string format_noul_json(const DecisionOutput& out);

std::string format_choice_human(const DecisionOutput& out);
std::string format_choice_json(const DecisionOutput& out);

std::string format_score_human(const DecisionOutput& out);
std::string format_score_json(const DecisionOutput& out,
                               const std::vector<std::pair<std::string, std::string>>& options);

} // namespace pjev
#endif // INC_PJEV_CLI_H_
