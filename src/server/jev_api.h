#ifndef INC_PJEV_JEV_API_H_
#define INC_PJEV_JEV_API_H_
#include <cstdint>
#include <mutex>
#include <string>

namespace pjev
{
    class DecisionEngine;

class JevApiHandler {
public:
    JevApiHandler(DecisionEngine& engine, const std::string& model_name = "pseudojev");

    // Parse request JSON, run decision, return response JSON.
    // status_out is set to the HTTP status code (200, 400, 422, 500).
    void handle(const std::string& request_body,
                int32_t& status_out,
                std::string& response_out);

private:
    DecisionEngine& engine_;
    std::string     model_name_;
    std::mutex      mutex_;
};
}
#endif //INC_PJEV_JEV_API_H_
