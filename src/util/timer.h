#ifndef INC_PJEV_TIMER_H
#define INC_PJEV_TIMER_H
#include <chrono>

namespace pjev
{
class Timer
{
public:
    Timer() = default;

    void start();
    void stop();
    std::chrono::milliseconds elapsed() const;
private:
    std::chrono::high_resolution_clock::time_point start_ = std::chrono::high_resolution_clock::time_point::min();
    std::chrono::high_resolution_clock::time_point end_ = std::chrono::high_resolution_clock::time_point::min();
};
} // namespace pjev
#endif // INC_PJEV_TIMER_H