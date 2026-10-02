#include "timer.h"

namespace pjev
{
void Timer::start()
{
    start_ = end_ = std::chrono::high_resolution_clock::now();
}

void Timer::stop()
{
    end_ = std::chrono::high_resolution_clock::now();
}

std::chrono::milliseconds Timer::elapsed() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
}
} // namespace pjev
