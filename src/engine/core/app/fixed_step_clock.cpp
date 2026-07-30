#include "engine/core/app/fixed_step_clock.hpp"

#include <algorithm>
#include <cmath>

namespace KE {

void FixedStepClock::setStepHz(double stepHz) {
    _stepHz = std::isfinite(stepHz) && stepHz > 0.0 ? stepHz : 0.0;
    _accumulator = 0.0;
    _singleStepRequested = false;
}

double FixedStepClock::getStepInterval() const {
    return _stepHz > 0.0 ? 1.0 / _stepHz : 0.0;
}

void FixedStepClock::setMaxCatchUpSteps(int count) {
    _maxCatchUpSteps = std::max(1, count);
}

void FixedStepClock::setMaxFrameDelta(double seconds) {
    _maxFrameDelta = std::isfinite(seconds) && seconds > 0.0 ? seconds : 0.0;
}

void FixedStepClock::setPaused(bool paused) {
    if (_paused == paused)
        return;
    _paused = paused;
    _accumulator = 0.0;
    _singleStepRequested = false;
}

void FixedStepClock::requestSingleStep() {
    if (_paused)
        _singleStepRequested = true;
}

int FixedStepClock::advance(double wallDeltaSeconds) {
    if (_paused) {
        _accumulator = 0.0;
        if (_singleStepRequested && _stepHz > 0.0) {
            _singleStepRequested = false;
            return 1;
        }
        return 0;
    }

    const double stepInterval = getStepInterval();
    if (stepInterval <= 0.0)
        return 0;

    double wallDelta = std::isfinite(wallDeltaSeconds) && wallDeltaSeconds > 0.0
                           ? wallDeltaSeconds
                           : 0.0;
    if (_maxFrameDelta > 0.0 && wallDelta > _maxFrameDelta) {
        _droppedWallTime += wallDelta - _maxFrameDelta;
        wallDelta = _maxFrameDelta;
    }
    _accumulator += wallDelta;

    const double epsilon = stepInterval * 1.0e-9;
    const int dueSteps =
        static_cast<int>(std::floor((_accumulator + epsilon) / stepInterval));
    if (dueSteps <= 0)
        return 0;

    const int steps = std::min(dueSteps, _maxCatchUpSteps);
    _accumulator -= static_cast<double>(dueSteps) * stepInterval;
    _accumulator = std::max(0.0, _accumulator);
    if (dueSteps > steps) {
        _droppedWallTime +=
            static_cast<double>(dueSteps - steps) * stepInterval;
    }
    return steps;
}

void FixedStepClock::reset() {
    _accumulator = 0.0;
    _droppedWallTime = 0.0;
    _singleStepRequested = false;
}

} // namespace KE
