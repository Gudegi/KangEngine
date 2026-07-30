#pragma once

namespace KE {

/// Converts elapsed wall-clock time into a bounded number of fixed updates.
///
/// This class only schedules updates. It does not know about physics, input,
/// rendering, or simulation state, so callers remain responsible for applying
/// one deterministic simulation/control step per returned update.
class FixedStepClock {
  public:
    void setStepHz(double stepHz);
    double getStepHz() const { return _stepHz; }
    double getStepInterval() const;

    void setMaxCatchUpSteps(int count);
    int getMaxCatchUpSteps() const { return _maxCatchUpSteps; }

    void setMaxFrameDelta(double seconds);
    double getMaxFrameDelta() const { return _maxFrameDelta; }

    void setPaused(bool paused);
    bool isPaused() const { return _paused; }
    void requestSingleStep();

    /// Return how many fixed updates are due for this rendered frame.
    int advance(double wallDeltaSeconds);
    void reset();

    double getAccumulator() const { return _accumulator; }
    double getDroppedWallTime() const { return _droppedWallTime; }

  private:
    double _stepHz = 0.0;
    double _accumulator = 0.0;
    double _maxFrameDelta = 0.25;
    double _droppedWallTime = 0.0;
    int _maxCatchUpSteps = 8;
    bool _paused = false;
    bool _singleStepRequested = false;
};

} // namespace KE
