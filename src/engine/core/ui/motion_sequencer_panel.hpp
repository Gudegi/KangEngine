#ifndef _MOTION_SEQUENCER_PANEL_HPP_
#define _MOTION_SEQUENCER_PANEL_HPP_

#include "panel.hpp"
#include "sequencer_widget.hpp"

#include <functional>
#include <string>

namespace KE {

class MotionSequencerPanel : public Panel {
  public:
    using FrameChangedCallback = std::function<void(int)>;
    using PlayingChangedCallback = std::function<void(bool)>;

    MotionSequencerPanel();
    ~MotionSequencerPanel();

    void setMotion(std::string motionName, int numFrames, float fps);
    void setCurrentTime(float time);
    float currentTime() const { return _time; }
    float duration() const;
    void setPlaying(bool playing);
    bool isPlaying() const { return _playing; }
    void setLoop(bool loop) { _loop = loop; }
    bool loop() const { return _loop; }
    void setTimeScale(float timeScale);
    float timeScale() const { return _timeScale; }
    void setFrameChangedCallback(FrameChangedCallback callback);
    void setPlayingChangedCallback(PlayingChangedCallback callback);
    void buildPanel() override;

  private:
    class SingleMotionSequence;

    int currentFrame() const;
    void wrapOrClampTime();
    void setFrame(int frame);
    void emitPlayingChanged();

    std::string _motionName = "Motion";
    int _numFrames = 1;
    float _fps = 30.0f;
    float _time = 0.0f;
    bool _playing = true;
    bool _loop = true;
    bool _fitToContent = true;
    bool _expanded = true;
    float _timeScale = 1.0f;
    int _firstFrame = 0;
    int _selectedTrack = -1;
    FrameChangedCallback _onFrameChanged;
    PlayingChangedCallback _onPlayingChanged;
};

} // namespace KE

#endif // _MOTION_SEQUENCER_PANEL_HPP_
