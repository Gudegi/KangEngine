#ifndef _MOTION_SEQUENCER_PANEL_HPP_
#define _MOTION_SEQUENCER_PANEL_HPP_

#include "panel.hpp"
#include "sequencer_widget.hpp"
#include "ui_scale.hpp"

#include <functional>
#include <string>
#include <vector>

namespace KE {

class MotionSequencerPanel : public Panel {
  public:
    using FrameChangedCallback = std::function<void(int)>;
    using PlayingChangedCallback = std::function<void(bool)>;

    MotionSequencerPanel();
    ~MotionSequencerPanel();

    void setMotion(std::string motionName, int numFrames, float fps);
    void setMotions(std::vector<std::string> motionNames,
                    std::vector<int> numFrames, std::vector<float> fps);
    void setCurrentTime(float time);
    float currentTime() const { return _time; }
    float duration() const;
    void setPlaying(bool playing);
    bool isPlaying() const { return _playing; }
    void setLoop(bool loop) { _loop = loop; }
    bool loop() const { return _loop; }
    void setTimeScale(float timeScale);
    float timeScale() const { return _timeScale; }
    void setTransparent(bool transparent) { _transparent = transparent; }
    bool transparent() const { return _transparent; }
    void setOverlay(bool overlay) { _overlay = overlay; }
    bool overlay() const { return _overlay; }
    void setOverlayWidthRatio(float ratio);
    float overlayWidthRatio() const { return _overlayWidthRatio; }
    void setLegendWidth(float width);
    float legendWidth() const { return _legendWidth; }
    void setUiScale(float scale) { _uiScale.setUserScale(scale); }
    float uiScale() const { return _uiScale.value(); }
    void setFolded(bool folded) { _folded = folded; }
    bool folded() const { return _folded; }
    void setShowProgressBar(bool show) { _showProgressBar = show; }
    bool showProgressBar() const { return _showProgressBar; }
    void setFrameChangedCallback(FrameChangedCallback callback);
    void setPlayingChangedCallback(PlayingChangedCallback callback);
    void buildPanel() override;

  private:
    class MotionSequence;

    int currentFrame() const;
    float playbackDuration() const;
    void wrapOrClampTime();
    void setFrame(int frame);
    void handleFrameShortcuts();
    void emitPlayingChanged();

    std::string _motionName = "Motion";
    std::vector<std::string> _trackNames{"Motion"};
    std::vector<int> _trackEndFrames{0};
    int _numFrames = 1;
    float _fps = 30.0f;
    float _time = 0.0f;
    bool _playing = true;
    bool _loop = true;
    bool _fitToContent = true;
    bool _expanded = true;
    bool _transparent = false;
    bool _overlay = true;
    bool _folded = false;
    bool _showProgressBar = false;
    float _overlayWidthRatio = 0.70f;
    float _legendWidth = 200.0f;
    float _timeScale = 1.0f;
    float _leftArrowHoldTime = 0.0f;
    float _rightArrowHoldTime = 0.0f;
    int _firstFrame = 0;
    int _selectedTrack = -1;
    UIScale _uiScale;
    FrameChangedCallback _onFrameChanged;
    PlayingChangedCallback _onPlayingChanged;
};

} // namespace KE

#endif // _MOTION_SEQUENCER_PANEL_HPP_
