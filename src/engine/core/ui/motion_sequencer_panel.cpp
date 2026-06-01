#include "motion_sequencer_panel.hpp"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

namespace KE {

class MotionSequencerPanel::SingleMotionSequence
    : public UI::SequenceInterface {
  public:
    SingleMotionSequence(int frameMin, int frameMax, const std::string& label)
        : _frameMin(frameMin), _frameMax(frameMax), _label(label),
          _start(frameMin), _end(frameMax) {}

    int GetFrameMin() const override { return _frameMin; }
    int GetFrameMax() const override { return _frameMax; }
    int GetItemCount() const override { return 1; }
    const char* GetItemLabel(int) const override { return _label.c_str(); }
    const char* GetCollapseFmt() const override { return "%d Frames"; }

    void Get(int, int** start, int** end, int* type,
             unsigned int* color) override {
        if (start)
            *start = &_start;
        if (end)
            *end = &_end;
        if (type)
            *type = 0;
        if (color)
            *color = IM_COL32(96, 180, 255, 255);
    }

  private:
    int _frameMin = 0;
    int _frameMax = 0;
    std::string _label;
    int _start = 0;
    int _end = 0;
};

MotionSequencerPanel::MotionSequencerPanel() : Panel("Motion Sequencer") {}

MotionSequencerPanel::~MotionSequencerPanel() {}

void MotionSequencerPanel::setMotion(std::string motionName, int numFrames,
                                     float fps) {
    _motionName = std::move(motionName);
    _numFrames = std::max(1, numFrames);
    _fps = std::max(1e-6f, fps);
    _time = 0.0f;
    _firstFrame = 0;
    _selectedTrack = -1;
}

void MotionSequencerPanel::setCurrentTime(float time) {
    _time = std::max(0.0f, time);
    wrapOrClampTime();
}

void MotionSequencerPanel::setPlaying(bool playing) {
    if (_playing == playing)
        return;
    _playing = playing;
    emitPlayingChanged();
}

void MotionSequencerPanel::setTimeScale(float timeScale) {
    _timeScale = std::max(0.0f, timeScale);
}

void MotionSequencerPanel::setFrameChangedCallback(
    FrameChangedCallback callback) {
    _onFrameChanged = std::move(callback);
}

void MotionSequencerPanel::setPlayingChangedCallback(
    PlayingChangedCallback callback) {
    _onPlayingChanged = std::move(callback);
}

void MotionSequencerPanel::buildPanel() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float panelHeight =
        std::min(280.0f, std::max(180.0f, viewport->WorkSize.y * 0.28f));
    ImGui::SetNextWindowPos(
        ImVec2(viewport->WorkPos.x,
               viewport->WorkPos.y + viewport->WorkSize.y - panelHeight),
        ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, panelHeight),
                             ImGuiCond_FirstUseEver);

    ImGui::Begin(name().c_str());
    ImGui::Text("%s", _motionName.c_str());
    const float d = duration();
    const float displayTime = d > 1e-6f ? std::fmod(_time, d) : 0.0f;
    ImGui::Text("Frame %d/%d  %.3fs / %.3fs  |  %s", currentFrame() + 1,
                _numFrames, displayTime, d, _playing ? "running" : "paused");
    if (ImGui::Button(_playing ? "Pause" : "Play"))
        setPlaying(!_playing);
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
        setFrame(0);
    ImGui::SameLine();
    ImGui::Checkbox("loop", &_loop);
    ImGui::SameLine();
    ImGui::Checkbox("fit whole motion", &_fitToContent);
    ImGui::SliderFloat("playback speed", &_timeScale, 0.0f, 4.0f);

    int frame = currentFrame();
    SingleMotionSequence sequence(0, std::max(_numFrames - 1, 0), _motionName);
    UI::SequencerConfig config;
    config.fitToContent = _fitToContent;
    const bool changed =
        UI::sequencer(&sequence, &frame, &_expanded, &_selectedTrack,
                      &_firstFrame, UI::SequencerChangeFrame, config);
    if (changed)
        setFrame(frame);

    const float progress = d > 1e-6f ? displayTime / d : 0.0f;
    char overlay[32];
    std::snprintf(overlay, sizeof(overlay), "%.1f%%", progress * 100.0f);
    ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay);
    ImGui::End();
}

float MotionSequencerPanel::duration() const {
    if (_numFrames <= 1)
        return 0.0f;
    return static_cast<float>(_numFrames - 1) / _fps;
}

int MotionSequencerPanel::currentFrame() const {
    if (_numFrames <= 1)
        return 0;
    const float d = duration();
    const float t = d > 1e-6f ? std::fmod(_time, d) : 0.0f;
    return std::clamp(static_cast<int>(std::round(t * _fps)), 0,
                      _numFrames - 1);
}

void MotionSequencerPanel::wrapOrClampTime() {
    const float d = duration();
    if (d <= 1e-6f) {
        _time = 0.0f;
        return;
    }
    if (_loop)
        _time = std::fmod(_time, d);
    else
        _time = std::clamp(_time, 0.0f, d);
}

void MotionSequencerPanel::setFrame(int frame) {
    frame = std::clamp(frame, 0, _numFrames - 1);
    _time = static_cast<float>(frame) / _fps;
    if (_onFrameChanged)
        _onFrameChanged(frame);
}

void MotionSequencerPanel::emitPlayingChanged() {
    if (_onPlayingChanged)
        _onPlayingChanged(_playing);
}

} // namespace KE
