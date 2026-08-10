#include "motion_sequencer_panel.hpp"
#include "engine/graphics/material/colors.hpp"
#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace KE {

class MotionSequencerPanel::MotionSequence : public UI::SequenceInterface {
  public:
    MotionSequence(int frameMax, const std::vector<std::string>& labels,
                   const std::vector<int>& ends, int currentFrame, bool loop)
        : _frameMax(frameMax), _labels(labels), _ends(ends),
          _starts(labels.size(), 0), _currentFrame(currentFrame), _loop(loop) {}

    int GetFrameMin() const override { return 0; }
    int GetFrameMax() const override { return _frameMax; }
    int GetItemCount() const override {
        return static_cast<int>(_labels.size());
    }
    const char* GetItemLabel(int index) const override {
        return _labels.at(static_cast<size_t>(index)).c_str();
    }
    int GetItemCurrentFrame(int index) const override {
        if (_labels.size() <= 1)
            return -1;
        const int end = _ends.at(static_cast<size_t>(index));
        const int playbackFrames = end + 1;
        if (_loop && playbackFrames > 0)
            return _currentFrame % playbackFrames;
        return std::min(_currentFrame, end);
    }
    const char* GetCollapseFmt() const override { return "%d Frames"; }

    void Get(int index, int** start, int** end, int* type,
             unsigned int* color) override {
        const size_t item = static_cast<size_t>(index);
        if (start)
            *start = &_starts.at(item);
        if (end)
            *end = &_ends.at(item);
        if (type)
            *type = 0;
        if (color) {
            constexpr ColorType colors[] = {
                ColorType::PASTEL_SKY,    ColorType::PASTEL_GREEN,
                ColorType::PASTEL_CORAL,  ColorType::PASTEL_PURPLE,
                ColorType::PASTEL_YELLOW,
            };
            const Color& pg = ColorLibrary::get(
                colors[item % (sizeof(colors) / sizeof(colors[0]))]);
            *color =
                ImGui::ColorConvertFloat4ToU32(ImVec4(pg.r, pg.g, pg.b, pg.a));
        }
    }

  private:
    int _frameMax = 0;
    const std::vector<std::string>& _labels;
    std::vector<int> _ends;
    std::vector<int> _starts;
    int _currentFrame = 0;
    bool _loop = true;
};

MotionSequencerPanel::MotionSequencerPanel() : Panel("Motion Sequencer") {}

MotionSequencerPanel::~MotionSequencerPanel() {}

void MotionSequencerPanel::setMotion(std::string motionName, int numFrames,
                                     float fps) {
    setMotions({std::move(motionName)}, {numFrames}, {fps});
}

void MotionSequencerPanel::setMotions(std::vector<std::string> motionNames,
                                      std::vector<int> numFrames,
                                      std::vector<float> fps) {
    if (motionNames.empty() || motionNames.size() != numFrames.size() ||
        motionNames.size() != fps.size())
        throw std::invalid_argument(
            "setMotions requires equally sized, non-empty track arrays");

    _fps = 1e-6f;
    for (float rate : fps)
        _fps = std::max(_fps, rate);
    _trackNames = std::move(motionNames);
    _trackEndFrames.clear();
    _trackEndFrames.reserve(numFrames.size());
    _numFrames = 1;
    for (size_t i = 0; i < numFrames.size(); ++i) {
        const int frames = std::max(1, numFrames[i]);
        const float rate = std::max(1e-6f, fps[i]);
        const int end = std::max(
            0, static_cast<int>(
                   std::ceil(static_cast<float>(frames) * _fps / rate)) -
                   1);
        _trackEndFrames.push_back(end);
        _numFrames = std::max(_numFrames, end + 1);
    }
    _motionName = _trackNames.size() == 1
                      ? _trackNames.front()
                      : std::to_string(_trackNames.size()) + " motions";
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

void MotionSequencerPanel::setOverlayWidthRatio(float ratio) {
    _overlayWidthRatio = std::clamp(ratio, 0.1f, 1.0f);
}

void MotionSequencerPanel::setLegendWidth(float width) {
    _legendWidth = std::clamp(width, 96.0f, 420.0f);
}

void MotionSequencerPanel::setFrameChangedCallback(
    FrameChangedCallback callback) {
    _onFrameChanged = std::move(callback);
}

void MotionSequencerPanel::setPlayingChangedCallback(
    PlayingChangedCallback callback) {
    _onPlayingChanged = std::move(callback);
}

void MotionSequencerPanel::handleFrameShortcuts() {
    const ImGuiIO& io = ImGui::GetIO();
    _leftArrowHoldTime = ImGui::IsKeyDown(ImGuiKey_LeftArrow)
                             ? _leftArrowHoldTime + io.DeltaTime
                             : 0.0f;
    _rightArrowHoldTime = ImGui::IsKeyDown(ImGuiKey_RightArrow)
                              ? _rightArrowHoldTime + io.DeltaTime
                              : 0.0f;

    const int leftStep = _leftArrowHoldTime >= 5.0f ? 30 : 1;
    const int rightStep = _rightArrowHoldTime >= 5.0f ? 30 : 1;
    const int leftPresses = ImGui::GetKeyPressedAmount(
        ImGuiKey_LeftArrow, io.KeyRepeatDelay, io.KeyRepeatRate);
    const int rightPresses = ImGui::GetKeyPressedAmount(
        ImGuiKey_RightArrow, io.KeyRepeatDelay, io.KeyRepeatRate);
    const int stepOffset = rightPresses * rightStep - leftPresses * leftStep;
    if (stepOffset == 0)
        return;
    setPlaying(false);
    setFrame(currentFrame() + stepOffset);
}

void MotionSequencerPanel::buildPanel() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    _uiScale.update(static_cast<int>(viewport->WorkSize.x * viewport->DpiScale),
                    static_cast<int>(viewport->WorkSize.y * viewport->DpiScale),
                    static_cast<int>(viewport->WorkSize.x),
                    static_cast<int>(viewport->WorkSize.y), viewport->DpiScale);
    const auto logicalPx = [this](float value) {
        return _uiScale.logicalPx(value);
    };
    const ImGuiStyle& style = ImGui::GetStyle();
    const float panelPaddingY = logicalPx(10.0f);
    const float itemSpacingY = logicalPx(6.0f);
    const float rowHeight = static_cast<float>(std::ceil(std::max(
        logicalPx(20.0f), ImGui::GetTextLineHeight() + logicalPx(8.0f))));
    const float sequencerMinHeight =
        rowHeight * (static_cast<float>(_trackNames.size()) + 1.0f) +
        logicalPx(8.0f);
    const float expandedMinHeight = panelPaddingY * 2.0f +
                                    ImGui::GetFrameHeight() * 2.0f +
                                    sequencerMinHeight + itemSpacingY * 2.0f;
    const float expandedMaxHeight =
        std::max(logicalPx(190.0f), expandedMinHeight);
    const float expandedHeight = std::clamp(
        viewport->WorkSize.y * 0.20f, expandedMinHeight, expandedMaxHeight);
    const float foldedContentHeight =
        ImGui::GetFrameHeight() + panelPaddingY * 2.0f;
    const float foldedHeight = std::max(logicalPx(42.0f), foldedContentHeight);
    const float panelHeight = _folded ? foldedHeight : expandedHeight;
    float availableLeft = viewport->WorkPos.x;
    float availableRight = viewport->WorkPos.x + viewport->WorkSize.x;
    ImGuiWindow* existingWindow = ImGui::FindWindowByName(name().c_str());
    const bool panelIsDocked = existingWindow && existingWindow->DockIsActive;
    bool foundRightDock = false;
    if (!panelIsDocked) {
        constexpr const char* rightPanelNames[] = {
            "Scene", "Renderer Debug", "Performance", "Inspector"};
        const float rightHalfStart =
            viewport->WorkPos.x + viewport->WorkSize.x * 0.5f;
        for (const char* panelName : rightPanelNames) {
            ImGuiWindow* panel = ImGui::FindWindowByName(panelName);
            if (!panel || !panel->DockIsActive || panel->Pos.x < rightHalfStart)
                continue;
            availableRight = std::min(availableRight, panel->Pos.x);
            foundRightDock = true;
        }
    }
    const float availableWidth = std::max(1.0f, availableRight - availableLeft);
    const float minPanelWidth =
        logicalPx(96.0f) +
        std::max(logicalPx(240.0f), availableWidth * 0.35f);
    const float preferredPanelWidth =
        availableWidth * (_overlay ? _overlayWidthRatio : 1.0f);
    const float panelWidth = std::min(
        availableWidth, std::max(preferredPanelWidth, minPanelWidth));
    const float panelX =
        availableLeft + (availableWidth - panelWidth) * 0.5f;
    const ImGuiCond placementCond =
        (_overlay || foundRightDock) ? ImGuiCond_Always
                                     : ImGuiCond_FirstUseEver;
    ImGui::SetNextWindowPos(
        ImVec2(panelX,
               viewport->WorkPos.y + viewport->WorkSize.y - panelHeight),
        placementCond);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), placementCond);

    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
    if (_overlay)
        windowFlags |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                       ImGuiWindowFlags_NoSavedSettings;
    if (_transparent)
        windowFlags |= ImGuiWindowFlags_NoBackground;

    if (!_transparent)
        ImGui::SetNextWindowBgAlpha(_overlay ? 0.74f : 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, logicalPx(8.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, _overlay ? 0.0f : 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                        ImVec2(logicalPx(14.0f), panelPaddingY));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(logicalPx(8.0f), itemSpacingY));
    ImGui::Begin(name().c_str(), nullptr, windowFlags);
    handleFrameShortcuts();

    const float d = duration();
    const float playbackD = playbackDuration();
    const float playbackTime = _loop && playbackD > 1e-6f
                                   ? std::fmod(_time, playbackD)
                                   : std::clamp(_time, 0.0f, d);
    const float displayTime = std::min(playbackTime, d);
    const auto buttonWidth = [&style](const char* label) {
        return ImGui::CalcTextSize(label).x + style.FramePadding.x * 2.0f;
    };
    const auto buttonMinWidth = [&](const char* label, float minWidth) {
        return std::max(logicalPx(minWidth), buttonWidth(label));
    };
    const auto checkboxWidth = [&style](const char* label) {
        return ImGui::GetFrameHeight() + style.ItemInnerSpacing.x +
               ImGui::CalcTextSize(label).x;
    };
    const char* foldLabel = _folded ? "Show" : "Hide";
    if (ImGui::Button(foldLabel,
                      ImVec2(buttonMinWidth(foldLabel, 52.0f), 0.0f)))
        _folded = !_folded;
    ImGui::SameLine();
    ImGui::Text("%d HZ", static_cast<int>(_fps));
    ImGui::SameLine();
    const float statusWidth =
        ImGui::CalcTextSize("Frame 000000/000000  0000.000s / 0000.000s").x;
    const float statusStartX = ImGui::GetCursorPosX();
    const float playButtonWidth =
        std::max(buttonMinWidth("Play", 58.0f), buttonMinWidth("Pause", 58.0f));
    const float speedSliderWidth =
        std::max(logicalPx(75.0f), ImGui::CalcTextSize("4.00").x +
                                       ImGui::GetFrameHeight() +
                                       style.FramePadding.x * 4.0f);
    constexpr int inlineControlCount = 8;
    const float inlineControlsWidth =
        playButtonWidth + buttonWidth("Reset") + checkboxWidth("Loop") +
        checkboxWidth("fit whole motion") + buttonWidth("0.3") +
        buttonWidth("0.5") + buttonWidth("1.0") + speedSliderWidth +
        style.ItemInnerSpacing.x + ImGui::CalcTextSize("Speed").x +
        style.ItemSpacing.x * static_cast<float>(inlineControlCount - 1);
    const bool inlineControls =
        ImGui::GetContentRegionAvail().x >=
        statusWidth + style.ItemSpacing.x + inlineControlsWidth;
    ImGui::Text("Frame %d/%d  %.3fs / %.3fs", currentFrame(),
                std::max(_numFrames - 1, 0), displayTime, d);

    if (_folded) {
        ImGui::End();
        ImGui::PopStyleVar(4);
        return;
    }

    if (inlineControls) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(statusStartX + statusWidth + style.ItemSpacing.x);
    }
    if (ImGui::Button(_playing ? "Pause" : "Play",
                      ImVec2(playButtonWidth, 0.0f)))
        setPlaying(!_playing);
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
        setFrame(0);
    ImGui::SameLine();
    ImGui::Checkbox("Loop", &_loop);
    ImGui::SameLine();
    const bool fitToContentChanged =
        ImGui::Checkbox("fit whole motion", &_fitToContent);
    ImGui::SameLine();
    if (ImGui::Button("0.3"))
        _timeScale = 0.3f;
    ImGui::SameLine();
    if (ImGui::Button("0.5"))
        _timeScale = 0.5f;
    ImGui::SameLine();
    if (ImGui::Button("1.0"))
        _timeScale = 1.0f;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(speedSliderWidth);
    ImGui::SliderFloat("Speed", &_timeScale, 0.0f, 4.0f);

    int frame = currentFrame();
    MotionSequence sequence(std::max(_numFrames - 1, 0), _trackNames,
                            _trackEndFrames, frame, _loop);
    UI::SequencerConfig config;
    config.fitToContent = _fitToContent;
    config.requestFitToContent = fitToContentChanged && _fitToContent;
    config.requestRestoreView = fitToContentChanged && !_fitToContent;
    config.minFramePixelWidth = logicalPx(config.minFramePixelWidth);
    config.maxFramePixelWidth = logicalPx(config.maxFramePixelWidth);
    float legendWidth = logicalPx(_legendWidth);
    config.legendWidth = static_cast<int>(legendWidth);
    config.legendWidthValue = &legendWidth;
    config.minLegendWidth = logicalPx(96.0f);
    float motionNameLegendWidth = 0.0f;
    for (const std::string& name : _trackNames) {
        motionNameLegendWidth =
            std::max(motionNameLegendWidth,
                     ImGui::CalcTextSize(name.c_str()).x + logicalPx(24.0f));
    }
    config.maxLegendWidth = std::max(logicalPx(420.0f), motionNameLegendWidth);
    config.legendResizeHandleWidth = logicalPx(10.0f);
    config.itemHeight = static_cast<int>(rowHeight);
    config.scrollBarHeight = logicalPx(config.scrollBarHeight);
    config.childBottomPadding = logicalPx(config.childBottomPadding);
    config.minTickSpacing = logicalPx(config.minTickSpacing);
    config.cursorWidth = logicalPx(config.cursorWidth);
    config.minHandleWidth = logicalPx(config.minHandleWidth);
    config.minScrollBarWidth = logicalPx(config.minScrollBarWidth);
    config.rectRounding = logicalPx(config.rectRounding);
    if (config.requestFitToContent)
        _firstFrame = 0;
    const bool changed =
        UI::sequencer(&sequence, &frame, &_expanded, &_selectedTrack,
                      &_firstFrame, UI::SequencerChangeFrame, config);
    _legendWidth = legendWidth / std::max(_uiScale.value(), 1e-6f);
    if (changed)
        setFrame(frame);

    if (_showProgressBar) {
        const float progress = d > 1e-6f ? displayTime / d : 0.0f;
        char overlay[32];
        std::snprintf(overlay, sizeof(overlay), "%.1f%%", progress * 100.0f);
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay);
    }
    ImGui::End();
    ImGui::PopStyleVar(4);
}

float MotionSequencerPanel::duration() const {
    if (_numFrames <= 1)
        return 0.0f;
    return static_cast<float>(_numFrames - 1) / _fps;
}

float MotionSequencerPanel::playbackDuration() const {
    if (_numFrames <= 0)
        return 0.0f;
    return static_cast<float>(_numFrames) / _fps;
}

int MotionSequencerPanel::currentFrame() const {
    if (_numFrames <= 1)
        return 0;
    const float d = duration();
    const float playbackD = playbackDuration();
    const float t = _loop && playbackD > 1e-6f ? std::fmod(_time, playbackD)
                                               : std::clamp(_time, 0.0f, d);
    float framePosition = t * _fps;
    const float nearestFrame = std::round(framePosition);
    if (std::abs(framePosition - nearestFrame) < 1e-4f)
        framePosition = nearestFrame;
    return std::clamp(static_cast<int>(std::floor(framePosition)), 0,
                      _numFrames - 1);
}

void MotionSequencerPanel::wrapOrClampTime() {
    const float d = duration();
    const float playbackD = playbackDuration();
    if (playbackD <= 1e-6f) {
        _time = 0.0f;
        return;
    }
    if (_loop)
        _time = std::fmod(_time, playbackD);
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
