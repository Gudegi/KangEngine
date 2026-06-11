#include "motion_sequencer_panel.hpp"
#include "engine/graphics/material/colors.hpp"
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
        if (color) {
            const Color& pg = ColorLibrary::get(ColorType::PASTEL_GREEN);
            *color =
                ImGui::ColorConvertFloat4ToU32(ImVec4(pg.r, pg.g, pg.b, pg.a));
        }
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
    const float sequencerMinHeight = rowHeight * 2.0f + logicalPx(8.0f);
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
    const float minPanelWidth =
        logicalPx(96.0f) +
        std::max(logicalPx(240.0f), viewport->WorkSize.x * 0.35f);
    const float preferredPanelWidth =
        viewport->WorkSize.x * (_overlay ? _overlayWidthRatio : 1.0f);
    const float panelWidth = std::min(
        viewport->WorkSize.x, std::max(preferredPanelWidth, minPanelWidth));
    const float panelX =
        viewport->WorkPos.x + (viewport->WorkSize.x - panelWidth) * 0.5f;
    const ImGuiCond placementCond =
        _overlay ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
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
    SingleMotionSequence sequence(0, std::max(_numFrames - 1, 0), _motionName);
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
    const float motionNameLegendWidth =
        ImGui::CalcTextSize(_motionName.c_str()).x + logicalPx(24.0f);
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
    return std::clamp(static_cast<int>(std::floor(t * _fps)), 0,
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
