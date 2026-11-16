#include "engine/core/ui/motion_sequencer_panel.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <iostream>
#include <stdexcept>

int main() {
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(800, 600);
    io.DeltaTime = 1.0f / 60.0f;
    io.IniFilename = nullptr;
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    KE::MotionSequencerPanel panel;
    panel.setMotion("test", 100, 30.0f);
    panel.setCurrentTime(10.0f / 30.0f);
    panel.setPlaying(false);
    panel.setFolded(true);

    auto require = [](bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    };
    bool builtIn = false;
    auto tick = [&](ImGuiKey key, bool down, bool typing = false) {
        io.AddKeyEvent(key, down);
        ImGui::NewFrame();
        io.WantTextInput = typing;
        if (builtIn)
            panel.buildPanel();
        else
            panel.handlePlaybackShortcuts();
        // Multiple callers and hidden-UI frames must not repeat a key edge.
        panel.handlePlaybackShortcuts();
        ImGui::EndFrame();
        panel.handlePlaybackShortcuts();
    };
    tick(ImGuiKey_RightArrow, true);
    require(panel.currentFrame() == 11, "Right must advance exactly one frame");
    tick(ImGuiKey_RightArrow, false);
    tick(ImGuiKey_LeftArrow, true);
    require(panel.currentFrame() == 10, "Left must rewind exactly one frame");
    tick(ImGuiKey_LeftArrow, false);
    tick(ImGuiKey_Space, true);
    require(panel.isPlaying(), "Space must start playback");
    for (int i = 0; i < 60; ++i) tick(ImGuiKey_Space, true);
    require(panel.isPlaying(), "Holding Space must not toggle repeatedly");
    tick(ImGuiKey_Space, false);
    tick(ImGuiKey_Space, true);
    require(!panel.isPlaying(), "Second Space press must pause");
    tick(ImGuiKey_Space, false);
    tick(ImGuiKey_RightArrow, true, true);
    require(panel.currentFrame() == 10, "Typing must suppress frame stepping");
    tick(ImGuiKey_RightArrow, false);
    tick(ImGuiKey_Space, true, true);
    require(!panel.isPlaying(), "Typing must suppress Space toggle");
    tick(ImGuiKey_Space, false);
    panel.setPlaying(true);
    panel.setPlaybackManaged(true);
    tick(ImGuiKey_Space, true);
    require(panel.isPlaying(), "App-managed panels must not handle Space a second time");
    tick(ImGuiKey_Space, false);
    panel.setPlaybackManaged(false);
    panel.setCurrentTime(0);
    tick(ImGuiKey_LeftArrow, true);
    require(panel.currentFrame() == 0 && !panel.isPlaying(), "Step must pause and clamp at zero");
    tick(ImGuiKey_LeftArrow, false);
    panel.setCurrentTime(99.0f / 30.0f);
    tick(ImGuiKey_RightArrow, true);
    require(panel.currentFrame() == 99, "Step must clamp at the last frame");
    for (int i = 0; i < 400; ++i) {
        panel.setCurrentTime(50.0f / 30.0f);
        tick(ImGuiKey_RightArrow, true);
        require(panel.currentFrame() <= 51, "Long hold must not accelerate to 30-frame jumps");
    }
    tick(ImGuiKey_RightArrow, false);

    int notifiedFrame = -1;
    int notifications = 0;
    panel.setFrameChangedCallback([&](int frame) {
        notifiedFrame = frame;
        ++notifications;
    });
    panel.setPlaying(true);
    panel.setCurrentFrame(37);
    require(panel.currentFrame() == 37 && !panel.isPlaying() && notifiedFrame == 37,
            "Direct seeking must pause and notify listeners");
    require(notifications == 1, "Direct seeking must notify exactly once");
    panel.setCurrentFrame(-10);
    require(panel.currentFrame() == 0 && notifiedFrame == 0, "Direct seek must clamp below zero");
    panel.setCurrentFrame(1000);
    require(panel.currentFrame() == 99 && notifiedFrame == 99, "Direct seek must clamp above the last frame");

    for (bool useBuiltIn : {false, true}) {
        builtIn = useBuiltIn;
        panel.setPlaying(true);
        tick(ImGuiKey_Home, true);
        require(panel.currentFrame() == 0 && !panel.isPlaying(), "Home must pause and seek to the start");
        tick(ImGuiKey_Home, false);
        tick(ImGuiKey_End, true);
        require(panel.currentFrame() == 99 && notifiedFrame == 99, "End must seek to the last frame");
        tick(ImGuiKey_End, false);
        tick(ImGuiKey_Home, true, true);
        require(panel.currentFrame() == 99, "Typing must suppress Home");
        tick(ImGuiKey_Home, false);
        tick(ImGuiKey_LeftArrow, true, true);
        require(panel.currentFrame() == 99, "Typing must suppress arrows in both panels");
        tick(ImGuiKey_LeftArrow, false);
        const int before = notifications;
        tick(ImGuiKey_LeftArrow, true);
        require(panel.currentFrame() == 98 && notifications == before + 1,
                "Both entry points must share the once-per-frame guard");
        tick(ImGuiKey_LeftArrow, false);
        io.AddKeyEvent(ImGuiMod_Ctrl, true);
        tick(ImGuiKey_Home, true);
        require(panel.currentFrame() == 98, "Modified shortcuts must not move the playhead");
        io.AddKeyEvent(ImGuiMod_Ctrl, false);
        tick(ImGuiKey_Home, false);

        io.AddKeyEvent(ImGuiKey_Home, true);
        ImGui::NewFrame();
        ImGui::Begin("Popup guard");
        ImGui::OpenPopup("Menu");
        if (ImGui::BeginPopup("Menu")) {
            ImGui::TextUnformatted("Popup owns input");
            ImGui::EndPopup();
        }
        ImGui::End();
        if (builtIn)
            panel.buildPanel();
        else
            panel.handlePlaybackShortcuts();
        require(panel.currentFrame() == 98, "Open popups must suppress timeline shortcuts");
        ImGui::ClosePopupsOverWindow(nullptr, false);
        ImGui::EndFrame();
        tick(ImGuiKey_Home, false);
    }
    tick(ImGuiKey_Space, true);
    require(!panel.isPlaying(), "The built-in panel must leave Space to the application");
    tick(ImGuiKey_Space, false);
    bool accelerated = false;
    for (int i = 0; i < 400; ++i) {
        panel.setCurrentTime(50.0f / 30.0f);
        tick(ImGuiKey_RightArrow, true);
        accelerated |= panel.currentFrame() == 80;
    }
    require(accelerated, "The built-in panel must preserve long-hold acceleration");
    tick(ImGuiKey_RightArrow, false);
    panel.setCurrentFrame(10);
    panel.setPlaying(true);
    auto drawPanel = [&]() {
        ImGui::NewFrame();
        panel.buildPanel();
        ImGui::EndFrame();
    };
    drawPanel();
    const ImGuiID inputId = ImGui::FindWindowByName(panel.name().c_str())->GetID("##current_frame");
    ImGui::ActivateItemByID(inputId);
    drawPanel();
    require(ImGui::GetActiveID() == inputId, "Frame input must accept keyboard focus");
    io.AddInputCharactersUTF8("37");
    drawPanel();
    require(panel.currentFrame() == 10 && panel.isPlaying(), "Typing must not seek before Enter");
    io.AddKeyEvent(ImGuiKey_LeftArrow, true);
    drawPanel();
    require(panel.currentFrame() == 10 && panel.isPlaying(), "Editing the frame field must not step playback");
    io.AddKeyEvent(ImGuiKey_LeftArrow, false);
    drawPanel();
    const int beforeEnter = notifications;
    io.AddKeyEvent(ImGuiKey_Enter, true);
    drawPanel();
    require(panel.currentFrame() == 37 && !panel.isPlaying() && notifications == beforeEnter + 1,
            "Enter must commit the frame input, pause and notify once");
    io.AddKeyEvent(ImGuiKey_Enter, false);
    drawPanel();
    panel.setFolded(false);
    drawPanel();
    auto activate = [&](const char* label) {
        ImGui::ActivateItemByID(ImGui::FindWindowByName(panel.name().c_str())->GetID(label));
        drawPanel();
    };
    activate("Loop range");
    require(panel.loopRange() == std::make_pair(0, 99), "Range checkbox must start with the full motion");
    auto enterBound = [&](const char* label, const char* value) {
        activate(label);
        io.AddInputCharactersUTF8(value);
        drawPanel();
        io.AddKeyEvent(ImGuiKey_Enter, true);
        drawPanel();
        io.AddKeyEvent(ImGuiKey_Enter, false);
        drawPanel();
    };
    enterBound("Start", "10");
    enterBound("End", "12");
    require(panel.loopRange() == std::make_pair(10, 12), "Range fields must commit inclusive bounds");
    panel.setPlaying(true);
    panel.setCurrentTime(1.0f / 3.0f);
    panel.setCurrentTime(13.0f / 30.0f);
    require(panel.currentFrame() == 10, "Playback must wrap after the inclusive end frame");
    activate("Loop range");
    require(!panel.loopRange(), "Disabling the range must restore whole-motion looping");
    panel.clearMotions();
    builtIn = false;
    tick(ImGuiKey_Space, true);
    require(!panel.isPlaying(), "An empty timeline must ignore playback shortcuts");
    ImGui::DestroyContext();
    std::cout << "PASS playback keys, direct seeking, loop ranges, shared input guards and built-in acceleration\n";
}
