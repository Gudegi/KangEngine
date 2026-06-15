///
/// KangEngine Python Bindings
///

#include "../src/kangEngine.hpp"
#include "../src/engine/graphics/backend/graphics_factory.hpp"
#include "../src/engine/graphics/material/material.hpp"
#include "../src/engine/graphics/material/colors.hpp"
#include "py_array_view.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <iomanip>
#include <optional>

namespace py = pybind11;
using namespace KE;

// Forward declarations for submodule bindings
void bind_scene(py::module& m);
void bind_animation(py::module& m);
void bind_physics(py::module& m);

// Trampoline class for App - allows Python to override virtual methods
class PyApp : public App {
  public:
    using App::App;

    void setup() override { PYBIND11_OVERRIDE_PURE(void, App, setup); }
    void preRender() override { PYBIND11_OVERRIDE_PURE(void, App, preRender); }
    void render() override { PYBIND11_OVERRIDE_PURE(void, App, render); }
    void postRender() override {
        PYBIND11_OVERRIDE_PURE(void, App, postRender);
    }
    void onRayPicked(const RayPickResult& result) override {
        PYBIND11_OVERRIDE(void, App, onRayPicked, result);
    }
    void onRayPickHover(const RayPickResult& result) override {
        PYBIND11_OVERRIDE(void, App, onRayPickHover, result);
    }
    void onForceDragBegin(const RayPickResult& result,
                          const glm::vec3& target) override {
        PYBIND11_OVERRIDE(void, App, onForceDragBegin, result, target);
    }
    void onForceDragUpdate(const RayPickResult& result,
                           const glm::vec3& target) override {
        PYBIND11_OVERRIDE(void, App, onForceDragUpdate, result, target);
    }
    void onForceDragEnd() override {
        PYBIND11_OVERRIDE(void, App, onForceDragEnd);
    }
};

class SingleMotionSequence : public UI::SequenceInterface {
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

void bind_imgui(py::module& m) {
    py::module imgui =
        m.def_submodule("imgui", "Small Dear ImGui wrapper for Python apps");

    imgui.attr("WindowFlags_None") =
        py::int_(static_cast<int>(ImGuiWindowFlags_None));
    imgui.attr("WindowFlags_NoTitleBar") =
        py::int_(static_cast<int>(ImGuiWindowFlags_NoTitleBar));
    imgui.attr("WindowFlags_NoBackground") =
        py::int_(static_cast<int>(ImGuiWindowFlags_NoBackground));
    imgui.attr("WindowFlags_NoResize") =
        py::int_(static_cast<int>(ImGuiWindowFlags_NoResize));
    imgui.attr("WindowFlags_NoMove") =
        py::int_(static_cast<int>(ImGuiWindowFlags_NoMove));
    imgui.attr("WindowFlags_NoScrollbar") =
        py::int_(static_cast<int>(ImGuiWindowFlags_NoScrollbar));
    imgui.def(
        "begin",
        [](const std::string& name, int flags) {
            return ImGui::Begin(name.c_str(), nullptr,
                                static_cast<ImGuiWindowFlags>(flags));
        },
        py::arg("name"), py::arg("flags") = 0);
    imgui.def("end", []() { ImGui::End(); });
    imgui.def(
        "begin_child",
        [](const std::string& id, float width, float height, bool border) {
            return ImGui::BeginChild(id.c_str(), ImVec2(width, height), border);
        },
        py::arg("id"), py::arg("width") = 0.0f, py::arg("height") = 0.0f,
        py::arg("border") = false);
    imgui.def("end_child", []() { ImGui::EndChild(); });
    imgui.def("text", [](const std::string& text) {
        ImGui::TextUnformatted(text.c_str());
    });
    imgui.def("text_disabled", [](const std::string& text) {
        ImGui::TextDisabled("%s", text.c_str());
    });
    imgui.def("separator", []() { ImGui::Separator(); });
    imgui.def("same_line", []() { ImGui::SameLine(); });
    imgui.def("main_viewport_work_rect", []() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        return py::make_tuple(viewport->WorkPos.x, viewport->WorkPos.y,
                              viewport->WorkSize.x, viewport->WorkSize.y);
    });
    imgui.def(
        "set_next_window_pos",
        [](float x, float y) {
            ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
        },
        py::arg("x"), py::arg("y"));
    imgui.def(
        "set_next_window_size",
        [](float width, float height) {
            ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);
        },
        py::arg("width"), py::arg("height"));
    imgui.def("button", [](const std::string& label) {
        return ImGui::Button(label.c_str());
    });
    imgui.def(
        "checkbox",
        [](const std::string& label, bool value) {
            bool v = value;
            bool changed = ImGui::Checkbox(label.c_str(), &v);
            return py::make_tuple(changed, v);
        },
        py::arg("label"), py::arg("value"));
    imgui.def(
        "slider_float",
        [](const std::string& label, float value, float min, float max) {
            float v = value;
            bool changed = ImGui::SliderFloat(label.c_str(), &v, min, max);
            return py::make_tuple(changed, v);
        },
        py::arg("label"), py::arg("value"), py::arg("min"), py::arg("max"));
    imgui.def(
        "progress_bar",
        [](float fraction, float width, float height,
           const std::string& overlay) {
            ImGui::ProgressBar(fraction, ImVec2(width, height),
                               overlay.empty() ? nullptr : overlay.c_str());
        },
        py::arg("fraction"), py::arg("width") = -1.0f, py::arg("height") = 0.0f,
        py::arg("overlay") = "");
    imgui.def(
        "motion_sequencer",
        [](const std::string& label, int currentFrame, int frameMin,
           int frameMax, int firstFrame, bool expanded, int selectedEntry,
           const std::string& itemLabel, bool fitToContent) {
            if (frameMax < frameMin)
                frameMax = frameMin;
            currentFrame = std::clamp(currentFrame, frameMin, frameMax);
            firstFrame = std::clamp(firstFrame, frameMin, frameMax);

            SingleMotionSequence sequence(frameMin, frameMax, itemLabel);
            int current = currentFrame;
            int first = firstFrame;
            int selected = selectedEntry;
            bool isExpanded = expanded;

            ImGui::PushID(label.c_str());
            UI::SequencerConfig config;
            config.fitToContent = fitToContent;
            bool changed =
                UI::sequencer(&sequence, &current, &isExpanded, &selected,
                              &first, UI::SequencerChangeFrame, config);
            ImGui::PopID();

            current = std::clamp(current, frameMin, frameMax);
            first = std::clamp(first, frameMin, frameMax);
            return py::make_tuple(changed || current != currentFrame, current,
                                  first, isExpanded, selected);
        },
        py::arg("label"), py::arg("current_frame"), py::arg("frame_min"),
        py::arg("frame_max"), py::arg("first_frame") = 0,
        py::arg("expanded") = true, py::arg("selected_entry") = -1,
        py::arg("item_label") = "Motion", py::arg("fit_to_content") = false);
    imgui.def(
        "motion_sequencer_resizable",
        [](const std::string& label, int currentFrame, int frameMin,
           int frameMax, int firstFrame, bool expanded, int selectedEntry,
           const std::string& itemLabel, bool fitToContent, float legendWidth) {
            if (frameMax < frameMin)
                frameMax = frameMin;
            currentFrame = std::clamp(currentFrame, frameMin, frameMax);
            firstFrame = std::clamp(firstFrame, frameMin, frameMax);

            SingleMotionSequence sequence(frameMin, frameMax, itemLabel);
            int current = currentFrame;
            int first = firstFrame;
            int selected = selectedEntry;
            bool isExpanded = expanded;
            float liveLegendWidth = legendWidth;

            ImGui::PushID(label.c_str());
            UI::SequencerConfig config;
            config.fitToContent = fitToContent;
            config.legendWidth = static_cast<int>(liveLegendWidth);
            config.legendWidthValue = &liveLegendWidth;
            config.minLegendWidth = 96.0f;
            config.maxLegendWidth = 420.0f;
            config.legendResizeHandleWidth = 10.0f;
            bool changed =
                UI::sequencer(&sequence, &current, &isExpanded, &selected,
                              &first, UI::SequencerChangeFrame, config);
            ImGui::PopID();

            current = std::clamp(current, frameMin, frameMax);
            first = std::clamp(first, frameMin, frameMax);
            return py::make_tuple(changed || current != currentFrame, current,
                                  first, isExpanded, selected, liveLegendWidth);
        },
        py::arg("label"), py::arg("current_frame"), py::arg("frame_min"),
        py::arg("frame_max"), py::arg("first_frame") = 0,
        py::arg("expanded") = true, py::arg("selected_entry") = -1,
        py::arg("item_label") = "Motion", py::arg("fit_to_content") = false,
        py::arg("legend_width") = 200.0f);
}

void bind_keys(py::module& m) {
    py::module keys = m.def_submodule("keys", "GLFW keyboard key constants");

    keys.attr("SPACE") = GLFW_KEY_SPACE;
    keys.attr("APOSTROPHE") = GLFW_KEY_APOSTROPHE;
    keys.attr("COMMA") = GLFW_KEY_COMMA;
    keys.attr("MINUS") = GLFW_KEY_MINUS;
    keys.attr("PERIOD") = GLFW_KEY_PERIOD;
    keys.attr("SLASH") = GLFW_KEY_SLASH;
    keys.attr("SEMICOLON") = GLFW_KEY_SEMICOLON;
    keys.attr("EQUAL") = GLFW_KEY_EQUAL;
    keys.attr("LEFT_BRACKET") = GLFW_KEY_LEFT_BRACKET;
    keys.attr("BACKSLASH") = GLFW_KEY_BACKSLASH;
    keys.attr("RIGHT_BRACKET") = GLFW_KEY_RIGHT_BRACKET;
    keys.attr("GRAVE_ACCENT") = GLFW_KEY_GRAVE_ACCENT;
    keys.attr("WORLD_1") = GLFW_KEY_WORLD_1;
    keys.attr("WORLD_2") = GLFW_KEY_WORLD_2;

    keys.attr("NUM_0") = GLFW_KEY_0;
    keys.attr("NUM_1") = GLFW_KEY_1;
    keys.attr("NUM_2") = GLFW_KEY_2;
    keys.attr("NUM_3") = GLFW_KEY_3;
    keys.attr("NUM_4") = GLFW_KEY_4;
    keys.attr("NUM_5") = GLFW_KEY_5;
    keys.attr("NUM_6") = GLFW_KEY_6;
    keys.attr("NUM_7") = GLFW_KEY_7;
    keys.attr("NUM_8") = GLFW_KEY_8;
    keys.attr("NUM_9") = GLFW_KEY_9;

    for (char c = 'A'; c <= 'Z'; ++c) {
        keys.attr(std::string(1, c).c_str()) = GLFW_KEY_A + (c - 'A');
    }

    keys.attr("ESCAPE") = GLFW_KEY_ESCAPE;
    keys.attr("ENTER") = GLFW_KEY_ENTER;
    keys.attr("TAB") = GLFW_KEY_TAB;
    keys.attr("BACKSPACE") = GLFW_KEY_BACKSPACE;
    keys.attr("INSERT") = GLFW_KEY_INSERT;
    keys.attr("DELETE") = GLFW_KEY_DELETE;
    keys.attr("RIGHT") = GLFW_KEY_RIGHT;
    keys.attr("LEFT") = GLFW_KEY_LEFT;
    keys.attr("DOWN") = GLFW_KEY_DOWN;
    keys.attr("UP") = GLFW_KEY_UP;
    keys.attr("PAGE_UP") = GLFW_KEY_PAGE_UP;
    keys.attr("PAGE_DOWN") = GLFW_KEY_PAGE_DOWN;
    keys.attr("HOME") = GLFW_KEY_HOME;
    keys.attr("END") = GLFW_KEY_END;
    keys.attr("CAPS_LOCK") = GLFW_KEY_CAPS_LOCK;
    keys.attr("SCROLL_LOCK") = GLFW_KEY_SCROLL_LOCK;
    keys.attr("NUM_LOCK") = GLFW_KEY_NUM_LOCK;
    keys.attr("PRINT_SCREEN") = GLFW_KEY_PRINT_SCREEN;
    keys.attr("PAUSE") = GLFW_KEY_PAUSE;

    for (int i = 1; i <= 25; ++i) {
        keys.attr(("F" + std::to_string(i)).c_str()) = GLFW_KEY_F1 + (i - 1);
    }

    keys.attr("KP_0") = GLFW_KEY_KP_0;
    keys.attr("KP_1") = GLFW_KEY_KP_1;
    keys.attr("KP_2") = GLFW_KEY_KP_2;
    keys.attr("KP_3") = GLFW_KEY_KP_3;
    keys.attr("KP_4") = GLFW_KEY_KP_4;
    keys.attr("KP_5") = GLFW_KEY_KP_5;
    keys.attr("KP_6") = GLFW_KEY_KP_6;
    keys.attr("KP_7") = GLFW_KEY_KP_7;
    keys.attr("KP_8") = GLFW_KEY_KP_8;
    keys.attr("KP_9") = GLFW_KEY_KP_9;
    keys.attr("KP_DECIMAL") = GLFW_KEY_KP_DECIMAL;
    keys.attr("KP_DIVIDE") = GLFW_KEY_KP_DIVIDE;
    keys.attr("KP_MULTIPLY") = GLFW_KEY_KP_MULTIPLY;
    keys.attr("KP_SUBTRACT") = GLFW_KEY_KP_SUBTRACT;
    keys.attr("KP_ADD") = GLFW_KEY_KP_ADD;
    keys.attr("KP_ENTER") = GLFW_KEY_KP_ENTER;
    keys.attr("KP_EQUAL") = GLFW_KEY_KP_EQUAL;

    keys.attr("LEFT_SHIFT") = GLFW_KEY_LEFT_SHIFT;
    keys.attr("LEFT_CONTROL") = GLFW_KEY_LEFT_CONTROL;
    keys.attr("LEFT_ALT") = GLFW_KEY_LEFT_ALT;
    keys.attr("LEFT_SUPER") = GLFW_KEY_LEFT_SUPER;
    keys.attr("RIGHT_SHIFT") = GLFW_KEY_RIGHT_SHIFT;
    keys.attr("RIGHT_CONTROL") = GLFW_KEY_RIGHT_CONTROL;
    keys.attr("RIGHT_ALT") = GLFW_KEY_RIGHT_ALT;
    keys.attr("RIGHT_SUPER") = GLFW_KEY_RIGHT_SUPER;
    keys.attr("MENU") = GLFW_KEY_MENU;
}

PYBIND11_MODULE(_kangengine, m) {
    m.doc() = "KangEngine Python bindings";

    // Enums first — submodule bindings may use them as default arguments
    py::enum_<UpAxis>(m, "UpAxis")
        .value("X", UpAxis::X)
        .value("Y", UpAxis::Y)
        .value("Z", UpAxis::Z)
        .export_values();

    py::enum_<Backend::BackendType>(m, "BackendType")
        .value("OpenGL", Backend::BackendType::OpenGL)
        .value("Vulkan", Backend::BackendType::Vulkan)
        .value("WebGPU", Backend::BackendType::WebGPU)
        .export_values();

    py::enum_<TransformSource>(m, "TransformSource")
        .value("SceneGraph", TransformSource::SceneGraph)
        .value("ExternalBuffer", TransformSource::ExternalBuffer);

    py::enum_<InteractionMode>(m, "InteractionMode")
        .value("Inspect", InteractionMode::Inspect)
        .value("Edit", InteractionMode::Edit)
        .value("Force", InteractionMode::Force)
        .export_values();

    py::class_<RayPickResult>(m, "RayPickResult")
        .def_readonly("hit", &RayPickResult::hit)
        .def_readonly("handle", &RayPickResult::handle)
        .def_readonly("instance_index", &RayPickResult::instanceIndex)
        .def_readonly("transform_source", &RayPickResult::transformSource)
        .def_readonly("distance", &RayPickResult::distance)
        .def_readonly("position", &RayPickResult::position);

    bind_scene(m);
    bind_animation(m);
    bind_imgui(m);
    bind_keys(m);
    // bind_physics is called after GLM types are registered (uses glm defaults)

    py::class_<MotionSequencerPanel>(m, "MotionSequencerPanel")
        .def(py::init<>())
        .def("set_motion", &MotionSequencerPanel::setMotion,
             py::arg("motion_name"), py::arg("num_frames"), py::arg("fps"))
        .def("set_current_time", &MotionSequencerPanel::setCurrentTime)
        .def("current_time", &MotionSequencerPanel::currentTime)
        .def("duration", &MotionSequencerPanel::duration)
        .def("set_playing", &MotionSequencerPanel::setPlaying)
        .def("is_playing", &MotionSequencerPanel::isPlaying)
        .def("set_loop", &MotionSequencerPanel::setLoop)
        .def("loop", &MotionSequencerPanel::loop)
        .def("set_time_scale", &MotionSequencerPanel::setTimeScale)
        .def("time_scale", &MotionSequencerPanel::timeScale)
        .def("set_transparent", &MotionSequencerPanel::setTransparent)
        .def("set_overlay", &MotionSequencerPanel::setOverlay)
        .def("set_overlay_width_ratio",
             &MotionSequencerPanel::setOverlayWidthRatio)
        .def("set_legend_width", &MotionSequencerPanel::setLegendWidth)
        .def("legend_width", &MotionSequencerPanel::legendWidth)
        .def("set_ui_scale", &MotionSequencerPanel::setUiScale)
        .def("ui_scale", &MotionSequencerPanel::uiScale)
        .def("set_folded", &MotionSequencerPanel::setFolded)
        .def("set_show_progress_bar", &MotionSequencerPanel::setShowProgressBar)
        .def("build_panel", &MotionSequencerPanel::buildPanel);

    // Materials & Colors
    py::class_<Color>(m, "Color")
        .def(py::init<>())
        .def_readwrite("r", &Color::r)
        .def_readwrite("g", &Color::g)
        .def_readwrite("b", &Color::b)
        .def_readwrite("a", &Color::a);

    py::enum_<ColorType>(m, "ColorType")
        .value("WHITE", ColorType::WHITE)
        .value("BLACK", ColorType::BLACK)
        .value("RED", ColorType::RED)
        .value("GREEN", ColorType::GREEN)
        .value("BLUE", ColorType::BLUE)
        .value("YELLOW", ColorType::YELLOW)
        .value("CYAN", ColorType::CYAN)
        .value("MAGENTA", ColorType::MAGENTA)
        .value("ORANGE", ColorType::ORANGE)
        .value("PURPLE", ColorType::PURPLE)
        .value("PINK", ColorType::PINK)
        .value("BROWN", ColorType::BROWN)
        .value("GRAY", ColorType::GRAY)
        .value("LIGHT_GRAY", ColorType::LIGHT_GRAY)
        .value("DARK_GRAY", ColorType::DARK_GRAY)
        .value("CORAL", ColorType::CORAL)
        .value("SALMON", ColorType::SALMON)
        .value("TOMATO", ColorType::TOMATO)
        .value("CRIMSON", ColorType::CRIMSON)
        .value("FOREST_GREEN", ColorType::FOREST_GREEN)
        .value("LIME_GREEN", ColorType::LIME_GREEN)
        .value("SEA_GREEN", ColorType::SEA_GREEN)
        .value("TEAL", ColorType::TEAL)
        .value("NAVY", ColorType::NAVY)
        .value("SKY_BLUE", ColorType::SKY_BLUE)
        .value("STEEL_BLUE", ColorType::STEEL_BLUE)
        .value("ROYAL_BLUE", ColorType::ROYAL_BLUE)
        .value("INDIGO", ColorType::INDIGO)
        .value("VIOLET", ColorType::VIOLET)
        .value("ORCHID", ColorType::ORCHID)
        .value("GOLD", ColorType::GOLD)
        .value("KHAKI", ColorType::KHAKI)
        .value("OLIVE", ColorType::OLIVE)
        .value("MAROON", ColorType::MAROON)
        .value("BEIGE", ColorType::BEIGE)
        .value("IVORY", ColorType::IVORY)
        .value("MINT", ColorType::MINT)
        .value("LAVENDER", ColorType::LAVENDER)
        .value("SLATE_GRAY", ColorType::SLATE_GRAY)
        .value("PASTEL_RED", ColorType::PASTEL_RED)
        .value("PASTEL_ORANGE", ColorType::PASTEL_ORANGE)
        .value("PASTEL_YELLOW", ColorType::PASTEL_YELLOW)
        .value("PASTEL_GREEN", ColorType::PASTEL_GREEN)
        .value("PASTEL_MINT", ColorType::PASTEL_MINT)
        .value("PASTEL_CYAN", ColorType::PASTEL_CYAN)
        .value("PASTEL_BLUE", ColorType::PASTEL_BLUE)
        .value("PASTEL_PURPLE", ColorType::PASTEL_PURPLE)
        .value("PASTEL_PINK", ColorType::PASTEL_PINK)
        .value("PASTEL_ROSE", ColorType::PASTEL_ROSE)
        .value("PASTEL_PEACH", ColorType::PASTEL_PEACH)
        .value("PASTEL_LAVENDER", ColorType::PASTEL_LAVENDER)
        .value("PASTEL_LILAC", ColorType::PASTEL_LILAC)
        .value("PASTEL_CORAL", ColorType::PASTEL_CORAL)
        .value("PASTEL_CREAM", ColorType::PASTEL_CREAM)
        .value("PASTEL_SKY", ColorType::PASTEL_SKY);

    py::class_<ColorLibrary>(m, "ColorLibrary")
        .def_static("get", &ColorLibrary::get, py::arg("type"));

    py::class_<PhongMaterial>(m, "PhongMaterial").def(py::init<>());
    // .def_readwrite("diffuse", &PhongMaterial::diffuse)

    // Backend::Shader
    py::class_<Backend::Shader>(m, "Shader")
        .def("use", &Backend::Shader::use)
        .def("bind", &Backend::Shader::bind)
        .def("unbind", &Backend::Shader::unbind)
        .def("setBool", &Backend::Shader::setBool)
        .def("setInt", &Backend::Shader::setInt)
        .def("setFloat", &Backend::Shader::setFloat)
        .def("setColor", &Backend::Shader::setColor)
        .def("setVec2", py::overload_cast<const std::string&, const glm::vec2&>(
                            &Backend::Shader::setVec2))
        .def("setVec2", py::overload_cast<const std::string&, float, float>(
                            &Backend::Shader::setVec2))
        .def("setVec3", py::overload_cast<const std::string&, const glm::vec3&>(
                            &Backend::Shader::setVec3))
        .def("setVec3",
             py::overload_cast<const std::string&, float, float, float>(
                 &Backend::Shader::setVec3))
        .def("setVec4", py::overload_cast<const std::string&, const glm::vec4&>(
                            &Backend::Shader::setVec4))
        .def("setVec4",
             py::overload_cast<const std::string&, float, float, float, float>(
                 &Backend::Shader::setVec4))
        .def("setMat2", &Backend::Shader::setMat2)
        .def("setMat3", &Backend::Shader::setMat3)
        .def("setMat4", &Backend::Shader::setMat4)
        .def("setUniformBlockBinding", &Backend::Shader::setUniformBlockBinding,
             py::arg("block_name"), py::arg("binding_point"));

    // Backend::Texture
    py::class_<Backend::Texture>(m, "Texture")
        .def("bind", &Backend::Texture::bind, py::arg("slot") = 0)
        .def("unbind", &Backend::Texture::unbind)
        .def("getWidth", &Backend::Texture::getWidth)
        .def("getHeight", &Backend::Texture::getHeight);

    // Backend::GraphicsDevice
    py::class_<Backend::GraphicsDevice,
               std::shared_ptr<Backend::GraphicsDevice>>(m, "GraphicsDevice")
        .def("createShader",
             py::overload_cast<const std::string&, const std::string&>(
                 &Backend::GraphicsDevice::createShader),
             py::return_value_policy::take_ownership)
        .def("createShaderFromFile",
             &Backend::GraphicsDevice::createShaderFromFile,
             py::arg("vert_path"), py::arg("frag_path"),
             py::return_value_policy::take_ownership)
        .def("createTexture",
             py::overload_cast<const std::string, bool, float, float, float>(
                 &Backend::GraphicsDevice::createTexture),
             py::arg("path"), py::arg("flip") = false,
             py::arg("warpParam") = GL_REPEAT,
             py::arg("minFilterParam") = GL_LINEAR_MIPMAP_LINEAR,
             py::arg("maxFilterParam") = GL_LINEAR,
             py::return_value_policy::take_ownership);

    py::class_<Renderer>(m, "Renderer")
        .def(
            "device", [](Renderer& self) { return self.device(); },
            py::return_value_policy::reference)
        .def(
            "update_renderable_transforms",
            [](Renderer& self, uint32_t handle, const FloatArray& transforms,
               py::object colors) {
                auto transformVec = mat4Array(transforms, "transforms");
                std::vector<glm::vec4> colorVec;
                const std::vector<glm::vec4>* colorPtr = nullptr;
                if (!colors.is_none()) {
                    auto colorArray = colors.cast<FloatArray>();
                    colorVec = vec4Array(colorArray, "colors");
                    if (colorVec.size() != 1 &&
                        colorVec.size() != transformVec.size()) {
                        throw py::value_error(
                            "colors must have length 1 or match transforms");
                    }
                    if (colorVec.size() == 1 && transformVec.size() > 1)
                        colorVec.resize(transformVec.size(), colorVec[0]);
                    colorPtr = &colorVec;
                }
                self.updateRenderableTransforms(handle, transformVec, colorPtr);
            },
            py::arg("handle"), py::arg("transforms"),
            py::arg("colors") = py::none())
        .def(
            "set_renderable_colors",
            [](Renderer& self, uint32_t handle, const FloatArray& colors) {
                self.setRenderableColors(handle, vec4Array(colors, "colors"));
            },
            py::arg("handle"), py::arg("colors"))
        .def("set_renderable_double_sided", &Renderer::setRenderableDoubleSided,
             py::arg("handle"), py::arg("double_sided") = true)
        .def("set_renderable_casts_shadow", &Renderer::setRenderableCastsShadow,
             py::arg("handle"), py::arg("casts_shadow") = true)
        .def("set_renderable_texture", &Renderer::setRenderableTexture,
             py::arg("handle"), py::arg("texture"), py::arg("slot") = 0)
        .def(
            "update_renderable_geometry",
            [](Renderer& self, uint32_t handle, const FloatArray& positions,
               py::object normals) {
                auto positionVec = vec3Array(positions, "positions");
                std::vector<glm::vec3> normalVec;
                if (!normals.is_none()) {
                    auto normalArray = normals.cast<FloatArray>();
                    normalVec = vec3Array(normalArray, "normals");
                    if (normalVec.size() != positionVec.size()) {
                        throw py::value_error(
                            "normals must match positions length");
                    }
                }
                self.updateRenderableGeometry(handle, positionVec, normalVec);
            },
            py::arg("handle"), py::arg("positions"),
            py::arg("normals") = py::none())
        .def(
            "update_renderable_skinning_matrices",
            [](Renderer& self, uint32_t handle, const FloatArray& matrices) {
                self.updateRenderableSkinningMatrices(
                    handle, mat4Array(matrices, "bone_matrices"));
            },
            py::arg("handle"), py::arg("bone_matrices"));

    // GLM types for matrix operations
    // Support implicit conversion from PyGLM types (tuple/list with x,y,z or
    // indexable)
    /*
py::class_<glm::vec3>(m, "vec3")
    .def(py::init<float, float, float>())
    .def(py::init([](py::object obj) {
        // Accept PyGLM vec3 or any object with x, y, z attributes
        if (py::hasattr(obj, "x") && py::hasattr(obj, "y") &&
            py::hasattr(obj, "z")) {
            return glm::vec3(obj.attr("x").cast<float>(),
                             obj.attr("y").cast<float>(),
                             obj.attr("z").cast<float>());
        }
        // Accept tuple/list
        if (py::isinstance<py::sequence>(obj) && py::len(obj) == 3) {
            auto seq = obj.cast<py::sequence>();
            return glm::vec3(seq[0].cast<float>(), seq[1].cast<float>(),
                             seq[2].cast<float>());
        }
        throw std::runtime_error("Cannot convert to vec3");
    }))
    .def_readwrite("x", &glm::vec3::x)
    .def_readwrite("y", &glm::vec3::y)
    .def_readwrite("z", &glm::vec3::z)
    .def("__repr__", [](const glm::vec3& v) {
        return "vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) +
               ", " + std::to_string(v.z) + ")";
    });
    */
    py::class_<glm::vec2>(m, "vec2", py::buffer_protocol())
        .def(py::init<float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f)
        .def(py::init([](py::handle obj) {
            if (py::isinstance<glm::vec2>(obj))
                return obj.cast<glm::vec2>();

            try {
                if (py::isinstance<py::buffer>(obj)) {
                    auto buf = obj.cast<py::buffer>().request();
                    if (buf.size == 2) {
                        auto read = [&](int i) -> float {
                            const char* p = static_cast<const char*>(buf.ptr) +
                                            i * buf.strides[0];
                            if (buf.format ==
                                py::format_descriptor<double>::format())
                                return static_cast<float>(
                                    *reinterpret_cast<const double*>(p));
                            return *reinterpret_cast<const float*>(p);
                        };
                        return glm::vec2(read(0), read(1));
                    }
                }
            } catch (...) {
            }

            if (py::hasattr(obj, "x") && py::hasattr(obj, "y"))
                return glm::vec2(obj.attr("x").cast<float>(),
                                 obj.attr("y").cast<float>());

            if (py::len_hint(obj) == 2)
                return glm::vec2(obj[py::int_(0)].cast<float>(),
                                 obj[py::int_(1)].cast<float>());

            throw py::value_error("Cannot convert input to ke.vec2");
        }))
        .def_buffer([](glm::vec2& v) -> py::buffer_info {
            return py::buffer_info(&v.x, sizeof(float),
                                   py::format_descriptor<float>::format(), 1,
                                   {2}, {sizeof(float)});
        })
        .def_readwrite("x", &glm::vec2::x)
        .def_readwrite("y", &glm::vec2::y)
        .def("__repr__", [](const glm::vec2& v) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);
            oss << "ke.vec2(" << v.x << ", " << v.y << ")";
            return oss.str();
        });

    py::implicitly_convertible<py::object, glm::vec2>();

    py::class_<glm::vec3>(m, "vec3", py::buffer_protocol())
        .def(py::init<float, float, float>(), py::arg("x") = 0.0f,
             py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def(py::init([](py::handle obj) {
            if (py::isinstance<glm::vec3>(obj))
                return obj.cast<glm::vec3>();

            // Buffer protocol: numpy (float32 or float64), PyGLM, etc.
            try {
                if (py::isinstance<py::buffer>(obj)) {
                    auto buf = obj.cast<py::buffer>().request();
                    if (buf.size == 3) {
                        auto read = [&](int i) -> float {
                            const char* p = static_cast<const char*>(buf.ptr) +
                                            i * buf.strides[0];
                            if (buf.format ==
                                py::format_descriptor<double>::format())
                                return static_cast<float>(
                                    *reinterpret_cast<const double*>(p));
                            return *reinterpret_cast<const float*>(p);
                        };
                        return glm::vec3(read(0), read(1), read(2));
                    }
                }
            } catch (...) {
            }

            // Attribute access (PyGLM objects, etc.)
            if (py::hasattr(obj, "x") && py::hasattr(obj, "y") &&
                py::hasattr(obj, "z"))
                return glm::vec3(obj.attr("x").cast<float>(),
                                 obj.attr("y").cast<float>(),
                                 obj.attr("z").cast<float>());

            // Sequence fallback (list, tuple, numpy array via __getitem__)
            if (py::len_hint(obj) == 3)
                return glm::vec3(obj[py::int_(0)].cast<float>(),
                                 obj[py::int_(1)].cast<float>(),
                                 obj[py::int_(2)].cast<float>());

            throw py::value_error("Cannot convert input to ke.vec3");
        }))
        // Buffer Protocol 설정 (Python -> C++ 데이터 전송용)
        .def_buffer([](glm::vec3& v) -> py::buffer_info {
            return py::buffer_info(&v.x, sizeof(float),
                                   py::format_descriptor<float>::format(), 1,
                                   {3}, {sizeof(float)});
        })
        .def_readwrite("x", &glm::vec3::x)
        .def_readwrite("y", &glm::vec3::y)
        .def_readwrite("z", &glm::vec3::z)
        .def("__repr__", [](const glm::vec3& v) {
            std::ostringstream oss;
            // 소수점 한자리만 출력하게 설정하면 보기 편합니다.
            oss << std::fixed << std::setprecision(2);
            oss << "ke.vec3(" << v.x << ", " << v.y << ", " << v.z << ")";
            return oss.str();
        });

    // 암시적 형변환 등록
    py::implicitly_convertible<py::object, glm::vec3>();

    py::class_<glm::vec4>(m, "vec4", py::buffer_protocol())
        .def(py::init<float, float, float, float>())
        .def(py::init([](py::object obj) {
            // Buffer protocol
            try {
                if (py::isinstance<py::buffer>(obj)) {
                    auto buf = obj.cast<py::buffer>().request();
                    if (buf.size == 4) {
                        float* ptr = static_cast<float*>(buf.ptr);
                        return glm::vec4(ptr[0], ptr[1], ptr[2], ptr[3]);
                    }
                }
            } catch (...) {
            }
            // Attribute access
            if (py::hasattr(obj, "x") && py::hasattr(obj, "y") &&
                py::hasattr(obj, "z") && py::hasattr(obj, "w")) {
                return glm::vec4(
                    obj.attr("x").cast<float>(), obj.attr("y").cast<float>(),
                    obj.attr("z").cast<float>(), obj.attr("w").cast<float>());
            }
            // Sequence
            if (py::isinstance<py::sequence>(obj) && py::len(obj) == 4) {
                auto seq = obj.cast<py::sequence>();
                return glm::vec4(seq[0].cast<float>(), seq[1].cast<float>(),
                                 seq[2].cast<float>(), seq[3].cast<float>());
            }
            throw std::runtime_error("Cannot convert to vec4");
        }))
        .def_buffer([](glm::vec4& v) -> py::buffer_info {
            return py::buffer_info(&v.x, sizeof(float),
                                   py::format_descriptor<float>::format(), 1,
                                   {4}, {sizeof(float)});
        })
        .def_readwrite("x", &glm::vec4::x)
        .def_readwrite("y", &glm::vec4::y)
        .def_readwrite("z", &glm::vec4::z)
        .def_readwrite("w", &glm::vec4::w)
        .def("__repr__", [](const glm::vec4& v) {
            return "vec4(" + std::to_string(v.x) + ", " + std::to_string(v.y) +
                   ", " + std::to_string(v.z) + ", " + std::to_string(v.w) +
                   ")";
        });

    py::implicitly_convertible<py::object, glm::vec4>();

    py::class_<glm::quat>(m, "quat", py::buffer_protocol())
        .def(py::init<float, float, float, float>(), py::arg("w"), py::arg("x"),
             py::arg("y"), py::arg("z"))
        .def(py::init([](py::object obj) {
            // Buffer protocol (GLM quat memory layout: x, y, z, w)
            try {
                if (py::isinstance<py::buffer>(obj)) {
                    auto buf = obj.cast<py::buffer>().request();
                    if (buf.size == 4) {
                        float* ptr = static_cast<float*>(buf.ptr);
                        // GLM stores as x,y,z,w but constructor is w,x,y,z
                        return glm::quat(ptr[3], ptr[0], ptr[1], ptr[2]);
                    }
                }
            } catch (...) {
            }
            // Attribute access
            if (py::hasattr(obj, "w") && py::hasattr(obj, "x") &&
                py::hasattr(obj, "y") && py::hasattr(obj, "z")) {
                return glm::quat(
                    obj.attr("w").cast<float>(), obj.attr("x").cast<float>(),
                    obj.attr("y").cast<float>(), obj.attr("z").cast<float>());
            }
            // Sequence
            if (py::isinstance<py::sequence>(obj) && py::len(obj) == 4) {
                auto seq = obj.cast<py::sequence>();
                // PyGLM quat: (w, x, y, z) order
                return glm::quat(seq[0].cast<float>(), seq[1].cast<float>(),
                                 seq[2].cast<float>(), seq[3].cast<float>());
            }
            throw std::runtime_error("Cannot convert to quat");
        }))
        .def_buffer([](glm::quat& q) -> py::buffer_info {
            // GLM quat memory layout: x, y, z, w (연속 메모리)
            return py::buffer_info(&q.x, sizeof(float),
                                   py::format_descriptor<float>::format(), 1,
                                   {4}, {sizeof(float)});
        })
        .def_readwrite("w", &glm::quat::w)
        .def_readwrite("x", &glm::quat::x)
        .def_readwrite("y", &glm::quat::y)
        .def_readwrite("z", &glm::quat::z)
        .def("__repr__", [](const glm::quat& q) {
            return "quat(w=" + std::to_string(q.w) +
                   ", x=" + std::to_string(q.x) + ", y=" + std::to_string(q.y) +
                   ", z=" + std::to_string(q.z) + ")";
        });

    py::implicitly_convertible<py::object, glm::quat>();

    py::class_<glm::mat3>(m, "mat3", py::buffer_protocol())
        .def(py::init<float>(), py::arg("value") = 1.0f)
        .def(py::init([](py::handle obj) {
            glm::mat3 m(1.0f);

            // A. Buffer Protocol (PyGLM, Numpy 등)
            try {
                if (py::isinstance<py::buffer>(obj)) {
                    auto buf = obj.cast<py::buffer>().request();
                    if (buf.size == 9) {
                        std::memcpy(&m[0][0], buf.ptr, 9 * sizeof(float));
                        return m;
                    }
                }
            } catch (...) {
            }

            // B. Sequence (List, Tuple)
            if (py::isinstance<py::sequence>(obj) && py::len(obj) == 3) {
                auto seq = obj.cast<py::sequence>();
                for (int i = 0; i < 3; ++i) {
                    auto col = seq[i].cast<py::sequence>();
                    for (int j = 0; j < 3; ++j) {
                        m[i][j] = col[j].cast<float>();
                    }
                }
                return m;
            }

            throw py::value_error("Cannot convert to ke.mat3");
        }))
        .def_buffer([](glm::mat3& m) -> py::buffer_info {
            return py::buffer_info(
                &m[0][0], sizeof(float), py::format_descriptor<float>::format(),
                2, {3, 3}, {sizeof(float), sizeof(float) * 3}
                // Column-major: column 연속, row stride = 3 floats
            );
        })
        .def("__repr__", [](const glm::mat3& m) {
            std::ostringstream oss;
            oss << "ke.mat3(\n";
            for (int i = 0; i < 3; ++i) {
                oss << "  [" << m[0][i] << ", " << m[1][i] << ", " << m[2][i]
                    << "]" << (i < 2 ? ",\n" : "\n");
            }
            oss << ")";
            return oss.str();
        });

    py::implicitly_convertible<py::object, glm::mat3>();

    // 1. mat4 클래스 정의 및 버퍼 프로토콜 활성화
    py::class_<glm::mat4>(m, "mat4", py::buffer_protocol())
        // 기본 생성자 (Identity)
        .def(py::init<float>(), py::arg("value") = 1.0f)

        // 통합 생성자: PyGLM, 리스트, 튜플, 넘파이 모두 처리
        .def(py::init([](py::handle obj) {
            glm::mat4 m(1.0f);

            // 1. Buffer Protocol 시도 (PyGLM, Numpy 등)
            // obj.cast<py::object>()를 통해 handle을 object로 변환 후 buffer로
            // 접근합니다.
            try {
                if (py::isinstance<py::buffer>(obj)) {
                    auto buf = obj.cast<py::buffer>().request();
                    if (buf.size == 16) {
                        std::memcpy(&m[0][0], buf.ptr, 16 * sizeof(float));
                        return m;
                    }
                }
            } catch (...) {
                // 버퍼 요청 실패 시 다음으로 진행
            }

            // 2. Sequence 시도 (List, Tuple)
            if (py::isinstance<py::sequence>(obj) && py::len(obj) == 4) {
                auto seq = obj.cast<py::sequence>();
                for (int i = 0; i < 4; ++i) {
                    auto col = seq[i].cast<py::sequence>();
                    if (py::len(col) == 4) {
                        for (int j = 0; j < 4; ++j) {
                            m[i][j] = col[j].cast<float>();
                        }
                    }
                }
                return m;
            }

            throw py::value_error("Cannot convert input to ke.mat4");
        }))

        // C. 버퍼 정보 제공 (엔진 -> 파이썬 데이터 전송용)
        .def_buffer([](glm::mat4& m) -> py::buffer_info {
            return py::buffer_info(
                &m[0][0], sizeof(float), py::format_descriptor<float>::format(),
                2,
                // {4, 4}, {sizeof(float) * 4, sizeof(float)} // row-major
                {4, 4}, {sizeof(float), sizeof(float) * 4}
                // Column-major: column 연속, row stride = 4 floats
            );
        })

        // D. 파이썬 스타일의 문자열 출력
        .def("__repr__",
             [](const glm::mat4& m) {
                 std::ostringstream oss;
                 oss << "ke.mat4(\n";
                 for (int i = 0; i < 4; ++i) {
                     oss << "  [" << m[0][i] << ", " << m[1][i] << ", "
                         << m[2][i] << ", " << m[3][i] << "]"
                         << (i < 3 ? ",\n" : "\n");
                 }
                 oss << ")";
                 return oss.str();
             })
        .def("__getitem__", [](const glm::mat4& m, int index) {
            if (index < 0 || index >= 4)
                throw py::index_error();
            return m[index]; // glm::vec4 반환
        });

    py::implicitly_convertible<py::object, glm::mat4>();

    // Physics bindings after GLM types — reset_root uses glm::vec3/quat
    // defaults
    bind_physics(m);

    // GLM helper functions
    m.def(
        "translate",
        [](const glm::mat4& mat, const glm::vec3& vec) {
            return glm::translate(mat, vec);
        },
        "Translate a matrix by a vector");

    m.def(
        "scale",
        [](const glm::mat4& mat, const glm::vec3& vec) {
            return glm::scale(mat, vec);
        },
        "Scale a matrix by a vector");

    py::class_<Camera>(m, "Camera")
        .def("get_camera_pos", &Camera::getCameraPos)
        .def("get_target_pos", &Camera::getTargetPos)
        .def("get_camera_look_dir", &Camera::getCameraLookDir)
        .def("get_camera_up_dir", &Camera::getCameraUpDir)
        .def("get_camera_right_dir", &Camera::getCameraRightDir)
        .def("get_fov", &Camera::getFoV)
        .def("get_near_plane", &Camera::getNearPlane)
        .def("get_far_plane", &Camera::getFarPlane)
        .def("set_camera_pos", &Camera::setCameraPos, py::arg("camera_pos"))
        .def("set_target_pos", &Camera::setTargetPos, py::arg("target_pos"))
        .def("set_fov", &Camera::setFoV, py::arg("fov"))
        .def("set_near_plane", &Camera::setNearPlane, py::arg("distance"))
        .def("set_far_plane", &Camera::setFarPlane, py::arg("distance"));

    // App class with trampoline for Python overrides
    py::class_<App, PyApp>(m, "App")
        .def(py::init<>())
        .def("initialize", &App::initialize, py::arg("width"),
             py::arg("height"), py::arg("hideUi") = false,
             py::arg("upAxis") = UpAxis::Y,
             py::arg("graphicsBackendType") = Backend::BackendType::OpenGL,
             py::arg("sceneBackendType") = Scene::BackendType::Native,
             py::arg("headless") = false)
        .def("setRenderHz", &App::setRenderHz, py::arg("renderHz"))
        .def("get_delta_time", &App::getDeltaTime)
        .def("get_render_hz", &App::getRenderHz)
        .def("get_ui_scale",
             [](const App& self) { return self.getUiScale().value(); })
        .def("set_camera_move_speed", &App::setCameraMoveSpeed,
             py::arg("speed"))
        .def("get_camera_move_speed", &App::getCameraMoveSpeed)
        .def("start", &App::start)
        .def("render_frame_once", &App::renderFrameOnce)
        .def(
            "read_rgb_pixels",
            [](App& self, bool flipY) {
                const int width = self.getWidth();
                const int height = self.getHeight();
                py::array_t<uint8_t> out({height, width, 3});
                auto view = out.mutable_unchecked<3>();

                std::vector<uint8_t> pixels = self.readRgbPixels(flipY);
                if (pixels.size() != static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) * 3)
                    return out;
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const std::size_t src =
                            (static_cast<std::size_t>(y) * width + x) * 3;
                        view(y, x, 0) = pixels[src + 0];
                        view(y, x, 1) = pixels[src + 1];
                        view(y, x, 2) = pixels[src + 2];
                    }
                }
                return out;
            },
            py::arg("flip_y") = true)
        .def("write_pixels_png", &App::writePixelsPNG, py::arg("path"),
             py::arg("flip_y") = true)
        .def("should_close", &App::shouldClose)
        .def("setup", &App::setup)
        .def("preRender", &App::preRender)
        .def("render", &App::render)
        .def("postRender", &App::postRender)
        .def("onRayPicked", &App::onRayPicked)
        .def("onRayPickHover", &App::onRayPickHover)
        .def("onForceDragBegin", &App::onForceDragBegin)
        .def("onForceDragUpdate", &App::onForceDragUpdate)
        .def("onForceDragEnd", &App::onForceDragEnd)
        .def("get_interaction_mode", &App::getInteractionMode)
        .def("set_interaction_mode", &App::setInteractionMode)
        .def("has_selection", &App::hasSelection)
        .def("clear_selection", &App::clearSelection)
        .def("getGraphicsDevice", &App::getGraphicsDevice,
             py::return_value_policy::reference)
        .def(
            "getRenderer", [](App& self) { return &self.getRenderer(); },
            py::return_value_policy::reference)
        .def(
            "get_renderer", [](App& self) { return &self.getRenderer(); },
            py::return_value_policy::reference)
        .def(
            "add_renderable",
            [](App* self, Backend::Shader* shader, Scene::Prim* prim,
               TransformSource transformSource) {
                return self->addRenderable(shader, prim, transformSource);
            },
            py::arg("shader"), py::arg("prim"),
            py::arg("transform_source") = TransformSource::SceneGraph)
        .def(
            "add_skinned_renderable",
            [](App* self, Backend::Shader* shader, Scene::Prim* prim,
               std::shared_ptr<Scene::SkinnedMeshData> skinnedMesh,
               TransformSource transformSource) {
                if (!skinnedMesh)
                    throw py::value_error("skinned_mesh_data is None");
                return self->addSkinnedRenderable(shader, prim, *skinnedMesh,
                                                  transformSource);
            },
            py::arg("shader"), py::arg("prim"), py::arg("skinned_mesh_data"),
            py::arg("transform_source") = TransformSource::SceneGraph)
        .def(
            "add_renderable",
            [](App* self, PhongMaterial* material, Scene::Prim* prim,
               TransformSource transformSource) {
                return self->addRenderable(material, prim, transformSource);
            },
            py::arg("material"), py::arg("prim"),
            py::arg("transform_source") = TransformSource::SceneGraph)
        .def(
            "update_renderable_transforms",
            [](App* self, uint32_t handle, const FloatArray& transforms,
               py::object colors) {
                auto t = mat4ArrayView(transforms, "transforms");
                const float* colorData = nullptr;
                size_t colorCount = 0;
                if (!colors.is_none()) {
                    auto colorArray = colors.cast<FloatArray>();
                    auto c = vec4ArrayView(colorArray, "colors");
                    if (c.count != 1 && c.count != t.count) {
                        throw py::value_error(
                            "colors must have length 1 or match transforms");
                    }
                    colorData = c.data;
                    colorCount = c.count;
                }
                self->updateRenderableTransforms(handle, t.data, colorData,
                                                 t.count, colorCount);
            },
            py::arg("handle"), py::arg("transforms"),
            py::arg("colors") = py::none())
        .def(
            "set_renderable_colors",
            [](App* self, uint32_t handle, const FloatArray& colors) {
                auto c = vec4ArrayView(colors, "colors");
                self->setRenderableColors(handle, c.data, c.count);
            },
            py::arg("handle"), py::arg("colors"))
        .def(
            "set_renderable_double_sided",
            [](App* self, uint32_t handle, bool doubleSided) {
                self->setRenderableDoubleSided(handle, doubleSided);
            },
            py::arg("handle"), py::arg("double_sided") = true)
        .def(
            "set_renderable_casts_shadow",
            [](App* self, uint32_t handle, bool castsShadow) {
                self->setRenderableCastsShadow(handle, castsShadow);
            },
            py::arg("handle"), py::arg("casts_shadow") = true)
        .def(
            "set_renderable_texture",
            [](App* self, uint32_t handle, Backend::Texture* texture,
               int slot) { self->setRenderableTexture(handle, texture, slot); },
            py::arg("handle"), py::arg("texture"), py::arg("slot") = 0)
        .def(
            "update_renderable_geometry",
            [](App* self, uint32_t handle, const FloatArray& positions,
               py::object normals) {
                auto p = vec3ArrayView(positions, "positions");
                const float* normalData = nullptr;
                size_t normalCount = 0;
                if (!normals.is_none()) {
                    auto normalArray = normals.cast<FloatArray>();
                    auto n = vec3ArrayView(normalArray, "normals");
                    if (n.count != p.count) {
                        throw py::value_error(
                            "normals must match positions length");
                    }
                    normalData = n.data;
                    normalCount = n.count;
                }
                self->updateRenderableGeometry(handle, p.data, normalData,
                                               p.count, normalCount);
            },
            py::arg("handle"), py::arg("positions"),
            py::arg("normals") = py::none())
        .def(
            "update_renderable_skinning_matrices",
            [](App* self, uint32_t handle, const FloatArray& matrices) {
                auto m = mat4ArrayView(matrices, "bone_matrices");
                self->updateRenderableSkinningMatrices(handle, m.data, m.count);
            },
            py::arg("handle"), py::arg("bone_matrices"))
        .def(
            "log_debug_lines",
            [](App* self, const std::string& path, const FloatArray& starts,
               const FloatArray& ends, py::object colors, float width,
               bool hidden) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                std::vector<glm::vec4> c;
                if (!colors.is_none()) {
                    auto colorArray = colors.cast<FloatArray>();
                    c = vec4Array(colorArray, "colors");
                    if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                        throw py::value_error(
                            "colors must be empty, length 1, or match lines");
                    }
                }
                self->logDebugLines(path, s, e, c, width, hidden);
            },
            py::arg("path"), py::arg("starts"), py::arg("ends"),
            py::arg("colors") = py::none(), py::arg("width") = 1.0f,
            py::arg("hidden") = false)
        .def(
            "log_debug_axes",
            [](App* self, const std::string& path, const FloatArray& transform,
               float length, float width, bool hidden) {
                auto t = mat4ArrayView(transform, "transform");
                if (t.count != 1) {
                    throw py::value_error(
                        "transform must be a single matrix with shape [16] "
                        "or [4, 4]");
                }

                const float* p = t.data;
                glm::mat4 m(1.0f);
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col)
                        m[col][row] = p[row * 4 + col];
                }
                self->logDebugAxes(path, m, length, width, hidden);
            },
            py::arg("path"), py::arg("transform"), py::arg("length") = 1.0f,
            py::arg("width") = 1.0f, py::arg("hidden") = false)
        .def("clear_debug_lines", &App::clearDebugLines, py::arg("path"))
        .def(
            "log_debug_points",
            [](App* self, const std::string& path, const FloatArray& points,
               py::object colors, float size, bool hidden) {
                auto p = vec3Array(points, "points");
                std::vector<glm::vec4> c;
                if (!colors.is_none()) {
                    auto colorArray = colors.cast<FloatArray>();
                    c = vec4Array(colorArray, "colors");
                    if (!c.empty() && c.size() != 1 && c.size() != p.size()) {
                        throw py::value_error(
                            "colors must be empty, length 1, or match points");
                    }
                }
                self->logDebugPoints(path, p, c, size, hidden);
            },
            py::arg("path"), py::arg("points"), py::arg("colors") = py::none(),
            py::arg("size") = 6.0f, py::arg("hidden") = false)
        .def("clear_debug_points", &App::clearDebugPoints, py::arg("path"))
        .def("setLightDirection", &App::setLightDirection, py::arg("direction"))
        .def("setLightColor", &App::setLightColor, py::arg("color"))
        .def("setLightIntensity", &App::setLightIntensity, py::arg("intensity"))
        .def("setLightAmbient", &App::setLightAmbient, py::arg("ambient"))
        .def("checkError", &App::checkError)
        .def(
            "is_key_pressed",
            [](App& self, int key) {
                return glfwGetKey(self.getWindow(), key) == GLFW_PRESS;
            },
            py::arg("key"))
        .def("getCamera", &App::getCamera, py::return_value_policy::reference)
        .def("getViewMatrix", &App::getViewMatrix)
        .def("getProjectionMatrix", &App::getProjectionMatrix)
        .def("getWidth", &App::getWidth)
        .def("getHeight", &App::getHeight)
        .def("getScene", &App::getScene, py::return_value_policy::reference);

    py::class_<Bridge::SkinnedCharacterBridge,
               std::unique_ptr<Bridge::SkinnedCharacterBridge>>(
        m, "SkinnedCharacterBridge")
        .def_static(
            "from_fbx",
            [](App* app, Backend::Shader* shader, const std::string& fbxPath,
               const std::optional<std::string>& bindFbxPath,
               const std::string& primBasePath, int clipIndex, float fps,
               float scale, bool useMaterials) {
                const std::string& resolvedBindPath =
                    bindFbxPath.has_value() ? bindFbxPath.value() : fbxPath;
                return std::make_unique<Bridge::SkinnedCharacterBridge>(
                    Bridge::SkinnedCharacterBridge::fromFBXWithBind(
                        app, shader, fbxPath, resolvedBindPath, primBasePath,
                        clipIndex, fps, scale, useMaterials));
            },
            py::arg("app"), py::arg("shader"), py::arg("fbx_path"),
            py::arg("bind_fbx_path") = py::none(),
            py::arg("prim_base_path") = "/fbx_character",
            py::arg("clip_index") = -1, py::arg("fps") = -1.0f,
            py::arg("scale") = 0.01f, py::arg("use_materials") = true)
        .def("apply_time", &Bridge::SkinnedCharacterBridge::applyTime,
             py::arg("time"), py::arg("loop") = true)
        .def(
            "apply_pose",
            [](Bridge::SkinnedCharacterBridge& self,
               py::array_t<float, py::array::c_style | py::array::forcecast>
                   rootTranslation,
               py::array_t<float, py::array::c_style | py::array::forcecast>
                   localRotationsWxyz) {
                py::buffer_info rootInfo = rootTranslation.request();
                if (rootInfo.size != 3) {
                    throw py::value_error(
                        "root_translation expected shape [3]");
                }
                const float* root = static_cast<const float*>(rootInfo.ptr);

                py::buffer_info rotInfo = localRotationsWxyz.request();
                const int joints = self.motion().numJoints();
                if (!((rotInfo.ndim == 2 && rotInfo.shape[0] == joints &&
                       rotInfo.shape[1] == 4) ||
                      (rotInfo.ndim == 1 && rotInfo.shape[0] == joints * 4))) {
                    throw py::value_error(
                        "local_rotations_wxyz expected shape [num_joints, 4] "
                        "or flat [num_joints * 4]");
                }
                const float* q = static_cast<const float*>(rotInfo.ptr);
                std::vector<Eigen::Quaternionf> rotations;
                rotations.reserve(static_cast<size_t>(joints));
                for (int i = 0; i < joints; ++i) {
                    const float* p = q + i * 4;
                    rotations.emplace_back(p[0], p[1], p[2], p[3]);
                }

                return self.applyPose(
                    Eigen::Vector3f(root[0], root[1], root[2]), rotations);
            },
            py::arg("root_translation"), py::arg("local_rotations_wxyz"))
        .def("set_visible", &Bridge::SkinnedCharacterBridge::setVisible,
             py::arg("visible"))
        .def("set_color", &Bridge::SkinnedCharacterBridge::setColor,
             py::arg("color"))
        .def("set_casts_shadow",
             &Bridge::SkinnedCharacterBridge::setCastsShadow,
             py::arg("casts_shadow"))
        .def("motion", &Bridge::SkinnedCharacterBridge::motion,
             py::return_value_policy::reference_internal)
        .def("num_meshes", [](const Bridge::SkinnedCharacterBridge& self) {
            return self.meshes().size();
        });
}
