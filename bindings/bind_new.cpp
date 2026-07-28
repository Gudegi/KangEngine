///
/// KangEngine Python Bindings
///

#include "../src/kangEngine.hpp"
#include "../src/engine/graphics/backend/graphics_factory.hpp"
#include "../src/engine/graphics/material/material.hpp"
#include "../src/engine/graphics/material/colors.hpp"
#include "engine/graphics/material/phongMaterials.hpp"
#include "py_array_view.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
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
void bind_character(py::module& m);
void bind_asset(py::module& m);
void bind_sim(py::module& m);
void bind_physics(py::module& m);
void bind_physics_gpu(py::module& m);

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
    imgui.def("cursor_screen_pos", []() {
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        return py::make_tuple(pos.x, pos.y);
    });
    imgui.def(
        "image",
        [](Backend::Texture* texture, float width, float height,
           float opacity) {
            if (!texture)
                return;
            ImGui::Image(
                static_cast<ImTextureID>(texture->getNativeHandle()),
                ImVec2(width, height), ImVec2(0.0f, 1.0f),
                ImVec2(1.0f, 0.0f),
                ImVec4(1.0f, 1.0f, 1.0f,
                       std::clamp(opacity, 0.0f, 1.0f)),
                ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        },
        py::arg("texture"), py::arg("width"), py::arg("height"),
        py::arg("opacity") = 1.0f);
    imgui.def(
        "draw_circle_filled",
        [](float x, float y, float radius, const glm::vec4& color) {
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(x, y), radius,
                ImGui::ColorConvertFloat4ToU32(
                    ImVec4(color.r, color.g, color.b, color.a)));
        },
        py::arg("x"), py::arg("y"), py::arg("radius"), py::arg("color"));
    imgui.def(
        "draw_rect_filled",
        [](float x1, float y1, float x2, float y2,
           const glm::vec4& color) {
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(x1, y1), ImVec2(x2, y2),
                ImGui::ColorConvertFloat4ToU32(
                    ImVec4(color.r, color.g, color.b, color.a)));
        },
        py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
        py::arg("color"));
    imgui.def(
        "draw_convex_polygon_filled",
        [](const py::sequence& points, const glm::vec4& color) {
            std::vector<ImVec2> vertices;
            vertices.reserve(points.size());
            for (const py::handle item : points) {
                const auto point =
                    py::reinterpret_borrow<py::sequence>(item);
                if (point.size() != 2)
                    throw py::value_error(
                        "Each polygon point must contain x and y.");
                vertices.emplace_back(point[0].cast<float>(),
                                      point[1].cast<float>());
            }
            if (vertices.size() >= 3) {
                ImGui::GetWindowDrawList()->AddConvexPolyFilled(
                    vertices.data(), static_cast<int>(vertices.size()),
                    ImGui::ColorConvertFloat4ToU32(
                        ImVec4(color.r, color.g, color.b, color.a)));
            }
        },
        py::arg("points"), py::arg("color"));
    imgui.def(
        "draw_line",
        [](float x1, float y1, float x2, float y2, const glm::vec4& color,
           float thickness) {
            ImGui::GetWindowDrawList()->AddLine(
                ImVec2(x1, y1), ImVec2(x2, y2),
                ImGui::ColorConvertFloat4ToU32(
                    ImVec4(color.r, color.g, color.b, color.a)),
                thickness);
        },
        py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
        py::arg("color"), py::arg("thickness") = 1.0f);
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

    // External render descriptors below depend on GpuArrayView registration.
    bind_sim(m);

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

    py::enum_<ExternalBufferFormat>(m, "ExternalBufferFormat")
        .value("MAT4", ExternalBufferFormat::Mat4)
        .value("POSITION_ROTATION", ExternalBufferFormat::PositionRotation)
        .value("POSITION_ROTATION_SCALE",
               ExternalBufferFormat::PositionRotationScale)
        .value("CUSTOM", ExternalBufferFormat::Custom);

    py::enum_<ExternalSyncPolicy>(m, "ExternalSyncPolicy")
        .value("NONE", ExternalSyncPolicy::None)
        .value("VERSIONED", ExternalSyncPolicy::Versioned)
        .value("FENCE", ExternalSyncPolicy::Fence)
        .value("EVENT", ExternalSyncPolicy::Event);

    py::class_<ExternalBufferDesc>(m, "ExternalBufferDesc")
        .def(py::init<>())
        .def_readwrite("view", &ExternalBufferDesc::view)
        .def_readwrite("format", &ExternalBufferDesc::format)
        .def_readwrite("count", &ExternalBufferDesc::count)
        .def_readwrite("stride_bytes", &ExternalBufferDesc::strideBytes)
        .def_readwrite("sync_policy", &ExternalBufferDesc::syncPolicy);

    py::enum_<InteractionMode>(m, "InteractionMode")
        .value("Inspect", InteractionMode::Inspect)
        .value("Edit", InteractionMode::Edit)
        .value("Force", InteractionMode::Force)
        .export_values();

    py::enum_<ToneMapMode>(m, "ToneMapMode")
        .value("Off", ToneMapMode::None)
        .value("Reinhard", ToneMapMode::Reinhard)
        .value("Exponential", ToneMapMode::Exponential)
        .value("AcesNarkowicz", ToneMapMode::AcesNarkowicz)
        .value("AcesFitted", ToneMapMode::AcesFitted)
        .export_values();

    py::enum_<TextureRole>(
        m, "TextureRole",
        "Renderer texture binding roles used by material shaders.")
        .value("BaseColor", TextureRole::BaseColor)
        .value("Diffuse", TextureRole::Diffuse)
        .value("Normal", TextureRole::Normal)
        .value("MetallicRoughness", TextureRole::MetallicRoughness)
        .value("AmbientOcclusion", TextureRole::AmbientOcclusion)
        .value("Emissive", TextureRole::Emissive)
        .value("Metallic", TextureRole::Metallic)
        .value("Roughness", TextureRole::Roughness)
        .value("OcclusionRoughnessMetallic",
               TextureRole::OcclusionRoughnessMetallic)
        .export_values();

    py::enum_<AlphaMode>(
        m, "AlphaMode",
        "How a renderable handles fragment alpha in scene and depth passes.")
        .value("Opaque", AlphaMode::Opaque)
        .value("Mask", AlphaMode::Mask)
        .value("Blend", AlphaMode::Blend)
        .export_values();

    py::enum_<TextAlignment>(m, "TextAlignment")
        .value("Left", TextAlignment::Left)
        .value("Center", TextAlignment::Center)
        .value("Right", TextAlignment::Right);

    py::enum_<TextDepthMode>(m, "TextDepthMode")
        .value("DepthTested", TextDepthMode::DepthTested)
        .value("Overlay", TextDepthMode::Overlay);

    py::enum_<ScreenAnchor>(m, "ScreenAnchor")
        .value("TopLeft", ScreenAnchor::TopLeft)
        .value("TopCenter", ScreenAnchor::TopCenter)
        .value("TopRight", ScreenAnchor::TopRight)
        .value("CenterLeft", ScreenAnchor::CenterLeft)
        .value("Center", ScreenAnchor::Center)
        .value("CenterRight", ScreenAnchor::CenterRight)
        .value("BottomLeft", ScreenAnchor::BottomLeft)
        .value("BottomCenter", ScreenAnchor::BottomCenter)
        .value("BottomRight", ScreenAnchor::BottomRight);

    py::enum_<PhongMaterialType>(m, "PhongMaterialType",
                                 "Built-in blin-phong based material presets.")
        .value("EMERALD", PhongMaterialType::EMERALD)
        .value("JADE", PhongMaterialType::JADE)
        .value("OBSIDIAN", PhongMaterialType::OBSIDIAN)
        .value("PEARL", PhongMaterialType::PEARL)
        .value("RUBY", PhongMaterialType::RUBY)
        .value("TURQUOISE", PhongMaterialType::TURQUOISE)
        .value("BRASS", PhongMaterialType::BRASS)
        .value("BRONZE", PhongMaterialType::BRONZE)
        .value("CHROME", PhongMaterialType::CHROME)
        .value("COPPER", PhongMaterialType::COPPER)
        .value("GOLD", PhongMaterialType::GOLD)
        .value("SILVER", PhongMaterialType::SILVER)
        .value("BLACK_PLASTIC", PhongMaterialType::BLACK_PLASTIC)
        .value("CYAN_PLASTIC", PhongMaterialType::CYAN_PLASTIC)
        .value("GREEN_PLASTIC", PhongMaterialType::GREEN_PLASTIC)
        .value("RED_PLASTIC", PhongMaterialType::RED_PLASTIC)
        .value("WHITE_PLASTIC", PhongMaterialType::WHITE_PLASTIC)
        .value("YELLOW_PLASTIC", PhongMaterialType::YELLOW_PLASTIC)
        .value("BLACK_RUBBER", PhongMaterialType::BLACK_RUBBER)
        .value("CYAN_RUBBER", PhongMaterialType::CYAN_RUBBER)
        .value("GREEN_RUBBER", PhongMaterialType::GREEN_RUBBER)
        .value("RED_RUBBER", PhongMaterialType::RED_RUBBER)
        .value("WHITE_RUBBER", PhongMaterialType::WHITE_RUBBER)
        .value("YELLOW_RUBBER", PhongMaterialType::YELLOW_RUBBER)
        .export_values();

    py::enum_<PBRMaterialType>(m, "PBRMaterialType",
                               "Built-in physically based material presets.")
        .value("GRAY_CARD", PBRMaterialType::GRAY_CARD)
        .value("WHITE_PLASTIC", PBRMaterialType::WHITE_PLASTIC)
        .value("BLACK_PLASTIC", PBRMaterialType::BLACK_PLASTIC)
        .value("BLACK_RUBBER", PBRMaterialType::BLACK_RUBBER)
        .value("CHARCOAL", PBRMaterialType::CHARCOAL)
        .value("CARROT", PBRMaterialType::CARROT)
        .value("CONCRETE", PBRMaterialType::CONCRETE)
        .value("RED_BRICK", PBRMaterialType::RED_BRICK)
        .value("ALUMINUM", PBRMaterialType::ALUMINUM)
        .value("CHROME", PBRMaterialType::CHROME)
        .value("COPPER", PBRMaterialType::COPPER)
        .value("GOLD", PBRMaterialType::GOLD)
        .value("EMISSIVE_BLUE", PBRMaterialType::EMISSIVE_BLUE)
        .export_values();

    py::class_<RayPickResult>(m, "RayPickResult")
        .def_readonly("hit", &RayPickResult::hit)
        .def_readonly("handle", &RayPickResult::handle)
        .def_readonly("instance_index", &RayPickResult::instanceIndex)
        .def_readonly("transform_source", &RayPickResult::transformSource)
        .def_readonly("prim", &RayPickResult::prim,
                      py::return_value_policy::reference)
        .def_readonly("distance", &RayPickResult::distance)
        .def_readonly("position", &RayPickResult::position);

    bind_scene(m);
    bind_animation(m);
    bind_character(m);
    bind_asset(m);
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
        .value("PASTEL_SKY", ColorType::PASTEL_SKY)
        .value("DARK_GREEN", ColorType::DARK_GREEN);

    py::class_<ColorLibrary>(m, "ColorLibrary")
        .def_static("get", &ColorLibrary::get, py::arg("type"));

    py::class_<Material>(m, "Material", "Base class for renderer materials.")
        .def(
            "set_shader",
            [](Material& self, Backend::Shader* shader) -> Material& {
                self.setShader(shader);
                return self;
            },
            py::arg("shader"), py::return_value_policy::reference_internal,
            "Attach the shader used when this material is bound and return "
            "this material.")
        .def("get_shader", &Material::getShader,
             py::return_value_policy::reference,
             "Return the shader currently attached to this material.");

    py::class_<VertexColorMaterial, Material>(
        m, "VertexColorMaterial",
        "Compatibility material for vertex/display-color shader rendering.")
        .def(py::init<>(), "Create a vertex-color material without a shader.")
        .def(py::init<Backend::Shader*>(), py::arg("shader"),
             "Create a vertex-color material attached to a shader.");

    py::class_<PhongMaterial, Material>(
        m, "PhongMaterial",
        "Classic Blinn-Phong material with diffuse/specular factors and "
        "optional textures.")
        .def(py::init<>(), "Create a Phong material with default factors.")
        .def(py::init<Backend::Shader*>(), py::arg("shader"),
             "Create a Phong material attached to a shader.")
        .def(
            "load_from_preset",
            [](PhongMaterial& self, PhongMaterialType type) -> PhongMaterial& {
                self.loadFromPreset(type);
                return self;
            },
            py::arg("type"), py::return_value_policy::reference_internal,
            "Load material factors from a built-in Phong preset and return "
            "this material.")
        .def("set_ambient", &PhongMaterial::setAmbient, py::arg("ambient"),
             py::return_value_policy::reference_internal)
        .def("set_diffuse", &PhongMaterial::setDiffuse, py::arg("diffuse"),
             py::return_value_policy::reference_internal)
        .def("set_specular", &PhongMaterial::setSpecular, py::arg("specular"),
             py::return_value_policy::reference_internal)
        .def("set_shininess", &PhongMaterial::setShininess,
             py::arg("shininess"), py::return_value_policy::reference_internal)
        .def("set_diffuse_map", &PhongMaterial::setDiffuseMap,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_specular_map", &PhongMaterial::setSpecularMap,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_alpha_map", &PhongMaterial::setAlphaMap, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_normal_map", &PhongMaterial::setNormalMap, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def_readwrite("ambient", &PhongMaterial::ambient,
                       "Ambient RGB factor.")
        .def_readwrite("diffuse", &PhongMaterial::diffuse,
                       "Diffuse RGB factor.")
        .def_readwrite("specular", &PhongMaterial::specular,
                       "Specular RGB factor.")
        .def_readwrite("shininess", &PhongMaterial::shininess,
                       "Specular highlight exponent.")
        .def_readwrite("diffuse_map", &PhongMaterial::diffuseMap,
                       "Optional diffuse texture.")
        .def_readwrite("specular_map", &PhongMaterial::specularMap,
                       "Optional specular texture.")
        .def_readwrite("alpha_map", &PhongMaterial::alphaMap,
                       "Optional alpha mask texture, such as OBJ map_d.")
        .def_readwrite("normal_map", &PhongMaterial::normalMap,
                       "Optional tangent-space normal map.");

    py::class_<PBRMaterial, Material>(
        m, "PBRMaterial",
        "Physically based material using metallic-roughness parameters.")
        .def(py::init<>(), "Create a PBR material with default factors.")
        .def(py::init<Backend::Shader*>(), py::arg("shader"),
             "Create a PBR material attached to a shader.")
        .def(
            "load_from_preset",
            [](PBRMaterial& self, PBRMaterialType type) -> PBRMaterial& {
                self.loadFromPreset(type);
                return self;
            },
            py::arg("type"), py::return_value_policy::reference_internal,
            "Load material factors from a built-in PBR preset and return "
            "this material.")
        .def("set_base_color", &PBRMaterial::setBaseColor,
             py::arg("base_color"), py::return_value_policy::reference_internal)
        .def("set_metallic", &PBRMaterial::setMetallic, py::arg("metallic"),
             py::return_value_policy::reference_internal)
        .def("set_roughness", &PBRMaterial::setRoughness, py::arg("roughness"),
             py::return_value_policy::reference_internal)
        .def("set_emissive_color", &PBRMaterial::setEmissiveColor,
             py::arg("emissive_color"),
             py::return_value_policy::reference_internal)
        .def("set_emissive_strength", &PBRMaterial::setEmissiveStrength,
             py::arg("emissive_strength"),
             py::return_value_policy::reference_internal)
        .def("set_base_color_texture", &PBRMaterial::setBaseColorTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_normal_texture", &PBRMaterial::setNormalTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_metallic_roughness_texture",
             &PBRMaterial::setMetallicRoughnessTexture, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_metallic_texture", &PBRMaterial::setMetallicTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_roughness_texture", &PBRMaterial::setRoughnessTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_ao_texture", &PBRMaterial::setAoTexture, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_orm_texture", &PBRMaterial::setOrmTexture, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_emissive_texture", &PBRMaterial::setEmissiveTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def_readwrite("base_color", &PBRMaterial::baseColor,
                       "Base color factor as RGBA.")
        .def_readwrite("metallic", &PBRMaterial::metallic,
                       "Metallic factor in the range 0..1.")
        .def_readwrite("roughness", &PBRMaterial::roughness,
                       "Roughness factor in the range 0..1.")
        .def_readwrite("emissive_color", &PBRMaterial::emissiveColor,
                       "Emissive RGB color factor.")
        .def_readwrite("emissive_strength", &PBRMaterial::emissiveStrength,
                       "Multiplier for emissive_color.")
        .def_readwrite("base_color_texture", &PBRMaterial::baseColorTexture,
                       "Optional base color texture.")
        .def_readwrite("normal_texture", &PBRMaterial::normalTexture,
                       "Optional tangent-space normal texture.")
        .def_readwrite("metallic_roughness_texture",
                       &PBRMaterial::metallicRoughnessTexture,
                       "Optional packed metallic-roughness texture.")
        .def_readwrite("metallic_texture", &PBRMaterial::metallicTexture,
                       "Optional single-channel metallic texture.")
        .def_readwrite("roughness_texture", &PBRMaterial::roughnessTexture,
                       "Optional single-channel roughness texture.")
        .def_readwrite("ao_texture", &PBRMaterial::aoTexture,
                       "Optional ambient occlusion texture.")
        .def_readwrite("orm_texture", &PBRMaterial::ormTexture,
                       "Optional packed occlusion-roughness-metallic texture.")
        .def_readwrite("emissive_texture", &PBRMaterial::emissiveTexture,
                       "Optional emissive texture.");

    // Backend::Shader
    py::class_<Backend::Shader>(
        m, "Shader", "Compiled GPU shader program used by renderer materials.")
        .def("use", &Backend::Shader::use,
             "Bind this shader program for subsequent draw or uniform calls.")
        .def("bind", &Backend::Shader::bind, "Bind this shader program.")
        .def("unbind", &Backend::Shader::unbind,
             "Unbind the current shader program.")
        .def("set_bool", &Backend::Shader::setBool, py::arg("name"),
             py::arg("value"), "Set a boolean uniform.")
        .def("set_int", &Backend::Shader::setInt, py::arg("name"),
             py::arg("value"), "Set an integer uniform.")
        .def("set_float", &Backend::Shader::setFloat, py::arg("name"),
             py::arg("value"), "Set a float uniform.")
        .def("set_color", &Backend::Shader::setColor, py::arg("name"),
             py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"),
             "Set a color uniform from RGBA components.")
        .def(
            "set_color",
            [](Backend::Shader& self, const std::string& name,
               const glm::vec4& value) {
                self.setColor(name, value.r, value.g, value.b, value.a);
            },
            py::arg("name"), py::arg("value"),
            "Set a color uniform from an RGBA vector.")
        .def("set_vec2",
             py::overload_cast<const std::string&, const glm::vec2&>(
                 &Backend::Shader::setVec2),
             py::arg("name"), py::arg("value"), "Set a vec2 uniform.")
        .def("set_vec2",
             py::overload_cast<const std::string&, float, float>(
                 &Backend::Shader::setVec2),
             py::arg("name"), py::arg("x"), py::arg("y"),
             "Set a vec2 uniform from components.")
        .def("set_vec3",
             py::overload_cast<const std::string&, const glm::vec3&>(
                 &Backend::Shader::setVec3),
             py::arg("name"), py::arg("value"), "Set a vec3 uniform.")
        .def("set_vec3",
             py::overload_cast<const std::string&, float, float, float>(
                 &Backend::Shader::setVec3),
             py::arg("name"), py::arg("x"), py::arg("y"), py::arg("z"),
             "Set a vec3 uniform from components.")
        .def("set_vec4",
             py::overload_cast<const std::string&, const glm::vec4&>(
                 &Backend::Shader::setVec4),
             py::arg("name"), py::arg("value"), "Set a vec4 uniform.")
        .def("set_vec4",
             py::overload_cast<const std::string&, float, float, float, float>(
                 &Backend::Shader::setVec4),
             py::arg("name"), py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("w"), "Set a vec4 uniform from components.")
        .def("set_mat2", &Backend::Shader::setMat2, py::arg("name"),
             py::arg("value"), "Set a mat2 uniform.")
        .def("set_mat3", &Backend::Shader::setMat3, py::arg("name"),
             py::arg("value"), "Set a mat3 uniform.")
        .def("set_mat4", &Backend::Shader::setMat4, py::arg("name"),
             py::arg("value"), "Set a mat4 uniform.")
        .def("set_uniform_block_binding",
             &Backend::Shader::setUniformBlockBinding, py::arg("block_name"),
             py::arg("binding_point"),
             "Bind a named uniform block to a binding point.");

    // Backend::Texture
    py::class_<Backend::Texture>(
        m, "Texture", "GPU texture object created by a GraphicsDevice.")
        .def("bind", &Backend::Texture::bind, py::arg("slot") = 0,
             "Bind this texture to a texture unit.")
        .def("unbind", &Backend::Texture::unbind,
             "Unbind this texture from the active context.")
        .def("get_width", &Backend::Texture::getWidth,
             "Return the texture width in pixels.")
        .def("get_height", &Backend::Texture::getHeight,
             "Return the texture height in pixels.")
        .def_property_readonly("width", &Backend::Texture::getWidth,
                               "Texture width in pixels.")
        .def_property_readonly("height", &Backend::Texture::getHeight,
                               "Texture height in pixels.");

    py::enum_<Backend::TextureWrap>(
        m, "TextureWrap", "Backend-neutral texture coordinate wrapping mode.")
        .value("Repeat", Backend::TextureWrap::Repeat)
        .value("ClampToEdge", Backend::TextureWrap::ClampToEdge)
        .value("MirroredRepeat", Backend::TextureWrap::MirroredRepeat);

    py::enum_<Backend::TextureFilter>(
        m, "TextureFilter", "Backend-neutral texture sampling filter.")
        .value("Nearest", Backend::TextureFilter::Nearest)
        .value("Linear", Backend::TextureFilter::Linear)
        .value("LinearMipmapLinear",
               Backend::TextureFilter::LinearMipmapLinear);

    py::class_<Backend::SamplerDesc>(
        m, "SamplerDesc", "Texture sampler settings independent of GL/WebGPU.")
        .def(py::init<>())
        .def_readwrite("wrap_u", &Backend::SamplerDesc::wrapU)
        .def_readwrite("wrap_v", &Backend::SamplerDesc::wrapV)
        .def_readwrite("min_filter", &Backend::SamplerDesc::minFilter)
        .def_readwrite("mag_filter", &Backend::SamplerDesc::magFilter);

    // Backend::GraphicsDevice
    py::class_<Backend::GraphicsDevice,
               std::shared_ptr<Backend::GraphicsDevice>>(
        m, "GraphicsDevice",
        "Factory for backend graphics resources such as shaders and textures.")
        .def("create_shader",
             py::overload_cast<const std::string&, const std::string&>(
                 &Backend::GraphicsDevice::createShader),
             py::arg("vertex_source"), py::arg("fragment_source"),
             py::return_value_policy::take_ownership,
             "Create a shader from vertex and fragment shader source strings.")
        .def("create_shader_from_file",
             &Backend::GraphicsDevice::createShaderFromFile,
             py::arg("vert_path"), py::arg("frag_path"),
             py::return_value_policy::take_ownership,
             "Create a shader from vertex and fragment shader files.")
        .def(
            "create_texture",
            [](Backend::GraphicsDevice& device, const std::string& path,
               bool flip, const Backend::SamplerDesc& sampler) {
                return device.createTexture(path, flip, sampler);
            },
            py::arg("path"), py::arg("flip") = false,
            py::arg("sampler") = Backend::SamplerDesc(),
            py::return_value_policy::take_ownership,
            "Load a texture from an image file.");

    py::class_<DirectionalLight>(
        m, "DirectionalLight",
        "Infinite-distance light used as the renderer's main sun light.")
        .def(py::init<>())
        .def_readwrite("direction", &DirectionalLight::direction)
        .def_readwrite("color", &DirectionalLight::color)
        .def_readwrite("intensity", &DirectionalLight::intensity)
        .def_readwrite("ambient", &DirectionalLight::ambient);

    py::class_<PointLight>(
        m, "PointLight",
        "Finite local light with position, color, intensity, and range.")
        .def(py::init<>())
        .def_readwrite("position", &PointLight::position)
        .def_readwrite("color", &PointLight::color)
        .def_readwrite("intensity", &PointLight::intensity)
        .def_readwrite("range", &PointLight::range);

    py::class_<SpotLight>(
        m, "SpotLight",
        "Finite cone light with position, direction, color, intensity, and "
        "range.")
        .def(py::init<>())
        .def_readwrite("position", &SpotLight::position)
        .def_readwrite("direction", &SpotLight::direction)
        .def_readwrite("color", &SpotLight::color)
        .def_readwrite("intensity", &SpotLight::intensity)
        .def_readwrite("range", &SpotLight::range)
        .def_readwrite("inner_cone_angle", &SpotLight::innerConeAngle)
        .def_readwrite("outer_cone_angle", &SpotLight::outerConeAngle);

    py::class_<Renderer>(
        m, "Renderer",
        "Renderer facade for updating renderable resources and draw settings.")
        .def(
            "device", [](Renderer& self) { return self.device(); },
            py::return_value_policy::reference,
            "Return the graphics device owned by this renderer.")
        .def("set_point_lights", &Renderer::setPointLights, py::arg("lights"),
             "Store point lights for renderers/shaders that support them.")
        .def("point_lights", &Renderer::pointLights,
             py::return_value_policy::reference_internal,
             "Return stored point lights.")
        .def("set_spot_lights", &Renderer::setSpotLights, py::arg("lights"),
             "Store spot lights for renderers/shaders that support them.")
        .def("spot_lights", &Renderer::spotLights,
             py::return_value_policy::reference_internal,
             "Return stored spot lights.")
        .def("sync_scene_lights", &Renderer::syncSceneLights, py::arg("scene"),
             "Sync renderer lights from /lights scene prims. Intended for "
             "diagnostics and tests.")
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
            py::arg("colors") = py::none(),
            "Replace instance transforms, optionally with per-instance colors.")
        .def(
            "set_renderable_colors",
            [](Renderer& self, uint32_t handle, const FloatArray& colors) {
                self.setRenderableColors(handle, vec4Array(colors, "colors"));
            },
            py::arg("handle"), py::arg("colors"),
            "Set per-instance colors for a renderable.")
        .def("set_renderable_external_buffer",
             &Renderer::setRenderableExternalBuffer, py::arg("handle"),
             py::arg("descriptor"),
             "Attach an external CPU/GPU transform buffer to a renderable.")
        .def("map_renderable_cuda_transform_buffers",
             &Renderer::mapRenderableCudaTransformBuffers, py::arg("handles"),
             py::arg("count"), py::arg("device_id"),
             py::arg("stream_handle") = 0,
             "Map multiple renderable transform VBOs for direct CUDA writes.")
        .def("unmap_renderable_cuda_transform_buffers",
             &Renderer::unmapRenderableCudaTransformBuffers, py::arg("handles"),
             py::arg("device_id"), py::arg("stream_handle") = 0,
             "Unmap transform VBOs after direct CUDA writes.")
        .def("set_renderable_double_sided", &Renderer::setRenderableDoubleSided,
             py::arg("handle"), py::arg("double_sided") = true,
             "Enable or disable double-sided rendering for a renderable.")
        .def("set_renderable_casts_shadow", &Renderer::setRenderableCastsShadow,
             py::arg("handle"), py::arg("casts_shadow") = true,
             "Enable or disable shadow casting for a renderable.")
        .def("set_renderable_alpha_mode", &Renderer::setRenderableAlphaMode,
             py::arg("handle"), py::arg("mode"), py::arg("cutoff") = 0.5f,
             "Select opaque, alpha-mask, or alpha-blend rendering.")
        .def(
            "set_renderable_texture",
            [](Renderer& self, uint32_t handle, Backend::Texture* texture,
               TextureRole role) {
                self.setRenderableTexture(handle, texture, role);
            },
            py::arg("handle"), py::arg("texture"), py::arg("role"),
            "Attach a texture to a renderable using a material texture role.")
        .def(
            "set_renderable_texture",
            [](Renderer& self, uint32_t handle, Backend::Texture* texture,
               int slot) { self.setRenderableTexture(handle, texture, slot); },
            py::arg("handle"), py::arg("texture"), py::arg("slot") = 0,
            "Attach a texture to a renderable using a raw texture slot.")
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
            py::arg("normals") = py::none(),
            "Update dynamic vertex positions and optional normals.")
        .def(
            "update_renderable_skinning_matrices",
            [](Renderer& self, uint32_t handle, const FloatArray& matrices) {
                self.updateRenderableSkinningMatrices(
                    handle, mat4RowMajorArray(matrices, "bone_matrices"));
            },
            py::arg("handle"), py::arg("bone_matrices"),
            "Update bone matrices for a skinned renderable.");

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

            if (auto values = fixedFloatArray<2>(obj, "vec2"))
                return glm::vec2((*values)[0], (*values)[1]);

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
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * float())
        .def(float() * py::self)
        .def(py::self / float())
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
            if (auto values = fixedFloatArray<3>(obj, "vec3"))
                return glm::vec3((*values)[0], (*values)[1], (*values)[2]);

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
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * float())
        .def(float() * py::self)
        .def(py::self / float())
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
            if (auto values = fixedFloatArray<4>(obj, "vec4"))
                return glm::vec4((*values)[0], (*values)[1], (*values)[2],
                                 (*values)[3]);
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
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self * float())
        .def(float() * py::self)
        .def(py::self / float())
        .def("__repr__", [](const glm::vec4& v) {
            return "vec4(" + std::to_string(v.x) + ", " + std::to_string(v.y) +
                   ", " + std::to_string(v.z) + ", " + std::to_string(v.w) +
                   ")";
        });

    py::implicitly_convertible<py::object, glm::vec4>();

    py::class_<glm::quat>(m, "quat")
        .def(py::init<float, float, float, float>(), py::arg("w"), py::arg("x"),
             py::arg("y"), py::arg("z"))
        .def(py::init([](py::object obj) {
            // Python quaternion values consistently use wxyz order,
            // independent of GLM's internal memory layout.
            if (auto values = fixedFloatArray<4>(obj, "quat"))
                return glm::quat((*values)[0], (*values)[1], (*values)[2],
                                 (*values)[3]);
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
                return glm::quat(seq[0].cast<float>(), seq[1].cast<float>(),
                                 seq[2].cast<float>(), seq[3].cast<float>());
            }
            throw std::runtime_error("Cannot convert to quat");
        }))
        .def_static(
            "from_wxyz",
            [](py::object obj) {
                auto values = fixedFloatArray<4>(obj, "wxyz quaternion");
                if (!values && py::len_hint(obj) == 4) {
                    values =
                        std::array<float, 4>{obj[py::int_(0)].cast<float>(),
                                             obj[py::int_(1)].cast<float>(),
                                             obj[py::int_(2)].cast<float>(),
                                             obj[py::int_(3)].cast<float>()};
                }
                if (!values)
                    throw py::value_error(
                        "wxyz quaternion expected exactly 4 values");
                return glm::quat((*values)[0], (*values)[1], (*values)[2],
                                 (*values)[3]);
            },
            py::arg("values"), "Create a quaternion from wxyz values.")
        .def_static(
            "from_xyzw",
            [](py::object obj) {
                auto values = fixedFloatArray<4>(obj, "xyzw quaternion");
                if (!values && py::len_hint(obj) == 4) {
                    values =
                        std::array<float, 4>{obj[py::int_(0)].cast<float>(),
                                             obj[py::int_(1)].cast<float>(),
                                             obj[py::int_(2)].cast<float>(),
                                             obj[py::int_(3)].cast<float>()};
                }
                if (!values)
                    throw py::value_error(
                        "xyzw quaternion expected exactly 4 values");
                return glm::quat((*values)[3], (*values)[0], (*values)[1],
                                 (*values)[2]);
            },
            py::arg("values"), "Create a quaternion from xyzw values.")
        .def(
            "to_wxyz",
            [](const glm::quat& q) {
                py::array_t<float> result(4);
                auto values = result.mutable_unchecked<1>();
                values(0) = q.w;
                values(1) = q.x;
                values(2) = q.y;
                values(3) = q.z;
                return result;
            },
            "Return a copied wxyz NumPy array.")
        .def(
            "to_xyzw",
            [](const glm::quat& q) {
                py::array_t<float> result(4);
                auto values = result.mutable_unchecked<1>();
                values(0) = q.x;
                values(1) = q.y;
                values(2) = q.z;
                values(3) = q.w;
                return result;
            },
            "Return a copied xyzw NumPy array.")
        .def(
            "__array__",
            [](const glm::quat& q, py::object dtype, py::object) -> py::object {
                py::array_t<float> result(4);
                auto values = result.mutable_unchecked<1>();
                values(0) = q.w;
                values(1) = q.x;
                values(2) = q.y;
                values(3) = q.z;
                if (!dtype.is_none())
                    return result.attr("astype")(dtype,
                                                 py::arg("copy") = false);
                return result;
            },
            py::arg("dtype") = py::none(), py::arg("copy") = py::none(),
            "Return a copied wxyz NumPy array.")
        .def_readwrite("w", &glm::quat::w)
        .def_readwrite("x", &glm::quat::x)
        .def_readwrite("y", &glm::quat::y)
        .def_readwrite("z", &glm::quat::z)
        .def(
            "__mul__",
            [](const glm::quat& rotation, const glm::vec3& vector) {
                return rotation * vector;
            },
            py::arg("vector"), "Rotate a vec3 by this quaternion.")
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
            if (auto values = fixedFloatArray<9>(obj, "mat3")) {
                std::copy(values->begin(), values->end(), &m[0][0]);
                return m;
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
            if (auto values = fixedFloatArray<16>(obj, "mat4")) {
                std::copy(values->begin(), values->end(), &m[0][0]);
                return m;
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
    bind_physics_gpu(m);

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

    py::class_<Camera>(m, "Camera",
                       "View camera used by KangEngine applications.")
        .def("get_camera_pos", &Camera::getCameraPos,
             "Return the camera position.")
        .def("get_target_pos", &Camera::getTargetPos,
             "Return the current camera target point.")
        .def("get_camera_look_dir", &Camera::getCameraLookDir,
             "Return the normalized forward/look direction.")
        .def("get_camera_up_dir", &Camera::getCameraUpDir,
             "Return the normalized up direction.")
        .def("get_camera_right_dir", &Camera::getCameraRightDir,
             "Return the normalized right direction.")
        .def("get_fov", &Camera::getFoV,
             "Return the vertical field of view in degrees.")
        .def("get_near_plane", &Camera::getNearPlane,
             "Return the near clipping distance.")
        .def("get_far_plane", &Camera::getFarPlane,
             "Return the far clipping distance.")
        .def("set_camera_pos", &Camera::setCameraPos, py::arg("camera_pos"),
             "Set the camera position.")
        .def("set_target_pos", &Camera::setTargetPos, py::arg("target_pos"),
             "Set the camera target point.")
        .def("set_fov", &Camera::setFoV, py::arg("fov"),
             "Set the vertical field of view in degrees.")
        .def("set_near_plane", &Camera::setNearPlane, py::arg("distance"),
             "Set the near clipping distance.")
        .def("set_far_plane", &Camera::setFarPlane, py::arg("distance"),
             "Set the far clipping distance.");

    // App class with trampoline for Python overrides
    py::class_<App, PyApp>(
        m, "App",
        "Native application shell with a window, scene, renderer, and input.")
        .def(py::init<>(), "Create an uninitialized application.")
        .def("initialize", &App::initialize, py::arg("width"),
             py::arg("height"), py::arg("hideUi") = false,
             py::arg("upAxis") = UpAxis::Y,
             py::arg("graphicsBackendType") = Backend::BackendType::OpenGL,
             py::arg("sceneBackendType") = Scene::BackendType::Native,
             py::arg("headless") = false,
             "Initialize the window, renderer, input, and scene backend.")
        .def("set_render_hz", &App::setRenderHz, py::arg("render_hz"),
             "Set the target render/update frequency in Hz.")
        .def("get_delta_time", &App::getDeltaTime,
             "Return elapsed seconds between recent frames.")
        .def("get_render_hz", &App::getRenderHz,
             "Return the target render/update frequency in Hz.")
        .def(
            "get_ui_scale",
            [](const App& self) { return self.getUiScale().value(); },
            "Return the current UI scale factor.")
        .def("set_camera_move_speed", &App::setCameraMoveSpeed,
             py::arg("speed"), "Set interactive camera movement speed.")
        .def("get_camera_move_speed", &App::getCameraMoveSpeed,
             "Return interactive camera movement speed.")
        .def("start", &App::start, "Enter the application render loop.")
        .def("render_frame_once", &App::renderFrameOnce,
             "Render and process one frame without entering the main loop.")
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
            py::arg("flip_y") = true,
            "Read the current framebuffer as a uint8 RGB numpy array.")
        .def("write_pixels_png", &App::writePixelsPNG, py::arg("path"),
             py::arg("flip_y") = true,
             "Write the current framebuffer to a PNG file.")
        .def("should_close", &App::shouldClose,
             "Return true when the application window should close.")
        .def("setup", &App::setup,
             "User override called once after initialization.")
        .def("preRender", &App::preRender,
             "User override called before each frame is rendered.")
        .def("render", &App::render,
             "User override called during each frame render.")
        .def("postRender", &App::postRender,
             "User override called after each frame is rendered.")
        .def("onRayPicked", &App::onRayPicked, py::arg("result"),
             "User override called when ray picking selects an object.")
        .def("onRayPickHover", &App::onRayPickHover, py::arg("result"),
             "User override called when ray picking hovers an object.")
        .def("onForceDragBegin", &App::onForceDragBegin, py::arg("result"),
             py::arg("target"),
             "User override called when force dragging begins.")
        .def("onForceDragUpdate", &App::onForceDragUpdate, py::arg("result"),
             py::arg("target"),
             "User override called while force dragging updates.")
        .def("onForceDragEnd", &App::onForceDragEnd,
             "User override called when force dragging ends.")
        .def("get_interaction_mode", &App::getInteractionMode,
             "Return the active viewport interaction mode.")
        .def("set_interaction_mode", &App::setInteractionMode, py::arg("mode"),
             "Set the active viewport interaction mode.")
        .def("has_selection", &App::hasSelection,
             "Return true when a scene object is selected.")
        .def("clear_selection", &App::clearSelection,
             "Clear the current scene selection.")
        .def(
            "ray_pick",
            [](const App& self, const glm::vec3& origin,
               const glm::vec3& direction) {
                return self.rayPick(Geometry::Ray(origin, direction));
            },
            py::arg("origin"), py::arg("direction"),
            "Pick the closest renderable intersected by a world-space ray.")
        .def("get_graphics_device", &App::getGraphicsDevice,
             py::return_value_policy::reference,
             "Return the application's graphics device.")
        .def(
            "get_renderer", [](App& self) { return &self.getRenderer(); },
            py::return_value_policy::reference,
            "Return the application's renderer.")
        .def(
            "get_scene_render_system",
            [](App& self) { return &self.getSceneRenderSystem(); },
            py::return_value_policy::reference_internal,
            "Return the scene-to-renderer component registry.")
        .def(
            "get_scene_resource_manager",
            [](App& self) { return &self.getSceneResourceManager(); },
            py::return_value_policy::reference_internal,
            "Return the scene-scoped resource manager used by Resource prim "
            "mirrors.")
        .def(
            "add_renderable",
            [](App* self, Backend::Shader* shader, Scene::Prim* prim,
               TransformSource transformSource) {
                return self->addRenderable(shader, prim, transformSource);
            },
            py::arg("shader"), py::arg("prim"),
            py::arg("transform_source") = TransformSource::SceneGraph,
            "Compatibility/low-level path: create a renderable for a scene "
            "prim using a shader and return its internal renderer handle. "
            "Prefer app.scene.add_renderable(...) for authored scene objects.")
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
            py::arg("transform_source") = TransformSource::SceneGraph,
            "Compatibility/low-level path: create a skinned renderable and "
            "return its internal renderer handle. Prefer higher-level "
            "character bridges for authored scene use.")
        .def(
            "add_skinned_renderable",
            [](App* self, Material* material, Scene::Prim* prim,
               std::shared_ptr<Scene::SkinnedMeshData> skinnedMesh,
               TransformSource transformSource) {
                if (!skinnedMesh)
                    throw py::value_error("skinned_mesh_data is None");
                return self->addSkinnedRenderable(material, prim, *skinnedMesh,
                                                  transformSource);
            },
            py::arg("material"), py::arg("prim"), py::arg("skinned_mesh_data"),
            py::arg("transform_source") = TransformSource::SceneGraph,
            "Compatibility/low-level path: create a skinned renderable using "
            "a material and return its internal renderer handle.")
        .def(
            "add_renderable",
            [](App* self, Material* material, Scene::Prim* prim,
               TransformSource transformSource) {
                return self->addRenderable(material, prim, transformSource);
            },
            py::arg("material"), py::arg("prim"),
            py::arg("transform_source") = TransformSource::SceneGraph,
            "Compatibility/low-level path: create a renderable for a scene "
            "prim using a material and return its internal renderer handle. "
            "Prefer app.scene.add_renderable(...) for authored scene objects.")
        .def(
            "remove_prim",
            [](App* self, const std::string& path) {
                return self->removePrim(path);
            },
            py::arg("path"),
            "Remove a scene prim subtree and detach its renderable instances.")
        .def(
            "remove_prim",
            [](App* self, Scene::Prim* prim) { return self->removePrim(prim); },
            py::arg("prim"),
            "Remove a scene prim subtree and detach its renderable instances.")
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
            py::arg("colors") = py::none(),
            "Low-level handle path: replace instance transforms, optionally "
            "with per-instance colors. Prefer RenderablePrimView or "
            "SimVisualBatch helpers when available.")
        .def(
            "set_renderable_colors",
            [](App* self, uint32_t handle, const FloatArray& colors) {
                auto c = vec4ArrayView(colors, "colors");
                self->setRenderableColors(handle, c.data, c.count);
            },
            py::arg("handle"), py::arg("colors"),
            "Low-level handle path: set per-instance colors for a renderable.")
        .def(
            "set_renderable_double_sided",
            [](App* self, uint32_t handle, bool doubleSided) {
                self->setRenderableDoubleSided(handle, doubleSided);
            },
            py::arg("handle"), py::arg("double_sided") = true,
            "Low-level handle path: enable or disable double-sided rendering. "
            "Prefer RenderablePrimView.set_double_sided() for scene objects.")
        .def(
            "set_renderable_casts_shadow",
            [](App* self, uint32_t handle, bool castsShadow) {
                self->setRenderableCastsShadow(handle, castsShadow);
            },
            py::arg("handle"), py::arg("casts_shadow") = true,
            "Low-level handle path: enable or disable shadow casting. Prefer "
            "RenderablePrimView.set_casts_shadow() for scene objects.")
        .def(
            "set_renderable_alpha_mode",
            [](App* self, uint32_t handle, AlphaMode mode, float cutoff) {
                self->setRenderableAlphaMode(handle, mode, cutoff);
            },
            py::arg("handle"), py::arg("mode"), py::arg("cutoff") = 0.5f,
            "Low-level handle path: select opaque, alpha-mask, or alpha-blend "
            "rendering. Prefer RenderablePrimView.set_alpha_mode().")
        .def(
            "set_renderable_texture",
            [](App* self, uint32_t handle, Backend::Texture* texture,
               TextureRole role) {
                self->setRenderableTexture(handle, texture, role);
            },
            py::arg("handle"), py::arg("texture"), py::arg("role"),
            "Low-level handle path: attach a texture using a material texture "
            "role. Prefer RenderablePrimView.set_texture().")
        .def(
            "set_renderable_texture",
            [](App* self, uint32_t handle, Backend::Texture* texture,
               int slot) { self->setRenderableTexture(handle, texture, slot); },
            py::arg("handle"), py::arg("texture"), py::arg("slot") = 0,
            "Low-level handle path: attach a texture using a raw texture slot.")
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
            py::arg("normals") = py::none(),
            "Low-level handle path: update dynamic vertex positions and "
            "optional normals.")
        .def(
            "update_renderable_skinning_matrices",
            [](App* self, uint32_t handle, const FloatArray& matrices) {
                auto m = mat4ArrayView(matrices, "bone_matrices");
                self->updateRenderableSkinningMatrices(handle, m.data, m.count);
            },
            py::arg("handle"), py::arg("bone_matrices"),
            "Low-level handle path: update bone matrices for a skinned "
            "renderable.")
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
            py::arg("hidden") = false,
            "Draw persistent debug line segments at a debug path.")
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
            py::arg("width") = 1.0f, py::arg("hidden") = false,
            "Draw XYZ debug axes from a transform matrix.")
        .def("clear_debug_lines", &App::clearDebugLines, py::arg("path"),
             "Clear debug lines stored at a debug path.")
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
            py::arg("size") = 6.0f, py::arg("hidden") = false,
            "Draw persistent debug points at a debug path.")
        .def("clear_debug_points", &App::clearDebugPoints, py::arg("path"),
             "Clear debug points stored at a debug path.")
        .def(
            "set_world_text",
            [](App& self, const std::string& path, const std::string& text,
               const glm::vec3& position, const glm::vec4& color,
               float pixelSize, TextAlignment alignment,
               TextDepthMode depthMode, bool hidden) {
                WorldTextDesc desc;
                desc.text = text;
                desc.position = position;
                desc.color = color;
                desc.pixelSize = pixelSize;
                desc.alignment = alignment;
                desc.depthMode = depthMode;
                desc.hidden = hidden;
                self.setWorldText(path, desc);
            },
            py::arg("path"), py::arg("text"), py::arg("position"),
            py::arg("color") = glm::vec4(1.0f), py::arg("pixel_size") = 18.0f,
            py::arg("alignment") = TextAlignment::Center,
            py::arg("depth_mode") = TextDepthMode::DepthTested,
            py::arg("hidden") = false,
            "Create or replace screen-aligned text at a world position.")
        .def("set_world_text_string", &App::setWorldTextString, py::arg("path"),
             py::arg("text"),
             "Update the string of an existing world-text entry.")
        .def("set_world_text_position", &App::setWorldTextPosition,
             py::arg("path"), py::arg("position"),
             "Update the position of an existing world-text entry.")
        .def("set_world_text_hidden", &App::setWorldTextHidden, py::arg("path"),
             py::arg("hidden"), "Show or hide an existing world-text entry.")
        .def("remove_world_text", &App::removeWorldText, py::arg("path"),
             "Remove one world-text entry.")
        .def("clear_world_text", &App::clearWorldText,
             "Remove every world-text entry.")
        .def(
            "set_screen_text",
            [](App& self, const std::string& path, const std::string& text,
               const glm::vec2& position, const glm::vec4& color,
               float pixelSize, TextAlignment alignment, ScreenAnchor anchor,
               bool hidden) {
                ScreenTextDesc desc;
                desc.text = text;
                desc.position = position;
                desc.color = color;
                desc.pixelSize = pixelSize;
                desc.alignment = alignment;
                desc.anchor = anchor;
                desc.hidden = hidden;
                self.setScreenText(path, desc);
            },
            py::arg("path"), py::arg("text"), py::arg("position"),
            py::arg("color") = glm::vec4(1.0f), py::arg("pixel_size") = 18.0f,
            py::arg("alignment") = TextAlignment::Left,
            py::arg("anchor") = ScreenAnchor::TopLeft,
            py::arg("hidden") = false,
            "Create or replace text anchored in viewport pixel coordinates.")
        .def("set_screen_text_string", &App::setScreenTextString,
             py::arg("path"), py::arg("text"),
             "Update the string of an existing screen-text entry.")
        .def("set_screen_text_position", &App::setScreenTextPosition,
             py::arg("path"), py::arg("position"),
             "Update the pixel offset of an existing screen-text entry.")
        .def("set_screen_text_hidden", &App::setScreenTextHidden,
             py::arg("path"), py::arg("hidden"),
             "Show or hide an existing screen-text entry.")
        .def("remove_screen_text", &App::removeScreenText, py::arg("path"),
             "Remove one screen-text entry.")
        .def("clear_screen_text", &App::clearScreenText,
             "Remove every screen-text entry.")
        .def("set_light_direction", &App::setLightDirection,
             py::arg("direction"), "Set the main directional light direction.")
        .def("set_light_color", &App::setLightColor, py::arg("color"),
             "Set the main light color.")
        .def("set_light_intensity", &App::setLightIntensity,
             py::arg("intensity"), "Set the main light intensity.")
        .def("set_light_ambient", &App::setLightAmbient, py::arg("ambient"),
             "Set ambient light intensity.")
        .def(
            "set_gamma",
            [](App& self, float gamma) {
                self.getRenderer().settings().gamma = std::max(0.001f, gamma);
            },
            py::arg("gamma"), "Set display gamma used by post processing.")
        .def(
            "set_tone_map",
            [](App& self, ToneMapMode mode, float exposure) {
                RendererSettings& settings = self.getRenderer().settings();
                settings.toneMapMode = mode;
                settings.toneMapExposure = std::max(0.0f, exposure);
            },
            py::arg("mode"), py::arg("exposure") = 1.0f,
            "Configure tone mapping mode and exposure.")
        .def(
            "set_bloom",
            [](App& self, bool enabled, float threshold, float intensity,
               int iterations, int downsample) {
                BloomConfig& bloom = self.getRenderer().settings().bloom;
                bloom.enabled = enabled;
                bloom.threshold = std::max(0.0f, threshold);
                bloom.intensity = std::max(0.0f, intensity);
                bloom.iterations = std::clamp(iterations, 0, 32);
                bloom.downsample = std::clamp(downsample, 1, 16);
            },
            py::arg("enabled"), py::arg("threshold") = 1.0f,
            py::arg("intensity") = 0.08f, py::arg("iterations") = 6,
            py::arg("downsample") = 2, "Configure bloom post processing.")
        .def(
            "set_background_shader",
            [](App& self, Backend::Shader* shader) {
                self.getRenderer().setBackgroundShader(shader);
            },
            py::arg("shader"),
            "Register the checker/grid background shader controlled by "
            "renderer background settings.")
        .def("check_error", &App::checkError,
             "Check and report backend graphics errors.")
        .def(
            "is_key_pressed",
            [](App& self, int key) {
                return glfwGetKey(self.getWindow(), key) == GLFW_PRESS;
            },
            py::arg("key"), "Return true while a keyboard key is pressed.")
        .def(
            "_is_gamepad_connected",
            [](App&, int index) {
                if (index < 0 || index > GLFW_JOYSTICK_LAST)
                    return false;
                int mappedIndex = 0;
                for (int joystick = GLFW_JOYSTICK_1;
                     joystick <= GLFW_JOYSTICK_LAST; ++joystick) {
                    if (glfwJoystickIsGamepad(joystick) != GLFW_TRUE)
                        continue;
                    int axisCount = 0;
                    const float* axes =
                        glfwGetJoystickAxes(joystick, &axisCount);
                    if (axisCount >= 4 && axes[0] <= -0.999f &&
                        axes[1] <= -0.999f && axes[2] <= -0.999f &&
                        axes[3] <= -0.999f)
                        continue;
                    if (mappedIndex++ == index)
                        return true;
                }
                return false;
            },
            py::arg("index") = 0,
            "Return whether a GLFW-mapped gamepad is connected.")
        .def(
            "_list_joysticks",
            [](App&) {
                py::list result;
                for (int joystick = GLFW_JOYSTICK_1;
                     joystick <= GLFW_JOYSTICK_LAST; ++joystick) {
                    if (glfwJoystickPresent(joystick) != GLFW_TRUE)
                        continue;
                    py::dict info;
                    const char* name = glfwGetJoystickName(joystick);
                    const char* guid = glfwGetJoystickGUID(joystick);
                    info["index"] = joystick - GLFW_JOYSTICK_1;
                    info["name"] = name ? name : "";
                    info["guid"] = guid ? guid : "";
                    info["mapped"] =
                        glfwJoystickIsGamepad(joystick) == GLFW_TRUE;
                    int axisCount = 0;
                    const float* axes =
                        glfwGetJoystickAxes(joystick, &axisCount);
                    py::list axisValues;
                    for (int axis = 0; axis < axisCount; ++axis)
                        axisValues.append(axes[axis]);
                    info["axes"] = std::move(axisValues);

                    int buttonCount = 0;
                    const unsigned char* buttons =
                        glfwGetJoystickButtons(joystick, &buttonCount);
                    py::list buttonValues;
                    for (int button = 0; button < buttonCount; ++button)
                        buttonValues.append(buttons[button] == GLFW_PRESS);
                    info["buttons"] = std::move(buttonValues);
                    result.append(std::move(info));
                }
                return result;
            },
            "List raw GLFW joysticks and whether each has a gamepad mapping.")
        .def(
            "_get_gamepad_state",
            [](App&, int index) -> py::dict {
                py::dict result;
                result["connected"] = false;
                result["name"] = "";
                if (index < 0 || index > GLFW_JOYSTICK_LAST)
                    return result;
                int joystick = -1;
                int mappedIndex = 0;
                for (int candidate = GLFW_JOYSTICK_1;
                     candidate <= GLFW_JOYSTICK_LAST; ++candidate) {
                    if (glfwJoystickIsGamepad(candidate) != GLFW_TRUE)
                        continue;
                    int axisCount = 0;
                    const float* axes =
                        glfwGetJoystickAxes(candidate, &axisCount);
                    if (axisCount >= 4 && axes[0] <= -0.999f &&
                        axes[1] <= -0.999f && axes[2] <= -0.999f &&
                        axes[3] <= -0.999f)
                        continue;
                    if (mappedIndex++ == index) {
                        joystick = candidate;
                        break;
                    }
                }
                if (joystick < 0)
                    return result;
                GLFWgamepadstate state{};
                if (glfwGetGamepadState(joystick, &state) != GLFW_TRUE)
                    return result;

                const char* name = glfwGetGamepadName(joystick);
                result["connected"] = true;
                result["name"] = name ? name : "";
                result["a"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
                result["b"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
                result["x"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_X] == GLFW_PRESS;
                result["y"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_Y] == GLFW_PRESS;
                result["left_bumper"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] ==
                    GLFW_PRESS;
                result["right_bumper"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] ==
                    GLFW_PRESS;
                result["back"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_BACK] == GLFW_PRESS;
                result["start"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
                result["guide"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_GUIDE] == GLFW_PRESS;
                result["left_thumb"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB] ==
                    GLFW_PRESS;
                result["right_thumb"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_THUMB] ==
                    GLFW_PRESS;
                result["dpad_up"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS;
                result["dpad_right"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS;
                result["dpad_down"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] == GLFW_PRESS;
                result["dpad_left"] =
                    state.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS;
                result["left_x"] = state.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
                result["left_y"] = state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
                result["right_x"] = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
                result["right_y"] = state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];
                result["left_trigger"] =
                    state.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
                result["right_trigger"] =
                    state.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];
                return result;
            },
            py::arg("index") = 0,
            "Poll one GLFW-mapped gamepad and return its current state.")
        .def("get_camera", &App::getCamera, py::return_value_policy::reference,
             "Return the application camera.")
        .def("set_active_scene_camera", &App::setActiveSceneCamera,
             py::arg("prim"),
             "Use a Camera Prim's CameraComponent as the viewport render "
             "camera.")
        .def("clear_active_scene_camera", &App::clearActiveSceneCamera,
             "Return viewport rendering to the editor navigation camera.")
        .def("has_active_scene_camera", &App::hasActiveSceneCamera,
             "Return whether a scene Camera Prim is active.")
        .def("active_scene_camera_path", &App::activeSceneCameraPath,
             "Return the active scene Camera Prim path, or an empty string.")
        .def("get_view_matrix", &App::getViewMatrix,
             "Return the current camera view matrix.")
        .def("get_projection_matrix", &App::getProjectionMatrix,
             "Return the current camera projection matrix.")
        .def("get_width", &App::getWidth,
             "Return the framebuffer width in pixels.")
        .def("get_height", &App::getHeight,
             "Return the framebuffer height in pixels.")
        .def("get_scene", &App::getScene, py::return_value_policy::reference,
             "Return the active scene backend.");

    py::class_<Bridge::SkinVisualBridge,
               std::unique_ptr<Bridge::SkinVisualBridge>>(m, "SkinVisual")
        .def_static(
            "from_fbx",
            [](App* app, Backend::Shader* shader, const std::string& fbxPath,
               const std::optional<std::string>& bindFbxPath,
               const std::string& primBasePath, int clipIndex, float fps,
               float scale, bool useMaterials) {
                const std::string& resolvedBindPath =
                    bindFbxPath.has_value() ? bindFbxPath.value() : fbxPath;
                return std::make_unique<Bridge::SkinVisualBridge>(
                    Bridge::SkinVisualBridge::fromFBXWithBind(
                        app, shader, fbxPath, resolvedBindPath, primBasePath,
                        clipIndex, fps, scale, useMaterials));
            },
            py::arg("app"), py::arg("shader"), py::arg("fbx_path"),
            py::arg("bind_fbx_path") = py::none(),
            py::arg("prim_base_path") = "/fbx_character",
            py::arg("clip_index") = -1, py::arg("fps") = -1.0f,
            py::arg("scale") = 0.01f, py::arg("use_materials") = true)
        .def("apply_time", &Bridge::SkinVisualBridge::applyTime,
             py::arg("time"), py::arg("loop") = true)
        .def(
            "apply_pose",
            [](Bridge::SkinVisualBridge& self,
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
        .def("set_visible", &Bridge::SkinVisualBridge::setVisible,
             py::arg("visible"))
        .def("set_color", &Bridge::SkinVisualBridge::setColor, py::arg("color"))
        .def("set_casts_shadow", &Bridge::SkinVisualBridge::setCastsShadow,
             py::arg("casts_shadow"))
        .def("motion", &Bridge::SkinVisualBridge::motion,
             py::return_value_policy::reference_internal)
        .def("num_meshes", [](const Bridge::SkinVisualBridge& self) {
            return self.meshes().size();
        });
}
