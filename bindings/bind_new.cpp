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
#include <cstring>
#include <iomanip>
#include <optional>

namespace py = pybind11;
using namespace KE;

// Forward declarations for submodule bindings
void bind_scene(py::module& m);
void bind_animation(py::module& m);
void bind_articulation_desc(py::module& m);
void bind_asset(py::module& m);
void bind_sim(py::module& m);
void bind_physics(py::module& m);
void bind_physics_gpu(py::module& m);
void bind_render(py::module& m);
void bind_material(py::module& m);

// Trampoline class for App - allows Python to override virtual methods
class PyApp : public App {
  public:
    using App::App;

    void setup() override { PYBIND11_OVERRIDE_PURE(void, App, setup); }
    void preUpdate() override {
        PYBIND11_OVERRIDE_NAME(void, App, "pre_update", preUpdate);
    }
    void fixedUpdate(double fixedDt) override {
        PYBIND11_OVERRIDE_NAME(void, App, "fixed_update", fixedUpdate, fixedDt);
    }
    void preRender() override {
        PYBIND11_OVERRIDE_NAME(void, App, "pre_render", preRender);
    }
    void render() override { PYBIND11_OVERRIDE_PURE(void, App, render); }
    void postRender() override {
        PYBIND11_OVERRIDE_NAME(void, App, "post_render", postRender);
    }
    void onFrameRenderedInternal() override {
        PYBIND11_OVERRIDE_NAME(void, App, "_on_frame_rendered_internal",
                               onFrameRenderedInternal);
    }
    void onRayPicked(const RayPickResult& result) override {
        PYBIND11_OVERRIDE_NAME(void, App, "on_ray_picked", onRayPicked, result);
    }
    void onRayPickHover(const RayPickResult& result) override {
        PYBIND11_OVERRIDE_NAME(void, App, "on_ray_pick_hover", onRayPickHover,
                               result);
    }
    void onForceDragBegin(const RayPickResult& result,
                          const glm::vec3& target) override {
        PYBIND11_OVERRIDE_NAME(void, App, "on_force_drag_begin",
                               onForceDragBegin, result, target);
    }
    void onForceDragUpdate(const RayPickResult& result,
                           const glm::vec3& target) override {
        PYBIND11_OVERRIDE_NAME(void, App, "on_force_drag_update",
                               onForceDragUpdate, result, target);
    }
    void onForceDragEnd() override {
        PYBIND11_OVERRIDE_NAME(void, App, "on_force_drag_end", onForceDragEnd);
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
                ImVec2(width, height), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f),
                ImVec4(1.0f, 1.0f, 1.0f, std::clamp(opacity, 0.0f, 1.0f)),
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
        [](float x1, float y1, float x2, float y2, const glm::vec4& color) {
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
                const auto point = py::reinterpret_borrow<py::sequence>(item);
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
    bind_render(m);
    bind_material(m);

    // Enums first — submodule bindings may use them as default arguments
    py::enum_<UpAxis>(m, "UpAxis")
        .value("X", UpAxis::X)
        .value("Y", UpAxis::Y)
        .value("Z", UpAxis::Z);

    py::enum_<InteractionMode>(m, "InteractionMode")
        .value("INSPECT", InteractionMode::Inspect)
        .value("EDIT", InteractionMode::Edit)
        .value("FORCE", InteractionMode::Force);

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
    bind_asset(m);
    bind_articulation_desc(m);
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

    // GLM types for matrix operations
    // Support implicit conversion from PyGLM types (tuple/list with x,y,z or
    // indexable)
    /*
py::class_<glm::vec3>(m, "Vec3")
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
    py::class_<glm::vec2>(m, "Vec2", py::buffer_protocol())
        .def(py::init<float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f)
        .def(py::init([](py::handle obj) {
            if (py::isinstance<glm::vec2>(obj))
                return obj.cast<glm::vec2>();

            if (auto values = fixedFloatArray<2>(obj, "Vec2"))
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

    py::class_<glm::vec3>(m, "Vec3", py::buffer_protocol())
        .def(py::init<float, float, float>(), py::arg("x") = 0.0f,
             py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def(py::init([](py::handle obj) {
            if (py::isinstance<glm::vec3>(obj))
                return obj.cast<glm::vec3>();

            // Buffer protocol: numpy (float32 or float64), PyGLM, etc.
            if (auto values = fixedFloatArray<3>(obj, "Vec3"))
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

    py::class_<glm::vec4>(m, "Vec4", py::buffer_protocol())
        .def(py::init<float, float, float, float>())
        .def(py::init([](py::object obj) {
            // Buffer protocol
            if (auto values = fixedFloatArray<4>(obj, "Vec4"))
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

    py::class_<glm::quat>(m, "Quat")
        .def(py::init<float, float, float, float>(), py::arg("w"), py::arg("x"),
             py::arg("y"), py::arg("z"))
        .def(py::init([](py::object obj) {
            // Python quaternion values consistently use wxyz order,
            // independent of GLM's internal memory layout.
            if (auto values = fixedFloatArray<4>(obj, "Quat"))
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

    py::class_<glm::mat3>(m, "Mat3", py::buffer_protocol())
        .def(py::init<float>(), py::arg("value") = 1.0f)
        .def(py::init([](py::handle obj) {
            glm::mat3 m(1.0f);

            // A. Buffer Protocol (PyGLM, Numpy 등)
            if (auto values = fixedFloatArray<9>(obj, "Mat3")) {
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
    py::class_<glm::mat4>(m, "Mat4", py::buffer_protocol())
        // 기본 생성자 (Identity)
        .def(py::init<float>(), py::arg("value") = 1.0f)

        // 통합 생성자: PyGLM, 리스트, 튜플, 넘파이 모두 처리
        .def(py::init([](py::handle obj) {
            glm::mat4 m(1.0f);

            // 1. Buffer Protocol 시도 (PyGLM, Numpy 등)
            // obj.cast<py::object>()를 통해 handle을 object로 변환 후 buffer로
            // 접근합니다.
            if (auto values = fixedFloatArray<16>(obj, "Mat4")) {
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

    py::class_<FixedStepClock>(
        m, "FixedStepClock",
        "Wall-clock accumulator that schedules bounded fixed updates.")
        .def(py::init<>())
        .def("set_step_hz", &FixedStepClock::setStepHz, py::arg("step_hz"))
        .def("get_step_hz", &FixedStepClock::getStepHz)
        .def("get_step_interval", &FixedStepClock::getStepInterval)
        .def("set_max_catch_up_steps", &FixedStepClock::setMaxCatchUpSteps,
             py::arg("count"))
        .def("get_max_catch_up_steps", &FixedStepClock::getMaxCatchUpSteps)
        .def("set_max_frame_delta", &FixedStepClock::setMaxFrameDelta,
             py::arg("seconds"))
        .def("get_max_frame_delta", &FixedStepClock::getMaxFrameDelta)
        .def("set_paused", &FixedStepClock::setPaused, py::arg("paused"))
        .def("is_paused", &FixedStepClock::isPaused)
        .def("request_single_step", &FixedStepClock::requestSingleStep)
        .def("advance", &FixedStepClock::advance, py::arg("wall_delta_seconds"))
        .def("reset", &FixedStepClock::reset)
        .def("get_accumulator", &FixedStepClock::getAccumulator)
        .def("get_dropped_wall_time", &FixedStepClock::getDroppedWallTime);

    // App class with trampoline for Python overrides
    py::class_<App, PyApp>(
        m, "App",
        "Native application shell with a window, scene, renderer, and input.")
        .def(py::init<>(), "Create an uninitialized application.")
        .def("initialize", &App::initialize, py::arg("width"),
             py::arg("height"), py::arg("hide_ui") = false,
             py::arg("up_axis") = UpAxis::Y,
             py::arg("graphics_backend_type") = Backend::BackendType::OpenGL,
             py::arg("scene_backend_type") = Scene::BackendType::Native,
             py::arg("headless") = false,
             "Initialize the window, renderer, input, and scene backend.")
        .def("set_render_hz", &App::setRenderHz, py::arg("render_hz"),
             "Set the target render/update frequency in Hz.")
        .def("set_vsync", &App::setVSync, py::arg("enabled"),
             "Enable or disable vertical synchronization.")
        .def("get_vsync", &App::getVSync)
        .def("get_delta_time", &App::getDeltaTime,
             "Return elapsed seconds between recent frames.")
        .def("get_render_hz", &App::getRenderHz,
             "Return the target render/update frequency in Hz.")
        .def("set_frame_capture_active", &App::setFrameCaptureActive,
             py::arg("active"))
        .def("get_frame_capture_active", &App::getFrameCaptureActive)
        .def("consume_video_recording_toggle_requested",
             &App::consumeVideoRecordingToggleRequested)
        .def("set_fixed_update_hz", &App::setFixedUpdateHz,
             py::arg("update_hz"),
             "Enable fixed updates at the requested wall-clock frequency.")
        .def("get_fixed_update_hz", &App::getFixedUpdateHz)
        .def("set_max_catch_up_steps", &App::setMaxCatchUpSteps,
             py::arg("count"))
        .def("get_max_catch_up_steps", &App::getMaxCatchUpSteps)
        .def("set_max_frame_delta", &App::setMaxFrameDelta, py::arg("seconds"))
        .def("get_max_frame_delta", &App::getMaxFrameDelta)
        .def("set_simulation_paused", &App::setSimulationPaused,
             py::arg("paused"))
        .def("is_simulation_paused", &App::isSimulationPaused)
        .def("request_simulation_step", &App::requestSimulationStep)
        .def("set_simulation_hotkeys_enabled",
             &App::setSimulationHotkeysEnabled, py::arg("enabled"),
             "Enable Enter play/pause and Space pause/single-step shortcuts.")
        .def("get_simulation_hotkeys_enabled",
             &App::getSimulationHotkeysEnabled)
        .def("get_dropped_wall_time", &App::getDroppedWallTime)
        .def(
            "get_ui_scale",
            [](const App& self) { return self.getUiScale().value(); },
            "Return the current UI scale factor.")
        .def("set_camera_move_speed", &App::setCameraMoveSpeed,
             py::arg("speed"), "Set interactive camera movement speed.")
        .def("get_camera_move_speed", &App::getCameraMoveSpeed,
             "Return interactive camera movement speed.")
        .def("set_skybox",
             py::overload_cast<const std::string&>(&App::setSkybox),
             py::arg("cross_image_path"),
             "Set a skybox from one cross-layout cubemap image.")
        .def(
            "set_skybox",
            py::overload_cast<const std::vector<std::string>&>(&App::setSkybox),
            py::arg("face_paths"),
            "Set a skybox from six cubemap face image paths.")
        .def("start", &App::start, "Enter the application render loop.")
        .def("render_frame_once", &App::renderFrameOnce,
             "Render and process one frame without entering the main loop.")
        .def(
            "read_rgb_pixels",
            [](App& self, bool flipY) {
                const int width = self.getWidth();
                const int height = self.getHeight();
                py::array_t<uint8_t> out({height, width, 3});

                std::vector<uint8_t> pixels = self.readRgbPixels(flipY);
                if (pixels.size() != static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) * 3)
                    return out;
                std::memcpy(out.mutable_data(), pixels.data(), pixels.size());
                return out;
            },
            py::arg("flip_y") = true,
            "Read the current framebuffer as a uint8 RGB numpy array.")
        .def(
            "read_rgb_pixels_resized",
            [](App& self, int width, int height, bool flipY) {
                py::array_t<uint8_t> out({height, width, 3});
                std::vector<uint8_t> pixels =
                    self.readRgbPixelsResized(width, height, flipY);
                if (pixels.size() != static_cast<std::size_t>(width) *
                                         static_cast<std::size_t>(height) * 3)
                    return out;
                std::memcpy(out.mutable_data(), pixels.data(), pixels.size());
                return out;
            },
            py::arg("width"), py::arg("height"), py::arg("flip_y") = true,
            "Scale the current framebuffer on the GPU and return uint8 RGB.")
        .def("write_pixels_png", &App::writePixelsPNG, py::arg("path"),
             py::arg("flip_y") = true,
             "Write the current framebuffer to a PNG file.")
        .def("should_close", &App::shouldClose,
             "Return true when the application window should close.")
        .def("request_close", &App::requestClose,
             "Request a clean exit from the application loop.")
        .def("setup", &App::setup,
             "User override called once after initialization.")
        .def("pre_update", &App::preUpdate,
             "User override called once before fixed updates each frame.")
        .def("fixed_update", &App::fixedUpdate, py::arg("fixed_dt"),
             "User override called zero or more times at a fixed timestep.")
        .def("pre_render", &App::preRender,
             "User override called before each frame is rendered.")
        .def("render", &App::render,
             "User override called during each frame render.")
        .def("post_render", &App::postRender,
             "User override called after each frame is rendered.")
        .def("on_ray_picked", &App::onRayPicked, py::arg("result"),
             "User override called when ray picking selects an object.")
        .def("on_ray_pick_hover", &App::onRayPickHover, py::arg("result"),
             "User override called when ray picking hovers an object.")
        .def("on_force_drag_begin", &App::onForceDragBegin, py::arg("result"),
             py::arg("target"),
             "User override called when force dragging begins.")
        .def("on_force_drag_update", &App::onForceDragUpdate, py::arg("result"),
             py::arg("target"),
             "User override called while force dragging updates.")
        .def("on_force_drag_end", &App::onForceDragEnd,
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
               py::object colors, float size, bool hidden, bool overlay) {
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
                self->logDebugPoints(path, p, c, size, hidden, overlay);
            },
            py::arg("path"), py::arg("points"), py::arg("colors") = py::none(),
            py::arg("size") = 6.0f, py::arg("hidden") = false,
            py::arg("overlay") = false,
            "Draw persistent debug points, optionally over scene depth.")
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
            "set_background_color",
            [](App& self, const glm::vec4& color) {
                self.getRenderer().settings().background.backgroundColor =
                    color;
            },
            py::arg("color"), "Set the renderer clear/background color.")
        .def(
            "set_ground_style",
            [](App& self, const glm::vec4& checkerA, const glm::vec4& checkerB,
               const glm::vec4& gridColor, bool showGrid, bool usePbr,
               float metallic, float roughness, float gridScale,
               float lineWidth, float gridEmission) {
                auto& background = self.getRenderer().settings().background;
                background.checkerColor1 = checkerA;
                background.checkerColor2 = checkerB;
                background.gridColor = gridColor;
                background.showGrid = showGrid;
                background.groundShadingModel = usePbr
                                                    ? GroundShadingModel::Pbr
                                                    : GroundShadingModel::Phong;
                background.groundMetallic = std::clamp(metallic, 0.0f, 1.0f);
                background.groundRoughness = std::clamp(roughness, 0.04f, 1.0f);
                background.gridScale = std::max(gridScale, 0.0001f);
                background.gridLineWidth =
                    std::clamp(lineWidth, 0.0001f, 0.49f);
                background.gridEmission = std::max(gridEmission, 0.0f);
            },
            py::arg("checker_a"), py::arg("checker_b"), py::arg("grid_color"),
            py::arg("show_grid") = true, py::arg("use_pbr") = false,
            py::arg("metallic") = 0.0f, py::arg("roughness") = 0.8f,
            py::arg("grid_scale") = 1.0f, py::arg("line_width") = 0.005f,
            py::arg("grid_emission") = 0.0f,
            "Configure the standard ground shading and procedural grid.")
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
                    state.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB] == GLFW_PRESS;
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
               std::unique_ptr<Bridge::SkinVisualBridge>>(
        m, "SkinVisual",
        "Skinned-character visual bridge created from an FBX asset.")
        .def_static(
            "from_fbx",
            [](App* app, Material* material, const std::string& fbxPath,
               const std::optional<std::string>& bindFbxPath,
               const std::string& primBasePath, int clipIndex, float fps,
               float scale, bool useMaterials) {
                const std::string& resolvedBindPath =
                    bindFbxPath.has_value() ? bindFbxPath.value() : fbxPath;
                return std::make_unique<Bridge::SkinVisualBridge>(
                    Bridge::SkinVisualBridge::fromFBXWithBind(
                        app, material, fbxPath, resolvedBindPath, primBasePath,
                        clipIndex, fps, scale, useMaterials));
            },
            py::arg("app"), py::arg("material"), py::arg("fbx_path"),
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
