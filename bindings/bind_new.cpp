///
/// KangEngine Python Bindings
///

#include "../src/kangEngine.hpp"
#include "../src/backend/graphics_factory.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iomanip>

namespace py = pybind11;
using namespace KE;

// Forward declaration for scene bindings
void bind_scene(py::module& m);

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
};

PYBIND11_MODULE(_kangengine, m) {
    m.doc() = "KangEngine Python bindings";

    // Bind scene system first (needed for Scene::BackendType in App.initialize)
    bind_scene(m);

    // Backend type enum
    py::enum_<Backend::BackendType>(m, "BackendType")
        .value("OpenGL", Backend::BackendType::OpenGL)
        .value("Vulkan", Backend::BackendType::Vulkan)
        .value("WebGPU", Backend::BackendType::WebGPU)
        .export_values();

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
        .def("setMat4", &Backend::Shader::setMat4);

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
        .def("createTexture",
             py::overload_cast<const std::string, bool, float, float, float>(
                 &Backend::GraphicsDevice::createTexture),
             py::arg("path"), py::arg("flip") = false,
             py::arg("warpParam") = GL_REPEAT,
             py::arg("minFilterParam") = GL_LINEAR_MIPMAP_LINEAR,
             py::arg("maxFilterParam") = GL_LINEAR,
             py::return_value_policy::take_ownership);

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
    py::class_<glm::vec3>(m, "vec3", py::buffer_protocol())
        .def(py::init<float, float, float>(), py::arg("x") = 0.0f,
             py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def(py::init([](py::handle obj) {
            // 1. 이미 ke.vec3 타입인 경우
            if (py::isinstance<glm::vec3>(obj)) {
                return obj.cast<glm::vec3>();
            }

            // 2. Buffer Protocol (PyGLM, Numpy, etc.) - 가장 빠름
            try {
                if (py::isinstance<py::buffer>(obj)) {
                    auto buf = obj.cast<py::buffer>().request();
                    if (buf.size == 3) {
                        float* ptr = static_cast<float*>(buf.ptr);
                        return glm::vec3(ptr[0], ptr[1], ptr[2]);
                    }
                }
            } catch (...) {
            }

            // 3. Attribute 접근 (PyGLM의 일반적인 객체 형태 또는 유저 정의
            // 객체)
            if (py::hasattr(obj, "x") && py::hasattr(obj, "y") &&
                py::hasattr(obj, "z")) {
                return glm::vec3(obj.attr("x").cast<float>(),
                                 obj.attr("y").cast<float>(),
                                 obj.attr("z").cast<float>());
            }

            // 4. Sequence (List, Tuple)
            if (py::isinstance<py::sequence>(obj) && py::len(obj) == 3) {
                auto seq = obj.cast<py::sequence>();
                return glm::vec3(seq[0].cast<float>(), seq[1].cast<float>(),
                                 seq[2].cast<float>());
            }

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

    // App class with trampoline for Python overrides
    py::class_<App, PyApp>(m, "App")
        .def(py::init<>())
        .def("initialize", &App::initialize, py::arg("width"),
             py::arg("height"), py::arg("hideUi") = false,
             py::arg("graphicsBackendType") = Backend::BackendType::OpenGL,
             py::arg("sceneBackendType") = Scene::BackendType::Native)
        .def("start", &App::start)
        .def("setup", &App::setup)
        .def("preRender", &App::preRender)
        .def("render", &App::render)
        .def("postRender", &App::postRender)
        .def("getGraphicsDevice", &App::getGraphicsDevice,
             py::return_value_policy::reference)
        .def("addShape",
             [](App* self, Backend::Shader* shader,
                std::shared_ptr<Scene::MeshData> meshData) {
                 return self->addShape(shader, meshData);
             })
        .def("addShapePrim",
             [](App* self, Backend::Shader* shader,
                std::shared_ptr<Scene::Prim> prim) {
                 return self->addShape(shader, prim);
             })
        .def("checkError", &App::checkError)
        .def("getCamera", &App::getCamera, py::return_value_policy::reference)
        .def("getViewMatrix", &App::getViewMatrix)
        .def("getProjectionMatrix", &App::getProjectionMatrix)
        .def("getWidth", &App::getWidth)
        .def("getHeight", &App::getHeight);
}
