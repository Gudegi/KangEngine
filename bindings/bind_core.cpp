// Shared math, timing, and coordinate-conversion bindings.
#include "utils/types.hpp"
#include "utils/coordinate_system.hpp"
#include "engine/core/app/fixed_step_clock.hpp"
#include "py_array_view.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <sstream>
#include <iomanip>

namespace py = pybind11;
using namespace KE;

void bind_core(py::module& m) {
    py::enum_<UpAxis>(m, "UpAxis")
        .value("X", UpAxis::X)
        .value("Y", UpAxis::Y)
        .value("Z", UpAxis::Z);
    // GLM types for matrix operations
    // Support implicit conversion from PyGLM types (tuple/list with x,y,z or
    // indexable)

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

    py::module utils =
        m.def_submodule("utils", "Shared KangEngine utility functions.");
    utils.def(
        "coordinate_conversion_matrix",
        [](const std::string& source, const std::string& target) {
            const Eigen::Matrix3f matrix = Utils::coordinateConversionMatrix(
                Utils::coordinateSystemFromString(source),
                Utils::coordinateSystemFromString(target));
            py::array_t<float> result({py::ssize_t(3), py::ssize_t(3)});
            auto view = result.mutable_unchecked<2>();
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    view(row, column) = matrix(row, column);
            return result;
        },
        py::arg("source"), py::arg("target"),
        "Return the rotation matrix converting vectors between coordinate "
        "systems.");
}
