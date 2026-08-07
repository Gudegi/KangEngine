#ifndef _PY_ARRAY_VIEW_HPP_
#define _PY_ARRAY_VIEW_HPP_

// Shared pybind helpers for accepting numpy arrays and CPU torch tensors
// without requiring Python callers to build GLM objects or std::vector lists.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <glm/glm.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;

using FloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

template <std::size_t N>
inline std::optional<std::array<float, N>> fixedFloatArray(py::handle obj,
                                                           const char* name) {
    if (!py::isinstance<py::buffer>(obj))
        return std::nullopt;

    // NumPy defaults to float64, while KangEngine stores GLM values as float32.
    // Force-casting through FloatArray avoids interpreting double bytes as
    // float bytes, which can scramble compact values such as RGBA colors.
    FloatArray array = FloatArray::ensure(obj);
    if (!array)
        return std::nullopt;

    py::buffer_info info = array.request();
    if (info.size != static_cast<py::ssize_t>(N))
        throw py::value_error(std::string(name) + " expected exactly " +
                              std::to_string(N) + " values");

    const float* ptr = static_cast<const float*>(info.ptr);
    std::array<float, N> values{};
    std::copy_n(ptr, N, values.begin());
    return values;
}

struct Vec3ArrayView {
    const float* data = nullptr;
    size_t count = 0;
};

struct Vec4ArrayView {
    const float* data = nullptr;
    size_t count = 0;
};

struct Mat4ArrayView {
    const float* data = nullptr;
    size_t count = 0;
};

struct FloatVectorView {
    const float* data = nullptr;
    size_t count = 0;
};

inline FloatVectorView floatVectorView(const FloatArray& array,
                                       const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim == 0)
        throw py::value_error(std::string(name) + " expected a 1D array");
    if (info.ndim == 1) {
        return {static_cast<const float*>(info.ptr),
                static_cast<size_t>(info.shape[0])};
    }
    if (info.ndim == 2 && info.shape[0] == 1) {
        return {static_cast<const float*>(info.ptr),
                static_cast<size_t>(info.shape[1])};
    }
    if (info.ndim == 2 && info.shape[1] == 1) {
        return {static_cast<const float*>(info.ptr),
                static_cast<size_t>(info.shape[0])};
    }
    throw py::value_error(std::string(name) + " expected shape [N], [1, N], "
                                              "or [N, 1]");
}

inline std::vector<float> floatVectorArray(const FloatArray& array,
                                           const char* name) {
    FloatVectorView view = floatVectorView(array, name);
    return std::vector<float>(view.data, view.data + view.count);
}

inline py::array_t<float>
floatArrayFromVector(const std::vector<float>& values) {
    py::array_t<float> array(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(),
                    sizeof(float) * values.size());
    }
    return array;
}

inline py::tuple intVectorTuple(const std::vector<int64_t>& values) {
    py::tuple result(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        result[i] = values[i];
    return result;
}

inline py::array_t<float>
floatArrayFromVec2Vector(const std::vector<std::array<float, 2>>& values) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(2)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        view(i, 0) = values[static_cast<size_t>(i)][0];
        view(i, 1) = values[static_cast<size_t>(i)][1];
    }
    return array;
}

inline glm::vec3 vec3FromSequence(const py::sequence& values,
                                  const char* name = "vec3") {
    if (py::len(values) != 3)
        throw py::value_error(std::string(name) +
                              " input must have exactly 3 values");
    return glm::vec3(values[0].cast<float>(), values[1].cast<float>(),
                     values[2].cast<float>());
}

inline glm::quat quatFromXYZWSequence(const py::sequence& values,
                                      const char* name = "quat") {
    if (py::len(values) != 4)
        throw py::value_error(std::string(name) +
                              " input must have exactly 4 xyzw values");
    return glm::quat(values[3].cast<float>(), values[0].cast<float>(),
                     values[1].cast<float>(), values[2].cast<float>());
}

inline py::tuple vec3Tuple(const glm::vec3& value) {
    return py::make_tuple(value.x, value.y, value.z);
}

inline py::tuple quatXYZWTuple(const glm::quat& value) {
    return py::make_tuple(value.x, value.y, value.z, value.w);
}

inline py::tuple mat4Tuple(const glm::mat4& value) {
    py::tuple result(16);
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row)
            result[static_cast<size_t>(col * 4 + row)] = value[col][row];
    }
    return result;
}

inline py::list mat4List(const std::vector<glm::mat4>& matrices) {
    py::list result;
    for (const glm::mat4& matrix : matrices)
        result.append(mat4Tuple(matrix));
    return result;
}

inline Vec3ArrayView vec3ArrayView(const FloatArray& array, const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim == 1) {
        if (info.shape[0] == 0)
            return {};
        if (info.shape[0] != 3)
            throw py::value_error(std::string(name) +
                                  " expected shape [3] or [N, 3]");
        return {static_cast<const float*>(info.ptr), 1};
    }
    if (info.ndim != 2 || info.shape[1] != 3)
        throw py::value_error(std::string(name) +
                              " expected shape [3] or [N, 3]");
    return {static_cast<const float*>(info.ptr),
            static_cast<size_t>(info.shape[0])};
}

inline Vec4ArrayView vec4ArrayView(const FloatArray& array, const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim == 1) {
        if (info.shape[0] == 0)
            return {};
        if (info.shape[0] != 4)
            throw py::value_error(std::string(name) +
                                  " expected shape [4] or [N, 4]");
        return {static_cast<const float*>(info.ptr), 1};
    }
    if (info.ndim != 2 || info.shape[1] != 4)
        throw py::value_error(std::string(name) +
                              " expected shape [4] or [N, 4]");
    return {static_cast<const float*>(info.ptr),
            static_cast<size_t>(info.shape[0])};
}

inline Mat4ArrayView mat4ArrayView(const FloatArray& array, const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim == 1) {
        if (info.shape[0] == 0)
            return {};
        if (info.shape[0] != 16)
            throw py::value_error(std::string(name) +
                                  " expected shape [16], [4, 4], [N, 16], "
                                  "or [N, 4, 4]");
        return {static_cast<const float*>(info.ptr), 1};
    }
    if (info.ndim == 2) {
        if (info.shape[0] == 4 && info.shape[1] == 4)
            return {static_cast<const float*>(info.ptr), 1};
        if (info.shape[1] == 16)
            return {static_cast<const float*>(info.ptr),
                    static_cast<size_t>(info.shape[0])};
    }
    if (info.ndim == 3 && info.shape[1] == 4 && info.shape[2] == 4)
        return {static_cast<const float*>(info.ptr),
                static_cast<size_t>(info.shape[0])};
    throw py::value_error(std::string(name) +
                          " expected shape [16], [4, 4], [N, 16], or "
                          "[N, 4, 4]");
}

inline std::vector<glm::vec3> vec3Array(const FloatArray& array,
                                        const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim == 1) {
        if (info.shape[0] == 0)
            return {};
        if (info.shape[0] != 3)
            throw py::value_error(std::string(name) +
                                  " expected shape [3] or [N, 3]");
        const auto* p = static_cast<const float*>(info.ptr);
        return {glm::vec3(p[0], p[1], p[2])};
    }
    if (info.ndim != 2 || info.shape[1] != 3)
        throw py::value_error(std::string(name) +
                              " expected shape [3] or [N, 3]");
    const auto* p = static_cast<const float*>(info.ptr);
    std::vector<glm::vec3> result;
    result.reserve(static_cast<size_t>(info.shape[0]));
    for (py::ssize_t i = 0; i < info.shape[0]; ++i)
        result.emplace_back(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
    return result;
}

inline std::vector<glm::vec4> vec4Array(const FloatArray& array,
                                        const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim == 1) {
        if (info.shape[0] == 0)
            return {};
        if (info.shape[0] != 4)
            throw py::value_error(std::string(name) +
                                  " expected shape [4] or [N, 4]");
        const auto* p = static_cast<const float*>(info.ptr);
        return {glm::vec4(p[0], p[1], p[2], p[3])};
    }
    if (info.ndim != 2 || info.shape[1] != 4)
        throw py::value_error(std::string(name) +
                              " expected shape [4] or [N, 4]");
    const auto* p = static_cast<const float*>(info.ptr);
    std::vector<glm::vec4> result;
    result.reserve(static_cast<size_t>(info.shape[0]));
    for (py::ssize_t i = 0; i < info.shape[0]; ++i)
        result.emplace_back(p[i * 4], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]);
    return result;
}

inline std::vector<glm::ivec4> intVec4Array(const IntArray& array,
                                            const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim != 2 || info.shape[1] != 4)
        throw py::value_error(std::string(name) + " expected shape [N, 4]");
    const auto* p = static_cast<const int*>(info.ptr);
    std::vector<glm::ivec4> result;
    result.reserve(static_cast<size_t>(info.shape[0]));
    for (py::ssize_t i = 0; i < info.shape[0]; ++i)
        result.emplace_back(p[i * 4], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]);
    return result;
}

inline std::vector<glm::mat4> mat4Array(const FloatArray& array,
                                        const char* name) {
    Mat4ArrayView view = mat4ArrayView(array, name);
    std::vector<glm::mat4> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        glm::mat4 m(1.0f);
        std::memcpy(&m[0][0], view.data + i * 16, sizeof(float) * 16);
        result.push_back(m);
    }
    return result;
}

inline std::vector<glm::mat4> mat4RowMajorArray(const FloatArray& array,
                                                const char* name) {
    Mat4ArrayView view = mat4ArrayView(array, name);
    std::vector<glm::mat4> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        const float* p = view.data + i * 16;
        glm::mat4 m(1.0f);
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col)
                m[col][row] = p[row * 4 + col];
        }
        result.push_back(m);
    }
    return result;
}

#endif
