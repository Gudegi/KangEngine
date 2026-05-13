#ifndef _PY_ARRAY_VIEW_HPP_
#define _PY_ARRAY_VIEW_HPP_

// Shared pybind helpers for accepting numpy arrays and CPU torch tensors
// without requiring Python callers to build GLM objects or std::vector lists.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <glm/glm.hpp>
#include <array>
#include <cstring>
#include <vector>

namespace py = pybind11;

using FloatArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

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

inline py::array_t<float> floatArrayFromVector(const std::vector<float>& values) {
    py::array_t<float> array(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(),
                    sizeof(float) * values.size());
    }
    return array;
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

#endif
