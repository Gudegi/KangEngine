#pragma once

// Copies Python image arrays into immutable scene texture sources.

#include "engine/scene/scene_resource_manager.hpp"
#include <pybind11/numpy.h>
#include <limits>

inline KE::Scene::TextureSourceResource
textureSourceFromArray(const pybind11::array& pixels, bool srgb, bool flip,
                       const KE::Backend::SamplerDesc& sampler = {}) {
    if (!pixels.dtype().is(pybind11::dtype::of<uint8_t>()) ||
        !(pixels.flags() & pybind11::array::c_style) || pixels.ndim() != 3 ||
        pixels.shape(0) <= 0 || pixels.shape(1) <= 0 ||
        pixels.shape(0) > std::numeric_limits<int>::max() ||
        pixels.shape(1) > std::numeric_limits<int>::max() ||
        (pixels.shape(2) != 3 && pixels.shape(2) != 4))
        throw pybind11::value_error(
            "pixels must be contiguous uint8 [height, width, 3 or 4]");
    KE::Scene::TextureSourceResource source;
    source.srgb = srgb;
    source.flip = flip;
    source.sampler = sampler;
    source.width = static_cast<int>(pixels.shape(1));
    source.height = static_cast<int>(pixels.shape(0));
    source.channels = static_cast<int>(pixels.shape(2));
    const auto* data = static_cast<const uint8_t*>(pixels.data());
    source.pixels.assign(data, data + pixels.size());
    return source;
}
