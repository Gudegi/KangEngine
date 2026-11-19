#pragma once

// Connects Python render adapters to App, including ovrtx CUDA image delivery.

#include "engine/graphics/renderer/render_backend/render_backend.hpp"
#include <pybind11/pybind11.h>
#include <memory>

// gpuOutput keeps rendered images on the GPU for window presentation.
std::unique_ptr<KE::RenderBackend>
makePythonRenderBackend(pybind11::object factory, bool gpuOutput = false);
