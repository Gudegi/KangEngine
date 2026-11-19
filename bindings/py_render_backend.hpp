#pragma once

// Connects Python render adapters to the C++ App frame lifecycle.

#include "engine/graphics/renderer/render_backend/render_backend.hpp"
#include <pybind11/pybind11.h>
#include <memory>

std::unique_ptr<KE::RenderBackend>
makePythonRenderBackend(pybind11::object factory);
