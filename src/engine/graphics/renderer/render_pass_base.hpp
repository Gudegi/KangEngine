#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace KE {

// Common lifecycle root for renderer passes. Pass-specific prepare data and
// encoder types intentionally remain on the concrete pass.

// initialize()
// prepare()       // render/device thread
// record() const  // read-only command recording
class RenderPassBase {
  public:
    void initialize(Backend::GraphicsDevice* device) {
        if (!device)
            throw std::invalid_argument(_passName +
                                        " requires a graphics device");
        if (_device)
            throw std::logic_error(_passName + " is already initialized");
        _device = device;
    }

    bool isInitialized() const { return _device != nullptr; }

  protected:
    explicit RenderPassBase(std::string passName)
        : _passName(std::move(passName)) {}

    void requireInitialized(const char* operation) const {
        if (!_device)
            throw std::logic_error(_passName + " must be initialized before " +
                                   operation);
    }

    Backend::GraphicsDevice* _device = nullptr;

  private:
    std::string _passName;
};

} // namespace KE
