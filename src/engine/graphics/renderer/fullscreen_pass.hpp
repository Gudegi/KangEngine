#pragma once

#include "engine/graphics/backend/base/graphics_device.hpp"

#include <cstdint>
#include <stdexcept>

namespace KE {

// Records the backend-neutral commands shared by fullscreen post passes.
// Pipeline and bindings are immutable resources owned by the caller, so this
// function remains safe to use from future worker/Taskflow recording tasks.
class FullscreenPass {
  public:
    static void record(Backend::RenderPassEncoder& pass,
                       Backend::GraphicsPipeline* pipeline, uint32_t width,
                       uint32_t height,
                       Backend::BindGroup* passBindGroup = nullptr,
                       uint32_t passGroupIndex = 3) {
        if (!pipeline)
            throw std::invalid_argument(
                "fullscreen pass requires a pipeline");
        if (width == 0 || height == 0)
            throw std::invalid_argument(
                "fullscreen pass extent must be non-zero");
        pass.setViewport(0.0f, 0.0f, static_cast<float>(width),
                         static_cast<float>(height));
        pass.setPipeline(pipeline);
        if (passBindGroup)
            pass.setBindGroup(passGroupIndex, passBindGroup);
        pass.draw(3);
    }
};

} // namespace KE
