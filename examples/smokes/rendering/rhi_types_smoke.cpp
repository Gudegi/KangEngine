#include "engine/graphics/backend/base/rhi_types.hpp"

#include <iostream>
#include <stdexcept>

using namespace KE::Backend;

static_assert(hasFlag(BufferUsage::Vertex | BufferUsage::CopyDst,
                      BufferUsage::Vertex));
static_assert(hasFlag(TextureUsage::TextureBinding |
                          TextureUsage::RenderAttachment,
                      TextureUsage::RenderAttachment));
static_assert(isDepthFormat(TextureFormat::Depth32Float));
static_assert(hasStencilAspect(TextureFormat::Depth24Stencil8));

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main() {
    BufferDesc buffer;
    require(!validate(buffer), "empty buffer descriptor was accepted");
    buffer.size = 64;
    buffer.usage = BufferUsage::Uniform | BufferUsage::CopyDst;
    require(validate(buffer).valid, "valid buffer descriptor was rejected");

    TextureResourceDesc texture;
    require(!validate(texture), "empty texture descriptor was accepted");
    texture.extent = {1280, 720, 1};
    texture.format = TextureFormat::RGBA16Float;
    texture.usage = TextureUsage::RenderAttachment |
                    TextureUsage::TextureBinding;
    require(validate(texture).valid, "valid texture descriptor was rejected");

    TextureViewDesc view;
    require(validate(view, texture).valid, "valid texture view was rejected");
    view.aspect = TextureAspect::DepthOnly;
    require(!validate(view, texture),
            "color texture accepted a depth-only view");

    TextureResourceDesc depth = texture;
    depth.format = TextureFormat::Depth32Float;
    depth.usage = TextureUsage::RenderAttachment;
    require(validate(depth).valid, "valid depth texture was rejected");
    require(validate(view, depth).valid, "valid depth view was rejected");

    TextureViewDesc invalidMipView;
    invalidMipView.baseMipLevel = 1;
    require(!validate(invalidMipView, texture),
            "out-of-range mip view was accepted");

    TextureResourceDesc multisampled = texture;
    multisampled.sampleCount = 4;
    multisampled.mipLevelCount = 2;
    require(!validate(multisampled),
            "multisampled mipmapped texture was accepted");
    multisampled.mipLevelCount = 1;
    require(!validate(multisampled),
            "directly sampled multisampled texture was accepted");
    multisampled.usage = TextureUsage::RenderAttachment;
    require(validate(multisampled).valid,
            "valid multisampled attachment was rejected");
    multisampled.sampleCount = 2;
    require(!validate(multisampled), "unsupported sample count was accepted");

    TextureResourceDesc invalidRenderDimension = texture;
    invalidRenderDimension.dimension = TextureDimension::D3;
    require(!validate(invalidRenderDimension),
            "3D render attachment was accepted by the bootstrap RHI");

    std::cout << "PASS: backend-neutral RHI type and descriptor validation"
              << std::endl;
    return 0;
}
