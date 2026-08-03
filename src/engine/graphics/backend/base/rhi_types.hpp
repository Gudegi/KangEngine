#ifndef KE_GRAPHICS_BACKEND_RHI_TYPES_HPP
#define KE_GRAPHICS_BACKEND_RHI_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace KE::Backend {

class TextureView;

enum class BufferUsage : uint32_t {
    None = 0,
    Vertex = 1u << 0,
    Index = 1u << 1,
    Uniform = 1u << 2,
    CopySrc = 1u << 3,
    CopyDst = 1u << 4,
};

enum class TextureUsage : uint32_t {
    None = 0,
    TextureBinding = 1u << 0,
    RenderAttachment = 1u << 1,
    CopySrc = 1u << 2,
    CopyDst = 1u << 3,
};

template <typename Enum>
constexpr std::enable_if_t<std::is_enum_v<Enum>, Enum> enumOr(Enum lhs,
                                                              Enum rhs) {
    using Value = std::underlying_type_t<Enum>;
    return static_cast<Enum>(static_cast<Value>(lhs) | static_cast<Value>(rhs));
}

template <typename Enum>
constexpr std::enable_if_t<std::is_enum_v<Enum>, Enum> enumAnd(Enum lhs,
                                                               Enum rhs) {
    using Value = std::underlying_type_t<Enum>;
    return static_cast<Enum>(static_cast<Value>(lhs) & static_cast<Value>(rhs));
}

template <typename Enum>
constexpr std::enable_if_t<std::is_enum_v<Enum>, bool> hasFlag(Enum value,
                                                               Enum flag) {
    using Value = std::underlying_type_t<Enum>;
    const Value flagValue = static_cast<Value>(flag);
    return flagValue != 0 &&
           (static_cast<Value>(value) & flagValue) == flagValue;
}

constexpr BufferUsage operator|(BufferUsage lhs, BufferUsage rhs) {
    return enumOr(lhs, rhs);
}

constexpr BufferUsage operator&(BufferUsage lhs, BufferUsage rhs) {
    return enumAnd(lhs, rhs);
}

constexpr BufferUsage& operator|=(BufferUsage& lhs, BufferUsage rhs) {
    lhs = lhs | rhs;
    return lhs;
}

constexpr TextureUsage operator|(TextureUsage lhs, TextureUsage rhs) {
    return enumOr(lhs, rhs);
}

constexpr TextureUsage operator&(TextureUsage lhs, TextureUsage rhs) {
    return enumAnd(lhs, rhs);
}

constexpr TextureUsage& operator|=(TextureUsage& lhs, TextureUsage rhs) {
    lhs = lhs | rhs;
    return lhs;
}

enum class TextureFormat : uint8_t {
    Undefined,
    R8Unorm,
    RGBA8Unorm,
    RGBA8UnormSrgb,
    RGBA16Float,
    Depth24Stencil8,
    Depth32Float,
};

enum class TextureDimension : uint8_t { D1, D2, D3 };
enum class TextureViewDimension : uint8_t { D2, Cube };

enum class TextureAspect : uint8_t { All, DepthOnly, StencilOnly };

enum class LoadOp : uint8_t { Load, Clear };

enum class StoreOp : uint8_t { Store, Discard };

enum class IndexFormat : uint8_t { Uint16, Uint32 };

enum class PrimitiveTopology : uint8_t {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
};

enum class VertexFormat : uint8_t {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Sint32x4,
};
enum class VertexStepMode : uint8_t { Vertex, Instance };
enum class FrontFace : uint8_t { CCW, CW };
enum class CullMode : uint8_t { None, Front, Back };
enum class CompareFunction : uint8_t {
    Never,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Always,
};
enum class ColorWriteMask : uint8_t {
    None = 0,
    Red = 1u << 0,
    Green = 1u << 1,
    Blue = 1u << 2,
    Alpha = 1u << 3,
    All = 0x0f,
};

enum class ShaderStageVisibility : uint8_t {
    None = 0,
    Vertex = 1u << 0,
    Fragment = 1u << 1,
    Compute = 1u << 2,
    AllGraphics = 0x03,
};
constexpr ShaderStageVisibility operator|(ShaderStageVisibility lhs,
                                          ShaderStageVisibility rhs) {
    return enumOr(lhs, rhs);
}
enum class BindingType : uint8_t { UniformBuffer, SampledTexture, Sampler };
enum class TextureSampleType : uint8_t { Float, Depth };

struct BindGroupLayoutEntry {
    uint32_t binding = 0;
    BindingType type = BindingType::UniformBuffer;
    ShaderStageVisibility visibility = ShaderStageVisibility::None;
    TextureFormat textureFormat = TextureFormat::Undefined;
    TextureSampleType textureSampleType = TextureSampleType::Float;
    TextureViewDimension textureViewDimension = TextureViewDimension::D2;
};

struct BindGroupLayoutDesc {
    std::vector<BindGroupLayoutEntry> entries;
    std::string label;
};

struct VertexAttributeDesc {
    VertexFormat format = VertexFormat::Float32;
    uint64_t offset = 0;
    uint32_t shaderLocation = 0;
};

struct VertexBufferLayout {
    uint64_t arrayStride = 0;
    VertexStepMode stepMode = VertexStepMode::Vertex;
    std::vector<VertexAttributeDesc> attributes;
};

struct PrimitiveState {
    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    FrontFace frontFace = FrontFace::CCW;
    CullMode cullMode = CullMode::Back;
};

struct RasterState {
    bool depthClamp = false;
};

struct DepthStencilState {
    TextureFormat format = TextureFormat::Depth32Float;
    bool depthWriteEnabled = true;
    CompareFunction depthCompare = CompareFunction::Less;
    // Portable pipeline depth bias. OpenGL maps constant/slope to
    // glPolygonOffset; WebGPU maps all three fields directly.
    int32_t depthBias = 0;
    float depthBiasSlopeScale = 0.0f;
    float depthBiasClamp = 0.0f;
};

enum class BlendOperation : uint8_t {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max
};
enum class BlendFactorValue : uint8_t {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
};

struct BlendComponent {
    BlendOperation operation = BlendOperation::Add;
    BlendFactorValue srcFactor = BlendFactorValue::One;
    BlendFactorValue dstFactor = BlendFactorValue::Zero;
};

struct BlendState {
    BlendComponent color;
    BlendComponent alpha;
};

struct ColorTargetState {
    TextureFormat format = TextureFormat::Undefined;
    std::optional<BlendState> blend;
    ColorWriteMask writeMask = ColorWriteMask::All;
};

struct Extent3D {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depthOrArrayLayers = 1;
};

constexpr bool isRenderableExtent(const Extent3D& extent) {
    return extent.width > 0 && extent.height > 0 &&
           extent.depthOrArrayLayers > 0;
}

struct BufferDesc {
    size_t size = 0;
    BufferUsage usage = BufferUsage::None;
    std::string label;
};

// New portable texture allocation descriptor. TextureDesc remains the legacy
// image-upload descriptor until existing renderer and bindings migrate.
struct TextureResourceDesc {
    Extent3D extent;
    TextureFormat format = TextureFormat::Undefined;
    TextureUsage usage = TextureUsage::None;
    TextureDimension dimension = TextureDimension::D2;
    uint32_t mipLevelCount = 1;
    uint32_t sampleCount = 1;
    std::string label;
};

struct TextureInitialData {
    const void* data = nullptr;
    size_t size = 0;
    // Zero selects tightly packed rows for the declared texture format.
    size_t bytesPerRow = 0;
};

struct TextureViewDesc {
    TextureFormat format = TextureFormat::Undefined;
    TextureAspect aspect = TextureAspect::All;
    uint32_t baseMipLevel = 0;
    uint32_t mipLevelCount = 1;
    uint32_t baseArrayLayer = 0;
    uint32_t arrayLayerCount = 1;
    std::string label;
    TextureViewDimension dimension = TextureViewDimension::D2;
};

struct ClearColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};

struct ColorAttachmentDesc {
    TextureView* view = nullptr;
    TextureView* resolveTarget = nullptr;
    LoadOp loadOp = LoadOp::Load;
    StoreOp storeOp = StoreOp::Store;
    ClearColor clearValue;
};

struct DepthStencilAttachmentDesc {
    TextureView* view = nullptr;
    LoadOp depthLoadOp = LoadOp::Load;
    StoreOp depthStoreOp = StoreOp::Store;
    float depthClearValue = 1.0f;
    LoadOp stencilLoadOp = LoadOp::Load;
    StoreOp stencilStoreOp = StoreOp::Store;
    uint32_t stencilClearValue = 0;
};

// Immutable attachment and load/store description copied into a RenderTarget.
// Worker threads may build descriptors, while backend object creation and use
// remain render-thread operations until command recording is introduced.
struct RenderPassDesc {
    std::vector<ColorAttachmentDesc> colorAttachments;
    std::optional<DepthStencilAttachmentDesc> depthStencilAttachment;
    std::string label;
};

// Synchronous diagnostic readback used by smoke tests and tooling. Production
// streaming should use future copy commands and asynchronous mapped buffers.
struct TextureReadback {
    uint32_t width = 0;
    uint32_t height = 0;
    TextureFormat format = TextureFormat::Undefined;
    TextureAspect aspect = TextureAspect::All;
    uint32_t componentCount = 0;
    std::vector<float> values;
};

struct DescriptorValidationResult {
    bool valid = true;
    const char* message = "";

    constexpr explicit operator bool() const { return valid; }
};

constexpr bool isDepthFormat(TextureFormat format) {
    return format == TextureFormat::Depth24Stencil8 ||
           format == TextureFormat::Depth32Float;
}

constexpr bool hasStencilAspect(TextureFormat format) {
    return format == TextureFormat::Depth24Stencil8;
}

constexpr DescriptorValidationResult validate(const BufferDesc& desc) {
    if (desc.size == 0)
        return {false, "buffer size must be greater than zero"};
    if (desc.usage == BufferUsage::None)
        return {false, "buffer usage must not be None"};
    return {};
}

constexpr DescriptorValidationResult validate(const TextureResourceDesc& desc) {
    if (desc.extent.width == 0 || desc.extent.height == 0 ||
        desc.extent.depthOrArrayLayers == 0)
        return {false, "texture extent dimensions must be greater than zero"};
    if (desc.format == TextureFormat::Undefined)
        return {false, "texture format must be explicit"};
    if (desc.usage == TextureUsage::None)
        return {false, "texture usage must not be None"};
    if (desc.mipLevelCount == 0)
        return {false, "texture mip level count must be greater than zero"};
    if (desc.sampleCount == 0)
        return {false, "texture sample count must be greater than zero"};
    if (desc.sampleCount != 1 && desc.sampleCount != 4)
        return {false, "initial RHI supports texture sample counts 1 and 4"};
    if (desc.sampleCount > 1 && desc.mipLevelCount > 1)
        return {false, "multisampled textures cannot have mip levels"};
    if (hasFlag(desc.usage, TextureUsage::RenderAttachment) &&
        desc.dimension != TextureDimension::D2)
        return {false, "initial RHI render attachments must be 2D"};
    if (desc.sampleCount > 1 &&
        hasFlag(desc.usage, TextureUsage::TextureBinding))
        return {false,
                "initial RHI cannot sample multisampled textures directly"};
    if (desc.dimension == TextureDimension::D1 && desc.extent.height != 1)
        return {false, "1D texture height must be one"};
    return {};
}

constexpr DescriptorValidationResult
validate(const TextureViewDesc& view, const TextureResourceDesc& texture) {
    const TextureFormat viewFormat =
        view.format == TextureFormat::Undefined ? texture.format : view.format;
    if (viewFormat != texture.format)
        return {false, "view format reinterpretation is not supported yet"};
    if (view.mipLevelCount == 0 || view.baseMipLevel >= texture.mipLevelCount ||
        view.mipLevelCount > texture.mipLevelCount - view.baseMipLevel)
        return {false, "texture view mip range is out of bounds"};
    if (view.arrayLayerCount == 0 ||
        view.baseArrayLayer >= texture.extent.depthOrArrayLayers ||
        view.arrayLayerCount >
            texture.extent.depthOrArrayLayers - view.baseArrayLayer)
        return {false, "texture view array-layer range is out of bounds"};
    if (view.aspect == TextureAspect::DepthOnly && !isDepthFormat(viewFormat))
        return {false, "DepthOnly aspect requires a depth format"};
    if (view.aspect == TextureAspect::StencilOnly &&
        !hasStencilAspect(viewFormat))
        return {false, "StencilOnly aspect requires a stencil format"};
    if (view.dimension == TextureViewDimension::Cube &&
        (texture.dimension != TextureDimension::D2 ||
         view.arrayLayerCount != 6))
        return {false, "cube view requires six layers of a 2D texture"};
    return {};
}

} // namespace KE::Backend

#endif
