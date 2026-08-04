#pragma once

#include "engine/graphics/renderer/raster_pipeline_library.hpp"
#include "engine/graphics/renderer/render_pass_base.hpp"

#include <array>
#include <optional>
#include <unordered_map>
#include <vector>

namespace KE {

class MeshInstancer;
class PhongMaterial;
class PBRMaterial;
class ShaderLibrary;
struct BackgroundSettings;

// Owns forward descriptors, immutable pipeline variants, shared material
// resources, preparation caches, and read-only draw recording. Rasterizer only
// supplies the visible opaque/transparent draw lists and submits commands.
class ForwardPass : public RenderPassBase {
  public:
    ForwardPass() : RenderPassBase("ForwardPass") {}
    void initializeResources(Backend::Buffer* cameraBuffer,
                             Backend::Buffer* lightBuffer,
                             Backend::Buffer* shadowBuffer,
                             Backend::BindGroupLayout* shadowSamplingLayout,
                             ShaderLibrary& shaderLibrary);
    void setBackgroundSettings(const BackgroundSettings& settings);
    Backend::BindGroupLayout* frameLayout() const { return _groupLayouts[0]; }
    Backend::BindGroup* frameBindGroup() const { return _frameBindGroup; }
    enum class DrawKind {
        Forward,
        Material,
        SkinnedForward,
        SkinnedMaterial,
    };
    struct PreparedDraw {
        MeshInstancer* instancer = nullptr;
        Backend::GraphicsPipeline* pipeline = nullptr;
        Backend::BindGroup* skinBindGroup = nullptr;
        Backend::BindGroup* materialBindGroup = nullptr;
        DrawKind kind = DrawKind::Forward;
        bool normalMapped = false;
    };
    struct PreparedDraws {
        std::vector<PreparedDraw> draws;
        std::vector<std::unique_ptr<Backend::BindGroup>> ownedBindGroups;
    };

    Backend::BindGroupLayout*
    createBindGroupLayout(const Backend::BindGroupLayoutDesc& desc);
    Backend::PipelineLayout*
    createPipelineLayout(const Backend::PipelineLayoutDesc& desc);
    Backend::BindGroup* createBindGroup(const Backend::BindGroupDesc& desc);
    Backend::Buffer* createBuffer(const Backend::BufferDesc& desc);
    Backend::Sampler* createSampler(const Backend::SamplerDesc& desc);
    Backend::Texture*
    createTexture(const Backend::TextureResourceDesc& desc,
                  const Backend::TextureInitialData* initialData = nullptr);
    void createPipeline(RasterPipelineKey key,
                        const Backend::GraphicsPipelineDesc& desc);
    Backend::GraphicsPipeline* pipelineFor(const MeshInstancer& inst,
                                           bool transparent) const;
    void configureMaterialBindings(
        Backend::BindGroupLayout* texturedVertexColorLayout,
        Backend::BindGroupLayout* phongLayout,
        Backend::BindGroupLayout* pbrLayout, Backend::Sampler* sampler,
        Backend::Texture* whiteTexture, Backend::Texture* normalTexture);
    Backend::BindGroup* updatePhongMaterial(PhongMaterial& material,
                                            const MeshInstancer& inst);
    Backend::BindGroup* updatePbrMaterial(PBRMaterial& material,
                                          const MeshInstancer& inst);
    Backend::BindGroup*
    updateTexturedVertexColorMaterial(const MeshInstancer& inst);
    PreparedDraws prepare(const std::vector<MeshInstancer*>& drawables,
                          bool transparent);
    void record(Backend::RenderPassEncoder& pass, const PreparedDraws& prepared,
                Backend::BindGroup* shadowBindGroup) const;

  private:
    struct TexturedVertexColorResources {
        std::unique_ptr<Backend::Buffer> params;
        std::array<std::unique_ptr<Backend::TextureView>, 2> views;
        std::unique_ptr<Backend::BindGroup> bindGroup;
        std::array<const Backend::Texture*, 2> textures{};
    };
    struct PhongResources {
        std::unique_ptr<Backend::Buffer> params;
        std::array<std::unique_ptr<Backend::TextureView>, 4> views;
        std::unique_ptr<Backend::BindGroup> bindGroup;
        std::array<uintptr_t, 4> textureHandles{};
    };
    struct PbrResources {
        std::unique_ptr<Backend::Buffer> params;
        std::array<std::unique_ptr<Backend::TextureView>, 8> views;
        std::unique_ptr<Backend::BindGroup> bindGroup;
        std::array<uintptr_t, 8> textureHandles{};
    };

    Backend::BindGroupLayout* _texturedVertexColorLayout = nullptr;
    Backend::BindGroupLayout* _phongLayout = nullptr;
    Backend::BindGroupLayout* _pbrLayout = nullptr;
    Backend::Sampler* _materialSampler = nullptr;
    Backend::Texture* _whiteTexture = nullptr;
    Backend::Texture* _normalTexture = nullptr;
    std::array<Backend::BindGroupLayout*, 3> _groupLayouts{};
    Backend::PipelineLayout* _forwardPipelineLayout = nullptr;
    Backend::BindGroup* _frameBindGroup = nullptr;
    Backend::BindGroupLayout* _skinGroupLayout = nullptr;
    Backend::PipelineLayout* _skinPipelineLayout = nullptr;
    Backend::BindGroupLayout* _texturedVertexColorGroupLayout = nullptr;
    std::array<Backend::PipelineLayout*, 2>
        _texturedVertexColorPipelineLayouts{};
    Backend::BindGroupLayout* _checkerboardGroupLayout = nullptr;
    Backend::PipelineLayout* _checkerboardPipelineLayout = nullptr;
    Backend::Buffer* _checkerboardParamsBuffer = nullptr;
    Backend::BindGroup* _checkerboardBindGroup = nullptr;
    Backend::BindGroupLayout* _phongMaterialGroupLayout = nullptr;
    Backend::PipelineLayout* _phongPipelineLayout = nullptr;
    Backend::BindGroupLayout* _pbrMaterialGroupLayout = nullptr;
    Backend::PipelineLayout* _pbrPipelineLayout = nullptr;
    std::array<Backend::PipelineLayout*, 2> _skinnedMaterialPipelineLayouts{};
    std::vector<std::unique_ptr<Backend::BindGroupLayout>> _bindGroupLayouts;
    std::vector<std::unique_ptr<Backend::PipelineLayout>> _pipelineLayouts;
    std::vector<std::unique_ptr<Backend::BindGroup>> _bindGroups;
    std::vector<std::unique_ptr<Backend::Buffer>> _buffers;
    std::vector<std::unique_ptr<Backend::Sampler>> _samplers;
    std::vector<std::unique_ptr<Backend::Texture>> _textures;
    RasterPipelineLibrary _pipelines;
    std::optional<RasterPassSignature> _passSignature;
    ShaderLibrary* _shaderLibrary = nullptr;
    uint64_t _shaderGeneration = 0;
    std::unordered_map<const MeshInstancer*, TexturedVertexColorResources>
        _texturedVertexColorResources;
    std::unordered_map<const PhongMaterial*, PhongResources> _phongResources;
    std::unordered_map<const PBRMaterial*, PbrResources> _pbrResources;
};

} // namespace KE
