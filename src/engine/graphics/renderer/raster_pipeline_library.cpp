#include "engine/graphics/renderer/raster_pipeline_library.hpp"

#include <stdexcept>

namespace KE {

RasterPassSignature RasterPassSignature::fromPipelineDesc(
    const Backend::GraphicsPipelineDesc& desc) {
    if (desc.colorTargets.size() > MaxColorTargets)
        throw std::invalid_argument(
            "Raster pass signature exceeds the supported color target count");
    if (desc.sampleCount == 0)
        throw std::invalid_argument(
            "Raster pass signature requires a non-zero sample count");

    RasterPassSignature signature;
    signature.colorTargetCount = static_cast<uint8_t>(desc.colorTargets.size());
    for (size_t i = 0; i < desc.colorTargets.size(); ++i) {
        if (desc.colorTargets[i].format == Backend::TextureFormat::Undefined)
            throw std::invalid_argument(
                "Raster pass signature has an undefined color format");
        signature.colorFormats[i] = desc.colorTargets[i].format;
    }
    signature.depthFormat = desc.depthStencil
                                ? desc.depthStencil->format
                                : Backend::TextureFormat::Undefined;
    signature.sampleCount = desc.sampleCount;
    signature.topology = desc.primitive.topology;
    return signature;
}

void RasterPipelineLibrary::add(
    RasterPipelineKey key,
    std::unique_ptr<Backend::GraphicsPipeline> pipeline) {
    if (!pipeline)
        throw std::invalid_argument(
            "RasterPipelineLibrary cannot register a null pipeline");
    const auto [it, inserted] = _pipelines.emplace(key, std::move(pipeline));
    if (!inserted)
        throw std::logic_error(
            "RasterPipelineLibrary received a duplicate pipeline key");
}

Backend::GraphicsPipeline*
RasterPipelineLibrary::find(const RasterPipelineKey& key) const {
    const auto it = _pipelines.find(key);
    return it == _pipelines.end() ? nullptr : it->second.get();
}

Backend::GraphicsPipeline*
RasterPipelineLibrary::getOrCreate(RasterPipelineKey key,
                                   const Factory& factory) {
    if (auto* existing = find(key))
        return existing;
    if (!factory)
        throw std::invalid_argument(
            "RasterPipelineLibrary requires a pipeline factory");
    auto pipeline = factory();
    if (!pipeline)
        throw std::runtime_error(
            "RasterPipelineLibrary factory returned a null pipeline");
    auto* result = pipeline.get();
    _pipelines.emplace(std::move(key), std::move(pipeline));
    return result;
}

Backend::GraphicsPipeline*
RasterPipelineLibrary::get(RasterPipelineKey key) const {
    auto* pipeline = find(key);
    if (!pipeline)
        throw std::logic_error(
            "RasterPipelineLibrary has no pipeline for the requested key");
    return pipeline;
}

void RasterPipelineLibrary::retireBeforeShaderGeneration(uint64_t generation) {
    for (auto it = _pipelines.begin(); it != _pipelines.end();) {
        if (it->first.shaderGeneration < generation)
            it = _pipelines.erase(it);
        else
            ++it;
    }
}

} // namespace KE
