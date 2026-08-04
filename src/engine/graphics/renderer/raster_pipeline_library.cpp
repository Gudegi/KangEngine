#include "engine/graphics/renderer/raster_pipeline_library.hpp"

#include <stdexcept>

namespace KE {

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
RasterPipelineLibrary::get(RasterPipelineKey key) const {
    const auto it = _pipelines.find(key);
    if (it == _pipelines.end())
        throw std::logic_error(
            "RasterPipelineLibrary has no pipeline for the requested key");
    return it->second.get();
}

} // namespace KE
