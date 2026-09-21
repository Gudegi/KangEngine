#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/renderer/renderer_profiler.hpp"
#include <iostream>
#include <stdexcept>

using namespace KE;
using namespace KE::Backend;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
class UnsupportedEncoder : public CommandEncoder {
  public:
    std::unique_ptr<RenderPassEncoder> beginRenderPass(RenderTarget*) override {
        return {};
    }
    std::unique_ptr<CommandBuffer> finish() override { return {}; }
};
class PassOnlyEncoder : public UnsupportedEncoder {
  public:
    std::vector<ProfileQuery> queries;
    std::unique_ptr<RenderPassEncoder>
    beginRenderPass(RenderTarget*, const ProfilePassOptions& options) override {
        if (auto query = options.context->gpuSample(
                options.path, ProfileSampleStatus::Pending))
            queries.push_back(*query);
        return {};
    }
};
int main() {
    auto context = std::make_shared<ProfileContext>();
    RendererProfiler profiler;
    profiler.bind(context);
    profiler.setEnabled(true);
    UnsupportedEncoder unsupported;
    PassOnlyEncoder passOnly;
    {
        auto frame = profiler.frame(20, BackendType::WebGPU);
        CommandEncoder& encoder = unsupported;
        encoder.beginRenderPass(nullptr, {context.get(), "render/unsupported"});
        context->recordBufferUpload(16);
    }
    require(profiler.latestFrameProfile()->finalized &&
                profiler.latestFrameProfile()->samples[1].status ==
                    ProfileSampleStatus::Unsupported &&
                profiler.latestFrameProfile()->counters.bufferUploadBytes == 16,
            "unsupported GPU path lost CPU counters");
    {
        auto frame = profiler.frame(21, BackendType::WebGPU);
        passOnly.beginRenderPass(nullptr, {context.get(), "render/first"});
        passOnly.beginRenderPass(nullptr, {context.get(), "render/second"});
    }
    const auto pending = profiler.latestFrameProfile();
    require(!pending->finalized && pending->revision == 0,
            "GPU results required synchronously");
    { auto frame = profiler.frame(22, BackendType::WebGPU); }
    profiler.applyGpuResults(
        {{passOnly.queries[0], ProfileSampleStatus::Ready, 1.5, 24}});
    auto partial = profiler.frameProfileHistory()[1];
    require(!partial->finalized && partial->revision == 1 &&
                !partial->gpuLatencyFrames,
            "partial result finalized prematurely");
    profiler.applyGpuResults(
        {{passOnly.queries[1], ProfileSampleStatus::Ready, 2.5, 28}});
    auto complete = profiler.frameProfileHistory()[1];
    require(complete->finalized && complete->revision == 2 &&
                complete->gpuLatencyFrames == 7,
            "delayed final result revision/latency mismatch");
    require(profiler.latestFrameProfile()->frameIndex == 22 &&
                !pending->finalized && !partial->finalized,
            "snapshot mutated or latest frame moved backward");
    for (int i = 23; i < 270; ++i) {
        auto frame = profiler.frame(i, BackendType::WebGPU);
    }
    profiler.applyGpuResults(
        {{passOnly.queries[0], ProfileSampleStatus::Ready, 99.0, 300}});
    require(profiler.frameProfileHistory().size() ==
                    RendererProfiler::HistoryCapacity &&
                complete->samples[1].durationMs == 1.5,
            "late result resurrected evicted frame");
    std::cout << "PASS: unsupported/pass-only API, partial results, immutable "
                 "revisions and eviction\n";
}
