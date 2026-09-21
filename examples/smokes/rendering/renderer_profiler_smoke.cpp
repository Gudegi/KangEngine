#include "engine/graphics/renderer/renderer_profiler.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace KE;
using namespace KE::Backend;
static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

static const ScopeProfileSummary& findScope(const ProfileSummary& summary,
                                            const std::string& path,
                                            ProfileTimingDomain domain) {
    for (const auto& scope : summary.scopes)
        if (scope.path == path && scope.domain == domain)
            return scope;
    throw std::runtime_error("missing summary scope " + path);
}
static void checkSummary() {
    double now = 0;
    auto context = std::make_shared<ProfileContext>([&] { return now; });
    RendererProfiler profiler;
    profiler.bind(context);
    require(profiler.profileSummary().frameCount == 0 &&
                !profiler.profileSummary().firstFrameIndex,
            "empty summary");
    bool rejected = false;
    try {
        profiler.profileSummary(0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid summary window accepted");
    profiler.setEnabled(true);
    std::vector<ProfileQueryResult> firstResults, secondResults;
    for (uint64_t i = 1; i <= 4; ++i) {
        auto frame = profiler.frame(i, BackendType::OpenGL);
        {
            auto scope = context->cpuScope("work");
            now += i;
        }
        {
            auto scope = context->cpuScope("work");
            now += 2 * i;
        }
        if (i == 1) {
            auto scope = context->cpuScope("rare");
            now += 7;
        }
        auto first = context->gpuSample("work", ProfileSampleStatus::Pending);
        auto second = context->gpuSample("work", ProfileSampleStatus::Pending);
        firstResults.push_back(
            {*first, ProfileSampleStatus::Ready, double(i), 10});
        secondResults.push_back(
            {*second,
             i == 4 ? ProfileSampleStatus::Dropped : ProfileSampleStatus::Ready,
             double(2 * i), 11});
    }
    const auto before = profiler.profileSummary();
    const auto& cpu = findScope(before, "work", ProfileTimingDomain::Cpu);
    require(cpu.readyFrames == 4 && cpu.sampleCount == 8 && cpu.meanMs == 7.5 &&
                cpu.medianMs == 7.5 && cpu.p95Ms == 12 && cpu.maxMs == 12,
            "repeated scopes were not summed per frame");
    require(
        findScope(before, "rare", ProfileTimingDomain::Cpu).meanMs == 7 &&
            findScope(before, "rare", ProfileTimingDomain::Cpu).readyFrames ==
                1,
        "missing scopes became zero samples");
    const auto window = profiler.profileSummary(2);
    require(window.firstFrameIndex == 3 && window.lastFrameIndex == 4 &&
                findScope(window, "work", ProfileTimingDomain::Cpu).meanMs ==
                    10.5,
            "summary window mismatch");
    profiler.applyGpuResults(firstResults);
    const auto partial = profiler.profileSummary();
    require(
        findScope(partial, "work", ProfileTimingDomain::Gpu).pendingFrames ==
                4 &&
            !findScope(partial, "work", ProfileTimingDomain::Gpu).meanMs,
        "partial frame totals polluted GPU summary");
    profiler.applyGpuResults(secondResults);
    const auto complete = profiler.profileSummary();
    const auto& gpu = findScope(complete, "work", ProfileTimingDomain::Gpu);
    require(gpu.readyFrames == 3 && gpu.unavailableFrames == 1 &&
                gpu.pendingFrames == 0 && gpu.meanMs == 6 &&
                gpu.medianMs == 6 && gpu.p95Ms == 9 && gpu.maxMs == 9,
            "GPU revisions/failure statistics mismatch");
    require(findScope(before, "work", ProfileTimingDomain::Gpu).pendingFrames ==
                4,
            "old summary mutated");
    profiler.setEnabled(false);
    { auto frame = profiler.frame(5, BackendType::OpenGL); }
    profiler.setEnabled(true);
    { auto frame = profiler.frame(6, BackendType::OpenGL); }
    const auto restarted = profiler.profileSummary();
    require(restarted.captureId == 2 && restarted.frameCount == 1 &&
                restarted.scopes.size() == 2,
            "summary mixed captures");
}

int main() {
    checkSummary();
    double now = 0;
    int clockCalls = 0;
    auto context = std::make_shared<ProfileContext>(
        [&] {
            ++clockCalls;
            return now;
        },
        4);
    RendererProfiler profiler;
    profiler.bind(context);
    {
        auto frame = profiler.frame(0, BackendType::OpenGL);
        auto scope = context->cpuScope("off");
        context->recordDraw(PrimitiveTopology::TriangleList, 6, 3, true);
    }
    require(clockCalls == 0 && !profiler.latestFrameProfile(),
            "disabled profiler did work");
    profiler.setEnabled(true);
    {
        auto frame = profiler.frame(1, BackendType::OpenGL);
        now = 1;
        auto parent = context->cpuScope("parent");
        now = 2;
        {
            auto child = context->cpuScope("child");
            now = 5;
        }
        now = 7;
        parent.end();
        auto duplicate = context->cpuScope("child");
        { auto overflow = context->cpuScope("overflow"); }
        context->recordDraw(PrimitiveTopology::TriangleList, 6, 3, true);
        context->recordDraw(PrimitiveTopology::TriangleStrip, 5, 2, false);
        context->recordDraw(PrimitiveTopology::LineList, 8, 1, false);
        context->recordDraw(PrimitiveTopology::TriangleList, 0, 100, true);
        context->recordBufferAllocation(128);
        context->recordBufferUpload(64);
        context->recordTextureUpload(16);
        context->recordExternalCopy(32);
        std::thread worker([&] {
            auto ignored = context->cpuScope("worker");
            context->recordBufferUpload(999);
        });
        worker.join();
        now = 10;
    }
    const auto first = profiler.latestFrameProfile();
    require(first->frameIndex == 1 && first->captureId == 1 && first->finalized,
            "frame identity");
    require(first->samples[0].durationMs == 10, "frame time");
    require(first->samples[1].durationMs == 6 &&
                first->samples[2].durationMs == 3,
            "nested timing");
    require(first->samples[2].parentSampleId == 1 &&
                first->samples[3].parentSampleId == 0,
            "parent identity");
    require(first->droppedSamples == 1, "sample capacity not bounded");
    require(
        findScope(profiler.profileSummary(), "child", ProfileTimingDomain::Cpu)
                .unavailableFrames == 1,
        "overflow frame accepted as complete summary");
    const auto& c = first->counters;
    require(c.drawCalls == 3 && c.indexedDrawCalls == 1 && c.instances == 6 &&
                c.triangles == 12,
            "draw counter semantics");
    require(c.bufferUploadBytes == 64 && c.textureUploadBytes == 16 &&
                c.externalBufferBytes == 32 && c.bufferAllocations == 1 &&
                c.bufferAllocatedBytes == 128,
            "upload counter semantics");
    require(first->samples.back().status == ProfileSampleStatus::Unsupported &&
                !first->samples.back().durationMs && !first->gpuLatencyFrames,
            "GPU fallback");

    context->recordBufferUpload(
        999); // Out-of-frame writes do not leak forward.
    for (uint64_t i = 2; i < 250; ++i) {
        auto frame = profiler.frame(i, BackendType::WebGPU);
        now += 1;
    }
    require(profiler.frameProfileHistory().size() ==
                RendererProfiler::HistoryCapacity,
            "history capacity");
    require(profiler.profileSummary().frameCount == 240 &&
                profiler.profileSummary().firstFrameIndex == 10,
            "summary retained evicted frames");
    require(first->frameIndex == 1 && first->counters.bufferUploadBytes == 64,
            "snapshot changed after eviction");
    require(profiler.latestFrameProfile()->counters.bufferUploadBytes == 0,
            "counter reset");
    profiler.setEnabled(false);
    const int previousCalls = clockCalls;
    { auto frame = profiler.frame(250, BackendType::OpenGL); }
    require(clockCalls == previousCalls, "disable did not stop clock reads");
    profiler.setEnabled(true);
    try {
        auto frame = profiler.frame(251, BackendType::OpenGL);
        auto scope = context->cpuScope("exception");
        throw std::runtime_error("expected");
    } catch (const std::runtime_error&) {
    }
    require(profiler.latestFrameProfile()->captureId == 2 &&
                profiler.latestFrameProfile()->samples[0].status ==
                    ProfileSampleStatus::Invalid,
            "aborted frame or capture restart");

    auto other = std::make_shared<ProfileContext>();
    other->beginFrame(1, 0, BackendType::OpenGL);
    auto stale = other->cpuScope("unclosed");
    require(other->endFrame()->samples[0].status ==
                ProfileSampleStatus::Invalid,
            "unclosed scope was accepted");
    other->beginFrame(1, 0, BackendType::OpenGL);
    auto fresh = other->cpuScope("fresh");
    stale.end();
    fresh.end();
    other->recordBufferUpload(17);
    require(other->endFrame()->counters.bufferUploadBytes == 17 &&
                first->counters.bufferUploadBytes == 64,
            "device isolation");
    std::cout << "PASS: CPU scopes, disabled path, counters, history, "
                 "exceptions and isolation\n";
}
