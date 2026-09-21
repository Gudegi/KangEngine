#include "profile_types.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <utility>

namespace KE::Backend {
ProfileContext::ProfileContext(Clock clock, size_t sampleCapacity)
    : _clock(std::move(clock)), _sampleCapacity(sampleCapacity),
      _owner(std::this_thread::get_id()) {
    if (!_clock)
        _clock = [] {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        };
    if (sampleCapacity == 0)
        throw std::invalid_argument("profile sample capacity must be positive");
}

ProfileContext::CpuScope::CpuScope(CpuScope&& other) noexcept
    : _context(std::exchange(other._context, nullptr)),
      _generation(other._generation), _index(other._index),
      _start(other._start) {}
ProfileContext::CpuScope::~CpuScope() noexcept { end(); }
void ProfileContext::CpuScope::end() noexcept {
    if (_context) {
        _context->endScope(_generation, _index, _start);
        _context = nullptr;
    }
}

ProfileContext::CpuScope ProfileContext::cpuScope(std::string_view path) {
    if (!active())
        return {};
    if (_frame->samples.size() >= _sampleCapacity) {
        ++_frame->droppedSamples;
        return {};
    }
    const double start = _clock();
    const size_t index = _frame->samples.size();
    ProfileSample sample;
    sample.sampleId = index;
    sample.parentSampleId = _parent;
    sample.path = path;
    _frame->samples.push_back(std::move(sample));
    _parent = index;
    return CpuScope(this, _generation, index, start);
}

void ProfileContext::endScope(uint64_t generation, size_t index,
                              double start) noexcept {
    if (!active() || generation != _generation ||
        index >= _frame->samples.size())
        return;
    auto& sample = _frame->samples[index];
    if (sample.status != ProfileSampleStatus::Pending)
        return;
    if (_parent != index) {
        // A parent closed early: invalidate its open descendants as well.
        while (_parent && *_parent != index) {
            auto& child = _frame->samples[*_parent];
            child.status = ProfileSampleStatus::Invalid;
            _parent = child.parentSampleId;
        }
        sample.status = ProfileSampleStatus::Invalid;
    } else {
        try {
            const double elapsed = _clock() - start;
            if (std::isfinite(elapsed) && elapsed >= 0) {
                sample.durationMs = elapsed;
                sample.status = ProfileSampleStatus::Ready;
            } else {
                sample.status = ProfileSampleStatus::Invalid;
            }
        } catch (...) {
            sample.status = ProfileSampleStatus::Invalid;
        }
    }
    _parent = sample.parentSampleId;
}

void ProfileContext::beginFrame(uint64_t captureId, uint64_t frameIndex,
                                BackendType backend) {
    if (_owner != std::this_thread::get_id())
        throw std::logic_error("profile capture must run on the owning thread");
    if (_frame)
        throw std::logic_error("a profile frame is already active");
    _frame = std::make_unique<FrameProfile>();
    ++_generation;
    _parent.reset();
    _frame->captureId = captureId;
    _frame->frameIndex = frameIndex;
    _frame->backend = backend;
    _frame->metadata = {{"timing_coverage", "cpu_only"},
                        {"counter_coverage", "rhi_draws_and_device_uploads"},
                        {"counter_exclusions",
                         "imgui,native_blit,cubemap_upload,mapped_cuda_writes"},
                        {"outside_frame_uploads", "excluded"}};
    _frame->metadata["gpu_timing_exclusions"] =
        "frame,ui,native_blit,cuda,worker_recording";
    _frame->metadata.insert(_deviceMetadata.begin(), _deviceMetadata.end());
}

std::shared_ptr<const FrameProfile> ProfileContext::endFrame(bool aborted) {
    if (!active())
        throw std::logic_error("no profile frame active on this thread");
    for (auto& sample : _frame->samples)
        if ((sample.domain == ProfileTimingDomain::Cpu &&
             sample.status == ProfileSampleStatus::Pending) ||
            aborted) {
            sample.status = ProfileSampleStatus::Invalid;
            sample.durationMs.reset();
        }
    if (std::none_of(_frame->samples.begin(), _frame->samples.end(),
                     [](const auto& sample) {
                         return sample.domain == ProfileTimingDomain::Gpu &&
                                sample.path == "frame";
                     })) {
        ProfileSample gpu;
        gpu.sampleId = _frame->samples.size();
        gpu.path = "frame";
        gpu.domain = ProfileTimingDomain::Gpu;
        gpu.status = ProfileSampleStatus::Unsupported;
        _frame->samples.push_back(std::move(gpu));
    }
    _frame->finalized = true;
    for (const auto& sample : _frame->samples)
        if (sample.status == ProfileSampleStatus::Pending)
            _frame->finalized = false;
    _frame->metadata["aborted"] = aborted ? "true" : "false";
    _parent.reset();
    return std::shared_ptr<const FrameProfile>(std::move(_frame));
}
void ProfileContext::cancelFrame() noexcept {
    _frame.reset();
    _parent.reset();
}
std::optional<ProfileQuery>
ProfileContext::gpuSample(std::string_view path, ProfileSampleStatus status) {
    if (!active() || path.empty())
        return std::nullopt;
    if (_frame->samples.size() >= _sampleCapacity) {
        ++_frame->droppedSamples;
        return std::nullopt;
    }
    ProfileSample sample;
    sample.sampleId = _frame->samples.size();
    sample.path = path;
    sample.domain = ProfileTimingDomain::Gpu;
    sample.status = status;
    _frame->samples.push_back(std::move(sample));
    if (status == ProfileSampleStatus::Pending)
        _frame->metadata["timing_coverage"] = "cpu_and_gpu_scopes";
    return ProfileQuery{_frame->captureId, _frame->frameIndex,
                        _frame->samples.back().sampleId};
}
void ProfileContext::setMetadata(std::string key, std::string value) {
    if (active())
        _frame->metadata[std::move(key)] = std::move(value);
}
void ProfileContext::setDeviceMetadata(
    std::map<std::string, std::string> metadata) {
    _deviceMetadata = std::move(metadata);
}
void ProfileContext::recordDraw(PrimitiveTopology topology, uint64_t count,
                                uint64_t instances, bool indexed) {
    if (!active() || !count || !instances)
        return;
    auto& c = _frame->counters;
    ++c.drawCalls;
    c.indexedDrawCalls += indexed;
    c.instances += instances;
    if (topology == PrimitiveTopology::TriangleList)
        c.triangles += count / 3 * instances;
    if (topology == PrimitiveTopology::TriangleStrip && count > 2)
        c.triangles += (count - 2) * instances;
}
void ProfileContext::recordBufferUpload(uint64_t bytes) {
    if (active())
        _frame->counters.bufferUploadBytes += bytes;
}
void ProfileContext::recordTextureUpload(uint64_t bytes) {
    if (active())
        _frame->counters.textureUploadBytes += bytes;
}
void ProfileContext::recordExternalCopy(uint64_t bytes) {
    if (active())
        _frame->counters.externalBufferBytes += bytes;
}
void ProfileContext::recordBufferAllocation(uint64_t bytes) {
    if (active()) {
        ++_frame->counters.bufferAllocations;
        _frame->counters.bufferAllocatedBytes += bytes;
    }
}
} // namespace KE::Backend
