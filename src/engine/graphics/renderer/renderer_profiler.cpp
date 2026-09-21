#include "renderer_profiler.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include <exception>
#include <fstream>
#include <iomanip>
#include <locale>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace KE {
RendererProfiler::FrameScope::FrameScope(RendererProfiler* profiler)
    : _profiler(profiler), _exceptions(std::uncaught_exceptions()),
      _cpu(profiler ? profiler->_context->cpuScope("frame")
                    : Backend::ProfileContext::CpuScope{}) {}
RendererProfiler::FrameScope::~FrameScope() noexcept {
    if (!_profiler)
        return;
    _cpu.end();
    try {
        _profiler->finishFrame(std::uncaught_exceptions() > _exceptions);
    } catch (...) {
        _profiler->_context->cancelFrame();
    }
}
void RendererProfiler::bind(std::shared_ptr<Backend::ProfileContext> context,
                            Backend::GraphicsDevice* device) {
    if (_context && _context->active())
        throw std::logic_error("cannot rebind an active profiler");
    _context = std::move(context);
    _device = device;
    _history.clear();
    _enabled = false;
}
RendererProfiler::FrameScope
RendererProfiler::frame(uint64_t index, Backend::BackendType backend) {
    if (!_context)
        throw std::logic_error("profiler has no device");
    if (_context->active())
        throw std::logic_error("nested profiler frame");
    if (_device)
        applyGpuResults(_device->pollProfileResults(index));
    if (_requested && !_enabled)
        ++_captureId;
    _enabled = _requested;
    if (!_enabled)
        return FrameScope(nullptr);
    if (_device)
        _device->prepareProfiler();
    _context->beginFrame(_captureId, index, backend);
    if (_device) {
        const auto capabilities = _device->profilerCapabilities();
        _context->setMetadata("gpu_external_timestamps",
                              capabilities.externalTimestamps ? "true"
                                                              : "false");
        _context->setMetadata("gpu_pass_timestamps",
                              capabilities.passTimestamps ? "true" : "false");
        _context->setMetadata("gpu_timestamp_capacity",
                              std::to_string(capabilities.timestampCapacity));
        _context->setMetadata("gpu_command_timestamps",
                              capabilities.commandTimestamps ? "true"
                                                             : "false");
    }
    return FrameScope(this);
}
void RendererProfiler::applyGpuResults(
    const std::vector<Backend::ProfileQueryResult>& results) {
    if (results.empty())
        return;
    using namespace Backend;
    for (auto& snapshot : _history) {
        std::shared_ptr<FrameProfile> updated;
        uint64_t resolvedFrame = snapshot->frameIndex;
        for (const auto& result : results) {
            const auto& q = result.query;
            if (q.captureId != snapshot->captureId ||
                q.frameIndex != snapshot->frameIndex ||
                q.sampleId >= snapshot->samples.size())
                continue;
            if (snapshot->samples[q.sampleId].domain !=
                    ProfileTimingDomain::Gpu ||
                snapshot->samples[q.sampleId].status !=
                    ProfileSampleStatus::Pending ||
                result.status == ProfileSampleStatus::Pending)
                continue;
            if (!updated)
                updated = std::make_shared<FrameProfile>(*snapshot);
            auto& sample = updated->samples[q.sampleId];
            if (sample.status != ProfileSampleStatus::Pending)
                continue;
            sample.status = result.status;
            sample.durationMs = result.status == ProfileSampleStatus::Ready
                                    ? result.durationMs
                                    : std::nullopt;
            if (sample.status == ProfileSampleStatus::Ready &&
                (!sample.durationMs || !std::isfinite(*sample.durationMs) ||
                 *sample.durationMs < 0)) {
                sample.status = ProfileSampleStatus::Invalid;
                sample.durationMs.reset();
            }
            resolvedFrame = std::max(resolvedFrame, result.resolvedFrameIndex);
        }
        if (!updated)
            continue;
        ++updated->revision;
        updated->finalized = true;
        bool ready = false, failed = false;
        for (const auto& sample : updated->samples) {
            if (sample.domain != ProfileTimingDomain::Gpu)
                continue;
            if (sample.status == ProfileSampleStatus::Pending)
                updated->finalized = false;
            if (sample.status == ProfileSampleStatus::Ready)
                ready = true;
            if (sample.status != ProfileSampleStatus::Ready &&
                sample.status != ProfileSampleStatus::Pending &&
                sample.status != ProfileSampleStatus::Unsupported)
                failed = true;
        }
        if (updated->finalized && ready && !failed)
            updated->gpuLatencyFrames =
                static_cast<uint32_t>(resolvedFrame - updated->frameIndex);
        snapshot = std::move(updated);
    }
}
void RendererProfiler::finishFrame(bool aborted) {
    _history.push_back(_context->endFrame(aborted));
    if (_history.size() > HistoryCapacity)
        _history.pop_front();
}
RendererProfiler::Snapshot RendererProfiler::latestFrameProfile() const {
    return _history.empty() ? nullptr : _history.back();
}
std::vector<RendererProfiler::Snapshot>
RendererProfiler::frameProfileHistory() const {
    return {_history.begin(), _history.end()};
}

Backend::ProfileSummary
RendererProfiler::profileSummary(size_t maxFrames) const {
    using namespace Backend;
    if (!maxFrames || maxFrames > HistoryCapacity)
        throw std::invalid_argument("max_frames must be between 1 and 240");
    ProfileSummary summary;
    if (_history.empty())
        return summary;
    summary.captureId = _history.back()->captureId;
    using Key = std::pair<ProfileTimingDomain, std::string>;
    struct Aggregate {
        ScopeProfileSummary scope;
        std::vector<double> values;
    };
    std::map<Key, Aggregate> aggregates;
    for (auto it = _history.rbegin();
         it != _history.rend() && summary.frameCount < maxFrames; ++it) {
        const auto& frame = **it;
        if (frame.captureId != summary.captureId)
            break;
        if (!summary.lastFrameIndex)
            summary.lastFrameIndex = frame.frameIndex;
        summary.firstFrameIndex = frame.frameIndex;
        ++summary.frameCount;
        struct FrameSum {
            double duration = 0;
            uint64_t samples = 0;
            bool pending = false, unavailable = false;
        };
        std::map<Key, FrameSum> sums;
        for (const auto& sample : frame.samples) {
            auto& sum = sums[{sample.domain, sample.path}];
            ++sum.samples;
            if (sample.status == ProfileSampleStatus::Pending)
                sum.pending = true;
            else if (sample.available() && sample.durationMs &&
                     std::isfinite(*sample.durationMs) &&
                     *sample.durationMs >= 0)
                sum.duration += *sample.durationMs;
            else
                sum.unavailable = true;
        }
        for (const auto& [key, sum] : sums) {
            auto& aggregate = aggregates[key];
            auto& scope = aggregate.scope;
            scope.domain = key.first;
            scope.path = key.second;
            scope.sampleCount += sum.samples;
            // A sample-capacity overflow can omit another occurrence of any
            // path, so no per-frame sum from that frame is trusted.
            if (sum.unavailable || frame.droppedSamples ||
                !std::isfinite(sum.duration))
                ++scope.unavailableFrames;
            else if (sum.pending)
                ++scope.pendingFrames;
            else {
                ++scope.readyFrames;
                aggregate.values.push_back(sum.duration);
            }
        }
    }
    for (auto& [key, aggregate] : aggregates) {
        auto& values = aggregate.values;
        auto& scope = aggregate.scope;
        if (!values.empty()) {
            std::sort(values.begin(), values.end());
            const size_t count = values.size();
            double mean = 0;
            for (double value : values)
                mean += value / count;
            scope.meanMs = mean;
            scope.medianMs =
                count % 2 ? values[count / 2]
                          : values[count / 2 - 1] / 2 + values[count / 2] / 2;
            scope.p95Ms = values[(95 * count + 99) / 100 - 1];
            scope.maxMs = values.back();
        }
        summary.scopes.push_back(std::move(scope));
    }
    return summary;
}

namespace {
void quoted(std::ostream& out, const std::string& text) {
    out << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else
            out << c;
    }
    out << '"';
}
const char* statusName(Backend::ProfileSampleStatus status) {
    using S = Backend::ProfileSampleStatus;
    switch (status) {
    case S::Pending:
        return "pending";
    case S::Ready:
        return "ready";
    case S::Unsupported:
        return "unsupported";
    case S::CapacityExceeded:
        return "capacity_exceeded";
    case S::Dropped:
        return "dropped";
    case S::Invalid:
        return "invalid";
    case S::DeviceLost:
        return "device_lost";
    }
    return "invalid";
}
} // namespace
void RendererProfiler::exportJson(const std::string& path,
                                  size_t maxFrames) const {
    const auto summary = profileSummary(maxFrames);
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open profiler export: " + path);
    out.imbue(std::locale::classic());
    out << std::setprecision(12) << "{\"schema_version\":1,\"frames\":[";
    bool first = true;
    const size_t start =
        _history.size() > maxFrames ? _history.size() - maxFrames : 0;
    for (size_t index = start; index < _history.size(); ++index) {
        const auto& f = _history[index];
        if (!first)
            out << ',';
        first = false;
        const auto& c = f->counters;
        out << "\n{\"capture_id\":" << f->captureId
            << ",\"frame_index\":" << f->frameIndex
            << ",\"revision\":" << f->revision
            << ",\"finalized\":" << (f->finalized ? "true" : "false")
            << ",\"backend\":";
        quoted(out, f->backend == Backend::BackendType::OpenGL   ? "OpenGL"
                    : f->backend == Backend::BackendType::WebGPU ? "WebGPU"
                                                                 : "Vulkan");
        out << ",\"gpu_latency_frames\":";
        if (f->gpuLatencyFrames)
            out << *f->gpuLatencyFrames;
        else
            out << "null";
        out << ",\"dropped_samples\":" << f->droppedSamples
            << ",\"metadata\":{";
        bool firstItem = true;
        for (const auto& [key, value] : f->metadata) {
            if (!firstItem)
                out << ',';
            firstItem = false;
            quoted(out, key);
            out << ':';
            quoted(out, value);
        }
        out << "},\"counters\":{\"draw_calls\":" << c.drawCalls
            << ",\"indexed_draw_calls\":" << c.indexedDrawCalls
            << ",\"instances\":" << c.instances
            << ",\"triangles\":" << c.triangles
            << ",\"buffer_upload_bytes\":" << c.bufferUploadBytes
            << ",\"texture_upload_bytes\":" << c.textureUploadBytes
            << ",\"external_buffer_bytes\":" << c.externalBufferBytes
            << ",\"buffer_allocations\":" << c.bufferAllocations
            << ",\"buffer_allocated_bytes\":" << c.bufferAllocatedBytes
            << "},\"samples\":[";
        firstItem = true;
        for (const auto& s : f->samples) {
            if (!firstItem)
                out << ',';
            firstItem = false;
            out << "{\"sample_id\":" << s.sampleId << ",\"parent_sample_id\":";
            if (s.parentSampleId)
                out << *s.parentSampleId;
            else
                out << "null";
            out << ",\"path\":";
            quoted(out, s.path);
            out << ",\"domain\":";
            quoted(out, s.domain == Backend::ProfileTimingDomain::Cpu ? "cpu"
                                                                      : "gpu");
            out << ",\"duration_ms\":";
            if (s.durationMs)
                out << *s.durationMs;
            else
                out << "null";
            out << ",\"available\":" << (s.available() ? "true" : "false")
                << ",\"status\":";
            quoted(out, statusName(s.status));
            out << '}';
        }
        out << "]}";
    }
    out << "\n],\"summary\":{\"capture_id\":" << summary.captureId
        << ",\"frame_count\":" << summary.frameCount;
    auto optionalNumber = [&](const char* name, const auto& value) {
        out << ",\"" << name << "\":";
        if (value)
            out << *value;
        else
            out << "null";
    };
    optionalNumber("first_frame_index", summary.firstFrameIndex);
    optionalNumber("last_frame_index", summary.lastFrameIndex);
    out << ",\"aggregation\":\"per_frame_inclusive_sum\",\"missing_scopes\":"
           "\"excluded\","
           "\"percentile_method\":\"nearest_rank\",\"scopes\":[";
    bool firstScope = true;
    for (const auto& scope : summary.scopes) {
        if (!firstScope)
            out << ',';
        firstScope = false;
        out << "{\"path\":";
        quoted(out, scope.path);
        out << ",\"domain\":";
        quoted(out, scope.domain == Backend::ProfileTimingDomain::Cpu ? "cpu"
                                                                      : "gpu");
        out << ",\"sample_count\":" << scope.sampleCount
            << ",\"ready_frames\":" << scope.readyFrames
            << ",\"pending_frames\":" << scope.pendingFrames
            << ",\"unavailable_frames\":" << scope.unavailableFrames;
        optionalNumber("mean_ms", scope.meanMs);
        optionalNumber("median_ms", scope.medianMs);
        optionalNumber("p95_ms", scope.p95Ms);
        optionalNumber("max_ms", scope.maxMs);
        out << '}';
    }
    out << "]}}\n";
    out.close();
    if (!out)
        throw std::runtime_error("failed to write profiler export: " + path);
}
} // namespace KE
