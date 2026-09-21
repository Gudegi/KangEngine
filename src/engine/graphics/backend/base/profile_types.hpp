#pragma once

#include "rhi_types.hpp"
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace KE::Backend {
enum class BackendType;
enum class ProfileTimingDomain { Cpu, Gpu };
enum class ProfileSampleStatus {
    Pending,
    Ready,
    Unsupported,
    CapacityExceeded,
    Dropped,
    Invalid,
    DeviceLost
};

struct ProfileSample {
    uint64_t sampleId = 0;
    std::optional<uint64_t> parentSampleId;
    std::string path;
    ProfileTimingDomain domain = ProfileTimingDomain::Cpu;
    std::optional<double> durationMs;
    ProfileSampleStatus status = ProfileSampleStatus::Pending;
    bool available() const { return status == ProfileSampleStatus::Ready; }
};

struct RenderCounters {
    uint64_t drawCalls = 0, indexedDrawCalls = 0, instances = 0, triangles = 0;
    uint64_t bufferUploadBytes = 0, textureUploadBytes = 0,
             externalBufferBytes = 0;
    uint64_t bufferAllocations = 0, bufferAllocatedBytes = 0;
};

struct FrameProfile {
    uint64_t captureId = 0, frameIndex = 0;
    uint32_t revision = 0;
    bool finalized = true;
    BackendType backend{};
    std::vector<ProfileSample> samples;
    RenderCounters counters;
    std::optional<uint32_t> gpuLatencyFrames;
    uint64_t droppedSamples = 0;
    std::map<std::string, std::string> metadata;
};

// Statistics of per-frame inclusive sums for one (domain, path). Missing
// scopes are not zero samples; incomplete scope frames are excluded.
struct ScopeProfileSummary {
    std::string path;
    ProfileTimingDomain domain = ProfileTimingDomain::Cpu;
    uint64_t sampleCount = 0;
    uint64_t readyFrames = 0, pendingFrames = 0, unavailableFrames = 0;
    std::optional<double> meanMs, medianMs, p95Ms, maxMs;
};
struct ProfileSummary {
    uint64_t captureId = 0, frameCount = 0;
    std::optional<uint64_t> firstFrameIndex, lastFrameIndex;
    std::vector<ScopeProfileSummary> scopes;
};

struct ProfilerCapabilities {
    bool cpuScopes = true;
    bool gpuTimestamps = false;
    bool externalTimestamps = false;
    bool passTimestamps = false;
    bool commandTimestamps = false;
    bool inPassTimestamps = false;
    bool debugGroups = false;
    uint32_t timestampCapacity = 0;
};

// Logical sample identity, never a native query object.
struct ProfileQuery {
    uint64_t captureId = 0, frameIndex = 0, sampleId = 0;
};
struct ProfileQueryResult {
    ProfileQuery query;
    ProfileSampleStatus status = ProfileSampleStatus::Invalid;
    std::optional<double> durationMs;
    uint64_t resolvedFrameIndex = 0;
};

class ProfileContext;
struct ProfilePassOptions {
    ProfileContext* context = nullptr;
    std::string_view path;
};

// Backend-owned immediate scope for native rendering outside command buffers.
// Destroy on the owning render thread/context, before the capture frame ends.
class ExternalProfileScope {
  public:
    virtual ~ExternalProfileScope() = default;
};

// Shared by one device and its resources. Recording never calls a GPU API.
// Capture/snapshot access is confined to the owning render thread.
class ProfileContext {
  public:
    using Clock = std::function<double()>;
    class CpuScope {
      public:
        CpuScope() = default;
        CpuScope(const CpuScope&) = delete;
        CpuScope& operator=(const CpuScope&) = delete;
        CpuScope(CpuScope&& other) noexcept;
        ~CpuScope() noexcept;
        void end() noexcept;

      private:
        friend class ProfileContext;
        CpuScope(ProfileContext* context, uint64_t generation, size_t index,
                 double start)
            : _context(context), _generation(generation), _index(index),
              _start(start) {}
        ProfileContext* _context = nullptr;
        uint64_t _generation = 0;
        size_t _index = 0;
        double _start = 0;
    };

    explicit ProfileContext(Clock clock = {}, size_t sampleCapacity = 4096);
    CpuScope cpuScope(std::string_view path);
    std::optional<ProfileQuery> gpuSample(std::string_view path,
                                          ProfileSampleStatus status);
    bool active() const {
        return _owner == std::this_thread::get_id() && _frame;
    }
    void beginFrame(uint64_t captureId, uint64_t frameIndex,
                    BackendType backend);
    std::shared_ptr<const FrameProfile> endFrame(bool aborted = false);
    void cancelFrame() noexcept;
    void setMetadata(std::string key, std::string value);
    void setDeviceMetadata(std::map<std::string, std::string> metadata);
    const std::map<std::string, std::string>& deviceMetadata() const {
        return _deviceMetadata;
    }
    void recordDraw(PrimitiveTopology topology, uint64_t count,
                    uint64_t instances, bool indexed);
    void recordBufferUpload(uint64_t bytes);
    void recordTextureUpload(uint64_t bytes);
    void recordExternalCopy(uint64_t bytes);
    void recordBufferAllocation(uint64_t bytes);

  private:
    void endScope(uint64_t generation, size_t index, double start) noexcept;
    Clock _clock;
    size_t _sampleCapacity;
    std::thread::id _owner;
    uint64_t _generation = 0;
    std::optional<size_t> _parent;
    std::unique_ptr<FrameProfile> _frame;
    std::map<std::string, std::string> _deviceMetadata;
};
} // namespace KE::Backend
