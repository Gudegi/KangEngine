#pragma once
#include "engine/graphics/backend/base/profile_types.hpp"
#include <deque>

namespace KE {
namespace Backend {
class GraphicsDevice;
}
class RendererProfiler {
  public:
    using Snapshot = std::shared_ptr<const Backend::FrameProfile>;
    class FrameScope {
      public:
        explicit FrameScope(RendererProfiler* profiler);
        FrameScope(const FrameScope&) = delete;
        FrameScope& operator=(const FrameScope&) = delete;
        ~FrameScope() noexcept;

      private:
        RendererProfiler* _profiler;
        int _exceptions;
        Backend::ProfileContext::CpuScope _cpu;
    };
    void bind(std::shared_ptr<Backend::ProfileContext> context,
              Backend::GraphicsDevice* device = nullptr);
    void
    applyGpuResults(const std::vector<Backend::ProfileQueryResult>& results);
    void setEnabled(bool enabled) { _requested = enabled; }
    bool enabled() const { return _requested; }
    FrameScope frame(uint64_t index, Backend::BackendType backend);
    Snapshot latestFrameProfile() const;
    std::vector<Snapshot> frameProfileHistory() const;
    static constexpr size_t HistoryCapacity = 240;
    Backend::ProfileSummary
    profileSummary(size_t maxFrames = HistoryCapacity) const;
    void exportJson(const std::string& path,
                    size_t maxFrames = HistoryCapacity) const;

  private:
    void finishFrame(bool aborted);
    std::shared_ptr<Backend::ProfileContext> _context;
    Backend::GraphicsDevice* _device = nullptr;
    std::deque<Snapshot> _history;
    bool _requested = false, _enabled = false;
    uint64_t _captureId = 0;
};
} // namespace KE
