#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace KE {
struct GpuResourceUsage {
    unsigned index = 0;
    std::string name, uuid, status;
    std::optional<unsigned> utilizationPercent;
    std::optional<uint64_t> memoryUsedBytes, memoryTotalBytes;
    bool memoryIncludesReserved = false; // Legacy NVML fallback only.
    std::optional<uint64_t> processMemoryBytes;
    std::string processMemoryStatus;
};
struct ResourceUsage {
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point sampledAt{};
    std::string cpuModel, os, status, gpuStatus, gpuDriver;
    unsigned logicalCpuCount = 0;
    // Process CPU: 100% is one logical CPU, so multithreaded use may exceed
    // 100%.
    std::optional<double> processCpuPercent, systemCpuPercent;
    std::optional<uint64_t> processRssBytes, ramTotalBytes, ramAvailableBytes;
    std::vector<GpuResourceUsage> gpus;
};

// Demand-driven worker: no GL calls, subprocesses or sampling on the UI thread.
// Call requestSample while visible. Requests coalesce to at most one per
// second; when requests stop, the worker finishes any queued sample then
// sleeps.
class ResourceMonitor {
  public:
    ResourceMonitor();
    ~ResourceMonitor();
    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;
    void requestSample();
    ResourceUsage snapshot() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace KE
