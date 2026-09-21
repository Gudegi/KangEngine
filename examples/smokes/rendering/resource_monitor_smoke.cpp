#include "engine/core/diagnostics/resource_monitor.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <cmath>

using namespace KE;
using Clock = std::chrono::steady_clock;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static ResourceUsage waitFor(ResourceMonitor& monitor, uint64_t sequence) {
    const auto deadline = Clock::now() + std::chrono::seconds(8);
    while (Clock::now() < deadline) {
        auto snapshot = monitor.snapshot();
        if (snapshot.sequence >= sequence)
            return snapshot;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("resource sample timeout");
}
int main(int argc, char** argv) {
    ResourceMonitor monitor;
    require(monitor.snapshot().sequence == 0, "monitor sampled without demand");
    monitor.requestSample();
    const auto first = waitFor(monitor, 1);
    require(!first.processCpuPercent && !first.systemCpuPercent,
            "first CPU delta is not available yet");
#ifdef __linux__
    require(first.logicalCpuCount > 0 && !first.os.empty(),
            "missing Linux hardware info");
    require(first.processRssBytes && *first.processRssBytes > 0 &&
                first.ramTotalBytes && *first.ramTotalBytes > 0 &&
                first.ramAvailableBytes &&
                *first.ramAvailableBytes <= *first.ramTotalBytes,
            "invalid Linux memory readings");
#endif
    for (int i = 0; i < 1000; ++i)
        monitor.requestSample();
    require(monitor.snapshot().sequence == first.sequence,
            "sampling not rate limited");
    const auto busyUntil = Clock::now() + std::chrono::milliseconds(40);
    while (Clock::now() < busyUntil) {
    }
    const auto second = waitFor(monitor, 2);
    require(second.sequence == 2 && first.sequence == 1,
            "snapshot changed or requests did not coalesce");
#ifdef __linux__
    require(second.processCpuPercent &&
                std::isfinite(*second.processCpuPercent) &&
                *second.processCpuPercent > 0,
            "process CPU clock did not measure work");
    require(second.systemCpuPercent && *second.systemCpuPercent >= 0 &&
                *second.systemCpuPercent <= 100,
            "invalid system CPU usage");
#endif
    if (argc > 1 && std::string(argv[1]) == "--require-gpu")
        require(!second.gpus.empty() && second.gpuStatus.empty(),
                "NVIDIA device unavailable");
    if (argc > 1 && std::string(argv[1]) == "--no-gpu")
        require(second.gpus.empty() && !second.gpuStatus.empty(),
                "missing NVIDIA fallback reason");
    for (const auto& gpu : second.gpus) {
        if (gpu.processMemoryBytes && gpu.memoryTotalBytes)
            require(*gpu.processMemoryBytes <= *gpu.memoryTotalBytes,
                    "invalid process GPU memory");
        if (!gpu.processMemoryBytes)
            require(!gpu.processMemoryStatus.empty(),
                    "missing process memory unavailable reason");
        if (gpu.utilizationPercent)
            require(*gpu.utilizationPercent <= 100, "invalid GPU percentage");
        if (gpu.memoryUsedBytes)
            require(gpu.memoryTotalBytes &&
                        *gpu.memoryUsedBytes <= *gpu.memoryTotalBytes,
                    "invalid GPU memory");
        if (argc > 1 && std::string(argv[1]) == "--require-gpu")
            require(gpu.utilizationPercent && gpu.memoryUsedBytes &&
                        !gpu.uuid.empty(),
                    "missing NVIDIA counters");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    require(monitor.snapshot().sequence == 2,
            "monitor kept sampling without requests");
    std::cout << "PASS: lazy/rate-limited resource monitor, CPU/RAM, immutable "
                 "snapshots, GPU fallback\n"
              << "CPU: " << second.cpuModel << " / " << second.logicalCpuCount
              << " logical CPUs\n"
              << "NVIDIA devices: " << second.gpus.size() << " / "
              << second.gpuStatus << '\n';
}
