#include "resource_monitor.hpp"
#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <limits>
#ifdef __linux__
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>
#endif
#if defined(__linux__) && defined(KANGENGINE_HAS_NVML_HEADER)
#include <dlfcn.h>
#include <nvml.h>
#endif

namespace KE {
namespace {
using Clock = std::chrono::steady_clock;

std::optional<uint64_t> readKiB(const std::string& text, const char* key) {
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream input(line);
        std::string name, unit;
        uint64_t value;
        if (input >> name && name == key && input >> value >> unit &&
            unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024)
            return value * 1024;
    }
    return {};
}
std::string readFile(const char* path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

class Sampler {
  public:
    ResourceUsage sample() {
        ResourceUsage result;
#ifdef __linux__
        if (_cpuModel.empty()) {
            std::istringstream input(readFile("/proc/cpuinfo"));
            std::string line;
            while (std::getline(input, line)) {
                const auto colon = line.find(':');
                if (colon != std::string::npos &&
                    (line.compare(0, 10, "model name") == 0 ||
                     line.compare(0, 8, "Hardware") == 0)) {
                    auto start = line.find_first_not_of(" \t", colon + 1);
                    if (start != std::string::npos)
                        _cpuModel = line.substr(start);
                    break;
                }
            }
        }
        result.cpuModel = _cpuModel;
        utsname info{};
        if (uname(&info) == 0)
            result.os = std::string(info.sysname) + " " + info.release + " " +
                        info.machine;
        const long count = sysconf(_SC_NPROCESSORS_ONLN);
        if (count > 0)
            result.logicalCpuCount = static_cast<unsigned>(count);
        const auto memory = readFile("/proc/meminfo");
        result.ramTotalBytes = readKiB(memory, "MemTotal:");
        result.ramAvailableBytes = readKiB(memory, "MemAvailable:");
        if (result.ramAvailableBytes && result.ramTotalBytes &&
            *result.ramAvailableBytes > *result.ramTotalBytes)
            result.ramAvailableBytes.reset();
        result.processRssBytes =
            readKiB(readFile("/proc/self/status"), "VmRSS:");

        const auto now = Clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - _previousAt).count();
        const bool continuous =
            _previousAt != Clock::time_point{} && elapsed > 0 && elapsed < 2.5;
        timespec processTime{};
        if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &processTime) == 0) {
            const double cpu = processTime.tv_sec + processTime.tv_nsec * 1e-9;
            if (continuous && _previousProcess && cpu >= *_previousProcess)
                result.processCpuPercent =
                    100 * (cpu - *_previousProcess) / elapsed;
            _previousProcess = cpu;
        } else
            _previousProcess.reset();

        std::istringstream cpuStat(readFile("/proc/stat"));
        std::string label;
        uint64_t user, nice, system, idle, wait, irq, softirq, steal;
        if (cpuStat >> label >> user >> nice >> system >> idle >> wait >> irq >>
                softirq >> steal &&
            label == "cpu") {
            // guest/guest_nice are already included in user/nice.
            const uint64_t total =
                user + nice + system + idle + wait + irq + softirq + steal;
            const uint64_t inactive = idle + wait;
            if (continuous && _previousTotal && total > *_previousTotal &&
                inactive >= _previousIdle) {
                const auto delta = total - *_previousTotal;
                const auto idleDelta = inactive - _previousIdle;
                if (idleDelta <= delta)
                    result.systemCpuPercent =
                        100.0 * (delta - idleDelta) / delta;
            }
            _previousTotal = total;
            _previousIdle = inactive;
        } else
            _previousTotal.reset();
        _previousAt = now;
        if (!result.ramTotalBytes || !result.ramAvailableBytes ||
            !result.processRssBytes)
            result.status = "Some /proc memory counters are unavailable";
#else
        result.status = "CPU/RAM monitoring currently supports Linux only";
#endif
        sampleGpu(result);
        result.sampledAt = Clock::now();
        return result;
    }
    ~Sampler() {
#if defined(__linux__) && defined(KANGENGINE_HAS_NVML_HEADER)
        if (_initialized)
            _shutdown();
        if (_library)
            dlclose(_library);
#endif
    }

  private:
    std::string _cpuModel;
    Clock::time_point _previousAt{};
    std::optional<double> _previousProcess;
    std::optional<uint64_t> _previousTotal;
    uint64_t _previousIdle = 0;
#if defined(__linux__) && defined(KANGENGINE_HAS_NVML_HEADER)
    void* _library = nullptr;
    bool _attempted = false, _initialized = false;
    std::string _gpuError;
    decltype(&nvmlInit_v2) _init = nullptr;
    decltype(&nvmlShutdown) _shutdown = nullptr;
    decltype(&nvmlErrorString) _error = nullptr;
    decltype(&nvmlDeviceGetCount_v2) _count = nullptr;
    decltype(&nvmlDeviceGetHandleByIndex_v2) _handle = nullptr;
    decltype(&nvmlDeviceGetName) _name = nullptr;
    decltype(&nvmlDeviceGetUUID) _uuid = nullptr;
    decltype(&nvmlDeviceGetMemoryInfo) _memory = nullptr;
#ifdef nvmlMemory_v2
    decltype(&nvmlDeviceGetMemoryInfo_v2) _memoryV2 = nullptr;
#endif
    decltype(&nvmlDeviceGetUtilizationRates) _utilization = nullptr;
    decltype(&nvmlSystemGetDriverVersion) _driver = nullptr;
    using ProcessQuery = nvmlReturn_t (*)(nvmlDevice_t, unsigned*,
                                          nvmlProcessInfo_t*);
    ProcessQuery _graphicsProcesses = nullptr, _computeProcesses = nullptr;
    template <typename T> bool load(T& target, const char* name) {
        target = reinterpret_cast<T>(dlsym(_library, name));
        return target != nullptr;
    }
    void initializeGpu() {
        if (_attempted)
            return;
        _attempted = true;
        _library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!_library) {
            _gpuError = "NVIDIA monitoring library unavailable";
            return;
        }
        if (!load(_init, "nvmlInit_v2") || !load(_shutdown, "nvmlShutdown") ||
            !load(_error, "nvmlErrorString") ||
            !load(_count, "nvmlDeviceGetCount_v2") ||
            !load(_handle, "nvmlDeviceGetHandleByIndex_v2") ||
            !load(_name, "nvmlDeviceGetName") ||
            !load(_uuid, "nvmlDeviceGetUUID") ||
            !load(_memory, "nvmlDeviceGetMemoryInfo") ||
            !load(_utilization, "nvmlDeviceGetUtilizationRates") ||
            !load(_driver, "nvmlSystemGetDriverVersion")) {
            _gpuError = "NVIDIA monitoring library lacks required functions";
            return;
        }
#ifdef nvmlMemory_v2
        // Optional: older libraries can still provide the legacy counters.
        load(_memoryV2, "nvmlDeviceGetMemoryInfo_v2");
#endif
        const auto status = _init();
        _initialized = status == NVML_SUCCESS;
        if (!_initialized)
            _gpuError = _error(status);
        if (!load(_graphicsProcesses,
                  "nvmlDeviceGetGraphicsRunningProcesses_v3"))
            load(_graphicsProcesses,
                 "nvmlDeviceGetGraphicsRunningProcesses_v2");
        if (!load(_computeProcesses, "nvmlDeviceGetComputeRunningProcesses_v3"))
            load(_computeProcesses, "nvmlDeviceGetComputeRunningProcesses_v2");
    }
    std::optional<uint64_t> processMemory(nvmlDevice_t device,
                                          ProcessQuery query,
                                          std::string& reason) {
        if (!query) {
            reason = "Process memory query unavailable";
            return {};
        }
        unsigned count = 0;
        auto code = query(device, &count, nullptr);
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (code == NVML_SUCCESS && !count)
                return 0;
            if (code != NVML_ERROR_INSUFFICIENT_SIZE && code != NVML_SUCCESS) {
                reason = _error(code);
                return {};
            }
            if (count > 65536) {
                reason = "Process list exceeds monitor capacity";
                return {};
            }
            std::vector<nvmlProcessInfo_t> processes(count + 8);
            count = static_cast<unsigned>(processes.size());
            code = query(device, &count, processes.data());
            if (code == NVML_ERROR_INSUFFICIENT_SIZE)
                continue;
            if (code != NVML_SUCCESS) {
                reason = _error(code);
                return {};
            }
            if (count > processes.size()) {
                reason = "Invalid process count";
                return {};
            }
            uint64_t memory = 0;
            for (unsigned i = 0; i < count; ++i) {
                const auto& process = processes[i];
                if (process.pid != static_cast<unsigned>(getpid()))
                    continue;
                if (process.usedGpuMemory ==
                    static_cast<unsigned long long>(NVML_VALUE_NOT_AVAILABLE)) {
                    reason = "Driver cannot report this process memory";
                    return {};
                }
                memory = std::max<uint64_t>(memory, process.usedGpuMemory);
            }
            return memory;
        }
        reason = "Process list changed during query";
        return {};
    }
    void sampleProcessMemory(nvmlDevice_t device, GpuResourceUsage& gpu) {
        auto graphics =
            processMemory(device, _graphicsProcesses, gpu.processMemoryStatus);
        auto compute =
            processMemory(device, _computeProcesses, gpu.processMemoryStatus);
        // The same PID can appear in both lists. These are whole-process
        // totals, not disjoint graphics/compute allocations; never add them
        // together.
        if (graphics && compute)
            gpu.processMemoryBytes = std::max(*graphics, *compute);
    }
#endif
    void sampleGpu(ResourceUsage& result) {
#if defined(__linux__) && defined(KANGENGINE_HAS_NVML_HEADER)
        initializeGpu();
        if (!_initialized) {
            result.gpuStatus = _gpuError;
            return;
        }
        char driver[96]{};
        if (_driver(driver, sizeof(driver)) == NVML_SUCCESS)
            result.gpuDriver = driver;
        unsigned count = 0;
        const auto status = _count(&count);
        if (status != NVML_SUCCESS) {
            result.gpuStatus = _error(status);
            return;
        }
        if (!count)
            result.gpuStatus = "No NVIDIA devices available";
        for (unsigned index = 0; index < count; ++index) {
            GpuResourceUsage gpu;
            gpu.index = index;
            nvmlDevice_t device{};
            auto code = _handle(index, &device);
            if (code != NVML_SUCCESS) {
                gpu.status = _error(code);
                gpu.processMemoryStatus = gpu.status;
                result.gpus.push_back(std::move(gpu));
                continue;
            }
            char name[128]{}, uuid[128]{};
            if (_name(device, name, sizeof(name)) == NVML_SUCCESS)
                gpu.name = name;
            if (_uuid(device, uuid, sizeof(uuid)) == NVML_SUCCESS)
                gpu.uuid = uuid;
            nvmlMemory_t memory{};
            bool legacyMemory = true;
#ifdef nvmlMemory_v2
            if (_memoryV2) {
                nvmlMemory_v2_t memoryV2{};
                memoryV2.version = nvmlMemory_v2;
                code = _memoryV2(device, &memoryV2);
                legacyMemory = code == NVML_ERROR_NOT_SUPPORTED ||
                               code == NVML_ERROR_FUNCTION_NOT_FOUND ||
                               code == NVML_ERROR_ARGUMENT_VERSION_MISMATCH;
                if (code == NVML_SUCCESS) {
                    // v1.used includes driver reservations. v2 separates
                    // allocated and reserved bytes, as in nvidia-smi.
                    memory.used = memoryV2.used;
                    memory.total = memoryV2.total;
                }
            }
#endif
            if (legacyMemory)
                code = _memory(device, &memory);
            if (code == NVML_SUCCESS && memory.used <= memory.total &&
                memory.total > 0) {
                gpu.memoryUsedBytes = memory.used;
                gpu.memoryTotalBytes = memory.total;
                gpu.memoryIncludesReserved = legacyMemory;
            } else
                gpu.status = "VRAM: " + std::string(code == NVML_SUCCESS
                                                        ? "invalid reading"
                                                        : _error(code));
            nvmlUtilization_t utilization{};
            code = _utilization(device, &utilization);
            if (code == NVML_SUCCESS && utilization.gpu <= 100)
                gpu.utilizationPercent = utilization.gpu;
            else {
                if (!gpu.status.empty())
                    gpu.status += "; ";
                gpu.status +=
                    "GPU usage: " + std::string(code == NVML_SUCCESS
                                                    ? "invalid reading"
                                                    : _error(code));
            }
            sampleProcessMemory(device, gpu);
            result.gpus.push_back(std::move(gpu));
        }
#else
        result.gpuStatus = "NVIDIA monitoring unavailable in this build";
#endif
    }
};
} // namespace

struct ResourceMonitor::Impl {
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool requested = false, stop = false;
    ResourceUsage latest;
    void run() {
        Sampler sampler;
        auto next = Clock::now();
        std::unique_lock<std::mutex> lock(mutex);
        while (!stop) {
            wake.wait(lock, [&] { return stop || requested; });
            if (stop)
                break;
            if (wake.wait_until(lock, next, [&] { return stop; }))
                break;
            requested = false;
            lock.unlock();
            ResourceUsage value;
            try {
                value = sampler.sample();
            } catch (const std::exception& error) {
                value.status = error.what();
                value.sampledAt = Clock::now();
            }
            lock.lock();
            value.sequence = latest.sequence + 1;
            latest = std::move(value);
            next = Clock::now() + std::chrono::seconds(1);
        }
    }
};
ResourceMonitor::ResourceMonitor() : _impl(std::make_unique<Impl>()) {}
ResourceMonitor::~ResourceMonitor() {
    {
        std::lock_guard<std::mutex> lock(_impl->mutex);
        _impl->stop = true;
    }
    _impl->wake.notify_one();
    if (_impl->worker.joinable())
        _impl->worker.join();
}
void ResourceMonitor::requestSample() {
    std::lock_guard<std::mutex> lock(_impl->mutex);
    if (!_impl->worker.joinable())
        _impl->worker = std::thread([this] { _impl->run(); });
    _impl->requested = true;
    _impl->wake.notify_one();
}
ResourceUsage ResourceMonitor::snapshot() const {
    std::lock_guard<std::mutex> lock(_impl->mutex);
    return _impl->latest;
}
} // namespace KE
