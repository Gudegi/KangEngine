#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace KE {

// Shared source/preprocess cache for renderer-owned shaders. File reads and
// #import expansion happen once per canonical root path. A global generation
// advances when any root or imported dependency is reloaded.
class ShaderLibrary {
  public:
    const std::string& load(const std::string& path);
    bool reloadChanged();
    void invalidate();
    uint64_t generation() const;
    size_t size() const;

  private:
    struct Entry {
        std::string source;
        std::vector<std::string> dependencies;
        std::vector<std::filesystem::file_time_type> writeTimes;
    };

    static std::string canonicalKey(const std::string& path);
    static Entry loadEntry(const std::string& path);
    static bool changed(const Entry& entry);

    mutable std::mutex _mutex;
    std::unordered_map<std::string, Entry> _entries;
    uint64_t _generation = 1;
};

} // namespace KE
