#include "engine/graphics/renderer/shader_library.hpp"
#include "engine/graphics/backend/base/shader_preprocessor.hpp"

#include <stdexcept>

namespace KE {

std::string ShaderLibrary::canonicalKey(const std::string& path) {
    if (path.empty())
        throw std::invalid_argument("ShaderLibrary requires a shader path");
    return std::filesystem::weakly_canonical(path).string();
}

ShaderLibrary::Entry ShaderLibrary::loadEntry(const std::string& path) {
    Backend::ShaderSourceLoadResult loaded =
        Backend::loadShaderSourceWithDependencies(path);
    Entry entry;
    entry.source = std::move(loaded.source);
    entry.dependencies = std::move(loaded.dependencies);
    entry.writeTimes.reserve(entry.dependencies.size());
    for (const std::string& dependency : entry.dependencies)
        entry.writeTimes.push_back(
            std::filesystem::last_write_time(dependency));
    return entry;
}

bool ShaderLibrary::changed(const Entry& entry) {
    if (entry.dependencies.size() != entry.writeTimes.size())
        return true;
    for (size_t i = 0; i < entry.dependencies.size(); ++i) {
        std::error_code error;
        const auto writeTime =
            std::filesystem::last_write_time(entry.dependencies[i], error);
        if (error || writeTime != entry.writeTimes[i])
            return true;
    }
    return false;
}

const std::string& ShaderLibrary::load(const std::string& path) {
    const std::string key = canonicalKey(path);
    std::lock_guard<std::mutex> lock(_mutex);
    const auto found = _entries.find(key);
    if (found != _entries.end())
        return found->second.source;
    const auto [it, inserted] = _entries.emplace(key, loadEntry(key));
    (void)inserted;
    return it->second.source;
}

bool ShaderLibrary::reloadChanged() {
    std::lock_guard<std::mutex> lock(_mutex);
    bool reloaded = false;
    for (auto& [path, entry] : _entries) {
        if (!changed(entry))
            continue;
        entry = loadEntry(path);
        reloaded = true;
    }
    if (reloaded)
        ++_generation;
    return reloaded;
}

void ShaderLibrary::invalidate() {
    std::lock_guard<std::mutex> lock(_mutex);
    _entries.clear();
    ++_generation;
}

uint64_t ShaderLibrary::generation() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _generation;
}

size_t ShaderLibrary::size() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _entries.size();
}

} // namespace KE
