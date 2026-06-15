#include "prim_path.hpp"

#include <algorithm>
#include <cctype>

namespace KE {
namespace Scene {
namespace PrimPath {

std::string safeToken(std::string value, const std::string& fallback) {
    for (char& ch : value) {
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'))
            ch = '_';
    }
    value.erase(std::remove(value.begin(), value.end(), '\0'), value.end());
    while (!value.empty() && value.front() == '_')
        value.erase(value.begin());
    while (!value.empty() && value.back() == '_')
        value.pop_back();
    return value.empty() ? fallback : value;
}

std::string normalize(std::string path) {
    while (path.size() > 1 && path.back() == '/')
        path.pop_back();
    return path;
}

std::string join(const std::string& parent, const std::string& child) {
    if (parent.empty())
        return child;
    if (parent == "/")
        return "/" + child;
    return parent + "/" + child;
}

std::string makeUniqueChildName(
    std::unordered_map<std::string, int>& siblingNameCounts,
    const std::string& baseName) {
    int& count = siblingNameCounts[baseName];
    if (count++ == 0)
        return baseName;
    return baseName + "_" + std::to_string(count - 1);
}

} // namespace PrimPath
} // namespace Scene
} // namespace KE

