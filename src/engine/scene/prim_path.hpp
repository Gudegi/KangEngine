#ifndef _PRIM_PATH_HPP_
#define _PRIM_PATH_HPP_

#include <string>
#include <unordered_map>

namespace KE {
namespace Scene {

// Helpers for turning imported names into stable Scene prim paths. These keep
// path formatting rules in one place instead of scattering string cleanup
// through asset loaders, bridges, and examples.
namespace PrimPath {

// Make one path segment safe to use as a prim name.
std::string safeToken(std::string value, const std::string& fallback);

// Remove trailing slashes while preserving the root path "/".
std::string normalize(std::string path);

// Join parent and child path segments without producing duplicate slashes.
std::string join(const std::string& parent, const std::string& child);

// Append _N when siblings sanitize to the same prim name.
std::string makeUniqueChildName(
    std::unordered_map<std::string, int>& siblingNameCounts,
    const std::string& baseName);

} // namespace PrimPath
} // namespace Scene
} // namespace KE

#endif
