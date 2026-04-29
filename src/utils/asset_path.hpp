#ifndef _ASSET_PATH_HPP_
#define _ASSET_PATH_HPP_

#include <cstdlib>
#include <filesystem>
#include <string>

namespace KE {

inline std::string getAssetRoot() {
    if (const char* envRoot = std::getenv("KANGENGINE_ASSETS_ROOT")) {
        if (envRoot[0] != '\0')
            return std::string(envRoot);
    }

#ifdef KANGENGINE_ASSETS_ROOT
    return std::string(KANGENGINE_ASSETS_ROOT);
#else
    return "./assets";
#endif
}

/// @brief Get full path to an asset file
/// @param relativePath Path relative to assets directory (e.g.,
/// "fonts/font.ttf")
/// @return Full absolute path to the asset
inline std::string getAssetPath(const std::string& relativePath) {
    std::filesystem::path root(getAssetRoot());
    std::filesystem::path full = root / relativePath;
    return full.string();
}

/// @brief Check if an asset file exists
/// @param relativePath Path relative to assets directory
/// @return true if file exists, false otherwise
inline bool assetExists(const std::string& relativePath) {
    return std::filesystem::exists(getAssetPath(relativePath));
}

} // namespace KE

#endif // _ASSET_PATH_HPP_
