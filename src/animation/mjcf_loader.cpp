#include "mjcf_loader.hpp"

#include <fmt/core.h>
#include <tinyxml2.h>

#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace KE {
namespace Animation {

// Resolve mesh directory relative to the MJCF file path
static std::string resolveMeshDir(const std::string& mjcfPath,
                                  const char* meshdir) {
    // Find directory of the MJCF file
    std::string dir;
    auto lastSlash = mjcfPath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        dir = mjcfPath.substr(0, lastSlash + 1);
    }

    if (!meshdir)
        return dir;

    std::string md(meshdir);
    // If meshdir starts with ./, resolve relative to MJCF dir
    if (md.size() >= 2 && md[0] == '.' && md[1] == '/') {
        return dir + md.substr(2);
    }
    // If absolute, return as-is
    if (!md.empty() && md[0] == '/') {
        return md;
    }
    return dir + md;
}

MJCFLoadResult MJCFLoader::load(const std::string& mjcfPath) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(mjcfPath.c_str()) != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error(
            fmt::format("Failed to load MJCF file: {}", mjcfPath));
    }

    auto* root = doc.RootElement();

    // Extract meshdir from <compiler>
    std::string meshDir;
    auto* compiler = root->FirstChildElement("compiler");
    meshDir =
        resolveMeshDir(mjcfPath, compiler ? compiler->Attribute("meshdir")
                                          : nullptr);

    // Build mesh name -> file mapping from <asset>
    std::unordered_map<std::string, std::string> meshNameToFile;
    auto* asset = root->FirstChildElement("asset");
    if (asset) {
        for (auto* mesh = asset->FirstChildElement("mesh"); mesh;
             mesh = mesh->NextSiblingElement("mesh")) {
            const char* name = mesh->Attribute("name");
            const char* file = mesh->Attribute("file");
            if (name && file) {
                meshNameToFile[name] = file;
            }
        }
    }

    // Parse skeleton via BFS
    SkeletonTree skeleton = SkeletonTree::fromMJCF(mjcfPath);

    // BFS again to extract visual geom mesh references per body
    // (Matches skeleton BFS order)
    auto* worldbody = root->FirstChildElement("worldbody");
    auto* bodyRoot = worldbody->FirstChildElement("body");

    std::vector<MJCFMeshInfo> meshInfos;

    struct BFSEntry {
        tinyxml2::XMLElement* element;
        int bodyIndex;
    };

    std::queue<BFSEntry> queue;
    int bodyIndex = 0;
    queue.push({bodyRoot, bodyIndex});

    while (!queue.empty()) {
        auto [element, idx] = queue.front();
        queue.pop();

        const char* bodyName = element->Attribute("name");

        // Find visual geom with mesh reference
        for (auto* geom = element->FirstChildElement("geom"); geom;
             geom = geom->NextSiblingElement("geom")) {
            const char* cls = geom->Attribute("class");
            const char* meshName = geom->Attribute("mesh");

            if (cls && std::string(cls) == "visual" && meshName) {
                auto it = meshNameToFile.find(meshName);
                if (it != meshNameToFile.end()) {
                    MJCFMeshInfo info;
                    info.bodyName =
                        bodyName ? bodyName
                                 : fmt::format("body_{}", idx);
                    info.meshFile = it->second;
                    info.bodyIndex = idx;
                    meshInfos.push_back(std::move(info));
                }
            }
        }

        // Enqueue children in same BFS order
        for (auto* child = element->FirstChildElement("body"); child;
             child = child->NextSiblingElement("body")) {
            ++bodyIndex;
            queue.push({child, bodyIndex});
        }
    }

    MJCFLoadResult result;
    result.skeleton = std::move(skeleton);
    result.meshInfos = std::move(meshInfos);
    result.meshDir = std::move(meshDir);
    return result;
}

} // namespace Animation
} // namespace KE
