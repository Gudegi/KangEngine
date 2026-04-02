#include "mjcf_loader.hpp"

#include <fmt/core.h>
#include <tinyxml2.h>

#include <cmath>
#include <queue>
#include <sstream>
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
    meshDir = resolveMeshDir(mjcfPath, compiler ? compiler->Attribute("meshdir")
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
                        bodyName ? bodyName : fmt::format("body_{}", idx);
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

// ── Collision geometry parsing
// ────────────────────────────────────────────────

namespace {

// Accumulated geom attributes from a <default class="X"> chain
struct DefaultGeomAttrs {
    std::string type;
    std::vector<float> size;
    std::vector<float> pos;
    std::vector<float> quat;
    std::vector<float> fromto;
    std::string parentClass;
};

static std::vector<float> splitFloats(const char* str) {
    if (!str)
        return {};
    std::vector<float> out;
    std::istringstream ss(str);
    float v;
    while (ss >> v)
        out.push_back(v);
    return out;
}

// Read <geom> child attributes into a DefaultGeomAttrs
static void readGeomDefaults(tinyxml2::XMLElement* defElem,
                             DefaultGeomAttrs& out) {
    auto* g = defElem->FirstChildElement("geom");
    if (!g)
        return;
    if (auto* t = g->Attribute("type"))
        out.type = t;
    auto sz = splitFloats(g->Attribute("size"));
    if (!sz.empty())
        out.size = sz;
    auto ps = splitFloats(g->Attribute("pos"));
    if (!ps.empty())
        out.pos = ps;
    auto qt = splitFloats(g->Attribute("quat"));
    if (!qt.empty())
        out.quat = qt;
    auto ft = splitFloats(g->Attribute("fromto"));
    if (!ft.empty())
        out.fromto = ft;
}

// Recursively collect all <default class="X"> entries into map
static void
collectDefaults(tinyxml2::XMLElement* elem, const std::string& parentClass,
                std::unordered_map<std::string, DefaultGeomAttrs>& map) {
    for (auto* def = elem->FirstChildElement("default"); def;
         def = def->NextSiblingElement("default")) {
        const char* cls = def->Attribute("class");
        if (!cls)
            continue;
        DefaultGeomAttrs attrs;
        attrs.parentClass = parentClass;
        readGeomDefaults(def, attrs);
        map[cls] = attrs;
        collectDefaults(def, cls, map);
    }
}

// Walk class → parent chain; first occurrence of each field wins
static DefaultGeomAttrs
resolveClass(const std::string& cls,
             const std::unordered_map<std::string, DefaultGeomAttrs>& map) {
    DefaultGeomAttrs out;
    std::string cur = cls;
    while (!cur.empty()) {
        auto it = map.find(cur);
        if (it == map.end())
            break;
        const auto& a = it->second;
        if (out.type.empty() && !a.type.empty())
            out.type = a.type;
        if (out.size.empty() && !a.size.empty())
            out.size = a.size;
        if (out.pos.empty() && !a.pos.empty())
            out.pos = a.pos;
        if (out.quat.empty() && !a.quat.empty())
            out.quat = a.quat;
        if (out.fromto.empty() && !a.fromto.empty())
            out.fromto = a.fromto;
        cur = a.parentClass;
    }
    return out;
}

// Build one MJCFCollisionGeom from a <geom> element; returns false if the
// geom should be skipped (visual, mesh, unknown type, etc.)
static bool buildCollisionGeom(
    tinyxml2::XMLElement* geomElem,
    const std::unordered_map<std::string, DefaultGeomAttrs>& defaultMap,
    MJCFCollisionGeom& out) {

    const char* clsAttr = geomElem->Attribute("class");
    DefaultGeomAttrs defs;
    if (clsAttr)
        defs = resolveClass(clsAttr, defaultMap);

    // Geom element attributes override class defaults
    auto typeStr = geomElem->Attribute("type")
                       ? std::string(geomElem->Attribute("type"))
                       : defs.type;
    if (typeStr.empty() || typeStr == "mesh" || typeStr == "plane")
        return false;

    auto size = splitFloats(geomElem->Attribute("size"));
    auto pos = splitFloats(geomElem->Attribute("pos"));
    auto quat = splitFloats(geomElem->Attribute("quat"));
    auto fromto = splitFloats(geomElem->Attribute("fromto"));
    if (size.empty())
        size = defs.size;
    if (pos.empty())
        pos = defs.pos;
    if (quat.empty())
        quat = defs.quat;
    if (fromto.empty())
        fromto = defs.fromto;

    if (typeStr == "capsule")
        out.type = MJCFCollisionGeom::Type::Capsule;
    else if (typeStr == "cylinder")
        out.type = MJCFCollisionGeom::Type::Cylinder;
    else if (typeStr == "sphere")
        out.type = MJCFCollisionGeom::Type::Sphere;
    else if (typeStr == "box")
        out.type = MJCFCollisionGeom::Type::Box;
    else
        return false;

    for (int i = 0; i < static_cast<int>(size.size()) && i < 3; ++i)
        out.size[i] = size[i];

    if (pos.size() >= 3)
        out.pos = Eigen::Vector3f(pos[0], pos[1], pos[2]);

    // MJCF quat: w x y z
    if (quat.size() >= 4) {
        float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
        float len = std::sqrt(w * w + x * x + y * y + z * z);
        if (len > 1e-6f)
            out.quat = Eigen::Quaternionf(w / len, x / len, y / len, z / len);
    }

    if (fromto.size() >= 6) {
        out.hasFromTo = true;
        out.from = Eigen::Vector3f(fromto[0], fromto[1], fromto[2]);
        out.to = Eigen::Vector3f(fromto[3], fromto[4], fromto[5]);
    }

    return true;
}

} // namespace

CollisionGeomMap parseMJCFCollision(const std::string& mjcfPath,
                                    const SkeletonTree& tree) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(mjcfPath.c_str()) != tinyxml2::XML_SUCCESS)
        return {};

    auto* root = doc.RootElement();

    // Build default class attribute map
    std::unordered_map<std::string, DefaultGeomAttrs> defaultMap;
    if (auto* def = root->FirstChildElement("default"))
        collectDefaults(def, "", defaultMap);

    CollisionGeomMap result;

    auto* worldbody = root->FirstChildElement("worldbody");
    if (!worldbody)
        return result;

    struct Entry {
        tinyxml2::XMLElement* elem;
    };
    std::queue<Entry> q;
    if (auto* b = worldbody->FirstChildElement("body"))
        q.push({b});

    while (!q.empty()) {
        auto [elem] = q.front();
        q.pop();

        const char* bodyName = elem->Attribute("name");
        if (bodyName) {
            try {
                int idx = tree.index(bodyName);
                for (auto* geom = elem->FirstChildElement("geom"); geom;
                     geom = geom->NextSiblingElement("geom")) {
                    // Skip visual geoms
                    const char* cls = geom->Attribute("class");
                    if (cls && std::string(cls) == "visual")
                        continue;

                    MJCFCollisionGeom g;
                    if (buildCollisionGeom(geom, defaultMap, g))
                        result[idx].push_back(g);
                }
            } catch (...) {
                // body name not in SkeletonTree — skip
            }
        }

        for (auto* c = elem->FirstChildElement("body"); c;
             c = c->NextSiblingElement("body"))
            q.push({c});
    }

    return result;
}

} // namespace Animation
} // namespace KE
