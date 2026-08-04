#include "asset/mjcf_loader.hpp"

#include <fmt/core.h>
#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace KE {
namespace Asset {

using namespace Animation;
using namespace Character;
using namespace Physics;

namespace {

std::vector<float> splitFloats(const char* str) {
    if (!str)
        return {};
    std::vector<float> out;
    std::istringstream ss(str);
    float v;
    while (ss >> v)
        out.push_back(v);
    return out;
}

Eigen::Vector3f
parseVec3(const char* str,
          const Eigen::Vector3f& fallback = Eigen::Vector3f::Zero()) {
    auto values = splitFloats(str);
    if (values.size() < 3)
        return fallback;
    return Eigen::Vector3f(values[0], values[1], values[2]);
}

Eigen::Vector4f parseVec4(const char* str,
                          const Eigen::Vector4f& fallback =
                              Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f)) {
    auto values = splitFloats(str);
    if (values.size() < 4)
        return fallback;
    return Eigen::Vector4f(values[0], values[1], values[2], values[3]);
}

Eigen::Quaternionf parseQuatWxyz(
    const char* str,
    const Eigen::Quaternionf& fallback = Eigen::Quaternionf::Identity()) {
    auto values = splitFloats(str);
    if (values.size() < 4)
        return fallback;
    return Eigen::Quaternionf(values[0], values[1], values[2], values[3])
        .normalized();
}

Eigen::Quaternionf quatFromZAxis(const Eigen::Vector3f& zaxis) {
    if (zaxis.squaredNorm() < 1e-12f)
        return Eigen::Quaternionf::Identity();
    return Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(),
                                              zaxis.normalized());
}

std::string resolveMeshDir(const std::string& mjcfPath, const char* meshdir) {
    std::string dir;
    auto lastSlash = mjcfPath.find_last_of("/\\");
    if (lastSlash != std::string::npos)
        dir = mjcfPath.substr(0, lastSlash + 1);

    if (!meshdir)
        return dir;

    std::string md(meshdir);
    if (md.size() >= 2 && md[0] == '.' && md[1] == '/')
        return dir + md.substr(2);
    if (!md.empty() && md[0] == '/')
        return md;
    return dir + md;
}

std::string meshAssetName(const char* name, const char* file) {
    if (name && name[0] != '\0')
        return name;
    if (!file || file[0] == '\0')
        return {};
    return std::filesystem::path(file).stem().string();
}

// Invokes callback(xml_node, tree_index, name, effectiveClass) for each body.
// effectiveClass is the childclass inherited from ancestor bodies.
template <typename Func>
void traverseBodies(tinyxml2::XMLElement* root, const SkeletonTree& tree,
                    bool logMissing, Func&& callback) {
    auto* worldbody = root->FirstChildElement("worldbody");
    if (!worldbody)
        return;

    struct Entry {
        tinyxml2::XMLElement* element;
        std::string inheritedClass;
    };

    std::queue<Entry> q;
    for (auto* b = worldbody->FirstChildElement("body"); b;
         b = b->NextSiblingElement("body"))
        q.push({b, ""});

    while (!q.empty()) {
        auto [elem, inherited] = q.front();
        q.pop();

        std::string forChildren = inherited;
        if (const char* cc = elem->Attribute("childclass"))
            forChildren = cc;

        const char* bodyName = elem->Attribute("name");
        if (bodyName) {
            try {
                int idx = tree.index(bodyName);
                callback(elem, idx, bodyName, forChildren);
            } catch (const std::exception& e) {
                if (logMissing)
                    fmt::print(stderr,
                               "Warning: body '{}' not in skeleton — {}\n",
                               bodyName, e.what());
            }
        }
        for (auto* c = elem->FirstChildElement("body"); c;
             c = c->NextSiblingElement("body"))
            q.push({c, forChildren});
    }
}

// Accumulated geom attributes from a <default class="X"> chain.
struct DefaultGeomAttrs {
    std::string type;
    std::vector<float> size;
    std::vector<float> pos;
    std::vector<float> quat;
    std::vector<float> fromto;
    std::vector<float> friction;
    std::vector<float> rgba;
    int condim = -1;
    float margin = -1.f;
    std::string parentClass;
};

struct DefaultSiteAttrs {
    std::string type;
    std::vector<float> size;
    std::vector<float> pos;
    std::vector<float> quat;
    std::vector<float> zaxis;
    std::vector<float> rgba;
    std::string parentClass;
};

struct DefaultJointAttrs {
    float stiffness = -1.f;
    float damping = -1.f;
    float armature = -1.f;
    std::string parentClass;
};

void readGeomDefaults(tinyxml2::XMLElement* defElem, DefaultGeomAttrs& out) {
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
    auto fr = splitFloats(g->Attribute("friction"));
    if (!fr.empty())
        out.friction = fr;
    auto rgba = splitFloats(g->Attribute("rgba"));
    if (!rgba.empty())
        out.rgba = rgba;
    g->QueryIntAttribute("condim", &out.condim);
    g->QueryFloatAttribute("margin", &out.margin);
}

void readSiteDefaults(tinyxml2::XMLElement* defElem, DefaultSiteAttrs& out) {
    auto* s = defElem->FirstChildElement("site");
    if (!s)
        return;
    if (auto* t = s->Attribute("type"))
        out.type = t;
    auto sz = splitFloats(s->Attribute("size"));
    if (!sz.empty())
        out.size = sz;
    auto ps = splitFloats(s->Attribute("pos"));
    if (!ps.empty())
        out.pos = ps;
    auto qt = splitFloats(s->Attribute("quat"));
    if (!qt.empty())
        out.quat = qt;
    auto za = splitFloats(s->Attribute("zaxis"));
    if (!za.empty())
        out.zaxis = za;
    auto rgba = splitFloats(s->Attribute("rgba"));
    if (!rgba.empty())
        out.rgba = rgba;
}

void readJointDefaults(tinyxml2::XMLElement* defElem, DefaultJointAttrs& out) {
    auto* joint = defElem->FirstChildElement("joint");
    if (!joint)
        return;
    joint->QueryFloatAttribute("stiffness", &out.stiffness);
    joint->QueryFloatAttribute("damping", &out.damping);
    joint->QueryFloatAttribute("armature", &out.armature);
}

bool geomHasZeroContact(tinyxml2::XMLElement* geom) {
    int contype = 1;
    int conaffinity = 1;
    const bool hasContype =
        geom->QueryIntAttribute("contype", &contype) == tinyxml2::XML_SUCCESS;
    const bool hasConaffinity =
        geom->QueryIntAttribute("conaffinity", &conaffinity) ==
        tinyxml2::XML_SUCCESS;
    return hasContype && hasConaffinity && contype == 0 && conaffinity == 0;
}

bool meshGeomIsVisualOnly(tinyxml2::XMLElement* geom,
                          const std::string& effectiveClass) {
    return effectiveClass == "visual" || geomHasZeroContact(geom);
}

void collectDefaults(
    tinyxml2::XMLElement* elem, const std::string& parentClass,
    std::unordered_map<std::string, DefaultGeomAttrs>& geomMap,
    std::unordered_map<std::string, DefaultSiteAttrs>& siteMap,
    std::unordered_map<std::string, DefaultJointAttrs>& jointMap) {
    for (auto* def = elem->FirstChildElement("default"); def;
         def = def->NextSiblingElement("default")) {
        const char* cls = def->Attribute("class");
        if (!cls)
            continue;
        DefaultGeomAttrs geomAttrs;
        geomAttrs.parentClass = parentClass;
        readGeomDefaults(def, geomAttrs);
        geomMap[cls] = geomAttrs;

        DefaultSiteAttrs siteAttrs;
        siteAttrs.parentClass = parentClass;
        readSiteDefaults(def, siteAttrs);
        siteMap[cls] = siteAttrs;

        DefaultJointAttrs jointAttrs;
        jointAttrs.parentClass = parentClass;
        readJointDefaults(def, jointAttrs);
        jointMap[cls] = jointAttrs;

        collectDefaults(def, cls, geomMap, siteMap, jointMap);
    }
}

DefaultJointAttrs resolveJointClass(
    const std::string& cls,
    const std::unordered_map<std::string, DefaultJointAttrs>& map) {
    DefaultJointAttrs out;
    std::string cur = cls;
    while (!cur.empty()) {
        auto it = map.find(cur);
        if (it == map.end())
            break;
        const auto& attrs = it->second;
        if (out.stiffness < 0.f && attrs.stiffness >= 0.f)
            out.stiffness = attrs.stiffness;
        if (out.damping < 0.f && attrs.damping >= 0.f)
            out.damping = attrs.damping;
        if (out.armature < 0.f && attrs.armature >= 0.f)
            out.armature = attrs.armature;
        cur = attrs.parentClass;
    }
    return out;
}

// Walk class -> parent chain; first occurrence of each field wins.
DefaultGeomAttrs
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
        if (out.friction.empty() && !a.friction.empty())
            out.friction = a.friction;
        if (out.rgba.empty() && !a.rgba.empty())
            out.rgba = a.rgba;
        if (out.condim < 0 && a.condim >= 0)
            out.condim = a.condim;
        if (out.margin < 0.f && a.margin >= 0.f)
            out.margin = a.margin;
        cur = a.parentClass;
    }
    return out;
}

DefaultSiteAttrs
resolveSiteClass(const std::string& cls,
                 const std::unordered_map<std::string, DefaultSiteAttrs>& map) {
    DefaultSiteAttrs out;
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
        if (out.zaxis.empty() && !a.zaxis.empty())
            out.zaxis = a.zaxis;
        if (out.rgba.empty() && !a.rgba.empty())
            out.rgba = a.rgba;
        cur = a.parentClass;
    }
    return out;
}

bool parseSite(
    tinyxml2::XMLElement* siteElem,
    const std::unordered_map<std::string, DefaultSiteAttrs>& defaultMap,
    SiteDesc& out, const std::string& inheritedClass = "") {
    const char* siteName = siteElem->Attribute("name");
    if (!siteName || siteName[0] == '\0')
        return false;

    const char* clsAttr = siteElem->Attribute("class");
    std::string effectiveCls = clsAttr ? clsAttr : inheritedClass;
    DefaultSiteAttrs defs;
    if (!effectiveCls.empty())
        defs = resolveSiteClass(effectiveCls, defaultMap);

    auto typeStr = siteElem->Attribute("type")
                       ? std::string(siteElem->Attribute("type"))
                       : defs.type;
    if (typeStr == "capsule")
        out.type = SiteDesc::Type::Capsule;
    else if (typeStr == "box")
        out.type = SiteDesc::Type::Box;
    else
        out.type = SiteDesc::Type::Sphere;

    auto size = splitFloats(siteElem->Attribute("size"));
    auto pos = splitFloats(siteElem->Attribute("pos"));
    auto quat = splitFloats(siteElem->Attribute("quat"));
    auto zaxis = splitFloats(siteElem->Attribute("zaxis"));
    auto rgba = splitFloats(siteElem->Attribute("rgba"));
    if (size.empty())
        size = defs.size;
    if (pos.empty())
        pos = defs.pos;
    if (quat.empty())
        quat = defs.quat;
    if (zaxis.empty())
        zaxis = defs.zaxis;
    if (rgba.empty())
        rgba = defs.rgba;

    out.name = siteName;
    if (pos.size() >= 3)
        out.pos = Eigen::Vector3f(pos[0], pos[1], pos[2]);
    for (int i = 0; i < static_cast<int>(size.size()) && i < 3; ++i)
        out.size[i] = size[i];
    if (rgba.size() >= 4)
        out.rgba = Eigen::Vector4f(rgba[0], rgba[1], rgba[2], rgba[3]);
    if (quat.size() >= 4) {
        out.quat =
            Eigen::Quaternionf(quat[0], quat[1], quat[2], quat[3]).normalized();
    } else if (zaxis.size() >= 3) {
        Eigen::Vector3f axis(zaxis[0], zaxis[1], zaxis[2]);
        if (axis.squaredNorm() > 1e-12f) {
            out.hasZAxis = true;
            out.zaxis = axis.normalized();
            out.quat = quatFromZAxis(out.zaxis);
        }
    }
    return true;
}

// Returns false if this loader cannot represent the geom as an explicit
// CollisionGeomDesc descriptor.
//
// Mesh collision geoms are intentionally unsupported here. Dynamic and
// articulation bodies should use primitive shapes or a future convex-cooked
// mesh path; visual MeshData cannot be attached directly as a safe PhysX link
// collision shape. Bodies with no supported descriptors may still receive
// KangEngine fallback boxes during Articulation::build().
bool buildCollisionGeom(
    tinyxml2::XMLElement* geomElem,
    const std::unordered_map<std::string, DefaultGeomAttrs>& defaultMap,
    CollisionGeomDesc& out, const std::string& inheritedClass = "") {

    const char* clsAttr = geomElem->Attribute("class");
    std::string effectiveCls = clsAttr ? clsAttr : inheritedClass;
    DefaultGeomAttrs defs;
    if (!effectiveCls.empty())
        defs = resolveClass(effectiveCls, defaultMap);

    auto typeStr = geomElem->Attribute("type")
                       ? std::string(geomElem->Attribute("type"))
                       : defs.type;
    if (typeStr.empty() || typeStr == "mesh" || typeStr == "plane")
        return false;

    auto size = splitFloats(geomElem->Attribute("size"));
    auto pos = splitFloats(geomElem->Attribute("pos"));
    auto quat = splitFloats(geomElem->Attribute("quat"));
    auto fromto = splitFloats(geomElem->Attribute("fromto"));
    auto friction = splitFloats(geomElem->Attribute("friction"));
    if (size.empty())
        size = defs.size;
    if (pos.empty())
        pos = defs.pos;
    if (quat.empty())
        quat = defs.quat;
    if (fromto.empty())
        fromto = defs.fromto;
    if (friction.empty())
        friction = defs.friction;

    if (typeStr == "capsule")
        out.type = CollisionGeomDesc::Type::Capsule;
    else if (typeStr == "cylinder")
        out.type = CollisionGeomDesc::Type::Cylinder;
    else if (typeStr == "sphere")
        out.type = CollisionGeomDesc::Type::Sphere;
    else if (typeStr == "box")
        out.type = CollisionGeomDesc::Type::Box;
    else
        return false;

    if (auto* name = geomElem->Attribute("name"))
        out.name = name;

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
    if (!friction.empty()) {
        out.friction = friction[0];
        out.physicsMaterial = mjcfFrictionToPhysX(friction);
    }
    out.condim = defs.condim;
    geomElem->QueryIntAttribute("condim", &out.condim);
    out.margin = defs.margin;
    geomElem->QueryFloatAttribute("margin", &out.margin);

    return true;
}

struct GeomMassData {
    float mass;
    Eigen::Vector3f center;
    Eigen::Vector3f iDiag;
};

GeomMassData geomMassContribution(const CollisionGeomDesc& g, float density) {
    using Type = CollisionGeomDesc::Type;
    float V = 0.f;
    Eigen::Vector3f center = g.pos;
    Eigen::Vector3f iDiag = Eigen::Vector3f::Zero();

    if (g.hasFromTo)
        center = (g.from + g.to) * 0.5f;

    switch (g.type) {
    case Type::Sphere: {
        float r = g.size[0];
        V = (4.f / 3.f) * static_cast<float>(M_PI) * r * r * r;
        float m = density * V;
        float I = (2.f / 5.f) * m * r * r;
        iDiag = Eigen::Vector3f(I, I, I);
        break;
    }
    case Type::Capsule:
    case Type::Cylinder: {
        float r = g.size[0];
        float halfLen = g.hasFromTo ? (g.to - g.from).norm() * 0.5f : g.size[1];
        float h = 2.f * halfLen;
        float Vcyl = static_cast<float>(M_PI) * r * r * h;
        float Vhemi = (g.type == Type::Capsule)
                          ? (4.f / 3.f) * static_cast<float>(M_PI) * r * r * r
                          : 0.f;
        V = Vcyl + Vhemi;
        float m = density * V;
        float Iz = m * r * r / 2.f;
        float Ixy = m * (3.f * r * r + h * h) / 12.f;
        iDiag = Eigen::Vector3f(Ixy, Ixy, Iz);
        break;
    }
    case Type::Box: {
        float hx = g.size[0], hy = g.size[1], hz = g.size[2];
        V = 8.f * hx * hy * hz;
        float m = density * V;
        iDiag = Eigen::Vector3f(m * (hy * hy + hz * hz) / 3.f,
                                m * (hx * hx + hz * hz) / 3.f,
                                m * (hx * hx + hy * hy) / 3.f);
        break;
    }
    }
    return {density * V, center, iDiag};
}

} // namespace

void MJCFLoader::parseIntoData(const std::string& mjcfPath, float scale,
                               const std::string& order) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(mjcfPath.c_str()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error(
            fmt::format("Failed to load MJCF file: {}", mjcfPath));

    auto* root = doc.RootElement();

    // 1. Compiler options
    // MJCF's default compiler angle unit is degrees. Only an explicit
    // angle="radian" leaves joint ranges unscaled.
    float degToRad = static_cast<float>(M_PI) / 180.f;
    bool inertiafromgeom = false;
    if (auto* compiler = root->FirstChildElement("compiler")) {
        _data.meshDir =
            resolveMeshDir(mjcfPath, compiler->Attribute("meshdir"));
        const char* angle = compiler->Attribute("angle");
        if (angle && std::string_view(angle) == "radian")
            degToRad = 1.f;
        const char* ifg = compiler->Attribute("inertiafromgeom");
        if (ifg && std::string(ifg) == "true")
            inertiafromgeom = true;
    } else {
        _data.meshDir = resolveMeshDir(mjcfPath, nullptr);
    }

    // 2. Asset name -> file map
    std::unordered_map<std::string, std::string> meshNameToFile;
    if (auto* asset = root->FirstChildElement("asset")) {
        for (auto* mesh = asset->FirstChildElement("mesh"); mesh;
             mesh = mesh->NextSiblingElement("mesh")) {
            const char* name = mesh->Attribute("name");
            const char* file = mesh->Attribute("file");
            const std::string assetName = meshAssetName(name, file);
            if (!assetName.empty() && file)
                meshNameToFile[assetName] = file;
        }
    }

    // 3. Skeleton
    SkeletonTree skelTree = SkeletonTree::skelFromMJCFElement(root, order);

    // 4. Default class maps for collision geoms and named sites
    std::unordered_map<std::string, DefaultGeomAttrs> defaultMap;
    std::unordered_map<std::string, DefaultSiteAttrs> siteDefaultMap;
    std::unordered_map<std::string, DefaultJointAttrs> jointDefaultMap;
    if (auto* def = root->FirstChildElement("default"))
        collectDefaults(def, "", defaultMap, siteDefaultMap, jointDefaultMap);

    // 5. Single traversal — mesh info, joints, collision, inertial
    traverseBodies(
        root, skelTree, true,
        [&](tinyxml2::XMLElement* elem, int idx, const char* bodyName,
            const std::string& inheritedClass) {
            std::vector<GeomMassData> geomMasses;
            std::unordered_set<std::string> visualOnlyMeshNames;

            // Some MJCF assets provide two mesh geoms for the same body/mesh:
            // one visual-only geom (e.g. contype=0 conaffinity=0 group=1) and
            // one collidable mesh geom.  Keep the visual-only mesh as the
            // render asset and avoid creating a duplicate visual_N prim for
            // the collidable copy.
            for (auto* geom = elem->FirstChildElement("geom"); geom;
                 geom = geom->NextSiblingElement("geom")) {
                const char* meshName = geom->Attribute("mesh");
                if (!meshName)
                    continue;
                const char* cls = geom->Attribute("class");
                std::string effectiveCls = cls ? cls : inheritedClass;
                if (meshGeomIsVisualOnly(geom, effectiveCls))
                    visualOnlyMeshNames.insert(meshName);
            }

            for (auto* geom = elem->FirstChildElement("geom"); geom;
                 geom = geom->NextSiblingElement("geom")) {
                const char* cls = geom->Attribute("class");
                std::string effectiveCls = cls ? cls : inheritedClass;
                const char* meshName = geom->Attribute("mesh");
                auto geomType = geom->Attribute("type")
                                    ? std::string(geom->Attribute("type"))
                                    : std::string();
                bool isCollisionMesh = effectiveCls == "collision";
                bool isVisualMesh = meshGeomIsVisualOnly(geom, effectiveCls);
                bool hasVisualOnlyDuplicate =
                    meshName && visualOnlyMeshNames.count(meshName) != 0;

                // Visual mesh
                if (meshName && !isCollisionMesh &&
                    (isVisualMesh ||
                     (geomType == "mesh" && !hasVisualOnlyDuplicate))) {
                    auto it = meshNameToFile.find(meshName);
                    if (it != meshNameToFile.end()) {
                        DefaultGeomAttrs defs;
                        if (!effectiveCls.empty())
                            defs = resolveClass(effectiveCls, defaultMap);
                        Eigen::Vector3f meshPos =
                            parseVec3(geom->Attribute("pos")) * scale;
                        Eigen::Quaternionf meshQuat =
                            parseQuatWxyz(geom->Attribute("quat"));
                        auto rgbaValues = splitFloats(geom->Attribute("rgba"));
                        if (rgbaValues.empty())
                            rgbaValues = defs.rgba;
                        Eigen::Vector4f rgba =
                            rgbaValues.size() >= 4
                                ? Eigen::Vector4f(rgbaValues[0], rgbaValues[1],
                                                  rgbaValues[2], rgbaValues[3])
                                : Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f);
                        _data.meshInfos.push_back({bodyName, it->second, idx,
                                                   meshPos, meshQuat, rgba});
                    }
                    continue;
                }

                // Collision geom
                // TODO : add convex mesh decomposition like COCAD
                if (meshName && !isVisualMesh) {
                    // Empty entry marks that the source authored a collidable
                    // mesh geom that KangEngine cannot cook yet. Articulation
                    // build may synthesize a fallback box only for these
                    // bodies, while visual-only mesh bodies stay collisionless.
                    _data.collisionGeoms.try_emplace(idx);
                    fmt::print(
                        stderr,
                        "Warning: MJCF mesh collision geom '{}' on body '{}' "
                        "is not supported yet; using KangEngine fallback "
                        "collision if this body has no supported primitive "
                        "collision geom.\n",
                        meshName, bodyName);
                    continue;
                }
                CollisionGeomDesc g;
                if (!buildCollisionGeom(geom, defaultMap, g, inheritedClass))
                    continue;

                // Apply uniform scale to all spatial quantities
                g.pos *= scale;
                for (int i = 0; i < 3; i++)
                    g.size[i] *= scale;
                if (g.hasFromTo) {
                    g.from *= scale;
                    g.to *= scale;
                }

                // MuJoCo assets commonly omit <inertial> and put density on
                // geoms instead. Use those densities whenever they are present,
                // even if compiler inertiafromgeom is not explicitly set.
                if (inertiafromgeom || geom->Attribute("density")) {
                    float density = 1000.f;
                    geom->QueryFloatAttribute("density", &density);
                    auto gmd = geomMassContribution(g, density);
                    if (gmd.mass > 1e-8f)
                        geomMasses.push_back(gmd);
                }

                _data.collisionGeoms[idx].push_back(g);
            }

            // Named body-local frames
            for (auto* sElem = elem->FirstChildElement("site"); sElem;
                 sElem = sElem->NextSiblingElement("site")) {
                SiteDesc site;
                if (!parseSite(sElem, siteDefaultMap, site, inheritedClass))
                    continue;
                site.bodyIndex = idx;
                site.pos *= scale;
                site.size *= scale;
                _data.sites[site.name] = std::move(site);
            }

            // Joints for this body
            for (auto* jElem = elem->FirstChildElement("joint"); jElem;
                 jElem = jElem->NextSiblingElement("joint")) {
                // A root <joint type="free"> is equivalent to <freejoint>.
                // The articulation's floating base represents it, so it must
                // not also be emitted as a revolute articulation DOF.
                const char* jointType = jElem->Attribute("type");
                if (jointType && std::string_view(jointType) == "free")
                    continue;

                JointDesc jd;
                jd.name =
                    jElem->Attribute("name") ? jElem->Attribute("name") : "";
                jd.type = JointDesc::Type::Revolute;
                if (!inheritedClass.empty()) {
                    const auto defaults =
                        resolveJointClass(inheritedClass, jointDefaultMap);
                    if (defaults.stiffness >= 0.f)
                        jd.kp = defaults.stiffness;
                    if (defaults.damping >= 0.f)
                        jd.kd = defaults.damping;
                    if (defaults.armature >= 0.f)
                        jd.armature = defaults.armature;
                }
                auto axisVals = splitFloats(jElem->Attribute("axis"));
                if (axisVals.size() >= 3)
                    jd.axis =
                        Eigen::Vector3f(axisVals[0], axisVals[1], axisVals[2])
                            .normalized();
                auto rangeVals = splitFloats(jElem->Attribute("range"));
                if (rangeVals.size() >= 2) {
                    jd.loLimit = rangeVals[0] * degToRad;
                    jd.hiLimit = rangeVals[1] * degToRad;
                }
                jElem->QueryFloatAttribute("stiffness", &jd.kp);
                jElem->QueryFloatAttribute("damping", &jd.kd);
                jElem->QueryFloatAttribute("armature", &jd.armature);
                auto forceRangeVals =
                    splitFloats(jElem->Attribute("actuatorfrcrange"));
                if (forceRangeVals.size() >= 2) {
                    jd.effortLimit = std::max(std::abs(forceRangeVals[0]),
                                              std::abs(forceRangeVals[1]));
                }
                _data.joints[idx].push_back(jd);
            }

            // InertialDesc: explicit element takes priority over geom-derived
            if (auto* ie = elem->FirstChildElement("inertial")) {
                InertialDesc inertial;
                ie->QueryFloatAttribute("mass", &inertial.mass);
                auto pos = splitFloats(ie->Attribute("pos"));
                if (pos.size() >= 3)
                    inertial.com =
                        Eigen::Vector3f(pos[0], pos[1], pos[2]) * scale;
                auto quat = splitFloats(ie->Attribute("quat"));
                if (quat.size() >= 4) {
                    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
                    float len = std::sqrt(w * w + x * x + y * y + z * z);
                    if (len > 1e-6f)
                        inertial.quat = Eigen::Quaternionf(w / len, x / len,
                                                           y / len, z / len);
                }
                auto di = splitFloats(ie->Attribute("diaginertia"));
                if (di.size() >= 3)
                    inertial.diagInertia = Eigen::Vector3f(di[0], di[1], di[2]);
                _data.inertials[idx] = inertial;
            } else if (!geomMasses.empty()) {
                float totalMass = 0.f;
                Eigen::Vector3f com = Eigen::Vector3f::Zero();
                for (const auto& gmd : geomMasses) {
                    totalMass += gmd.mass;
                    com += gmd.mass * gmd.center;
                }
                com /= totalMass;

                Eigen::Vector3f iTotal = Eigen::Vector3f::Zero();
                for (const auto& gmd : geomMasses) {
                    Eigen::Vector3f r = gmd.center - com;
                    iTotal.x() += gmd.iDiag.x() +
                                  gmd.mass * (r.y() * r.y() + r.z() * r.z());
                    iTotal.y() += gmd.iDiag.y() +
                                  gmd.mass * (r.x() * r.x() + r.z() * r.z());
                    iTotal.z() += gmd.iDiag.z() +
                                  gmd.mass * (r.x() * r.x() + r.y() * r.y());
                }

                InertialDesc inertial;
                inertial.mass = totalMass;
                inertial.com = com;
                inertial.diagInertia = iTotal.cwiseMax(1e-4f);
                _data.inertials[idx] = inertial;
            }
        });

    _data.skeletonTree =
        std::make_shared<const SkeletonTree>(std::move(skelTree));
}

MJCFImportResult MJCFLoader::parse(const std::string& mjcfPath, float scale,
                                   const std::string& order) {
    MJCFLoader loader;
    loader.parseIntoData(mjcfPath, scale, order);

    MJCFImportResult result;
    result.character = std::move(loader._data);
    result.diagnostics = std::move(loader._diagnostics);
    result.diagnostics.printWarnings("MJCFLoader " + mjcfPath);
    return result;
}

CharacterData MJCFLoader::load(const std::string& mjcfPath, float scale,
                               const std::string& order) {
    return std::move(parse(mjcfPath, scale, order).character);
}

} // namespace Asset
} // namespace KE
