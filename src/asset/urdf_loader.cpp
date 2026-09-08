#include "asset/urdf_loader.hpp"

#include "asset/mesh_loader.hpp"

#include <fmt/core.h>
#include <tinyxml2.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace KE {
namespace Asset {

using namespace Animation;

namespace {

struct Origin {
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
};

struct LinkRecord {
    std::string name;
    tinyxml2::XMLElement* element = nullptr;
};

struct JointRecord {
    std::string name;
    std::string type;
    std::string parent;
    std::string child;
    Origin origin;
    tinyxml2::XMLElement* element = nullptr;
};

std::vector<float> splitFloats(const char* text) {
    if (!text)
        return {};
    std::istringstream stream(text);
    std::vector<float> values;
    float value = 0.f;
    while (stream >> value)
        values.push_back(value);
    return values;
}

Eigen::Vector3f
parseVec3(const char* text,
          const Eigen::Vector3f& fallback = Eigen::Vector3f::Zero()) {
    const auto values = splitFloats(text);
    if (values.size() != 3)
        return fallback;
    return {values[0], values[1], values[2]};
}

Eigen::Quaternionf quaternionFromRpy(const Eigen::Vector3f& rpy) {
    return (Eigen::AngleAxisf(rpy.z(), Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(rpy.y(), Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(rpy.x(), Eigen::Vector3f::UnitX()))
        .normalized();
}

Origin parseOrigin(tinyxml2::XMLElement* parent, float scale) {
    Origin result;
    auto* origin = parent ? parent->FirstChildElement("origin") : nullptr;
    if (!origin)
        return result;
    result.translation = parseVec3(origin->Attribute("xyz")) * scale;
    result.rotation = quaternionFromRpy(parseVec3(origin->Attribute("rpy")));
    return result;
}

std::string requiredAttribute(tinyxml2::XMLElement* element,
                              const char* attribute,
                              const std::string& context) {
    const char* value = element ? element->Attribute(attribute) : nullptr;
    if (!value || value[0] == '\0')
        throw std::runtime_error(
            fmt::format("URDF {} requires attribute '{}'", context, attribute));
    return value;
}

std::string childLinkName(tinyxml2::XMLElement* joint, const char* tag,
                          const std::string& jointName) {
    auto* link = joint->FirstChildElement(tag);
    return requiredAttribute(link, "link",
                             fmt::format("joint '{}' <{}>", jointName, tag));
}

std::string normalizeMeshFilename(const std::string& filename) {
    constexpr std::string_view filePrefix = "file://";
    if (filename.rfind(filePrefix.data(), 0) == 0)
        return filename.substr(filePrefix.size());
    return filename;
}

std::vector<std::filesystem::path> environmentPaths(const char* name) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0')
        return {};

#ifdef _WIN32
    constexpr char separator = ';';
#else
    constexpr char separator = ':';
#endif

    std::vector<std::filesystem::path> paths;
    std::stringstream stream(value);
    std::string entry;
    while (std::getline(stream, entry, separator)) {
        if (!entry.empty())
            paths.emplace_back(entry);
    }
    return paths;
}

std::optional<std::filesystem::path>
resolveUrdfMeshPath(const std::string& filename,
                    const std::filesystem::path& urdfPath) {
    const std::string normalized = normalizeMeshFilename(filename);
    constexpr std::string_view packagePrefix = "package://";
    const std::filesystem::path urdfDirectory =
        std::filesystem::absolute(urdfPath).parent_path();

    if (normalized.rfind(packagePrefix.data(), 0) != 0) {
        const std::filesystem::path path(normalized);
        return (path.is_absolute() ? path : urdfDirectory / path)
            .lexically_normal();
    }

    const std::string packagePath =
        normalized.substr(packagePrefix.size());
    const size_t separator = packagePath.find('/');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 >= packagePath.size())
        return std::nullopt;

    const std::string packageName = packagePath.substr(0, separator);
    const std::filesystem::path relativePath =
        packagePath.substr(separator + 1);
    std::vector<std::filesystem::path> candidates;

    for (std::filesystem::path directory = urdfDirectory;
         !directory.empty();) {
        if (directory.filename() == packageName)
            candidates.push_back(directory / relativePath);
        candidates.push_back(directory / packageName / relativePath);

        const std::filesystem::path parent = directory.parent_path();
        if (parent == directory)
            break;
        directory = parent;
    }

    for (const auto& root : environmentPaths("ROS_PACKAGE_PATH")) {
        if (root.filename() == packageName)
            candidates.push_back(root / relativePath);
        candidates.push_back(root / packageName / relativePath);
    }
    for (const char* name : {"AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"}) {
        for (const auto& prefix : environmentPaths(name))
            candidates.push_back(prefix / "share" / packageName / relativePath);
    }

    for (const auto& candidate : candidates) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error))
            return std::filesystem::absolute(candidate).lexically_normal();
    }
    return std::nullopt;
}

std::string lowerExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension;
}

Scene::MeshData loadCollisionMesh(const std::filesystem::path& path) {
    const std::string extension = lowerExtension(path);
    if (extension == ".stl")
        return loadStl(path.string());
    if (extension == ".obj")
        return loadObj(path.string());
    throw std::runtime_error("Unsupported URDF collision mesh extension: " +
                             path.string());
}

Eigen::Vector4f
parseColor(tinyxml2::XMLElement* material,
           const std::unordered_map<std::string, Eigen::Vector4f>& colors) {
    const Eigen::Vector4f fallback(0.15f, 0.15f, 0.15f, 1.f);
    if (!material)
        return fallback;
    if (auto* color = material->FirstChildElement("color")) {
        const auto rgba = splitFloats(color->Attribute("rgba"));
        if (rgba.size() == 4)
            return {rgba[0], rgba[1], rgba[2], rgba[3]};
    }
    if (const char* name = material->Attribute("name")) {
        const auto found = colors.find(name);
        if (found != colors.end())
            return found->second;
    }
    return fallback;
}

void validateMeshScale(const Eigen::Vector3f& value,
                       const std::string& filename) {
    if (!value.allFinite() || (value.array() == 0.f).any())
        throw std::runtime_error("URDF mesh has an invalid scale: " + filename);
}

void parseInertial(tinyxml2::XMLElement* link, int bodyIndex, float scale,
                   ArticulationDesc& data, ImportDiagnostics& diagnostics) {
    auto* inertialElement = link->FirstChildElement("inertial");
    if (!inertialElement)
        return;

    InertialDesc inertial;
    if (auto* mass = inertialElement->FirstChildElement("mass"))
        mass->QueryFloatAttribute("value", &inertial.mass);
    const Origin origin = parseOrigin(inertialElement, scale);
    inertial.com = origin.translation;

    auto* tensor = inertialElement->FirstChildElement("inertia");
    if (!tensor) {
        diagnostics.warnings.push_back(
            fmt::format("link '{}' has an inertial without an inertia tensor",
                        requiredAttribute(link, "name", "link")));
        data.inertials[bodyIndex] = inertial;
        return;
    }

    float ixx = 0.f, ixy = 0.f, ixz = 0.f;
    float iyy = 0.f, iyz = 0.f, izz = 0.f;
    tensor->QueryFloatAttribute("ixx", &ixx);
    tensor->QueryFloatAttribute("ixy", &ixy);
    tensor->QueryFloatAttribute("ixz", &ixz);
    tensor->QueryFloatAttribute("iyy", &iyy);
    tensor->QueryFloatAttribute("iyz", &iyz);
    tensor->QueryFloatAttribute("izz", &izz);
    Eigen::Matrix3f matrix;
    matrix << ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(matrix);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Failed to diagonalize URDF inertia tensor");

    Eigen::Matrix3f axes = solver.eigenvectors();
    if (axes.determinant() < 0.f)
        axes.col(0) *= -1.f;
    inertial.diagInertia = solver.eigenvalues().cwiseMax(1e-8f);
    inertial.quat = (origin.rotation * Eigen::Quaternionf(axes)).normalized();
    data.inertials[bodyIndex] = inertial;
}

} // namespace

void URDFLoader::parseIntoData(const std::string& urdfPath, float scale,
                               const std::string& order,
                               Utils::CoordinateSystem targetCoordinateSystem) {
    if (order != "DFS" && order != "BFS")
        throw std::invalid_argument("URDF traversal order must be DFS or BFS");
    if (!(scale > 0.f) || !std::isfinite(scale))
        throw std::invalid_argument("URDF scale must be finite and positive");

    tinyxml2::XMLDocument document;
    if (document.LoadFile(urdfPath.c_str()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error(
            fmt::format("Failed to load URDF file: {}", urdfPath));
    auto* robot = document.RootElement();
    if (!robot || std::string_view(robot->Name()) != "robot")
        throw std::runtime_error("URDF root element must be <robot>");

    _data.assetDir = std::filesystem::path(urdfPath).parent_path().string();

    std::vector<LinkRecord> links;
    std::unordered_map<std::string, size_t> linkLookup;
    for (auto* link = robot->FirstChildElement("link"); link;
         link = link->NextSiblingElement("link")) {
        const std::string name = requiredAttribute(link, "name", "link");
        if (!linkLookup.emplace(name, links.size()).second)
            throw std::runtime_error("Duplicate URDF link name: " + name);
        links.push_back({name, link});
    }
    if (links.empty())
        throw std::runtime_error("URDF contains no links");

    std::vector<JointRecord> joints;
    std::unordered_set<std::string> jointNames;
    std::unordered_map<std::string, size_t> incoming;
    std::unordered_map<std::string, std::vector<size_t>> children;
    for (auto* joint = robot->FirstChildElement("joint"); joint;
         joint = joint->NextSiblingElement("joint")) {
        JointRecord record;
        record.name = requiredAttribute(joint, "name", "joint");
        record.type = requiredAttribute(joint, "type",
                                        fmt::format("joint '{}'", record.name));
        record.parent = childLinkName(joint, "parent", record.name);
        record.child = childLinkName(joint, "child", record.name);
        record.origin = parseOrigin(joint, scale);
        record.element = joint;
        if (!jointNames.insert(record.name).second)
            throw std::runtime_error("Duplicate URDF joint name: " +
                                     record.name);
        if (!linkLookup.count(record.parent) || !linkLookup.count(record.child))
            throw std::runtime_error(fmt::format(
                "joint '{}' references an unknown parent or child link",
                record.name));
        if (!incoming.emplace(record.child, joints.size()).second)
            throw std::runtime_error("URDF link has multiple parent joints: " +
                                     record.child);
        children[record.parent].push_back(joints.size());
        joints.push_back(std::move(record));
    }

    std::vector<std::string> roots;
    for (const LinkRecord& link : links)
        if (!incoming.count(link.name))
            roots.push_back(link.name);
    if (roots.size() != 1)
        throw std::runtime_error(fmt::format(
            "URDF must contain exactly one root link; found {}", roots.size()));

    std::vector<std::string> orderedNames;
    std::unordered_set<std::string> visited;
    if (order == "BFS") {
        std::queue<std::string> queue;
        queue.push(roots.front());
        while (!queue.empty()) {
            std::string name = std::move(queue.front());
            queue.pop();
            if (!visited.insert(name).second)
                throw std::runtime_error("Cycle detected in URDF link graph");
            orderedNames.push_back(name);
            for (size_t jointIndex : children[name])
                queue.push(joints[jointIndex].child);
        }
    } else {
        std::vector<std::string> stack{roots.front()};
        while (!stack.empty()) {
            std::string name = std::move(stack.back());
            stack.pop_back();
            if (!visited.insert(name).second)
                throw std::runtime_error("Cycle detected in URDF link graph");
            orderedNames.push_back(name);
            const auto& childJoints = children[name];
            for (auto it = childJoints.rbegin(); it != childJoints.rend(); ++it)
                stack.push_back(joints[*it].child);
        }
    }
    if (orderedNames.size() != links.size())
        throw std::runtime_error("URDF link graph is disconnected or cyclic");

    std::unordered_map<std::string, int> bodyIndices;
    for (size_t i = 0; i < orderedNames.size(); ++i)
        bodyIndices[orderedNames[i]] = static_cast<int>(i);
    std::vector<int> parentIndices(orderedNames.size(), -1);
    std::vector<Eigen::Vector3f> translations(orderedNames.size(),
                                              Eigen::Vector3f::Zero());
    std::vector<Eigen::Quaternionf> rotations(orderedNames.size(),
                                              Eigen::Quaternionf::Identity());
    std::vector<int> jointCounts(orderedNames.size(), 0);

    for (const JointRecord& joint : joints) {
        const int childIndex = bodyIndices.at(joint.child);
        parentIndices[childIndex] = bodyIndices.at(joint.parent);
        translations[childIndex] = joint.origin.translation;
        rotations[childIndex] = joint.origin.rotation;

        if (joint.type == "fixed")
            continue;
        if (joint.type != "revolute" && joint.type != "prismatic")
            throw std::runtime_error(
                fmt::format("URDF joint '{}' has unsupported type '{}'",
                            joint.name, joint.type));

        JointDesc descriptor;
        descriptor.name = joint.name;
        descriptor.type = joint.type == "prismatic"
                              ? JointDesc::Type::Prismatic
                              : JointDesc::Type::Revolute;
        descriptor.axis = parseVec3(
            joint.element->FirstChildElement("axis")
                ? joint.element->FirstChildElement("axis")->Attribute("xyz")
                : nullptr,
            Eigen::Vector3f::UnitX());
        if (descriptor.axis.squaredNorm() < 1e-12f)
            throw std::runtime_error("URDF joint has a zero axis: " +
                                     joint.name);
        descriptor.axis.normalize();
        auto* limit = joint.element->FirstChildElement("limit");
        if (!limit ||
            limit->QueryFloatAttribute("lower", &descriptor.loLimit) !=
                tinyxml2::XML_SUCCESS ||
            limit->QueryFloatAttribute("upper", &descriptor.hiLimit) !=
                tinyxml2::XML_SUCCESS)
            throw std::runtime_error("URDF movable joint requires limits: " +
                                     joint.name);
        if (descriptor.type == JointDesc::Type::Prismatic) {
            descriptor.loLimit *= scale;
            descriptor.hiLimit *= scale;
        }
        limit->QueryFloatAttribute("effort", &descriptor.effortLimit);
        limit->QueryFloatAttribute("velocity", &descriptor.velocityLimit);
        if (auto* dynamics = joint.element->FirstChildElement("dynamics"))
            dynamics->QueryFloatAttribute("damping", &descriptor.kd);
        _data.joints[childIndex].push_back(descriptor);
        jointCounts[childIndex] = 1;
    }

    std::unordered_map<std::string, Eigen::Vector4f> materialColors;
    for (auto* material = robot->FirstChildElement("material"); material;
         material = material->NextSiblingElement("material")) {
        const char* name = material->Attribute("name");
        auto* color = material->FirstChildElement("color");
        const auto rgba =
            splitFloats(color ? color->Attribute("rgba") : nullptr);
        if (name && rgba.size() == 4)
            materialColors[name] = {rgba[0], rgba[1], rgba[2], rgba[3]};
    }

    std::unordered_map<std::string, std::shared_ptr<const Scene::MeshData>>
        collisionMeshCache;
    for (const LinkRecord& link : links) {
        const int bodyIndex = bodyIndices.at(link.name);
        parseInertial(link.element, bodyIndex, scale, _data, _diagnostics);

        for (auto* visual = link.element->FirstChildElement("visual"); visual;
             visual = visual->NextSiblingElement("visual")) {
            auto* geometry = visual->FirstChildElement("geometry");
            auto* mesh =
                geometry ? geometry->FirstChildElement("mesh") : nullptr;
            if (!mesh) {
                _diagnostics.warnings.push_back(fmt::format(
                    "link '{}' has a non-mesh visual, which is not yet "
                    "represented by VisualGeomDesc",
                    link.name));
                continue;
            }
            const std::string filename =
                requiredAttribute(mesh, "filename", "visual mesh");
            const auto meshPath = resolveUrdfMeshPath(filename, urdfPath);
            if (!meshPath) {
                _diagnostics.warnings.push_back(fmt::format(
                    "could not resolve visual mesh '{}'",
                    filename));
                continue;
            }
            const Eigen::Vector3f meshScale =
                parseVec3(mesh->Attribute("scale"), Eigen::Vector3f::Ones());
            validateMeshScale(meshScale, filename);
            const Origin origin = parseOrigin(visual, scale);
            VisualGeomDesc descriptor{
                link.name,
                meshPath->string(),
                bodyIndex,
                origin.translation,
                origin.rotation,
                parseColor(visual->FirstChildElement("material"),
                           materialColors)};
            descriptor.scale = meshScale * scale;
            _data.visualGeoms.push_back(std::move(descriptor));
        }

        int collisionIndex = 0;
        for (auto* collision = link.element->FirstChildElement("collision");
             collision;
             collision = collision->NextSiblingElement("collision")) {
            auto* geometry = collision->FirstChildElement("geometry");
            if (!geometry)
                continue;
            CollisionGeomDesc descriptor;
            descriptor.name =
                collision->Attribute("name")
                    ? collision->Attribute("name")
                    : fmt::format("{}_collision_{}", link.name, collisionIndex);
            ++collisionIndex;
            const Origin origin = parseOrigin(collision, scale);
            descriptor.pos = origin.translation;
            descriptor.quat = origin.rotation;

            if (auto* sphere = geometry->FirstChildElement("sphere")) {
                descriptor.type = CollisionGeomDesc::Type::Sphere;
                sphere->QueryFloatAttribute("radius", &descriptor.size[0]);
                descriptor.size[0] *= scale;
            } else if (auto* cylinder =
                           geometry->FirstChildElement("cylinder")) {
                descriptor.type = CollisionGeomDesc::Type::Cylinder;
                float length = 0.f;
                cylinder->QueryFloatAttribute("radius", &descriptor.size[0]);
                cylinder->QueryFloatAttribute("length", &length);
                descriptor.size[0] *= scale;
                descriptor.size[1] = 0.5f * length * scale;
            } else if (auto* box = geometry->FirstChildElement("box")) {
                descriptor.type = CollisionGeomDesc::Type::Box;
                const Eigen::Vector3f size =
                    parseVec3(box->Attribute("size")) * (0.5f * scale);
                descriptor.size[0] = size.x();
                descriptor.size[1] = size.y();
                descriptor.size[2] = size.z();
            } else if (auto* mesh = geometry->FirstChildElement("mesh")) {
                descriptor.type = CollisionGeomDesc::Type::ConvexMesh;
                const std::string filename =
                    requiredAttribute(mesh, "filename", "collision mesh");
                const auto resolvedPath =
                    resolveUrdfMeshPath(filename, urdfPath);
                if (!resolvedPath) {
                    _diagnostics.warnings.push_back(fmt::format(
                        "could not resolve collision mesh '{}'; fallback "
                        "collision will be used",
                        filename));
                    _data.collisionGeoms.try_emplace(bodyIndex);
                    continue;
                }
                descriptor.meshFile = resolvedPath->string();
                const Eigen::Vector3f meshScale =
                    parseVec3(mesh->Attribute("scale"),
                              Eigen::Vector3f::Ones()) *
                    scale;
                validateMeshScale(meshScale, descriptor.meshFile);
                const auto& meshPath = *resolvedPath;
                const std::string cacheKey =
                    fmt::format("{}|{:.9g},{:.9g},{:.9g}", meshPath.string(),
                                meshScale.x(), meshScale.y(), meshScale.z());
                auto cached = collisionMeshCache.find(cacheKey);
                if (cached == collisionMeshCache.end()) {
                    try {
                        auto meshData = std::make_shared<Scene::MeshData>(
                            loadCollisionMesh(meshPath));
                        for (glm::vec3& vertex : meshData->vertices) {
                            vertex.x *= meshScale.x();
                            vertex.y *= meshScale.y();
                            vertex.z *= meshScale.z();
                        }
                        cached = collisionMeshCache
                                     .emplace(cacheKey, std::move(meshData))
                                     .first;
                    } catch (const std::exception& error) {
                        _diagnostics.warnings.push_back(fmt::format(
                            "failed to load collision mesh '{}' on link '{}': "
                            "{}; fallback collision will be used",
                            meshPath.string(), link.name, error.what()));
                        _data.collisionGeoms.try_emplace(bodyIndex);
                        continue;
                    }
                }
                descriptor.meshData = cached->second;
            } else {
                _diagnostics.warnings.push_back(
                    "unsupported URDF collision geometry on link '" +
                    link.name + "'");
                continue;
            }
            _data.collisionGeoms[bodyIndex].push_back(std::move(descriptor));
        }
    }

    SkeletonTree skeleton(std::move(orderedNames), std::move(parentIndices),
                          std::move(translations), std::move(rotations),
                          std::move(jointCounts));
    const Eigen::Quaternionf basis = Utils::coordinateSystemRotation(
        Utils::CoordinateSystem::ZUpXForward, targetCoordinateSystem);
    const Eigen::Quaternionf basisInverse = basis.conjugate();
    if (!basis.isApprox(Eigen::Quaternionf::Identity())) {
        auto convertedTranslations = skeleton.localTranslations();
        auto convertedRotations = skeleton.localRotations();
        std::vector<int> convertedJointCounts;
        convertedJointCounts.reserve(skeleton.numJoints());
        for (int i = 0; i < skeleton.numJoints(); ++i) {
            convertedTranslations[i] = basis * convertedTranslations[i];
            convertedRotations[i] =
                (basis * convertedRotations[i] * basisInverse).normalized();
            convertedJointCounts.push_back(skeleton.numJointsInBody(i));
        }
        skeleton = SkeletonTree(skeleton.nodeNames(), skeleton.parentIndices(),
                                std::move(convertedTranslations),
                                std::move(convertedRotations),
                                std::move(convertedJointCounts));
        for (VisualGeomDesc& geom : _data.visualGeoms) {
            geom.pos = basis * geom.pos;
            geom.quat = (basis * geom.quat).normalized();
        }
        for (auto& [_, bodyJoints] : _data.joints)
            for (JointDesc& joint : bodyJoints)
                joint.axis = basis * joint.axis;
        for (auto& [_, geoms] : _data.collisionGeoms) {
            for (CollisionGeomDesc& geom : geoms) {
                geom.pos = basis * geom.pos;
                geom.quat = (basis * geom.quat).normalized();
            }
        }
        for (auto& [_, inertial] : _data.inertials) {
            inertial.com = basis * inertial.com;
            inertial.quat = (basis * inertial.quat).normalized();
        }
    }

    _data.skeletonTree =
        std::make_shared<const SkeletonTree>(std::move(skeleton));
    _data.traversalOrder = order;
}

URDFImportResult
URDFLoader::parse(const std::string& urdfPath, float scale,
                  const std::string& order,
                  Utils::CoordinateSystem targetCoordinateSystem) {
    URDFLoader loader;
    loader.parseIntoData(urdfPath, scale, order, targetCoordinateSystem);
    URDFImportResult result;
    result.articulation = std::move(loader._data);
    result.diagnostics = std::move(loader._diagnostics);
    result.diagnostics.printWarnings("URDFLoader " + urdfPath);
    return result;
}

ArticulationDesc
URDFLoader::load(const std::string& urdfPath, float scale,
                 const std::string& order,
                 Utils::CoordinateSystem targetCoordinateSystem) {
    return std::move(
        parse(urdfPath, scale, order, targetCoordinateSystem).articulation);
}

} // namespace Asset
} // namespace KE
