///
/// Articulation description Python bindings.
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <cmath>

#include "animation/skeleton_math.hpp"
#include "asset/articulation_desc.hpp"

namespace py = pybind11;
using namespace KE;
using namespace KE::Asset;

void bind_articulation_desc(py::module& m) {
    py::module asset = m.attr("asset").cast<py::module>();

    py::enum_<JointDesc::Type>(asset, "JointDescType")
        .value("REVOLUTE", JointDesc::Type::Revolute)
        .value("PRISMATIC", JointDesc::Type::Prismatic);

    py::class_<JointDesc>(
        asset, "JointDesc",
        "Joint description imported from robot/character assets.")
        .def(py::init([](const std::string& name, JointDesc::Type type,
                         Eigen::Vector3f axis, Eigen::Vector3f jointOffset,
                         float lo, float hi, float kp, float kd) {
                 if (!axis.allFinite() || axis.norm() < 1e-6f ||
                     !jointOffset.allFinite() || !std::isfinite(lo) ||
                     !std::isfinite(hi) || lo >= hi || !std::isfinite(kp) ||
                     !std::isfinite(kd) || kp < 0 || kd < 0)
                     throw py::value_error(
                         "Invalid joint axis, jointOffset, limits, or gains");
                 JointDesc j;
                 j.name = name;
                 j.type = type;
                 j.axis = axis.normalized();
                 j.jointOffset = jointOffset;
                 j.loLimit = lo;
                 j.hiLimit = hi;
                 j.kp = kp;
                 j.kd = kd;
                 return j;
             }),
             py::arg("name"), py::kw_only(), py::arg("type"), py::arg("axis"),
             py::arg("joint_offset"), py::arg("lo_limit"), py::arg("hi_limit"),
             py::arg("kp") = 0.f, py::arg("kd") = 0.f)
        .def_readonly("type", &JointDesc::type)
        .def_property_readonly("joint_offset",
                               [](const JointDesc& j) { return j.jointOffset; })
        .def_readonly("name", &JointDesc::name, "Joint name.")
        .def_readonly("lo_limit", &JointDesc::loLimit, "Lower joint limit.")
        .def_readonly("hi_limit", &JointDesc::hiLimit, "Upper joint limit.")
        .def_readonly("armature", &JointDesc::armature, "Joint-space armature.")
        .def_readonly("kp", &JointDesc::kp, "Joint stiffness.")
        .def_readonly("kd", &JointDesc::kd, "Joint damping.")
        .def_readonly("effort_limit", &JointDesc::effortLimit,
                      "Maximum absolute joint effort.")
        .def_readonly("velocity_limit", &JointDesc::velocityLimit,
                      "Maximum absolute joint velocity.")
        .def_property_readonly(
            "axis", [](const JointDesc& j) { return Animation::toGlm(j.axis); },
            "Joint axis.");

    py::enum_<SiteDesc::Type>(asset, "SiteDescType",
                              "MJCF site geometry description type.")
        .value("SPHERE", SiteDesc::Type::Sphere)
        .value("CAPSULE", SiteDesc::Type::Capsule)
        .value("BOX", SiteDesc::Type::Box);

    py::class_<FixedFrameDesc>(asset, "FixedFrameDesc",
        "A frame fixed relative to its owning body, not to the world.")
        .def(py::init([](const std::string& name, int bodyIndex,
                         Eigen::Vector3f pos, Eigen::Vector4f xyzw) {
            FixedFrameDesc f;
            f.name = name;
            f.bodyIndex = bodyIndex;
            f.pos = pos;
            f.quat = Eigen::Quaternionf(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
            if (name.empty() || bodyIndex < 0 || !pos.allFinite() ||
                !xyzw.allFinite() || std::abs(f.quat.norm() - 1.f) > 1e-4f)
                throw py::value_error("Invalid fixed frame name, owner or pose");
            return f;
        }), py::arg("name"), py::kw_only(), py::arg("body_index"),
            py::arg("pos") = Eigen::Vector3f::Zero().eval(),
            py::arg("quat_xyzw") = Eigen::Vector4f(0, 0, 0, 1))
        .def_readonly("name", &FixedFrameDesc::name)
        .def_readonly("body_index", &FixedFrameDesc::bodyIndex)
        .def_property_readonly("pos", [](const FixedFrameDesc& f) { return Animation::toGlm(f.pos); })
        .def_property_readonly("quat", [](const FixedFrameDesc& f) { return Animation::toGlm(f.quat); });

    py::class_<SiteDesc, FixedFrameDesc>(
        asset, "SiteDesc",
        "Imported MJCF site description attached to a character body.")
        .def_readonly("type", &SiteDesc::type, "Site geometry type.")
        .def_readonly("name", &SiteDesc::name, "Site name.")
        .def_readonly("body_index", &SiteDesc::bodyIndex,
                      "Index of the body this site belongs to.")
        .def_property_readonly(
            "pos", [](const SiteDesc& s) { return Animation::toGlm(s.pos); },
            "Local site position.")
        .def_property_readonly(
            "quat", [](const SiteDesc& s) { return Animation::toGlm(s.quat); },
            "Local site orientation.")
        .def_property_readonly(
            "size", [](const SiteDesc& s) { return Animation::toGlm(s.size); },
            "Site size parameters.")
        .def_property_readonly(
            "rgba",
            [](const SiteDesc& s) {
                return glm::vec4(s.rgba.x(), s.rgba.y(), s.rgba.z(),
                                 s.rgba.w());
            },
            "Site display color.")
        .def_readonly("has_zaxis", &SiteDesc::hasZAxis,
                      "Whether this site has an explicit z-axis.")
        .def_property_readonly(
            "zaxis",
            [](const SiteDesc& s) { return Animation::toGlm(s.zaxis); },
            "Explicit site z-axis if present.");

    py::class_<InertialDesc>(asset, "InertialDesc",
                             "Imported body-local inertial properties.")
        .def(py::init([](float mass, Eigen::Vector3f diagonal,
                         Eigen::Vector3f com) {
                 if (!std::isfinite(mass) || mass <= 0 ||
                     !diagonal.allFinite() || (diagonal.array() <= 0).any() ||
                     !com.allFinite() ||
                     2 * diagonal.maxCoeff() > diagonal.sum() + 1e-6f)
                     throw py::value_error("Mass and physically valid diagonal "
                                           "inertia must be positive");
                 InertialDesc i;
                 i.mass = mass;
                 i.diagInertia = diagonal;
                 i.com = com;
                 return i;
             }),
             py::arg("mass"), py::kw_only(), py::arg("diag_inertia"),
             py::arg("com"))
        .def_readonly("mass", &InertialDesc::mass, "Body mass.")
        .def_property_readonly(
            "com",
            [](const InertialDesc& i) { return Animation::toGlm(i.com); },
            "Body-local center of mass.")
        .def_property_readonly(
            "quat",
            [](const InertialDesc& i) { return Animation::toGlm(i.quat); },
            "Body-local inertial frame orientation.")
        .def_property_readonly(
            "diag_inertia",
            [](const InertialDesc& i) {
                return Animation::toGlm(i.diagInertia);
            },
            "Diagonal inertia in the inertial frame.");

    py::class_<VisualGeomDesc>(
        asset, "VisualGeomDesc",
        "Visual mesh description imported from a character asset.")
        .def_readonly("body_name", &VisualGeomDesc::bodyName,
                      "Owning body name.")
        .def_readonly("mesh_file", &VisualGeomDesc::meshFile, "Mesh file path.")
        .def_readonly("body_index", &VisualGeomDesc::bodyIndex,
                      "Owning body index.")
        .def_property_readonly(
            "pos",
            [](const VisualGeomDesc& m) { return Animation::toGlm(m.pos); },
            "Local mesh position.")
        .def_property_readonly(
            "quat",
            [](const VisualGeomDesc& m) {
                return glm::quat(m.quat.w(), m.quat.x(), m.quat.y(),
                                 m.quat.z());
            },
            "Local mesh orientation.")
        .def_property_readonly(
            "rgba",
            [](const VisualGeomDesc& m) {
                return glm::vec4(m.rgba.x(), m.rgba.y(), m.rgba.z(),
                                 m.rgba.w());
            },
            "Mesh display color.")
        .def_property_readonly(
            "scale",
            [](const VisualGeomDesc& m) { return Animation::toGlm(m.scale); },
            "Scale applied to mesh-local vertices before the visual pose.");

    py::enum_<CollisionGeomDesc::Type>(
        asset, "CollisionGeomDescType",
        "Collision geometry description type imported from character assets.")
        .value("CAPSULE", CollisionGeomDesc::Type::Capsule)
        .value("CYLINDER", CollisionGeomDesc::Type::Cylinder)
        .value("SPHERE", CollisionGeomDesc::Type::Sphere)
        .value("BOX", CollisionGeomDesc::Type::Box)
        .value("CONVEX_MESH", CollisionGeomDesc::Type::ConvexMesh);

    py::class_<CollisionGeomDesc>(
        asset, "CollisionGeomDesc",
        "Imported body-local collision geometry description.")
        .def(py::init([](CollisionGeomDesc::Type type, Eigen::Vector3f size,
                         Eigen::Vector3f position,
                         std::array<float, 4> rotation) {
                 if (type != CollisionGeomDesc::Type::Box &&
                     type != CollisionGeomDesc::Type::Sphere)
                     throw py::value_error(
                         "Authored shapes currently support BOX and SPHERE");
                 Eigen::Quaternionf q(rotation[3], rotation[0], rotation[1],
                                      rotation[2]);
                 if (!size.allFinite() || size[0] <= 0 ||
                     !position.allFinite() || !q.coeffs().allFinite() ||
                     std::abs(q.squaredNorm() - 1.f) > 1e-3f ||
                     (type == CollisionGeomDesc::Type::Box &&
                      (size.array() <= 0).any()))
                     throw py::value_error(
                         "Invalid shape dimensions or local pose");
                 CollisionGeomDesc g;
                 g.type = type;
                 g.pos = position;
                 g.quat = q.normalized();
                 for (int i = 0; i < 3; ++i)
                     g.size[i] = size[i];
                 return g;
             }),
             py::kw_only(), py::arg("type"), py::arg("size"),
             py::arg("position"), py::arg("rotation_xyzw"))
        .def_readonly("type", &CollisionGeomDesc::type,
                      "Collision geometry type.")
        .def_readonly("name", &CollisionGeomDesc::name,
                      "Imported MJCF geom name, if present.")
        .def_readonly("mesh_file", &CollisionGeomDesc::meshFile,
                      "Source mesh file for convex mesh collision geoms.")
        .def_property_readonly(
            "mesh_data",
            [](const CollisionGeomDesc& g) {
                return g.meshData
                           ? std::make_shared<Scene::MeshData>(*g.meshData)
                           : std::shared_ptr<Scene::MeshData>{};
            },
            "Imported mesh payload used for convex cooking.")
        .def_property_readonly(
            "pos",
            [](const CollisionGeomDesc& g) { return Animation::toGlm(g.pos); },
            "Local collision position.")
        .def_property_readonly(
            "quat",
            [](const CollisionGeomDesc& g) { return Animation::toGlm(g.quat); },
            "Local collision orientation.")
        .def_property_readonly(
            "size",
            [](const CollisionGeomDesc& g) {
                return std::vector<float>{g.size[0], g.size[1], g.size[2]};
            },
            "Collision size parameters.")
        .def_readonly("has_from_to", &CollisionGeomDesc::hasFromTo,
                      "Whether capsule-style from/to endpoints are present.")
        .def_property_readonly(
            "from_pos",
            [](const CollisionGeomDesc& g) { return Animation::toGlm(g.from); },
            "Collision endpoint start position.")
        .def_property_readonly(
            "to_pos",
            [](const CollisionGeomDesc& g) { return Animation::toGlm(g.to); },
            "Collision endpoint end position.")
        .def_readonly("friction", &CollisionGeomDesc::friction,
                      "Imported MuJoCo sliding friction value.")
        .def_readonly("physics_material", &CollisionGeomDesc::physicsMaterial,
                      "PhysX-style material factors derived from this geom.")
        .def_readonly("condim", &CollisionGeomDesc::condim,
                      "Imported contact dimensionality.")
        .def_readonly("margin", &CollisionGeomDesc::margin,
                      "Imported collision margin.")
        .def_readonly("is_fallback", &CollisionGeomDesc::isFallback,
                      "Whether KangEngine synthesized this fallback shape.");

    py::class_<ArticulationDesc>(
        asset, "ArticulationDesc",
        "Imported articulation description with skeleton, visual, collision, "
        "joint, and site payloads.")
        .def(py::init([](std::shared_ptr<Animation::SkeletonTree> tree,
                         JointDescMap joints, CollisionGeomDescMap shapes,
                         InertialDescMap inertials) {
                 if (!tree || tree->numJoints() == 0)
                     throw py::value_error(
                         "Articulation description requires a nonempty tree");
                 const int n = tree->numJoints();
                 std::vector<int> counts(n, 0);
                 for (int i = 0; i < n; ++i) {
                     const int parent = tree->parentIndex(i);
                     if ((i == 0 && parent != -1) ||
                         (i > 0 && (parent < 0 || parent >= i)))
                         throw py::value_error(
                             "Links must form one parent-before-child tree");
                 }
                 for (const auto& [index, axes] : joints) {
                     if (index <= 0 || index >= n || axes.empty() ||
                         axes.size() > 3)
                         throw py::value_error(
                             "Invalid inbound joint body index or axis count");
                     for (const auto& axis : axes)
                         if (axes.size() > 1 &&
                             (axis.type != JointDesc::Type::Revolute ||
                              !axis.jointOffset.isApprox(axes[0].jointOffset)))
                             throw py::value_error("Spherical axes must share "
                                                   "a child-local jointOffset");
                     counts[index] = static_cast<int>(axes.size());
                 }
                 for (const auto& item : shapes)
                     if (item.first < 0 || item.first >= n)
                         throw py::value_error(
                             "Collision body index out of range");
                 for (const auto& item : inertials)
                     if (item.first < 0 || item.first >= n)
                         throw py::value_error(
                             "Inertial body index out of range");
                 ArticulationDesc d;
                 d.skeletonTree = std::make_shared<Animation::SkeletonTree>(
                     tree->nodeNames(), tree->parentIndices(),
                     tree->localTranslations(), tree->localRotations(),
                     std::move(counts));
                 d.traversalOrder = "AUTHORED";
                 d.joints = std::move(joints);
                 d.collisionGeoms = std::move(shapes);
                 d.inertials = std::move(inertials);
                 return d;
             }),
             py::arg("skeleton_tree"), py::kw_only(), py::arg("joints"),
             py::arg("collision_geoms"), py::arg("inertials"))
        .def_readonly("skeleton_tree", &ArticulationDesc::skeletonTree,
                      "Imported skeleton hierarchy.")
        .def_readonly("traversal_order", &ArticulationDesc::traversalOrder,
                      "Body traversal order used by the imported model.")
        .def_readonly("visual_geoms", &ArticulationDesc::visualGeoms,
                      "Visual mesh descriptions.")
        .def_readonly("asset_dir", &ArticulationDesc::assetDir,
                      "Directory used to resolve mesh files.")
        .def_readonly("sites", &ArticulationDesc::sites,
                      "Imported site markers.")
        .def_readonly("fixed_frames", &ArticulationDesc::fixedFrames)
        .def("add_fixed_frame", [](ArticulationDesc& d, const FixedFrameDesc& frame) {
            if (frame.bodyIndex >= d.skeletonTree->numJoints())
                throw py::value_error("Fixed frame body index out of range");
            if (d.sites.count(frame.name))
                throw py::value_error("Duplicate fixed frame name: " + frame.name);
            for (const auto& f : d.fixedFrames)
                if (f.name == frame.name)
                    throw py::value_error("Duplicate fixed frame name: " + frame.name);
            d.fixedFrames.push_back(frame);
        }, py::arg("frame"))
        .def_property_readonly(
            "joints",
            [](const ArticulationDesc& d) {
                py::dict result;
                for (const auto& [idx, jvec] : d.joints)
                    result[py::int_(idx)] = jvec;
                return result;
            },
            "JointDesc metadata keyed by body index.")
        .def_property_readonly(
            "collision_geoms",
            [](const ArticulationDesc& d) {
                py::dict result;
                for (const auto& [idx, geoms] : d.collisionGeoms)
                    result[py::int_(idx)] = geoms;
                return result;
            },
            "Collision geometry descriptions keyed by body index.")
        .def_property_readonly(
            "inertials",
            [](const ArticulationDesc& d) {
                py::dict result;
                for (const auto& [idx, inertial] : d.inertials)
                    result[py::int_(idx)] = inertial;
                return result;
            },
            "Inertial descriptions keyed by body index.");
}
