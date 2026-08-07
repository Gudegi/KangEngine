///
/// Articulation description Python bindings.
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "animation/skeleton_math.hpp"
#include "asset/articulation_desc.hpp"

namespace py = pybind11;
using namespace KE;
using namespace KE::Asset;

void bind_articulation_desc(py::module& m) {
    py::module asset = m.attr("asset").cast<py::module>();

    py::class_<JointDesc>(
        asset, "JointDesc",
        "Joint description imported from robot/character assets.")
        .def_readonly("name", &JointDesc::name, "Joint name.")
        .def_readonly("lo_limit", &JointDesc::loLimit, "Lower joint limit.")
        .def_readonly("hi_limit", &JointDesc::hiLimit, "Upper joint limit.")
        .def_readonly("armature", &JointDesc::armature, "Joint-space armature.")
        .def_property_readonly(
            "axis", [](const JointDesc& j) { return Animation::toGlm(j.axis); },
            "Joint axis.");

    py::enum_<SiteDesc::Type>(asset, "SiteDescType",
                              "MJCF site geometry description type.")
        .value("SPHERE", SiteDesc::Type::Sphere)
        .value("CAPSULE", SiteDesc::Type::Capsule)
        .value("BOX", SiteDesc::Type::Box);

    py::class_<SiteDesc>(
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
            "Mesh display color.");

    py::enum_<CollisionGeomDesc::Type>(
        asset, "CollisionGeomDescType",
        "Collision geometry description type imported from character assets.")
        .value("CAPSULE", CollisionGeomDesc::Type::Capsule)
        .value("CYLINDER", CollisionGeomDesc::Type::Cylinder)
        .value("SPHERE", CollisionGeomDesc::Type::Sphere)
        .value("BOX", CollisionGeomDesc::Type::Box);

    py::class_<CollisionGeomDesc>(
        asset, "CollisionGeomDesc",
        "Imported body-local collision geometry description.")
        .def_readonly("type", &CollisionGeomDesc::type,
                      "Collision geometry type.")
        .def_readonly("name", &CollisionGeomDesc::name,
                      "Imported MJCF geom name, if present.")
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
        .def_readonly("skeleton_tree", &ArticulationDesc::skeletonTree,
                      "Imported skeleton hierarchy.")
        .def_readonly("visual_geoms", &ArticulationDesc::visualGeoms,
                      "Visual mesh descriptions.")
        .def_readonly("asset_dir", &ArticulationDesc::assetDir,
                      "Directory used to resolve mesh files.")
        .def_readonly("sites", &ArticulationDesc::sites, "Imported site markers.")
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
