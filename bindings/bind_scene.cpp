///
/// Scene Backend Python Bindings
/// With USD Stage access for Python OpenUSD integration
///

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "engine/scene/scene_backend.hpp"
#include "engine/core/app/app.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/scene/debug_draw.hpp"
#include "engine/scene/component/transform_component.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/light_component.hpp"
#include "engine/scene/component/camera_component.hpp"
#include "engine/scene/component/material_binding_component.hpp"
#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/component/articulation_component.hpp"
#include "engine/scene/component/articulation_binding_component.hpp"
#include "engine/scene/component/collision_shape_component.hpp"
#include "engine/scene/component/scene_render_system.hpp"
#include "engine/scene/scene_resource_manager.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/native/token.hpp"
#include "engine/graphics/material/material.hpp"
#include "py_array_view.hpp"

#ifdef KANGENGINE_USE_USD
#include "engine/scene/usd/usd_scene.hpp"
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#endif

namespace py = pybind11;

void bind_scene(py::module& m) {
    py::module scene = m.def_submodule(
        "scene", "Scene graph, prim, mesh, and debug drawing APIs.");

    // Token class
    py::class_<KE::Scene::Token>(
        scene, "Token", "Interned-style attribute key used by scene prims.")
        .def(py::init<>(), "Create an empty token.")
        .def(py::init<const std::string&>(), py::arg("value"),
             "Create a token from a string.")
        .def("id", &KE::Scene::Token::id,
             "Return the token's stable numeric id.")
        .def("str", &KE::Scene::Token::str, "Return the token string.")
        .def("__eq__", &KE::Scene::Token::operator==)
        .def("__ne__", &KE::Scene::Token::operator!=)
        .def("__repr__", [](const KE::Scene::Token& t) {
            return "Token('" + t.str() + "')";
        });

    // PrimType enum
    py::enum_<KE::Scene::PrimType>(
        scene, "PrimType",
        "Scene prim categories used by KangEngine's scene graph.")
        .value("Root", KE::Scene::PrimType::Root)
        .value("Xform", KE::Scene::PrimType::Xform)
        .value("Mesh", KE::Scene::PrimType::Mesh)
        .value("MeshInstance", KE::Scene::PrimType::MeshInstance)
        .value("Camera", KE::Scene::PrimType::Camera)
        .value("Light", KE::Scene::PrimType::Light)
        .value("Resource", KE::Scene::PrimType::Resource)
        .export_values();

    py::enum_<KE::Scene::ResourceType>(
        scene, "ResourceType",
        "Kind of shared resource represented by a ResourceComponent.")
        .value("Unknown", KE::Scene::ResourceType::Unknown)
        .value("Mesh", KE::Scene::ResourceType::Mesh)
        .value("Material", KE::Scene::ResourceType::Material)
        .value("Texture", KE::Scene::ResourceType::Texture)
        .value("Shader", KE::Scene::ResourceType::Shader)
        .export_values();
    scene.attr("InvalidResourceHandle") =
        py::int_(KE::Scene::InvalidResourceHandle);

    py::enum_<KE::Scene::ArticulationPrimRole>(
        scene, "ArticulationPrimRole",
        "Role of a Prim inside an articulated character/robot.")
        .value("Root", KE::Scene::ArticulationPrimRole::Root)
        .value("BodyFrame", KE::Scene::ArticulationPrimRole::BodyFrame)
        .value("VisualGeom", KE::Scene::ArticulationPrimRole::VisualGeom)
        .value("CollisionGeom", KE::Scene::ArticulationPrimRole::CollisionGeom)
        .export_values();

    py::enum_<KE::Scene::CollisionShapeType>(
        scene, "CollisionShapeType",
        "Primitive collision shape type mirrored onto debug collision prims.")
        .value("Sphere", KE::Scene::CollisionShapeType::Sphere)
        .value("Capsule", KE::Scene::CollisionShapeType::Capsule)
        .value("Cylinder", KE::Scene::CollisionShapeType::Cylinder)
        .value("Box", KE::Scene::CollisionShapeType::Box)
        .export_values();

    py::enum_<KE::Scene::ManipulationPolicy>(
        scene, "ManipulationPolicy",
        "How viewport manipulation resolves from a picked prim.")
        .value("Inherit", KE::Scene::ManipulationPolicy::Inherit)
        .value("Self", KE::Scene::ManipulationPolicy::Self)
        .value("Parent", KE::Scene::ManipulationPolicy::Parent)
        .value("Root", KE::Scene::ManipulationPolicy::Root)
        .value("Disabled", KE::Scene::ManipulationPolicy::Disabled)
        .export_values();

    py::enum_<KE::Scene::LightType>(
        scene, "LightType",
        "Light prim subtype used by renderer scene-light sync.")
        .value("Directional", KE::Scene::LightType::Directional)
        .value("Point", KE::Scene::LightType::Point)
        .value("Spot", KE::Scene::LightType::Spot)
        .export_values();

    py::enum_<KE::Scene::CameraProjectionType>(
        scene, "CameraProjectionType",
        "Projection mode used by scene CameraComponent.")
        .value("Perspective", KE::Scene::CameraProjectionType::Perspective)
        .value("Orthographic", KE::Scene::CameraProjectionType::Orthographic)
        .export_values();

    py::class_<KE::Scene::RenderComponent,
               std::shared_ptr<KE::Scene::RenderComponent>>(
        scene, "RenderComponent",
        "Renderer-independent visual state attached to one scene prim. "
        "Renderer handles are owned by SceneRenderSystem, not by this object.")
        .def_property_readonly("attached",
                               &KE::Scene::RenderComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner", &KE::Scene::RenderComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property("visible", &KE::Scene::RenderComponent::isVisible,
                      &KE::Scene::RenderComponent::setVisible,
                      "Get or set local render visibility. Registered "
                      "components sync this change to the renderer.")
        .def_property("double_sided",
                      &KE::Scene::RenderComponent::isDoubleSided,
                      &KE::Scene::RenderComponent::setDoubleSided,
                      "Get or set double-sided rendering. Registered "
                      "components sync this change to the renderer.")
        .def_property("casts_shadow", &KE::Scene::RenderComponent::castsShadow,
                      &KE::Scene::RenderComponent::setCastsShadow,
                      "Get or set shadow casting. Registered components sync "
                      "this change to the renderer.")
        .def_property_readonly("alpha_mode",
                               &KE::Scene::RenderComponent::alphaMode,
                               "Return the alpha rendering mode.")
        .def_property_readonly("alpha_cutoff",
                               &KE::Scene::RenderComponent::alphaCutoff,
                               "Return the alpha-mask cutoff.")
        .def_property("transform_source",
                      &KE::Scene::RenderComponent::transformSource,
                      &KE::Scene::RenderComponent::setTransformSource,
                      "Get or set the transform data source.")
        .def_property_readonly("mesh_data",
                               &KE::Scene::RenderComponent::resolveMeshData,
                               "Resolve mesh data from the owning prim.")
        .def_property_readonly("version", &KE::Scene::RenderComponent::version,
                               "Return the visual state version.")
        .def("__repr__", [](const KE::Scene::RenderComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<RenderComponent path='" + path +
                   "' version=" + std::to_string(c.version()) + ">";
        });

    py::class_<KE::Scene::TransformComponent,
               std::shared_ptr<KE::Scene::TransformComponent>>(
        scene, "TransformComponent",
        "Local/world transform state attached to every scene prim.")
        .def_property_readonly("attached",
                               &KE::Scene::TransformComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner", &KE::Scene::TransformComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property_readonly("version",
                               &KE::Scene::TransformComponent::version,
                               "Return the transform state version.")
        .def("set_local_translation",
             &KE::Scene::TransformComponent::setLocalTranslation,
             py::arg("translation"), "Set local translation.")
        .def("set_local_scale", &KE::Scene::TransformComponent::setLocalScale,
             py::arg("scale"), "Set local scale.")
        .def("set_local_rotation",
             &KE::Scene::TransformComponent::setLocalRotation,
             py::arg("rotation"), "Set local rotation.")
        .def("set_local_matrix", &KE::Scene::TransformComponent::setLocalMatrix,
             py::arg("matrix"), "Set the local transform matrix.")
        .def("set_world_translation",
             &KE::Scene::TransformComponent::setWorldTranslation,
             py::arg("translation"), "Set world translation.")
        .def("set_world_rotation",
             &KE::Scene::TransformComponent::setWorldRotation,
             py::arg("rotation"), "Set world rotation.")
        .def("set_world_matrix", &KE::Scene::TransformComponent::setWorldMatrix,
             py::arg("matrix"), "Set the world transform matrix.")
        .def("compute_local_matrix",
             &KE::Scene::TransformComponent::computeLocalMatrix,
             "Return the cached/computed local matrix.")
        .def("compute_world_matrix",
             &KE::Scene::TransformComponent::computeWorldMatrix,
             "Return the cached/computed world matrix.")
        .def("compute_model_matrix",
             &KE::Scene::TransformComponent::computeModelMatrix,
             "Return the model matrix, currently equal to world matrix.")
        .def("__repr__", [](const KE::Scene::TransformComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<TransformComponent path='" + path +
                   "' version=" + std::to_string(c.version()) + ">";
        });

    py::class_<KE::Scene::MeshComponent,
               std::shared_ptr<KE::Scene::MeshComponent>>(
        scene, "MeshComponent",
        "Geometry payload/reference attached to a renderable mesh prim.")
        .def_property_readonly("attached",
                               &KE::Scene::MeshComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner", &KE::Scene::MeshComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property("mesh_data", &KE::Scene::MeshComponent::meshData,
                      &KE::Scene::MeshComponent::setMeshData,
                      "Get or set directly attached mesh data.")
        .def_property("mesh_source_path",
                      &KE::Scene::MeshComponent::meshSourcePath,
                      &KE::Scene::MeshComponent::setMeshSourcePath,
                      "Get or set the source path for MeshInstance prims.")
        .def_property("resource_handle",
                      &KE::Scene::MeshComponent::resourceHandle,
                      &KE::Scene::MeshComponent::setResourceHandle,
                      "Get or set the optional SceneResourceManager handle.")
        .def("resolve_mesh_data", &KE::Scene::MeshComponent::resolveMeshData,
             "Return direct mesh data or resolved source mesh data.")
        .def_property_readonly("version", &KE::Scene::MeshComponent::version,
                               "Return the mesh component version.")
        .def("__repr__", [](const KE::Scene::MeshComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<MeshComponent path='" + path +
                   "' version=" + std::to_string(c.version()) + ">";
        });

    py::class_<KE::Scene::MaterialBindingComponent,
               std::shared_ptr<KE::Scene::MaterialBindingComponent>>(
        scene, "MaterialBindingComponent",
        "Scene-level non-owning binding from one prim to a renderer material.")
        .def_property_readonly("attached",
                               &KE::Scene::MaterialBindingComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner",
                               &KE::Scene::MaterialBindingComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property(
            "material", &KE::Scene::MaterialBindingComponent::material,
            &KE::Scene::MaterialBindingComponent::setMaterial,
            py::return_value_policy::reference,
            "Get or set the bound material. The component does not own it.")
        .def("clear_material",
             &KE::Scene::MaterialBindingComponent::clearMaterial,
             "Clear the material binding.")
        .def_property_readonly("version",
                               &KE::Scene::MaterialBindingComponent::version,
                               "Return the material binding version.")
        .def("__repr__", [](const KE::Scene::MaterialBindingComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<MaterialBindingComponent path='" + path +
                   "' version=" + std::to_string(c.version()) + ">";
        });

    py::class_<KE::Scene::ResourceComponent,
               std::shared_ptr<KE::Scene::ResourceComponent>>(
        scene, "ResourceComponent",
        "Lightweight resource metadata attached to a Resource prim.")
        .def_property_readonly("attached",
                               &KE::Scene::ResourceComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner", &KE::Scene::ResourceComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property("type", &KE::Scene::ResourceComponent::type,
                      &KE::Scene::ResourceComponent::setType,
                      "Get or set the resource type.")
        .def_property("handle", &KE::Scene::ResourceComponent::handle,
                      &KE::Scene::ResourceComponent::setHandle,
                      "Get or set the SceneResourceManager handle.")
        .def_property("uri", &KE::Scene::ResourceComponent::uri,
                      &KE::Scene::ResourceComponent::setUri,
                      "Get or set the resource URI/path.")
        .def_property("display_name",
                      &KE::Scene::ResourceComponent::displayName,
                      &KE::Scene::ResourceComponent::setDisplayName,
                      "Get or set a user-facing resource name.")
        .def_property_readonly("version",
                               &KE::Scene::ResourceComponent::version,
                               "Return the resource metadata version.")
        .def("__repr__", [](const KE::Scene::ResourceComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<ResourceComponent path='" + path +
                   "' version=" + std::to_string(c.version()) + ">";
        });

    py::class_<KE::Scene::ArticulationComponent,
               std::shared_ptr<KE::Scene::ArticulationComponent>>(
        scene, "ArticulationComponent",
        "Root-level metadata for an articulated object subtree.")
        .def_property_readonly("attached",
                               &KE::Scene::ArticulationComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner",
                               &KE::Scene::ArticulationComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property("asset_path",
                      &KE::Scene::ArticulationComponent::assetPath,
                      &KE::Scene::ArticulationComponent::setAssetPath,
                      "Imported articulation asset path or URI.")
        .def_property("root_path", &KE::Scene::ArticulationComponent::rootPath,
                      &KE::Scene::ArticulationComponent::setRootPath,
                      "Scene root prim path for this articulation.")
        .def_property("mesh_asset_base_path",
                      &KE::Scene::ArticulationComponent::meshAssetBasePath,
                      &KE::Scene::ArticulationComponent::setMeshAssetBasePath,
                      "Shared mesh resource base path, if any.")
        .def_property("body_count",
                      &KE::Scene::ArticulationComponent::bodyCount,
                      &KE::Scene::ArticulationComponent::setBodyCount,
                      "Number of body frame prims.")
        .def_property("render_prim_count",
                      &KE::Scene::ArticulationComponent::renderPrimCount,
                      &KE::Scene::ArticulationComponent::setRenderPrimCount,
                      "Number of renderable visual prims.")
        .def_property("split_visual_geoms",
                      &KE::Scene::ArticulationComponent::splitVisualGeoms,
                      &KE::Scene::ArticulationComponent::setSplitVisualGeoms,
                      "Whether visual geoms were instantiated as split child "
                      "prims.")
        .def("set_articulation_metadata",
             &KE::Scene::ArticulationComponent::setArticulationMetadata,
             py::arg("root_path"), py::arg("asset_path"),
             py::arg("mesh_asset_base_path"), py::arg("body_count"),
             py::arg("render_prim_count"), py::arg("split_visual_geoms"),
             "Set all root articulation metadata fields at once.")
        .def_property_readonly("version",
                               &KE::Scene::ArticulationComponent::version,
                               "Return the articulation metadata version.")
        .def("__repr__", [](const KE::Scene::ArticulationComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<ArticulationComponent path='" + path +
                   "' body_count=" + std::to_string(c.bodyCount()) +
                   " render_prim_count=" + std::to_string(c.renderPrimCount()) +
                   ">";
        });

    py::class_<KE::Scene::ArticulationBindingComponent,
               std::shared_ptr<KE::Scene::ArticulationBindingComponent>>(
        scene, "ArticulationBindingComponent",
        "Metadata binding a Prim to an articulated body/frame.")
        .def_property_readonly(
            "attached", &KE::Scene::ArticulationBindingComponent::isAttached,
            "Return whether this component is attached.")
        .def_property_readonly("owner",
                               &KE::Scene::ArticulationBindingComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property("role", &KE::Scene::ArticulationBindingComponent::role,
                      &KE::Scene::ArticulationBindingComponent::setRole,
                      "Role of this prim inside the articulation.")
        .def_property("body_index",
                      &KE::Scene::ArticulationBindingComponent::bodyIndex,
                      &KE::Scene::ArticulationBindingComponent::setBodyIndex,
                      "Articulation body index this prim belongs to.")
        .def_property("body_name",
                      &KE::Scene::ArticulationBindingComponent::bodyName,
                      &KE::Scene::ArticulationBindingComponent::setBodyName,
                      "Articulation body name this prim belongs to.")
        .def_property(
            "articulation_root_path",
            &KE::Scene::ArticulationBindingComponent::articulationRootPath,
            &KE::Scene::ArticulationBindingComponent::setArticulationRootPath,
            "Root prim path of the owning articulation.")
        .def("set_binding",
             &KE::Scene::ArticulationBindingComponent::setBinding,
             py::arg("role"), py::arg("body_index"), py::arg("body_name"),
             py::arg("articulation_root_path"),
             "Set all binding fields at once.")
        .def_property_readonly(
            "version", &KE::Scene::ArticulationBindingComponent::version,
            "Return the articulation binding version.")
        .def("__repr__", [](const KE::Scene::ArticulationBindingComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<ArticulationBindingComponent path='" + path +
                   "' role=" + KE::Scene::articulationPrimRoleLabel(c.role()) +
                   " body_index=" + std::to_string(c.bodyIndex()) + ">";
        });

    py::class_<KE::Scene::CollisionShapeComponent,
               std::shared_ptr<KE::Scene::CollisionShapeComponent>>(
        scene, "CollisionShapeComponent",
        "Scene-side metadata for a collision shape debug prim.")
        .def_property_readonly("attached",
                               &KE::Scene::CollisionShapeComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner",
                               &KE::Scene::CollisionShapeComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property_readonly("shape_type",
                               &KE::Scene::CollisionShapeComponent::shapeType,
                               "Primitive collision shape type.")
        .def_property_readonly(
            "size",
            [](const KE::Scene::CollisionShapeComponent& c) {
                return c.size();
            },
            "Shape size descriptor. Interpretation depends on shape type.")
        .def_property_readonly(
            "local_position",
            [](const KE::Scene::CollisionShapeComponent& c) {
                return c.localPosition();
            },
            "Body-local collision shape position.")
        .def_property_readonly(
            "local_rotation",
            [](const KE::Scene::CollisionShapeComponent& c) {
                return c.localRotation();
            },
            "Body-local collision shape rotation.")
        .def_property_readonly("has_from_to",
                               &KE::Scene::CollisionShapeComponent::hasFromTo,
                               "Whether this shape was authored by from/to.")
        .def_property_readonly(
            "from_position",
            [](const KE::Scene::CollisionShapeComponent& c) {
                return c.fromPosition();
            },
            "Body-local from endpoint when has_from_to is true.")
        .def_property_readonly(
            "to_position",
            [](const KE::Scene::CollisionShapeComponent& c) {
                return c.toPosition();
            },
            "Body-local to endpoint when has_from_to is true.")
        .def_property_readonly(
            "static_friction",
            &KE::Scene::CollisionShapeComponent::staticFriction,
            "Reference PhysX static friction.")
        .def_property_readonly(
            "dynamic_friction",
            &KE::Scene::CollisionShapeComponent::dynamicFriction,
            "Reference PhysX dynamic friction.")
        .def_property_readonly("restitution",
                               &KE::Scene::CollisionShapeComponent::restitution,
                               "Reference PhysX restitution.")
        .def_property_readonly(
            "condim", &KE::Scene::CollisionShapeComponent::condim,
            "Imported MuJoCo contact dimensionality, if any.")
        .def_property_readonly(
            "margin", &KE::Scene::CollisionShapeComponent::margin,
            "Imported MuJoCo margin mapped to contactOffset, if any.")
        .def_property_readonly(
            "source_geom_index",
            &KE::Scene::CollisionShapeComponent::sourceGeomIndex,
            "Index of the source collision geom inside the body descriptor.")
        .def_property_readonly("version",
                               &KE::Scene::CollisionShapeComponent::version,
                               "Return the collision shape metadata version.")
        .def("__repr__", [](const KE::Scene::CollisionShapeComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<CollisionShapeComponent path='" + path + "' shape=" +
                   KE::Scene::collisionShapeTypeLabel(c.shapeType()) +
                   " geom_index=" + std::to_string(c.sourceGeomIndex()) + ">";
        });

    py::class_<KE::Scene::SceneResourceManager>(
        scene, "SceneResourceManager",
        "Scene resource manager mirrored into metadata-only /.Resources prims.")
        .def(py::init<KE::Scene::SceneBackend*>(), py::arg("scene") = nullptr,
             "Create a SceneResourceManager optionally bound to a scene "
             "backend.")
        .def("bind_scene", &KE::Scene::SceneResourceManager::bindScene,
             py::arg("scene"), "Bind the scene used for Resource prim mirrors.")
        .def("register_mesh", &KE::Scene::SceneResourceManager::registerMesh,
             py::arg("name"), py::arg("mesh"), py::arg("uri") = "",
             "Register mesh data and return a resource handle.")
        .def("register_material",
             &KE::Scene::SceneResourceManager::registerMaterial,
             py::arg("name"), py::arg("material"), py::arg("uri") = "",
             "Register a non-owned material and return a resource handle.")
        .def("register_texture",
             &KE::Scene::SceneResourceManager::registerTexture, py::arg("name"),
             py::arg("texture"), py::arg("uri") = "",
             "Register a non-owned texture and return a resource handle.")
        .def("register_shader",
             &KE::Scene::SceneResourceManager::registerShader, py::arg("name"),
             py::arg("shader"), py::arg("uri") = "",
             "Register a non-owned shader and return a resource handle.")
        .def("mesh", &KE::Scene::SceneResourceManager::mesh, py::arg("handle"),
             "Return mesh data for a handle, or None.")
        .def("material", &KE::Scene::SceneResourceManager::material,
             py::arg("handle"), py::return_value_policy::reference,
             "Return material for a handle, or None.")
        .def("texture", &KE::Scene::SceneResourceManager::texture,
             py::arg("handle"), py::return_value_policy::reference,
             "Return texture for a handle, or None.")
        .def("shader", &KE::Scene::SceneResourceManager::shader,
             py::arg("handle"), py::return_value_policy::reference,
             "Return shader for a handle, or None.")
        .def("resource_prim", &KE::Scene::SceneResourceManager::resourcePrim,
             py::arg("handle"), py::return_value_policy::reference,
             "Return mirrored Resource prim for a handle, or None.")
        .def("usage_count", &KE::Scene::SceneResourceManager::usageCount,
             py::arg("handle"),
             "Return how many scene prim bindings currently reference this "
             "resource.")
        .def("usage_paths", &KE::Scene::SceneResourceManager::usagePaths,
             py::arg("handle"),
             "Return scene prim paths that currently reference this resource.")
        .def("is_used", &KE::Scene::SceneResourceManager::isUsed,
             py::arg("handle"),
             "Return whether at least one scene prim binding references this "
             "resource.")
        .def("invalidate_usage_cache",
             &KE::Scene::SceneResourceManager::invalidateUsageCache,
             "Mark cached resource usage counts dirty after direct scene or "
             "material mutation.")
        .def("clear", &KE::Scene::SceneResourceManager::clear)
        .def("__len__", &KE::Scene::SceneResourceManager::size);

    py::class_<KE::Scene::LightComponent,
               std::shared_ptr<KE::Scene::LightComponent>>(
        scene, "LightComponent",
        "Renderer-independent light state attached to one Light prim.")
        .def_property_readonly("attached",
                               &KE::Scene::LightComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner", &KE::Scene::LightComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property_readonly("type", &KE::Scene::LightComponent::type,
                               "Return the immutable light subtype.")
        .def_property_readonly("version", &KE::Scene::LightComponent::version,
                               "Return the light state version.")
        .def("set_directional_light",
             &KE::Scene::LightComponent::setDirectionalLight, py::arg("light"),
             "Set this component from directional light data.")
        .def("directional_light", &KE::Scene::LightComponent::directionalLight,
             "Return directional light data in world space.")
        .def("set_point_light", &KE::Scene::LightComponent::setPointLight,
             py::arg("light"), "Set this component from point light data.")
        .def("point_light", &KE::Scene::LightComponent::pointLight,
             "Return point light data in world space.")
        .def("set_spot_light", &KE::Scene::LightComponent::setSpotLight,
             py::arg("light"), "Set this component from spot light data.")
        .def("spot_light", &KE::Scene::LightComponent::spotLight,
             "Return spot light data in world space.")
        .def("__repr__", [](const KE::Scene::LightComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<LightComponent path='" + path +
                   "' version=" + std::to_string(c.version()) + ">";
        });

    py::class_<KE::Scene::CameraComponent,
               std::shared_ptr<KE::Scene::CameraComponent>>(
        scene, "CameraComponent",
        "Renderer-independent authored camera state attached to one Camera "
        "prim.")
        .def_property_readonly("attached",
                               &KE::Scene::CameraComponent::isAttached,
                               "Return whether this component is attached.")
        .def_property_readonly("owner", &KE::Scene::CameraComponent::owner,
                               py::return_value_policy::reference,
                               "Return the owning prim, or None after detach.")
        .def_property_readonly("projection_type",
                               &KE::Scene::CameraComponent::projectionType,
                               "Return the camera projection mode.")
        .def_property_readonly("version", &KE::Scene::CameraComponent::version,
                               "Return the camera state version.")
        .def("set_perspective", &KE::Scene::CameraComponent::setPerspective,
             py::arg("vertical_fov_degrees"), py::arg("near_plane"),
             py::arg("far_plane"), "Set perspective projection settings.")
        .def("set_orthographic", &KE::Scene::CameraComponent::setOrthographic,
             py::arg("vertical_size"), py::arg("near_plane"),
             py::arg("far_plane"), "Set orthographic projection settings.")
        .def("vertical_fov_degrees",
             &KE::Scene::CameraComponent::verticalFovDegrees,
             "Return perspective vertical field of view in degrees.")
        .def("orthographic_size", &KE::Scene::CameraComponent::orthographicSize,
             "Return orthographic vertical size.")
        .def("near_plane", &KE::Scene::CameraComponent::nearPlane,
             "Return near clipping distance.")
        .def("far_plane", &KE::Scene::CameraComponent::farPlane,
             "Return far clipping distance.")
        .def("position", &KE::Scene::CameraComponent::position,
             "Return world-space camera position from the owning Prim "
             "transform.")
        .def("forward", &KE::Scene::CameraComponent::forward,
             "Return world-space camera forward direction.")
        .def("up", &KE::Scene::CameraComponent::up,
             "Return world-space camera up direction.")
        .def("view_matrix", &KE::Scene::CameraComponent::viewMatrix,
             "Return view matrix from the owning Prim transform.")
        .def("projection_matrix", &KE::Scene::CameraComponent::projectionMatrix,
             py::arg("aspect"), "Return projection matrix for an aspect ratio.")
        .def("view_projection_matrix",
             &KE::Scene::CameraComponent::viewProjectionMatrix,
             py::arg("aspect"),
             "Return projection * view matrix for an aspect ratio.")
        .def("__repr__", [](const KE::Scene::CameraComponent& c) {
            const KE::Scene::Prim* owner = c.owner();
            const std::string path = owner ? owner->getPath() : "<detached>";
            return "<CameraComponent path='" + path +
                   "' version=" + std::to_string(c.version()) + ">";
        });

    py::class_<KE::Scene::SceneRenderSystem>(
        scene, "SceneRenderSystem",
        "Registry connecting scene RenderComponents to renderer resources. "
        "This hides raw renderable handles for normal scene use.")
        .def_property_readonly(
            "registration_count",
            &KE::Scene::SceneRenderSystem::registrationCount,
            "Return the number of registered render components.")
        .def("is_registered", &KE::Scene::SceneRenderSystem::isRegistered,
             py::arg("component"),
             "Return whether a render component owns renderer resources.")
        .def("shares_batch", &KE::Scene::SceneRenderSystem::sharesBatch,
             py::arg("first"), py::arg("second"),
             "Return whether two components share one renderer batch without "
             "exposing the raw handle.")
        .def("set_double_sided", &KE::Scene::SceneRenderSystem::setDoubleSided,
             py::arg("component"), py::arg("enabled") = true,
             "Set double-sided rendering through component registration.")
        .def("set_casts_shadow", &KE::Scene::SceneRenderSystem::setCastsShadow,
             py::arg("component"), py::arg("enabled") = true,
             "Set shadow casting through component registration.")
        .def("set_alpha_mode", &KE::Scene::SceneRenderSystem::setAlphaMode,
             py::arg("component"), py::arg("mode"), py::arg("cutoff") = 0.5f,
             "Set alpha rendering through component registration.")
        .def("set_material", &KE::Scene::SceneRenderSystem::setMaterial,
             py::arg("component"), py::arg("material"),
             "Replace a registered SceneGraph renderable's material and move "
             "it to the matching renderer batch. ExternalBuffer renderables "
             "reject dynamic material replacement.")
        .def(
            "set_texture",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               KE::Backend::Texture* texture, KE::TextureRole role) {
                self.setTexture(component, texture, role);
            },
            py::arg("component"), py::arg("texture"), py::arg("role"),
            "Set a texture by material role through component registration.")
        .def(
            "set_texture",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               KE::Backend::Texture* texture,
               int slot) { self.setTexture(component, texture, slot); },
            py::arg("component"), py::arg("texture"), py::arg("slot") = 0,
            "Set a texture by raw renderer slot through component "
            "registration.")
        .def("set_external_buffer",
             &KE::Scene::SceneRenderSystem::setExternalBuffer,
             py::arg("component"), py::arg("descriptor"),
             "Attach an external transform buffer through registration.")
        .def(
            "update_instances",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               const FloatArray& transforms, py::object colors) {
                auto transformVec = mat4Array(transforms, "transforms");
                std::vector<glm::vec4> colorVec;
                const std::vector<glm::vec4>* colorPtr = nullptr;
                if (!colors.is_none()) {
                    auto colorArray = colors.cast<FloatArray>();
                    colorVec = vec4Array(colorArray, "colors");
                    if (!colorVec.empty() && colorVec.size() != 1 &&
                        colorVec.size() != transformVec.size()) {
                        throw py::value_error(
                            "colors must be empty, length 1, or match "
                            "transforms length");
                    }
                    colorPtr = &colorVec;
                }
                self.updateInstances(component, transformVec, colorPtr);
            },
            py::arg("component"), py::arg("transforms"),
            py::arg("colors") = py::none(),
            "Update instanced transforms and optional colors through "
            "component registration.")
        .def(
            "update_geometry",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               const FloatArray& positions, py::object normals) {
                auto positionVec = vec3Array(positions, "positions");
                std::vector<glm::vec3> normalVec;
                if (!normals.is_none()) {
                    auto normalArray = normals.cast<FloatArray>();
                    normalVec = vec3Array(normalArray, "normals");
                    if (normalVec.size() != positionVec.size()) {
                        throw py::value_error(
                            "normals must match positions length");
                    }
                }
                self.updateGeometry(component, positionVec, normalVec);
            },
            py::arg("component"), py::arg("positions"),
            py::arg("normals") = py::none(),
            "Update dynamic vertex positions and optional normals through "
            "component registration.")
        .def(
            "update_skinning",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               const FloatArray& boneMatrices) {
                self.updateSkinning(
                    component,
                    mat4RowMajorArray(boneMatrices, "bone_matrices"));
            },
            py::arg("component"), py::arg("bone_matrices"),
            "Update skinned bone matrices through component registration.");

    // Prim class
    py::class_<KE::Scene::Prim>(
        scene, "Prim",
        "USD-like scene graph node with hierarchy, transform, and mesh data.")
        .def(py::init<const std::string&, KE::Scene::PrimType,
                      KE::Scene::Prim*>(),
             py::arg("name"), py::arg("type"), py::arg("parent") = nullptr,
             "Create a prim with an optional parent.")
        // Getters
        .def("get_name", &KE::Scene::Prim::getName,
             "Return this prim's local name.")
        .def("get_path", &KE::Scene::Prim::getPath,
             "Return this prim's absolute scene path.")
        .def("get_type", &KE::Scene::Prim::getType, "Return this prim's type.")
        .def("get_parent", &KE::Scene::Prim::getParent,
             py::return_value_policy::reference,
             "Return this prim's parent, or None for the root.")
        // Hierarchy
        .def("add_child", &KE::Scene::Prim::addChild, py::arg("name"),
             py::arg("type"), py::return_value_policy::reference,
             "Create and return a child prim.")
        .def("get_child", &KE::Scene::Prim::getChild, py::arg("name"),
             py::return_value_policy::reference,
             "Return a direct child by name.")
        .def("get_prim_at_path", &KE::Scene::Prim::getPrimAtPath,
             py::arg("path"), py::return_value_policy::reference,
             "Find a descendant prim by absolute path.")
        .def("get_children", &KE::Scene::Prim::getChildren,
             "Return direct child prims.")
        // Components
        .def("add_render_component", &KE::Scene::Prim::addRenderComponent,
             "Attach and return this prim's render component.")
        .def("get_render_component", &KE::Scene::Prim::getRenderComponent,
             "Return this prim's render component, or None.")
        .def("has_render_component", &KE::Scene::Prim::hasRenderComponent,
             "Return whether this prim has a render component.")
        .def("remove_render_component", &KE::Scene::Prim::removeRenderComponent,
             "Detach this prim's render component.")
        .def("get_transform_component", &KE::Scene::Prim::getTransformComponent,
             "Return this prim's mandatory transform component.")
        .def("has_transform_component", &KE::Scene::Prim::hasTransformComponent,
             "Return whether this prim has a transform component.")
        .def("add_mesh_component", &KE::Scene::Prim::addMeshComponent,
             "Attach and return this Mesh/MeshInstance prim's mesh component.")
        .def("get_mesh_component", &KE::Scene::Prim::getMeshComponent,
             "Return this prim's mesh component, or None.")
        .def("has_mesh_component", &KE::Scene::Prim::hasMeshComponent,
             "Return whether this prim has a mesh component.")
        .def("remove_mesh_component", &KE::Scene::Prim::removeMeshComponent,
             "Detach this prim's mesh component.")
        .def("add_material_binding_component",
             &KE::Scene::Prim::addMaterialBindingComponent,
             "Attach and return this prim's material binding component.")
        .def("get_material_binding_component",
             &KE::Scene::Prim::getMaterialBindingComponent,
             "Return this prim's material binding component, or None.")
        .def("has_material_binding_component",
             &KE::Scene::Prim::hasMaterialBindingComponent,
             "Return whether this prim has a material binding component.")
        .def("remove_material_binding_component",
             &KE::Scene::Prim::removeMaterialBindingComponent,
             "Detach this prim's material binding component.")
        .def("add_light_component", &KE::Scene::Prim::addLightComponent,
             "Attach and return this Light prim's light component.")
        .def("get_light_component", &KE::Scene::Prim::getLightComponent,
             "Return this prim's light component, or None.")
        .def("has_light_component", &KE::Scene::Prim::hasLightComponent,
             "Return whether this prim has a light component.")
        .def("remove_light_component", &KE::Scene::Prim::removeLightComponent,
             "Detach this prim's light component.")
        .def("add_camera_component", &KE::Scene::Prim::addCameraComponent,
             "Attach and return this Camera prim's camera component.")
        .def("get_camera_component", &KE::Scene::Prim::getCameraComponent,
             "Return this prim's camera component, or None.")
        .def("has_camera_component", &KE::Scene::Prim::hasCameraComponent,
             "Return whether this prim has a camera component.")
        .def("remove_camera_component", &KE::Scene::Prim::removeCameraComponent,
             "Detach this prim's camera component.")
        .def("add_resource_component", &KE::Scene::Prim::addResourceComponent,
             "Attach and return this Resource prim's resource component.")
        .def("get_resource_component", &KE::Scene::Prim::getResourceComponent,
             "Return this prim's resource component, or None.")
        .def("has_resource_component", &KE::Scene::Prim::hasResourceComponent,
             "Return whether this prim has a resource component.")
        .def("remove_resource_component",
             &KE::Scene::Prim::removeResourceComponent,
             "Detach this prim's resource component.")
        .def("add_articulation_component",
             &KE::Scene::Prim::addArticulationComponent,
             "Attach and return this prim's articulation root component.")
        .def("get_articulation_component",
             &KE::Scene::Prim::getArticulationComponent,
             "Return this prim's articulation root component, or None.")
        .def("has_articulation_component",
             &KE::Scene::Prim::hasArticulationComponent,
             "Return whether this prim has an articulation root component.")
        .def("remove_articulation_component",
             &KE::Scene::Prim::removeArticulationComponent,
             "Detach this prim's articulation root component.")
        .def("add_articulation_binding_component",
             &KE::Scene::Prim::addArticulationBindingComponent,
             "Attach and return this prim's articulation binding component.")
        .def("get_articulation_binding_component",
             &KE::Scene::Prim::getArticulationBindingComponent,
             "Return this prim's articulation binding component, or None.")
        .def("has_articulation_binding_component",
             &KE::Scene::Prim::hasArticulationBindingComponent,
             "Return whether this prim has an articulation binding component.")
        .def("remove_articulation_binding_component",
             &KE::Scene::Prim::removeArticulationBindingComponent,
             "Detach this prim's articulation binding component.")
        .def("add_collision_shape_component",
             &KE::Scene::Prim::addCollisionShapeComponent,
             "Attach and return this prim's collision shape metadata "
             "component.")
        .def("get_collision_shape_component",
             &KE::Scene::Prim::getCollisionShapeComponent,
             "Return this prim's collision shape component, or None.")
        .def("has_collision_shape_component",
             &KE::Scene::Prim::hasCollisionShapeComponent,
             "Return whether this prim has a collision shape component.")
        .def("remove_collision_shape_component",
             &KE::Scene::Prim::removeCollisionShapeComponent,
             "Detach this prim's collision shape component.")
        // Mesh data
        .def("set_mesh_data", &KE::Scene::Prim::setMeshData,
             py::arg("mesh_data"), "Attach mesh data to this prim.")
        .def("get_mesh_data", &KE::Scene::Prim::getMeshData,
             "Return mesh data attached directly to this prim.")
        .def("set_mesh_source_path", &KE::Scene::Prim::setMeshSourcePath,
             py::arg("path"), "Set a source path for mesh instancing.")
        .def("get_mesh_source_path", &KE::Scene::Prim::getMeshSourcePath,
             "Return the mesh source path for mesh instances.")
        .def("resolve_mesh_data", &KE::Scene::Prim::resolveMeshData,
             "Return direct mesh data or resolved source mesh data.")
        // Static mesh creation (returns shared_ptr for set_mesh_data
        // compatibility)
        .def_static(
            "create_square_data",
            [](float scale) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createSquareData(scale));
            },
            py::arg("scale") = 1.0f, "Create square mesh data.")
        .def_static(
            "create_plane_data",
            [](float scale, KE::UpAxis upAxis) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createPlaneData(scale, upAxis));
            },
            py::arg("scale"), py::arg("up_axis") = KE::UpAxis::Y,
            "Create plane mesh data.")
        .def_static(
            "create_sphere_data",
            [](float radius, int numLongitudes, int numLatitudes) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createSphereData(radius, numLongitudes,
                                                      numLatitudes));
            },
            py::arg("radius"), py::arg("num_longitudes"),
            py::arg("num_latitudes"), "Create sphere mesh data.")
        .def_static(
            "create_rectangle_data",
            [](float xScale, float yScale, float zScale) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createRectangleData(xScale, yScale,
                                                         zScale));
            },
            py::arg("x_scale"), py::arg("y_scale"), py::arg("z_scale"),
            "Create box/rectangle mesh data.")
        .def_static(
            "create_cylinder_data",
            [](float radius, float length, KE::UpAxis upAxis, int segments) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createCylinderData(radius, length, upAxis,
                                                        segments));
            },
            py::arg("radius"), py::arg("length"),
            py::arg("up_axis") = KE::UpAxis::Y, py::arg("segments") = 32,
            "Create cylinder mesh data.")
        .def_static(
            "create_capsule_data",
            [](float radius, float height, KE::UpAxis upAxis, int segments) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createCapsuleData(radius, height, upAxis,
                                                       segments));
            },
            py::arg("radius"), py::arg("height"),
            py::arg("up_axis") = KE::UpAxis::Y, py::arg("segments") = 32,
            "Create capsule mesh data.")
        .def_static("define_manipulation_group",
                    &KE::Scene::Prim::defineManipulationGroup, py::arg("scene"),
                    py::arg("path"), py::return_value_policy::reference,
                    "Create an Xform prim that manipulates as a group.")
        // Transform ops
        .def("set_local_translation", &KE::Scene::Prim::setLocalTranslation,
             py::arg("translation"), "Set local translation.")
        .def("set_local_scale", &KE::Scene::Prim::setLocalScale,
             py::arg("scale"), "Set local scale.")
        .def("set_local_rotation", &KE::Scene::Prim::setLocalRotation,
             py::arg("rotation"), "Set local quaternion rotation.")
        .def("set_local_matrix", &KE::Scene::Prim::setLocalMatrix,
             py::arg("matrix"), "Set local transform matrix.")
        .def("set_world_translation", &KE::Scene::Prim::setWorldTranslation,
             py::arg("translation"), "Set world translation.")
        .def("set_world_rotation", &KE::Scene::Prim::setWorldRotation,
             py::arg("rotation"), "Set world quaternion rotation.")
        .def("set_world_matrix", &KE::Scene::Prim::setWorldMatrix,
             py::arg("matrix"), "Set world transform matrix.")
        .def("add_translate_op", &KE::Scene::Prim::addTranslateOp,
             py::arg("translation"),
             "Compatibility alias for local translation.")
        .def("add_scale_op", &KE::Scene::Prim::addScaleOp, py::arg("scale"),
             "Compatibility alias for local scale.")
        .def("add_rotate_quaternion_op",
             &KE::Scene::Prim::addRotateQuaternionOp, py::arg("rotation"),
             "Compatibility alias for local quaternion rotation.")
        // Display color
        .def("set_display_color", &KE::Scene::Prim::setDisplayColor,
             py::arg("color"), "Set RGB display color metadata.")
        .def("get_display_color", &KE::Scene::Prim::getDisplayColor,
             "Return RGB display color metadata if present.")
        .def("set_display_color_alpha", &KE::Scene::Prim::setDisplayColorAlpha,
             py::arg("color"), "Set RGBA display color metadata.")
        .def("get_display_color_alpha", &KE::Scene::Prim::getDisplayColorAlpha,
             "Return RGBA display color metadata if present.")
        .def("set_material", &KE::Scene::Prim::setMaterial, py::arg("material"),
             "Bind a non-owned material to this prim at the scene level.")
        .def("get_material", &KE::Scene::Prim::getMaterial,
             py::return_value_policy::reference,
             "Return the bound material, or None.")
        // Light data
        .def("get_light_type", &KE::Scene::Prim::getLightType,
             py::arg("default_type") = KE::Scene::LightType::Point,
             "Return the light subtype for a Light prim.")
        .def("set_directional_light", &KE::Scene::Prim::setDirectionalLight,
             py::arg("light"),
             "Attach directional light data to this Light prim.")
        .def("get_directional_light", &KE::Scene::Prim::getDirectionalLight,
             "Return directional light data from this Light prim.")
        .def("set_point_light", &KE::Scene::Prim::setPointLight,
             py::arg("light"), "Attach point light data to this Light prim.")
        .def("get_point_light", &KE::Scene::Prim::getPointLight,
             "Return point light data from this Light prim.")
        .def("set_spot_light", &KE::Scene::Prim::setSpotLight, py::arg("light"),
             "Attach spot light data to this Light prim.")
        .def("get_spot_light", &KE::Scene::Prim::getSpotLight,
             "Return spot light data from this Light prim.")
        // Model matrix
        .def("compute_model_matrix", &KE::Scene::Prim::computeModelMatrix,
             "Compute this prim's model matrix.")
        .def("compute_local_matrix", &KE::Scene::Prim::computeLocalMatrix,
             "Compute this prim's local transform matrix.")
        .def("compute_world_matrix", &KE::Scene::Prim::computeWorldMatrix,
             "Compute this prim's world transform matrix.")
        .def("is_visible", &KE::Scene::Prim::isVisible,
             "Return local visibility state.")
        .def("set_visible", &KE::Scene::Prim::setVisible, py::arg("visible"),
             "Set local visibility state.")
        .def("is_visible_in_hierarchy", &KE::Scene::Prim::isVisibleInHierarchy,
             "Return visibility after parent hierarchy is considered.")
        .def("is_active", &KE::Scene::Prim::isActive,
             "Return local active state.")
        .def("set_active", &KE::Scene::Prim::setActive, py::arg("active"),
             "Set local active state.")
        .def("is_active_in_hierarchy", &KE::Scene::Prim::isActiveInHierarchy,
             "Return active state after parent hierarchy is considered.")
        .def("get_manipulation_policy", &KE::Scene::Prim::getManipulationPolicy,
             "Return this prim's manipulation policy.")
        .def("set_manipulation_policy", &KE::Scene::Prim::setManipulationPolicy,
             py::arg("policy"), "Set this prim's manipulation policy.")
        .def("resolve_manipulation_target",
             py::overload_cast<>(&KE::Scene::Prim::resolveManipulationTarget),
             py::return_value_policy::reference,
             "Resolve which prim should be manipulated from this prim.")
        // setAttribute with specific types
        .def(
            "set_attribute_vec3",
            [](KE::Scene::Prim& self, const std::string& name,
               const glm::vec3& value) { self.setAttribute(name, value); },
            py::arg("name"), py::arg("value"), "Set a vec3 attribute.")
        .def(
            "set_attribute_vec4",
            [](KE::Scene::Prim& self, const std::string& name,
               const glm::vec4& value) { self.setAttribute(name, value); },
            py::arg("name"), py::arg("value"), "Set a vec4 attribute.")
        .def(
            "set_attribute_quat",
            [](KE::Scene::Prim& self, const std::string& name,
               const glm::quat& value) { self.setAttribute(name, value); },
            py::arg("name"), py::arg("value"), "Set a quaternion attribute.")
        .def(
            "set_attribute_float",
            [](KE::Scene::Prim& self, const std::string& name, float value) {
                self.setAttribute(name, value);
            },
            py::arg("name"), py::arg("value"), "Set a float attribute.")
        .def(
            "set_attribute_int",
            [](KE::Scene::Prim& self, const std::string& name, int value) {
                self.setAttribute(name, value);
            },
            py::arg("name"), py::arg("value"), "Set an integer attribute.")
        .def(
            "set_attribute_string",
            [](KE::Scene::Prim& self, const std::string& name,
               const std::string& value) { self.setAttribute(name, value); },
            py::arg("name"), py::arg("value"), "Set a string attribute.")
        // getAttribute with specific types
        .def(
            "get_attribute_vec3",
            [](KE::Scene::Prim& self, const std::string& name) {
                return self.getAttribute<glm::vec3>(name);
            },
            py::arg("name"), "Get a vec3 attribute.")
        .def(
            "get_attribute_vec4",
            [](KE::Scene::Prim& self, const std::string& name) {
                return self.getAttribute<glm::vec4>(name);
            },
            py::arg("name"), "Get a vec4 attribute.")
        .def(
            "get_attribute_quat",
            [](KE::Scene::Prim& self, const std::string& name) {
                return self.getAttribute<glm::quat>(name);
            },
            py::arg("name"), "Get a quaternion attribute.")
        .def(
            "get_attribute_float",
            [](KE::Scene::Prim& self, const std::string& name) {
                return self.getAttribute<float>(name);
            },
            py::arg("name"), "Get a float attribute.")
        .def(
            "get_attribute_int",
            [](KE::Scene::Prim& self, const std::string& name) {
                return self.getAttribute<int>(name);
            },
            py::arg("name"), "Get an integer attribute.")
        .def(
            "get_attribute_string",
            [](KE::Scene::Prim& self, const std::string& name) {
                return self.getAttribute<std::string>(name);
            },
            py::arg("name"), "Get a string attribute.")
        .def("has_attribute",
             py::overload_cast<const std::string&>(
                 &KE::Scene::Prim::hasAttribute, py::const_),
             py::arg("name"), "Return true when an attribute exists.")
        .def("traverse", &KE::Scene::Prim::traverse, py::arg("callback"),
             "Traverse this prim subtree and call callback for each prim.");

    // BackendType enum
    py::enum_<KE::Scene::BackendType>(scene, "BackendType",
                                      "Scene backend implementation type.")
        .value("Native", KE::Scene::BackendType::Native)
        .value("USD", KE::Scene::BackendType::USD)
        .export_values();

    // MeshData struct (with shared_ptr holder for set_mesh_data compatibility)
    py::class_<KE::Scene::MeshData, std::shared_ptr<KE::Scene::MeshData>>(
        scene, "MeshData",
        "Static mesh payload with vertex attributes and triangle indices.")
        .def(py::init<>(), "Create empty mesh data.")
        .def_readwrite("vertices", &KE::Scene::MeshData::vertices,
                       "Vertex positions.")
        .def_readwrite("normals", &KE::Scene::MeshData::normals,
                       "Vertex normals.")
        .def_readwrite("uvs", &KE::Scene::MeshData::uvs,
                       "Vertex texture coordinates.")
        .def_readwrite("indices", &KE::Scene::MeshData::indices,
                       "Triangle index buffer.");

    py::class_<KE::Scene::SkinnedMeshData,
               std::shared_ptr<KE::Scene::SkinnedMeshData>>(
        scene, "SkinnedMeshData",
        "Mesh payload with skinning attributes for skeletal animation.")
        .def(py::init<>(), "Create empty skinned mesh data.")
        .def_readwrite("mesh", &KE::Scene::SkinnedMeshData::mesh,
                       "Base static mesh data.")
        .def_readwrite("bone_node_indices",
                       &KE::Scene::SkinnedMeshData::boneNodeIndices,
                       "Skeleton node index for each bound bone.")
        .def(
            "has_valid_vertex_skinning",
            &KE::Scene::SkinnedMeshData::hasValidVertexSkinning,
            "Return true when vertex bone data matches the mesh vertex count.");

    py::class_<KE::Scene::DebugDraw>(
        scene, "DebugDraw",
        "Helpers for creating debug lines, arrows, and coordinate axes.")
        .def_static(
            "log_component_lines",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, const FloatArray& starts,
               const FloatArray& ends, py::object colors, float radius,
               int segments) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                return KE::Scene::DebugDraw::logLineComponent(
                    app, shader, path, s, e, c, radius, segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors") = py::none(),
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            "Create instanced debug line geometry and return its "
            "RenderComponent.")
        .def_static(
            "update_component_lines",
            [](KE::App* app, KE::Scene::RenderComponent& component,
               const FloatArray& starts, const FloatArray& ends,
               py::object colors) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                KE::Scene::DebugDraw::updateLines(app, component, s, e, c);
            },
            py::arg("app"), py::arg("component"), py::arg("starts"),
            py::arg("ends"), py::arg("colors") = py::none(),
            "Update component-backed debug line geometry.")
        .def_static(
            "log_component_arrows",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, const FloatArray& starts,
               const FloatArray& ends, py::object colors, float radius,
               int segments) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                return KE::Scene::DebugDraw::logArrowComponent(
                    app, shader, path, s, e, c, radius, segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors") = py::none(),
            py::arg("radius") = 0.02f, py::arg("segments") = 12,
            "Create instanced debug arrow geometry and return its "
            "RenderComponent.")
        .def_static(
            "update_component_arrows",
            [](KE::App* app, KE::Scene::RenderComponent& component,
               const FloatArray& starts, const FloatArray& ends,
               py::object colors) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                KE::Scene::DebugDraw::updateArrows(app, component, s, e, c);
            },
            py::arg("app"), py::arg("component"), py::arg("starts"),
            py::arg("ends"), py::arg("colors") = py::none(),
            "Update component-backed debug arrow geometry.")
        .def_static(
            "log_coordinate_axes",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, glm::vec3 origin, glm::quat orientation,
               float length, float radius, int segments) {
                return KE::Scene::DebugDraw::logCoordinateAxes(
                    app, shader, path, origin, orientation, length, radius,
                    segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("origin"), py::arg("orientation"), py::arg("length") = 1.0f,
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            "Create instanced XYZ coordinate axes.")
        .def_static(
            "log_scene_lines",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logLines(
                    sceneBackend, basePath, vec3Array(starts, "starts"),
                    vec3Array(ends, "ends"), vec4Array(colors, "colors"),
                    radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.005f,
            py::arg("segments") = 8, py::return_value_policy::reference,
            "Create scene-backed line prims from numpy arrays.")
        .def_static(
            "log_scene_lines",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath,
               const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logLines(sceneBackend, basePath,
                                                      starts, ends, colors,
                                                      radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.005f,
            py::arg("segments") = 8, py::return_value_policy::reference,
            "Create scene-backed line prims.")
        .def_static(
            "log_scene_arrows",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logArrows(
                    sceneBackend, basePath, vec3Array(starts, "starts"),
                    vec3Array(ends, "ends"), vec4Array(colors, "colors"),
                    radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.02f,
            py::arg("segments") = 12, py::return_value_policy::reference,
            "Create scene-backed arrow prims from numpy arrays.")
        .def_static(
            "log_scene_arrows",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath,
               const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logArrows(sceneBackend, basePath,
                                                       starts, ends, colors,
                                                       radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.02f,
            py::arg("segments") = 12, py::return_value_policy::reference,
            "Create scene-backed arrow prims.")
        .def_static(
            "log_scene_coordinate_axes",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath, glm::vec3 origin,
               glm::quat orientation, float length, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logCoordinateAxes(
                    sceneBackend, basePath, origin, orientation, length, radius,
                    segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("origin"),
            py::arg("orientation"), py::arg("length") = 1.0f,
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            py::return_value_policy::reference,
            "Create scene-backed XYZ coordinate axes.");

    // SceneBackend interface
    py::class_<KE::Scene::SceneBackend>(
        scene, "SceneBackend",
        "Abstract scene backend interface for native and USD scenes.")
        .def("get_backend_type", &KE::Scene::SceneBackend::getBackendType,
             "Return the backend implementation type.")
        .def("load_scene", &KE::Scene::SceneBackend::loadScene, py::arg("path"),
             "Load a scene from a file.")
        .def("save_scene", &KE::Scene::SceneBackend::saveScene, py::arg("path"),
             "Save a scene to a file.")
        .def("load_mesh", &KE::Scene::SceneBackend::loadMesh,
             py::arg("prim_path"), "Load mesh data for a prim path.")
        .def("list_meshes", &KE::Scene::SceneBackend::listMeshes,
             "Return mesh prim paths known by this scene.")
        .def("define_prim", &KE::Scene::SceneBackend::definePrim,
             py::arg("path"), py::arg("type"),
             py::return_value_policy::reference,
             "Define and return a prim at a scene path.")
        .def("get_prim_at_path", &KE::Scene::SceneBackend::getPrimAtPath,
             py::arg("path"), py::return_value_policy::reference,
             "Return a prim at a scene path, or None if it does not exist.")
        .def("remove_prim", &KE::Scene::SceneBackend::removePrim,
             py::arg("path"),
             "Remove a prim subtree. Components registered by an App detach "
             "their renderer resources during destruction.")
        .def("get_root_prim", &KE::Scene::SceneBackend::getRootPrim,
             py::return_value_policy::reference, "Return the scene root prim.");

#ifdef KANGENGINE_USE_USD
    // USDScene with direct USD Stage access!
    py::class_<KE::Scene::USDScene, KE::Scene::SceneBackend>(scene, "USDScene")
        .def(py::init<>(), "Create a new USD Scene")

        // Backend interface
        .def("get_backend_type", &KE::Scene::USDScene::getBackendType)
        .def("load_scene", &KE::Scene::USDScene::loadScene)
        .def("save_scene", &KE::Scene::USDScene::saveScene)
        .def("load_mesh", &KE::Scene::USDScene::loadMesh)
        .def("list_meshes", &KE::Scene::USDScene::listMeshes)

        // USD-specific API
        .def("create_new", &KE::Scene::USDScene::createNew,
             "Create new in-memory USD stage")
        // Note: get_prim, create_* methods return UsdPrim which requires
        // custom type casters. For now, use them for side effects only.
        .def(
            "create_xform",
            [](KE::Scene::USDScene& self, const std::string& path) {
                self.createXform(path);
            },
            "Create Xform (transform/group) at path")
        .def(
            "create_mesh",
            [](KE::Scene::USDScene& self, const std::string& path) {
                self.createMesh(path);
            },
            "Create Mesh at path")
        .def(
            "create_sphere",
            [](KE::Scene::USDScene& self, const std::string& path,
               double radius) { self.createSphere(path, radius); },
            "Create Sphere at path", py::arg("path"), py::arg("radius") = 1.0)
        .def(
            "create_cube",
            [](KE::Scene::USDScene& self, const std::string& path,
               double size) { self.createCube(path, size); },
            "Create Cube at path", py::arg("path"), py::arg("size") = 1.0)
        .def("print_hierarchy", &KE::Scene::USDScene::printHierarchy,
             "Print scene hierarchy (debug)");

    // NOTE: get_stage() would require custom USD type casters to work.
    // For now, use file-based workflow:
    //   1. Create scene with Python USD → Save to file
    //   2. Load file in KangEngine → Render
    //   3. Export from KangEngine → Edit with Python USD
    // This is actually the recommended workflow for production pipelines!
#endif

    // Factory
    scene.def("create_backend", &KE::Scene::SceneFactory::createBackend,
              "Create a scene backend", py::arg("type"));

    // Helper function
    scene.def(
        "load_mesh_from_usd",
        [](const std::string& usd_path,
           const std::string& prim_path) -> KE::Scene::MeshData {
#ifdef KANGENGINE_USE_USD
            auto backend = KE::Scene::SceneFactory::createBackend(
                KE::Scene::BackendType::USD);
            backend->loadScene(usd_path);
            return backend->loadMesh(prim_path);
#else
            throw std::runtime_error("USD support not compiled");
#endif
        },
        py::arg("usd_path"), py::arg("prim_path"),
        "Load mesh from USD file (helper function)");

    // USD 지원 확인
    scene.def("has_usd_support", []() -> bool {
#ifdef KANGENGINE_USE_USD
        return true;
#else
        return false;
#endif
    });
}
