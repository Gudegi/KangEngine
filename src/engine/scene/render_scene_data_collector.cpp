// Evaluates scene hierarchy and resource bindings for static image rendering.
#include "render_scene_data_collector.hpp"

#include "engine/graphics/material/material.hpp"
#include "engine/scene/component/camera_component.hpp"
#include "engine/scene/component/light_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/component/scene_render_system.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_render_bridge.hpp"

#include <set>
#include <stdexcept>

namespace KE {
namespace Scene {

RenderInstanceId RenderSceneDataCollector::instanceId(const Prim& prim) const {
    const auto found = _identities.find(prim.getPath());
    if (found == _identities.end() ||
        found->second.component.lock() != prim.getRenderComponent())
        return InvalidRenderInstance;
    return found->second.id;
}

void RenderSceneDataCollector::forget(const RenderComponent& component) {
    for (auto it = _identities.begin(); it != _identities.end();) {
        if (it->second.component.lock().get() == &component)
            it = _identities.erase(it);
        else
            ++it;
    }
}

std::shared_ptr<RenderComponent>
RenderSceneDataCollector::resolve(RenderInstanceId id) const {
    for (const auto& [path, identity] : _identities) {
        if (identity.id == id)
            return identity.component.lock();
    }
    return nullptr;
}

RenderSceneFrame
RenderSceneDataCollector::collect(SceneBackend& scene,
                                  const SceneResourceManager& resources,
                                  const std::string& cameraPath) {
    auto* camera = scene.getPrimAtPath(cameraPath);
    auto projection = camera ? camera->getCameraComponent() : nullptr;
    if (!projection || !camera->isActiveInHierarchy() ||
        projection->projectionType() != CameraProjectionType::Perspective)
        throw std::invalid_argument(
            "SceneRenderer requires an active perspective camera");

    auto frame = collectScene(scene, resources, nullptr);
    frame.camera = {camera->computeModelMatrix(),
                    projection->verticalFovDegrees(), projection->nearPlane(),
                    projection->farPlane()};
    return frame;
}

RenderSceneFrame RenderSceneDataCollector::collectRegistered(
    SceneBackend& scene, const SceneResourceManager& resources,
    const SceneRenderSystem& registrations) {
    return collectScene(scene, resources, &registrations);
}

RenderSceneFrame
RenderSceneDataCollector::collectScene(SceneBackend& scene,
                                       const SceneResourceManager& resources,
                                       const SceneRenderSystem* registrations) {
    if (resources.scene() != &scene)
        throw std::invalid_argument(
            "Renderer resources must belong to its scene");
    RenderSceneFrame frame;
    std::set<std::string> livePaths;
    auto* root = scene.getRootPrim();
    if (!root)
        throw std::invalid_argument("SceneRenderer requires a scene root");
    std::vector<Prim*> pending{root};
    while (!pending.empty()) {
        Prim* prim = pending.back();
        pending.pop_back();
        for (Prim* child : prim->getChildren())
            pending.push_back(child);

        if (auto light = prim->getLightComponent()) {
            if (!prim->isActiveInHierarchy() || !prim->isVisibleInHierarchy())
                continue;
            if (light->type() == LightType::Dome) {
                frame.domes.push_back({prim->getPath(), light->domeLight(),
                                       prim->computeWorldMatrix()});
            } else if (light->type() == LightType::Directional) {
                auto value = light->directionalLight();
                value.direction = glm::normalize(
                    glm::mat3(prim->computeModelMatrix()) * value.direction);
                frame.lights.push_back(value);
            } else {
                throw std::invalid_argument(
                    "SceneRenderer currently supports directional and dome lights only");
            }
        }

        auto component = prim->getRenderComponent();
        if (!component ||
            (registrations && !registrations->isRegistered(*component)))
            continue;
        if (component->transformSource() != TransformSource::SceneGraph ||
            component->alphaMode() != AlphaMode::Opaque ||
            !component->castsShadow())
            throw std::invalid_argument(
                "SceneRenderer supports opaque, shadow-casting SceneGraph "
                "meshes only");
        auto* material = prim->getMaterial();
        if (!material)
            throw std::invalid_argument("Renderable has no material: " +
                                        prim->getPath());

        RenderObjectSource object;
        object.path = prim->getPath();
        livePaths.insert(object.path);
        auto& identity = _identities[object.path];
        if (identity.component.lock() != component) {
            identity = {component, _nextId++};
        }
        object.id = identity.id;
        object.mesh = SceneRenderBridge::meshData(*prim);
        if (!object.mesh.mesh ||
            resources.mesh(object.mesh.resource) != object.mesh.mesh)
            throw std::invalid_argument(
                "Renderable mesh must be registered with its scene: " +
                object.path);
        auto materialSource =
            SceneRenderBridge::materialSource(*material, &resources);
        object.material = materialSource.values;
        object.materialResource = materialSource.resource;
        if (object.materialResource == InvalidResourceHandle)
            throw std::invalid_argument(
                "Renderable material must be registered with its scene: " +
                object.path);
        object.textures = std::move(materialSource.textures);
        object.world = prim->computeModelMatrix();
        object.color = prim->getDisplayColorAlpha().value_or(glm::vec4(1.0f));
        object.visible =
            prim->isActiveInHierarchy() && prim->isVisibleInHierarchy();
        const auto selection = prim->getSelectionComponent();
        object.pickable = !selection || selection->isPickable();
        object.doubleSided = component->isDoubleSided();
        frame.objects.push_back(std::move(object));
    }

    for (auto it = _identities.begin(); it != _identities.end();) {
        if (!livePaths.count(it->first))
            it = _identities.erase(it);
        else
            ++it;
    }
    return frame;
}

} // namespace Scene
} // namespace KE
