// Evaluates scene hierarchy and resource bindings for static image rendering.
#include "render_scene_data_collector.hpp"

#include "engine/graphics/material/material.hpp"
#include "engine/scene/component/camera_component.hpp"
#include "engine/scene/component/light_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_render_bridge.hpp"

#include <set>
#include <stdexcept>

namespace KE {
namespace Scene {

RenderSceneFrame
RenderSceneDataCollector::collect(SceneBackend& scene,
                                  const SceneResourceManager& resources,
                                  const std::string& cameraPath) {
    if (resources.scene() != &scene)
        throw std::invalid_argument(
            "Renderer resources must belong to its scene");
    auto* camera = scene.getPrimAtPath(cameraPath);
    auto projection = camera ? camera->getCameraComponent() : nullptr;
    if (!projection || !camera->isActiveInHierarchy() ||
        projection->projectionType() != CameraProjectionType::Perspective)
        throw std::invalid_argument(
            "SceneRenderer requires an active perspective camera");

    RenderSceneFrame frame;
    frame.camera = {camera->computeModelMatrix(),
                    projection->verticalFovDegrees(), projection->nearPlane(),
                    projection->farPlane()};
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
            if (light->type() != LightType::Directional)
                throw std::invalid_argument(
                    "SceneRenderer currently supports directional lights only");
            auto value = light->directionalLight();
            value.direction = glm::normalize(
                glm::mat3(prim->computeModelMatrix()) * value.direction);
            frame.lights.push_back(value);
        }

        auto component = prim->getRenderComponent();
        if (!component)
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
        object.material = SceneRenderBridge::materialData(*material);
        object.materialResource = resources.materialHandle(material);
        if (object.materialResource == InvalidResourceHandle)
            throw std::invalid_argument(
                "Renderable material must be registered with its scene: " +
                object.path);
        for (const auto& [slot, handle] : material->textureResources) {
            auto source = resources.textureSource(handle);
            if (!source)
                throw std::invalid_argument(
                    "Material texture handle has no source: " + slot);
            object.textures.emplace(slot,
                                    std::make_pair(handle, std::move(source)));
        }
        object.world = prim->computeModelMatrix();
        object.color = prim->getDisplayColorAlpha().value_or(glm::vec4(1.0f));
        object.visible =
            prim->isActiveInHierarchy() && prim->isVisibleInHierarchy();
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
