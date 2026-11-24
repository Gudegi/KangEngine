// Collects scene objects and lights and compares accepted render state.
#include "render_scene_data_collector.hpp"

#include "engine/graphics/material/material.hpp"
#include "engine/scene/component/camera_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/component/scene_render_system.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/render/scene_render_bridge.hpp"
#include "engine/scene/scene_resource_manager.hpp"

#include <set>
#include <stdexcept>
#include <type_traits>

namespace KE {
namespace Scene {
namespace {
template <class Predicate>
void collectPrims(Prim* root, std::vector<Prim*>& result, Predicate include) {
    // Membership includes inactive nodes; visibility is evaluated per frame.
    result.clear();
    std::vector<Prim*> pending{root};
    while (!pending.empty()) {
        auto* prim = pending.back();
        pending.pop_back();
        if (include(*prim))
            result.push_back(prim);
        const auto children = prim->getChildren();
        pending.insert(pending.end(), children.rbegin(), children.rend());
    }
}

bool sameEmission(const LightEmission& a, const LightEmission& b) {
    return a.color == b.color && a.intensity == b.intensity &&
           a.exposure == b.exposure && a.normalize == b.normalize &&
           a.enableColorTemperature == b.enableColorTemperature &&
           a.colorTemperature == b.colorTemperature && a.diffuse == b.diffuse &&
           a.specular == b.specular;
}

bool sameLighting(const RenderLightData& a, const RenderLightData& b) {
    if (a.authored != b.authored || a.lights.size() != b.lights.size())
        return false;
    for (size_t i = 0; i < a.lights.size(); ++i) {
        const auto& x = a.lights[i];
        const auto& y = b.lights[i];
        if (x.path != y.path || x.world != y.world ||
            x.light.index() != y.light.index())
            return false;
        const bool equal = std::visit(
            [&](const auto& light) {
                using T = std::decay_t<decltype(light)>;
                const auto& other = std::get<T>(y.light);
                if constexpr (std::is_same_v<T, DirectionalLight>) {
                    return light.direction == other.direction &&
                           light.color == other.color &&
                           light.intensity == other.intensity &&
                           light.ambient == other.ambient;
                } else if constexpr (std::is_same_v<T, PointLight>) {
                    return light.position == other.position &&
                           light.color == other.color &&
                           light.intensity == other.intensity &&
                           light.range == other.range;
                } else if constexpr (std::is_same_v<T, SpotLight>) {
                    return light.position == other.position &&
                           light.direction == other.direction &&
                           light.color == other.color &&
                           light.intensity == other.intensity &&
                           light.range == other.range &&
                           light.innerConeAngle == other.innerConeAngle &&
                           light.outerConeAngle == other.outerConeAngle;
                } else if constexpr (std::is_same_v<T, RectLight>) {
                    return sameEmission(light, other) &&
                           light.width == other.width &&
                           light.height == other.height &&
                           light.texturePath == other.texturePath;
                } else {
                    return sameEmission(light, other) &&
                           light.texturePath == other.texturePath &&
                           light.textureFormat == other.textureFormat;
                }
            },
            x.light);
        if (!equal)
            return false;
    }
    return true;
}

bool sameObject(const RenderObjectSource& a, const RenderObjectSource& b) {
    const auto& x = a.material;
    const auto& y = b.material;
    return a.path == b.path && a.mesh.mesh == b.mesh.mesh &&
           a.mesh.resource == b.mesh.resource && a.geometry == b.geometry &&
           a.materialResource == b.materialResource &&
           a.textures == b.textures && a.world == b.world &&
           a.color == b.color && a.visible == b.visible &&
           a.pickable == b.pickable && a.doubleSided == b.doubleSided &&
           a.alphaMode == b.alphaMode && a.alphaCutoff == b.alphaCutoff &&
           x.shadingModel == y.shadingModel && x.colorStyle == y.colorStyle &&
           x.unlit == y.unlit && x.hasNativeTextures == y.hasNativeTextures &&
           x.baseColor == y.baseColor && x.ambient == y.ambient &&
           x.specular == y.specular && x.shininess == y.shininess &&
           x.metallic == y.metallic && x.roughness == y.roughness &&
           x.emissiveColor == y.emissiveColor &&
           x.emissiveStrength == y.emissiveStrength;
}
} // namespace

RenderSceneFrame
RenderSceneUpdateTracker::prepare(const RenderSceneFrame& frame) const {
    RenderSceneFrame update;
    update.incremental = true;
    update.camera = frame.camera;
    update.lightingChanged =
        !_initialized || !sameLighting(frame.lighting, _lighting);
    if (update.lightingChanged)
        update.lighting = frame.lighting;
    std::set<RenderInstanceId> live;
    for (const auto& object : frame.objects) {
        live.insert(object.id);
        const auto found = _objects.find(object.id);
        if (found == _objects.end() || !sameObject(object, found->second))
            update.objects.push_back(object);
    }
    for (const auto& [id, object] : _objects) {
        if (!live.count(id))
            update.removedObjects.push_back(id);
    }
    return update;
}

void RenderSceneUpdateTracker::accept(const RenderSceneFrame& update) {
    if (!update.incremental)
        _objects.clear();
    for (auto id : update.removedObjects)
        _objects.erase(id);
    for (const auto& object : update.objects)
        _objects[object.id] = object;
    if (update.lightingChanged)
        _lighting = update.lighting;
    _initialized = true;
}

RenderLightData RenderLightCollector::collect(Prim* root) {
    RenderLightData result;
    if (!root)
        return result;
    if (_root.lock() != root->getTransformComponent() ||
        _structureVersion != root->structureVersion()) {
        collectPrims(root, _prims, [](const Prim& prim) {
            return prim.getType() == PrimType::Light;
        });
        _root = root->getTransformComponent();
        _structureVersion = root->structureVersion();
    }
    result.authored = !_prims.empty();
    result.lights.reserve(_prims.size());
    for (auto* prim : _prims) {
        if (!prim->isActiveInHierarchy() || !prim->isVisibleInHierarchy())
            continue;
        RenderLightSource source;
        source.path = prim->getPath();
        source.world = prim->computeWorldMatrix();
        switch (prim->getLightType()) {
        case LightType::Directional:
            source.light = prim->getDirectionalLight();
            break;
        case LightType::Point:
            source.light = prim->getPointLight();
            break;
        case LightType::Spot:
            source.light = prim->getSpotLight();
            break;
        case LightType::Rect:
            source.light = prim->getRectLight();
            break;
        case LightType::Dome:
            source.light = prim->getDomeLight();
            break;
        }
        result.lights.push_back(std::move(source));
    }
    return result;
}

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
    frame.lighting = _lights.collect(root);
    if (_root.lock() != root->getTransformComponent() ||
        _structureVersion != root->structureVersion()) {
        collectPrims(root, _prims, [](const Prim& prim) {
            return prim.hasRenderComponent();
        });
        _root = root->getTransformComponent();
        _structureVersion = root->structureVersion();
    }
    // Public material fields remain mutable; evaluate each shared material
    // once.
    std::map<const Material*, RenderMaterialSource> materials;
    for (Prim* prim : _prims) {
        auto component = prim->getRenderComponent();
        if (!component ||
            (registrations && !registrations->isRegistered(*component)))
            continue;
        // Registered skinning and external instances travel through the batch
        // API.
        if (registrations && registrations->hasBatch(*component))
            continue;
        if (component->transformSource() != TransformSource::SceneGraph ||
            !component->castsShadow())
            throw std::invalid_argument(
                "SceneRenderer supports shadow-casting SceneGraph "
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
        if (registrations)
            object.geometry = registrations->geometry(*component);
        if (!object.mesh.mesh ||
            resources.mesh(object.mesh.resource) != object.mesh.mesh)
            throw std::invalid_argument(
                "Renderable mesh must be registered with its scene: " +
                object.path);
        auto found = materials.find(material);
        if (found == materials.end())
            found = materials
                        .emplace(material, SceneRenderBridge::materialSource(
                                               *material, &resources))
                        .first;
        const auto& materialSource = found->second;
        object.material = materialSource.values;
        object.materialResource = materialSource.resource;
        if (object.materialResource == InvalidResourceHandle)
            throw std::invalid_argument(
                "Renderable material must be registered with its scene: " +
                object.path);
        object.textures = materialSource.textures;
        object.world = prim->computeModelMatrix();
        object.color = prim->getDisplayColorAlpha().value_or(glm::vec4(1.0f));
        object.visible =
            prim->isActiveInHierarchy() && prim->isVisibleInHierarchy();
        const auto selection = prim->getSelectionComponent();
        object.pickable = !selection || selection->isPickable();
        object.doubleSided = component->isDoubleSided();
        object.alphaMode = component->alphaMode();
        object.alphaCutoff = component->alphaCutoff();
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
