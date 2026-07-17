///
/// Scene Prim Implementation
///

#include "prim.hpp"
#include "engine/scene/component/transform_component.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/light_component.hpp"
#include "engine/scene/component/camera_component.hpp"
#include "engine/scene/component/material_binding_component.hpp"
#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/component/articulation_binding_component.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/types.hpp"
#include <Eigen/Geometry>
#include <glm/ext/quaternion_transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include "xform_token.hpp"

namespace KE {
namespace Scene {

namespace {
bool isRenderableType(PrimType type) {
    return type == PrimType::Mesh || type == PrimType::MeshInstance;
}

glm::vec3 safeDirection(glm::vec3 direction, glm::vec3 fallback) {
    if (glm::length(direction) < 1e-4f)
        return glm::normalize(fallback);
    return glm::normalize(direction);
}
} // namespace

Prim::Prim(const std::string& name, PrimType type, Prim* parent)
    : _name(name), _type(type), _parent(parent) {
    _renderable = isRenderableType(type);
    if (_type != PrimType::Resource) {
        _transformComponent =
            std::shared_ptr<TransformComponent>(new TransformComponent(this));
    }
    _selectionComponent =
        std::shared_ptr<SelectionComponent>(new SelectionComponent(this));
    if (_type == PrimType::Root) {
        _selectionComponent->setPickable(false);
        _selectionComponent->setSelectable(false);
        _selectionComponent->setManipulatable(false);
        _selectionComponent->setForceDraggable(false);
        _selectionComponent->setInteractionKind(InteractionKind::Helper);
    } else if (_type == PrimType::Resource) {
        _selectionComponent->setPickable(false);
        _selectionComponent->setManipulatable(false);
        _selectionComponent->setForceDraggable(false);
        _selectionComponent->setInteractionKind(InteractionKind::Resource);
    }

    // Initialize prim path
    if (parent == nullptr) {
        _path = "/"; // root
    } else if (parent->getPath() == "/") {
        _path = "/" + name;
    } else {
        _path = parent->getPath() + "/" + name;
    }
}

Prim::~Prim() {
    if (_transformComponent)
        _transformComponent->detach();
    if (_meshComponent)
        _meshComponent->detach();
    if (_renderComponent)
        _renderComponent->detach();
    if (_lightComponent)
        _lightComponent->detach();
    if (_cameraComponent)
        _cameraComponent->detach();
    if (_materialBindingComponent)
        _materialBindingComponent->detach();
    if (_resourceComponent)
        _resourceComponent->detach();
    if (_selectionComponent)
        _selectionComponent->detach();
    if (_articulationBindingComponent)
        _articulationBindingComponent->detach();
}

Prim* Prim::addChild(const std::string& name, PrimType type) {
    auto child = std::make_unique<Prim>(name, type, this);
    Prim* childPtr = child.get();

    _childrenMap[name] = childPtr;
    _children.emplace_back(std::move(child));

    return childPtr;
}

bool Prim::removeChild(const std::string& name) {
    auto mapIt = _childrenMap.find(name);
    if (mapIt == _childrenMap.end())
        return false;

    Prim* target = mapIt->second;
    _childrenMap.erase(mapIt);
    _children.erase(std::remove_if(_children.begin(), _children.end(),
                                   [target](const auto& child) {
                                       return child.get() == target;
                                   }),
                    _children.end());
    markWorldTransformDirtyRecursive();
    return true;
}

Prim* Prim::getChild(const std::string& name) const {
    auto it = _childrenMap.find(name);
    if (it != _childrenMap.end()) {
        return it->second;
    }
    return nullptr;
}

Prim* Prim::getPrimAtPath(const std::string& path) {
    // "/" → 루트
    if (path == "/" || path.empty()) {
        return this;
    }

    // "/World/Cube" 파싱
    std::string pathCopy = path;
    if (pathCopy[0] == '/') {
        pathCopy = pathCopy.substr(1); // "/" 제거
    }

    // 순차 탐색
    Prim* current = this;
    size_t start = 0;
    size_t end = pathCopy.find('/');

    while (end != std::string::npos) {
        std::string part = pathCopy.substr(start, end - start);
        if (!part.empty()) {
            current = current->getChild(part);
            if (!current)
                return nullptr;
        }
        start = end + 1;
        end = pathCopy.find('/', start);
    }

    // Last part
    std::string part = pathCopy.substr(start);
    if (!part.empty()) {
        current = current->getChild(part);
    }
    return current;
}

std::vector<Prim*> Prim::getChildren() const {
    std::vector<Prim*> result;
    result.reserve(_children.size());
    for (const auto& child : _children) {
        result.emplace_back(child.get());
    }
    return result;
}

void Prim::setMeshData(std::shared_ptr<MeshData> data) {
    if (!_meshComponent)
        addMeshComponent();
    _meshComponent->setMeshData(std::move(data));
}

std::shared_ptr<MeshData> Prim::getMeshData() const {
    return _meshComponent ? _meshComponent->meshData() : nullptr;
}

void Prim::setMeshSourcePath(const std::string& path) {
    if (!_meshComponent)
        addMeshComponent();
    _meshComponent->setMeshSourcePath(path);
}

const std::string& Prim::getMeshSourcePath() const {
    static const std::string empty;
    return _meshComponent ? _meshComponent->meshSourcePath() : empty;
}

std::shared_ptr<MeshData> Prim::resolveMeshData() const {
    return _meshComponent ? _meshComponent->resolveMeshData() : nullptr;
}

std::shared_ptr<MeshComponent> Prim::addMeshComponent() {
    if (_type != PrimType::Mesh && _type != PrimType::MeshInstance &&
        _type != PrimType::Resource)
        throw std::runtime_error(
            "Prim '" + _path +
            "' must be PrimType::Mesh, MeshInstance, or Resource to add a "
            "MeshComponent");
    if (_meshComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a MeshComponent");
    _meshComponent = std::shared_ptr<MeshComponent>(new MeshComponent(this));
    return _meshComponent;
}

std::shared_ptr<MeshComponent> Prim::getMeshComponent() const {
    return _meshComponent;
}

bool Prim::removeMeshComponent() {
    if (!_meshComponent)
        return false;
    _meshComponent->detach();
    _meshComponent.reset();
    if (_renderComponent)
        _renderComponent->markChanged();
    return true;
}

std::shared_ptr<TransformComponent> Prim::getTransformComponent() const {
    return _transformComponent;
}

std::shared_ptr<RenderComponent> Prim::addRenderComponent() {
    if (_renderComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a RenderComponent");
    _renderComponent =
        std::shared_ptr<RenderComponent>(new RenderComponent(this));
    return _renderComponent;
}

std::shared_ptr<RenderComponent> Prim::getRenderComponent() const {
    return _renderComponent;
}

bool Prim::removeRenderComponent() {
    if (!_renderComponent)
        return false;
    _renderComponent->detach();
    _renderComponent.reset();
    return true;
}

std::shared_ptr<MaterialBindingComponent> Prim::addMaterialBindingComponent() {
    if (_materialBindingComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a MaterialBindingComponent");
    _materialBindingComponent = std::shared_ptr<MaterialBindingComponent>(
        new MaterialBindingComponent(this));
    return _materialBindingComponent;
}

std::shared_ptr<MaterialBindingComponent>
Prim::getMaterialBindingComponent() const {
    return _materialBindingComponent;
}

bool Prim::removeMaterialBindingComponent() {
    if (!_materialBindingComponent)
        return false;
    _materialBindingComponent->detach();
    _materialBindingComponent.reset();
    return true;
}

void Prim::setMaterial(Material* material) {
    if (!_materialBindingComponent)
        addMaterialBindingComponent();
    _materialBindingComponent->setMaterial(material);
}

Material* Prim::getMaterial() const {
    return _materialBindingComponent ? _materialBindingComponent->material()
                                     : nullptr;
}

std::shared_ptr<LightComponent> Prim::addLightComponent() {
    if (_type != PrimType::Light)
        throw std::runtime_error(
            "Prim '" + _path +
            "' must be PrimType::Light to add a LightComponent");
    if (_lightComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a LightComponent");
    _lightComponent = std::shared_ptr<LightComponent>(new LightComponent(this));
    return _lightComponent;
}

std::shared_ptr<LightComponent> Prim::getLightComponent() const {
    return _lightComponent;
}

bool Prim::removeLightComponent() {
    if (!_lightComponent)
        return false;
    _lightComponent->detach();
    _lightComponent.reset();
    return true;
}

std::shared_ptr<CameraComponent> Prim::addCameraComponent() {
    if (_type != PrimType::Camera)
        throw std::runtime_error(
            "Prim '" + _path +
            "' must be PrimType::Camera to add a CameraComponent");
    if (_cameraComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a CameraComponent");
    _cameraComponent =
        std::shared_ptr<CameraComponent>(new CameraComponent(this));
    return _cameraComponent;
}

std::shared_ptr<CameraComponent> Prim::getCameraComponent() const {
    return _cameraComponent;
}

bool Prim::removeCameraComponent() {
    if (!_cameraComponent)
        return false;
    _cameraComponent->detach();
    _cameraComponent.reset();
    return true;
}

std::shared_ptr<ResourceComponent> Prim::addResourceComponent() {
    if (_type != PrimType::Resource)
        throw std::runtime_error(
            "Prim '" + _path +
            "' must be PrimType::Resource to add a ResourceComponent");
    if (_resourceComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a ResourceComponent");
    _resourceComponent =
        std::shared_ptr<ResourceComponent>(new ResourceComponent(this));
    return _resourceComponent;
}

std::shared_ptr<ResourceComponent> Prim::getResourceComponent() const {
    return _resourceComponent;
}

bool Prim::removeResourceComponent() {
    if (!_resourceComponent)
        return false;
    _resourceComponent->detach();
    _resourceComponent.reset();
    return true;
}

std::shared_ptr<SelectionComponent> Prim::addSelectionComponent() {
    if (_selectionComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a SelectionComponent");
    _selectionComponent =
        std::shared_ptr<SelectionComponent>(new SelectionComponent(this));
    return _selectionComponent;
}

std::shared_ptr<SelectionComponent> Prim::getSelectionComponent() const {
    return _selectionComponent;
}

bool Prim::removeSelectionComponent() {
    if (!_selectionComponent)
        return false;
    _selectionComponent->detach();
    _selectionComponent.reset();
    return true;
}

std::shared_ptr<ArticulationBindingComponent>
Prim::addArticulationBindingComponent() {
    if (_articulationBindingComponent)
        throw std::runtime_error(
            "Prim '" + _path + "' already has an ArticulationBindingComponent");
    _articulationBindingComponent =
        std::shared_ptr<ArticulationBindingComponent>(
            new ArticulationBindingComponent(this));
    return _articulationBindingComponent;
}

std::shared_ptr<ArticulationBindingComponent>
Prim::getArticulationBindingComponent() const {
    return _articulationBindingComponent;
}

bool Prim::removeArticulationBindingComponent() {
    if (!_articulationBindingComponent)
        return false;
    _articulationBindingComponent->detach();
    _articulationBindingComponent.reset();
    return true;
}

LightType Prim::getLightType(LightType defaultType) const {
    if (_lightComponent)
        return _lightComponent->type();
    if (!hasAttribute("light:type"))
        return defaultType;
    const int value = getAttribute<int>("light:type");
    if (value < static_cast<int>(LightType::Directional) ||
        value > static_cast<int>(LightType::Spot))
        return defaultType;
    return static_cast<LightType>(value);
}

void Prim::setDirectionalLight(const DirectionalLight& light) {
    if (!_lightComponent)
        addLightComponent();
    _lightComponent->setDirectionalLight(light);

    setAttribute("light:type", static_cast<int>(LightType::Directional));
    setAttribute("light:direction",
                 safeDirection(light.direction, glm::vec3(0.0f, 0.0f, -1.0f)));
    setAttribute("light:color", light.color);
    setAttribute("light:intensity", std::max(0.0f, light.intensity));
    setAttribute("light:ambient", light.ambient);
}

DirectionalLight Prim::getDirectionalLight() {
    if (_lightComponent)
        return _lightComponent->directionalLight();

    DirectionalLight light;
    const glm::vec3 localDirection =
        getAttribute<glm::vec3>("light:direction", light.direction);
    const glm::mat3 worldRotation(computeWorldMatrix());
    light.direction =
        safeDirection(worldRotation * localDirection, light.direction);
    light.color = getAttribute<glm::vec3>("light:color", light.color);
    light.intensity =
        std::max(0.0f, getAttribute<float>("light:intensity", light.intensity));
    light.ambient = getAttribute<glm::vec3>("light:ambient", light.ambient);
    return light;
}

void Prim::setPointLight(const PointLight& light) {
    if (!_lightComponent)
        addLightComponent();
    _lightComponent->setPointLight(light);

    setAttribute("light:type", static_cast<int>(LightType::Point));
    setAttribute("light:color", light.color);
    setAttribute("light:intensity", std::max(0.0f, light.intensity));
    setAttribute("light:range", std::max(0.0f, light.range));
}

PointLight Prim::getPointLight() {
    if (_lightComponent)
        return _lightComponent->pointLight();

    PointLight light;
    const glm::mat4 world = computeWorldMatrix();
    light.position = glm::vec3(world[3]);
    light.color = getAttribute<glm::vec3>("light:color", light.color);
    light.intensity =
        std::max(0.0f, getAttribute<float>("light:intensity", light.intensity));
    light.range =
        std::max(0.0f, getAttribute<float>("light:range", light.range));
    return light;
}

void Prim::setSpotLight(const SpotLight& light) {
    if (!_lightComponent)
        addLightComponent();
    _lightComponent->setSpotLight(light);

    setAttribute("light:type", static_cast<int>(LightType::Spot));
    setAttribute("light:direction",
                 safeDirection(light.direction, glm::vec3(0.0f, 0.0f, -1.0f)));
    setAttribute("light:color", light.color);
    setAttribute("light:intensity", std::max(0.0f, light.intensity));
    setAttribute("light:range", std::max(0.0f, light.range));
    setAttribute("light:innerConeAngle", std::max(0.0f, light.innerConeAngle));
    setAttribute("light:outerConeAngle",
                 std::max(light.innerConeAngle, light.outerConeAngle));
}

SpotLight Prim::getSpotLight() {
    if (_lightComponent)
        return _lightComponent->spotLight();

    SpotLight light;
    const glm::mat4 world = computeWorldMatrix();
    const glm::vec3 localDirection =
        getAttribute<glm::vec3>("light:direction", light.direction);
    light.position = glm::vec3(world[3]);
    light.direction =
        safeDirection(glm::mat3(world) * localDirection, light.direction);
    light.color = getAttribute<glm::vec3>("light:color", light.color);
    light.intensity =
        std::max(0.0f, getAttribute<float>("light:intensity", light.intensity));
    light.range =
        std::max(0.0f, getAttribute<float>("light:range", light.range));
    light.innerConeAngle =
        std::max(0.0f, getAttribute<float>("light:innerConeAngle",
                                           light.innerConeAngle));
    light.outerConeAngle = std::max(
        light.innerConeAngle,
        getAttribute<float>("light:outerConeAngle", light.outerConeAngle));
    return light;
}

bool Prim::isActiveInHierarchy() const {
    for (const Prim* prim = this; prim; prim = prim->_parent) {
        if (!prim->_active)
            return false;
    }
    return true;
}

bool Prim::isVisibleInHierarchy() const {
    for (const Prim* prim = this; prim; prim = prim->_parent) {
        if (!prim->_visible)
            return false;
    }
    return true;
}

void Prim::setVisible(bool visible) {
    if (_visible == visible)
        return;
    _visible = visible;
    if (_renderComponent)
        _renderComponent->markChanged();
}

void Prim::setActive(bool a) { _active = a; }

Prim* Prim::resolveManipulationTarget() {
    for (Prim* prim = this; prim; prim = prim->_parent) {
        switch (prim->_manipulationPolicy) {
        case ManipulationPolicy::Inherit:
            break;
        case ManipulationPolicy::Self:
            return prim;
        case ManipulationPolicy::Parent:
            return prim->_parent ? prim->_parent : prim;
        case ManipulationPolicy::Root:
            return prim;
        case ManipulationPolicy::Disabled:
            return nullptr;
        }
    }
    return this;
}

const Prim* Prim::resolveManipulationTarget() const {
    return const_cast<Prim*>(this)->resolveManipulationTarget();
}

Prim* Prim::defineManipulationGroup(SceneBackend* scene,
                                    const std::string& path) {
    if (!scene)
        return nullptr;
    Prim* prim = scene->definePrim(path, PrimType::Xform);
    if (prim)
        prim->setManipulationPolicy(ManipulationPolicy::Root);
    return prim;
}

MeshData Prim::createSquareData(float scale) { return createCubeData(scale); }

MeshData Prim::createCubeData(float scale) {
    // Scale means the length of one side.
    float half = scale / 2;
    //    v3----- v7
    //   /|      /|
    //  v2------v6|
    //  | |     | |
    //  | v0----|-v4
    //  |/      |/
    //  v1------v5
    //
    std::vector<glm::vec3> positions = {
        // v0, v1, v2, v3
        glm::vec3(-half, -half, -half),
        glm::vec3(-half, -half, half),
        glm::vec3(-half, half, half),
        glm::vec3(-half, half, -half),
        // v4, v5, v6, v7
        glm::vec3(half, -half, -half),
        glm::vec3(half, -half, half),
        glm::vec3(half, half, half),
        glm::vec3(half, half, -half),
        // v0, v1, v5, v4
        glm::vec3(-half, -half, -half),
        glm::vec3(-half, -half, half),
        glm::vec3(half, -half, half),
        glm::vec3(half, -half, -half),
        // v3, v2, v6, v7
        glm::vec3(-half, half, -half),
        glm::vec3(-half, half, half),
        glm::vec3(half, half, half),
        glm::vec3(half, half, -half),
        // v0, v3, v7, v4
        glm::vec3(-half, -half, -half),
        glm::vec3(-half, half, -half),
        glm::vec3(half, half, -half),
        glm::vec3(half, -half, -half),
        // v1, v2, v6, v5
        glm::vec3(-half, -half, half),
        glm::vec3(-half, half, half),
        glm::vec3(half, half, half),
        glm::vec3(half, -half, half),
    }; // positions.size() == 24

    std::vector<glm::vec3> normals = {
        glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0), glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),  glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 0, -1), glm::vec3(0, 0, -1),
        glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),
    };

    std::vector<glm::vec2> uvs = {
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
    };

    std::vector<unsigned int> indices = {
        0,  1,  2,  0,  2,  3,  // left
        4,  6,  5,  4,  7,  6,  // right
        8,  10, 9,  8,  11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 23, 22, 20, 22, 21, // front
    };

    MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);

    return meshData;
}

MeshData Prim::createPlaneData(float scale) {
    return createPlaneData(scale, UpAxis::Y);
}

MeshData Prim::createPlaneData(float scale, UpAxis upAxis) {
    float half = scale / 2;
    //
    // v2 ----- v3
    // |        |
    // |        |
    // v0 ----- v1
    //

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;

    if (upAxis == UpAxis::Y) {
        // XZ plane, Y-up normal, CCW from +Y with indices {0,1,3,0,3,2}
        // (v1-v0)x(v3-v0) = (2h,0,0)x(2h,0,-2h) -> +Y
        positions = {
            glm::vec3(-half, 0, half),
            glm::vec3(half, 0, half),
            glm::vec3(-half, 0, -half),
            glm::vec3(half, 0, -half),
        };
        normals = {
            glm::vec3(0, 1, 0),
            glm::vec3(0, 1, 0),
            glm::vec3(0, 1, 0),
            glm::vec3(0, 1, 0),
        };
    } else if (upAxis == UpAxis::Z) {
        // XY plane, Z-up normal
        positions = {
            glm::vec3(-half, -half, 0),
            glm::vec3(half, -half, 0),
            glm::vec3(-half, half, 0),
            glm::vec3(half, half, 0),
        };
        normals = {
            glm::vec3(0, 0, 1),
            glm::vec3(0, 0, 1),
            glm::vec3(0, 0, 1),
            glm::vec3(0, 0, 1),
        };
    } else { // UpAxis::X
        // YZ plane, X-up normal
        positions = {
            glm::vec3(0, -half, -half),
            glm::vec3(0, half, -half),
            glm::vec3(0, -half, half),
            glm::vec3(0, half, half),
        };
        normals = {
            glm::vec3(1, 0, 0),
            glm::vec3(1, 0, 0),
            glm::vec3(1, 0, 0),
            glm::vec3(1, 0, 0),
        };
    }

    std::vector<glm::vec2> uvs = {
        glm::vec2(0, 0),
        glm::vec2(scale, 0),
        glm::vec2(0, scale),
        glm::vec2(scale, scale),
    };
    std::vector<unsigned int> indices = {0, 1, 3, 0, 3, 2};

    MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);

    return meshData;
}

MeshData Prim::createSphereData(float radius, int numLongitudes,
                                int numLatitudes) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    int vertexCount = (numLatitudes) * (numLongitudes);
    int indexCount = (numLatitudes - 1) * (numLongitudes - 1) * 6;
    positions.reserve(vertexCount);
    normals.reserve(vertexCount);
    uvs.reserve(vertexCount);
    indices.reserve(indexCount);

    float thetaUnit = (glm::pi<float>() / (numLatitudes - 1));
    float phiUnit = (2 * glm::pi<float>() / (numLongitudes - 1));

    for (int i = 0; i < numLatitudes; i++) {
        float theta = i * thetaUnit;
        float sinTheta = glm::sin(theta);
        float cosTheta = glm::cos(theta);

        for (int j = 0; j < numLongitudes; j++) {
            float phi = j * phiUnit;
            float sinPhi = glm::sin(phi);
            float cosPhi = glm::cos(phi);

            float x = sinTheta * cosPhi;
            float y = sinTheta * sinPhi;
            float z = cosTheta;

            float u =
                static_cast<float>(j) / static_cast<float>(numLongitudes - 1);
            float v =
                static_cast<float>(i) / static_cast<float>(numLatitudes - 1);

            // positions.emplace_back(
            //     glm::vec3(radius * x, radius * y, radius * z));
            // normals.emplace_back(glm::vec3(x, y, z));
            // uvs.emplace_back(glm::vec2(u, v));
            positions.emplace_back(radius * x, radius * y, radius * z);
            normals.emplace_back(x, y, z);
            uvs.emplace_back(u, v);
        }
    }

    for (int i = 0; i < numLatitudes - 1; i++) {
        for (int j = 0; j < numLongitudes - 1; j++) {
            unsigned int first = i * numLongitudes + j;
            unsigned int second = first + numLongitudes;

            // counter clock-wise
            indices.emplace_back(first);
            indices.emplace_back(second);
            indices.emplace_back(first + 1);
            indices.emplace_back(second);
            indices.emplace_back(second + 1);
            indices.emplace_back(first + 1);

            /*
            // clock-wise
            indices.emplace_back(first);
            indices.emplace_back(first + 1);
            indices.emplace_back(second);
            indices.emplace_back(second);
            indices.emplace_back(first + 1);
            indices.emplace_back(second + 1);
            */
        }
    }

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

// Internal helper: truncated cylinder (topRadius==radius → cylinder,
// topRadius==0 → cone) Ported from references/render_opengl.py
// _create_cylinder_mesh
static MeshData makeCylinderMesh(float radius, float halfLength,
                                 float topRadius, const int perm[3],
                                 int segments) {
    auto P = [&](float x, float y, float z) -> glm::vec3 {
        float v[3] = {x, y, z};
        return {v[perm[0]], v[perm[1]], v[perm[2]]};
    };

    // Side normal slope: -arctan2(topRadius - radius, 2*halfLength)
    // 0 for straight cylinder, positive for cone (normals tilt outward-up)
    float sideSlope = -glm::atan(topRadius - radius, 2.0f * halfLength);
    const float pi2 = glm::two_pi<float>();

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    // Cap vertices: [0] bottom center, [1] top center
    positions.emplace_back(P(0.0f, -halfLength, 0.0f));
    normals.emplace_back(P(0.0f, -1.0f, 0.0f));
    uvs.emplace_back(0.5f, 0.5f);

    positions.emplace_back(P(0.0f, halfLength, 0.0f));
    normals.emplace_back(P(0.0f, 1.0f, 0.0f));
    uvs.emplace_back(0.5f, 0.5f);

    for (int j : {-1, 1}) {
        unsigned int ci = (j == 1) ? 1u : 0u;
        float y = j * halfLength;
        float r = (j == -1) ? radius : topRadius;

        for (int i = 0; i < segments; ++i) {
            float theta = pi2 * i / segments;
            float c = glm::cos(theta);
            float s = glm::sin(theta);

            positions.emplace_back(P(r * c, y, r * s));
            normals.emplace_back(P(0.0f, (float)j, 0.0f));
            uvs.emplace_back(c * 0.5f + 0.5f, s * 0.5f + 0.5f);

            int cs = ci * segments;
            unsigned int v_curr = 2u + i + cs;
            unsigned int v_next = 2u + (i + 1) % segments + cs;

            if (j == -1) {
                // bottom cap: CCW from -Y (outside)
                indices.emplace_back(ci);
                indices.emplace_back(v_curr);
                indices.emplace_back(v_next);
            } else {
                // top cap: CCW from +Y (outside)
                indices.emplace_back(v_next);
                indices.emplace_back(v_curr);
                indices.emplace_back(ci);
            }
        }
    }

    // Side vertices
    int sideStart = (int)positions.size();
    for (int j : {-1, 1}) {
        float y = j * halfLength;
        float r = (j == -1) ? radius : topRadius;
        float v = ((float)j + 1.0f) / 2.0f;

        for (int i = 0; i < segments; ++i) {
            float theta = pi2 * i / segments;
            float c = glm::cos(theta);
            float s = glm::sin(theta);

            positions.emplace_back(P(r * c, y, r * s));
            normals.emplace_back(glm::normalize(P(c, sideSlope, s)));
            float u = (float)i / (float)(segments - 1);
            uvs.emplace_back(u, v);
        }
    }

    for (int i = 0; i < segments; ++i) {
        unsigned int top_i = sideStart + i + segments;
        unsigned int top_next = sideStart + (i + 1) % segments + segments;
        unsigned int bot_i = sideStart + i;
        unsigned int bot_next = sideStart + (i + 1) % segments;

        indices.emplace_back(top_i);
        indices.emplace_back(top_next);
        indices.emplace_back(bot_i);
        indices.emplace_back(top_next);
        indices.emplace_back(bot_next);
        indices.emplace_back(bot_i);
    }

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

MeshData Prim::createCylinderData(float radius, float length, int segments) {
    return createCylinderData(radius, length, UpAxis::Y, segments);
}

MeshData Prim::createCylinderData(float radius, float length, UpAxis upAxis,
                                  int segments) {
    int perm[3];
    if (upAxis == UpAxis::X) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0; // long axis → X
    } else if (upAxis == UpAxis::Z) {
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1; // long axis → Z
    } else {
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2; // long axis → Y (default)
    }
    return makeCylinderMesh(radius, length / 2.0f, radius, perm, segments);
}

MeshData Prim::createArrowData(float baseRadius, float baseHeight,
                               int segments) {
    return createArrowData(baseRadius, baseHeight, UpAxis::Y, -1.0f, -1.0f,
                           segments);
}

// Helper: merge src into dst, offsetting src indices by dst.vertices.size()
static void mergeMesh(MeshData& dst, MeshData&& src) {
    auto base = static_cast<unsigned int>(dst.vertices.size());
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(),
                        src.vertices.end());
    dst.normals.insert(dst.normals.end(), src.normals.begin(),
                       src.normals.end());
    dst.uvs.insert(dst.uvs.end(), src.uvs.begin(), src.uvs.end());
    for (auto idx : src.indices)
        dst.indices.emplace_back(idx + base);
}

MeshData Prim::createArrowData(float baseRadius, float baseHeight,
                               UpAxis upAxis, float capRadius, float capHeight,
                               int segments) {
    if (capRadius < 0.0f)
        capRadius = baseRadius * 1.8f;
    if (capHeight < 0.0f)
        capHeight = baseHeight * 0.18f;

    glm::vec3 axisDir;
    if (upAxis == UpAxis::X)
        axisDir = {1, 0, 0};
    else if (upAxis == UpAxis::Z)
        axisDir = {0, 0, 1};
    else
        axisDir = {0, 1, 0};

    // Shaft: cylinder centered at origin → shift so bottom sits at 0
    MeshData shaft =
        createCylinderData(baseRadius, baseHeight, upAxis, segments);
    for (auto& v : shaft.vertices)
        v += axisDir * (baseHeight / 2.0f);

    // Cone tip: shift to sit on top of shaft (small epsilon to avoid
    // z-fighting)
    MeshData cone = createConeData(capRadius, capHeight, upAxis, segments);
    float coneCenter = baseHeight + capHeight / 2.0f - 1e-3f * baseHeight;
    for (auto& v : cone.vertices)
        v += axisDir * coneCenter;

    mergeMesh(shaft, std::move(cone));
    return shaft;
}
std::vector<Prim*> Prim::defineCoordinateAxes(SceneBackend* scene,
                                              const std::string& basePath,
                                              float length, float radius,
                                              int segments, glm::vec3 origin,
                                              glm::quat orientation) {
    const float safeLength = std::max(length, 1e-4f);
    const float safeRadius = std::max(radius, 1e-5f);
    const float capHeight = safeLength * 0.22f;
    const float shaftHeight = safeLength - capHeight;
    const float capRadius = safeRadius * 2.4f;

    struct AxisDesc {
        const char* name;
        UpAxis axis;
        glm::vec4 color;
    };
    const AxisDesc axes[] = {
        {"x", UpAxis::X, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)},
        {"y", UpAxis::Y, glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)},
        {"z", UpAxis::Z, glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)},
    };

    std::vector<Prim*> result;
    result.reserve(3);
    defineManipulationGroup(scene, basePath);
    for (const auto& axis : axes) {
        auto meshData = std::make_shared<MeshData>(
            createArrowData(safeRadius, shaftHeight, axis.axis, capRadius,
                            capHeight, segments));
        auto* prim =
            scene->definePrim(basePath + "/" + axis.name, PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(origin);
        prim->addRotateQuaternionOp(glm::normalize(orientation));
        prim->setDisplayColorAlpha(axis.color);
        result.push_back(prim);
    }
    return result;
}

MeshData Prim::createCapsuleData(float radius, float height, UpAxis upAxis,
                                 int segments) {
    // rings per hemisphere (at least 4)
    int rings = std::max(4, segments / 4);
    float halfH = height * 0.5f;

    int perm[3];
    if (upAxis == UpAxis::X) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0;
    } else if (upAxis == UpAxis::Z) {
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1;
    } else {
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2;
    }

    auto P = [&](float x, float y, float z) -> glm::vec3 {
        float v[3] = {x, y, z};
        return {v[perm[0]], v[perm[1]], v[perm[2]]};
    };

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    // Layout (internal Y-up):
    //   row 0        : top pole          (theta=0,      yOff=+halfH)
    //   row 1..rings : top hemisphere    (theta→π/2,    yOff=+halfH)
    //   row rings+1  : bottom equator    (theta=π/2,    yOff=-halfH)
    //   row rings+2..2*rings+1: bottom hemisphere→pole (theta→π, yOff=-halfH)
    // The quad strip between row 'rings' and 'rings+1' forms the cylinder wall.
    int totalRings = 2 * rings + 2;
    int vertsPerRing = segments + 1;

    for (int r = 0; r < totalRings; ++r) {
        float theta, yOff;
        if (r <= rings) {
            // top hemisphere: theta 0 → π/2
            theta = glm::half_pi<float>() * r / rings;
            yOff = halfH;
        } else {
            // bottom hemisphere: theta π/2 → π
            int br = r - rings - 1; // 0..rings
            theta = glm::half_pi<float>() + glm::half_pi<float>() * br / rings;
            yOff = -halfH;
        }

        float sinT = glm::sin(theta);
        float cosT = glm::cos(theta);
        float ringY = yOff + radius * cosT;
        float ringR = radius * sinT;

        float vCoord = (float)r / (totalRings - 1);
        for (int s = 0; s <= segments; ++s) {
            float phi = glm::two_pi<float>() * s / segments;
            float cosP = glm::cos(phi);
            float sinP = glm::sin(phi);

            positions.emplace_back(P(ringR * cosP, ringY, ringR * sinP));
            normals.emplace_back(P(sinT * cosP, cosT, sinT * sinP));
            uvs.emplace_back((float)s / segments, vCoord);
        }
    }

    for (int r = 0; r < totalRings - 1; ++r) {
        for (int s = 0; s < segments; ++s) {
            unsigned int a = r * vertsPerRing + s;
            unsigned int b = a + 1;
            unsigned int c = a + vertsPerRing;
            unsigned int d = c + 1;

            // CCW from outside
            indices.emplace_back(a);
            indices.emplace_back(b);
            indices.emplace_back(c);
            indices.emplace_back(b);
            indices.emplace_back(d);
            indices.emplace_back(c);
        }
    }

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

MeshData Prim::createConeData(float radius, float height, UpAxis upAxis,
                              int segments) {
    int perm[3];
    if (upAxis == UpAxis::X) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0;
    } else if (upAxis == UpAxis::Z) {
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1;
    } else {
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2;
    }
    return makeCylinderMesh(radius, height / 2.0f, 0.0f, perm, segments);
}

std::vector<Prim*> Prim::definePoints(SceneBackend* scene,
                                      const std::string& basePath,
                                      const std::vector<glm::vec3>& points,
                                      float radius, glm::vec4 color,
                                      int segments) {
    auto meshData = std::make_shared<MeshData>(
        createSphereData(radius, segments, segments / 2 + 1));

    std::vector<Prim*> result;
    result.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        std::string path = basePath + "/" + std::to_string(i);
        auto* prim = scene->definePrim(path, PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(points[i]);
        prim->setDisplayColorAlpha(color);
        result.push_back(prim);
    }
    return result;
}

std::vector<Prim*> Prim::defineLines(SceneBackend* scene,
                                     const std::string& basePath,
                                     const std::vector<glm::vec3>& vertices,
                                     const std::vector<unsigned int>& indices,
                                     float radius, glm::vec4 color,
                                     int segments) {

    auto meshData = std::make_shared<MeshData>(
        createCapsuleData(radius, 1.0f, UpAxis::Y, segments));

    std::vector<Prim*> result;
    int segIdx = 0;
    for (size_t i = 0; i + 1 < indices.size(); i += 2) {
        const glm::vec3& a = vertices[indices[i]];
        const glm::vec3& b = vertices[indices[i + 1]];
        glm::vec3 diff = b - a;
        float len = glm::length(diff);
        if (len < 1e-6f)
            continue;

        glm::vec3 dir = diff / len;
        glm::vec3 center = (a + b) * 0.5f;

        // Rotation: Y-axis -> diff direction
        Eigen::Quaternionf eq = Eigen::Quaternionf::FromTwoVectors(
            Eigen::Vector3f::UnitY(), Eigen::Vector3f(dir.x, dir.y, dir.z));
        glm::quat rot(eq.w(), eq.x(), eq.y(), eq.z());

        std::string path = basePath + "/" + std::to_string(segIdx++);
        auto* prim = scene->definePrim(path, PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(center);
        prim->addRotateQuaternionOp(rot);
        prim->addScaleOp(
            glm::vec3(1.f, len, 1.f)); // Y-stretch = segment length
        prim->setDisplayColorAlpha(color);
        result.push_back(prim);
    }
    return result;
}

MeshData Prim::createRectangleData(float xScale, float yScale, float zScale) {
    // Scale means the length of one side.
    float xHalf = xScale / 2;
    float yHalf = yScale / 2;
    float zHalf = zScale / 2;
    //    v3----- v7
    //   /|      /|
    //  v2------v6|
    //  | |     | |
    //  | v0----|-v4
    //  |/      |/
    //  v1------v5
    //
    std::vector<glm::vec3> positions = {
        // v0, v1, v2, v3
        glm::vec3(-xHalf, -yHalf, -zHalf),
        glm::vec3(-xHalf, -yHalf, zHalf),
        glm::vec3(-xHalf, yHalf, zHalf),
        glm::vec3(-xHalf, yHalf, -zHalf),
        // v4, v5, v6, v7
        glm::vec3(xHalf, -yHalf, -zHalf),
        glm::vec3(xHalf, -yHalf, zHalf),
        glm::vec3(xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, -zHalf),
        // v0, v1, v5, v4
        glm::vec3(-xHalf, -yHalf, -zHalf),
        glm::vec3(-xHalf, -yHalf, zHalf),
        glm::vec3(xHalf, -yHalf, zHalf),
        glm::vec3(xHalf, -yHalf, -zHalf),
        // v3, v2, v6, v7
        glm::vec3(-xHalf, yHalf, -zHalf),
        glm::vec3(-xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, -zHalf),
        // v0, v3, v7, v4
        glm::vec3(-xHalf, -yHalf, -zHalf),
        glm::vec3(-xHalf, yHalf, -zHalf),
        glm::vec3(xHalf, yHalf, -zHalf),
        glm::vec3(xHalf, -yHalf, -zHalf),
        // v1, v2, v6, v5
        glm::vec3(-xHalf, -yHalf, zHalf),
        glm::vec3(-xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, zHalf),
        glm::vec3(xHalf, -yHalf, zHalf),
    }; // positions.size() == 24

    std::vector<glm::vec3> normals = {
        glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0), glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),  glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 0, -1), glm::vec3(0, 0, -1),
        glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),
    };

    std::vector<glm::vec2> uvs = {
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
    };

    std::vector<unsigned int> indices = {
        0,  1,  2,  0,  2,  3,  // left
        4,  6,  5,  4,  7,  6,  // right
        8,  10, 9,  8,  11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 23, 22, 20, 22, 21, // front
    };

    MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);

    return meshData;
}

void Prim::traverse(std::function<void(Prim*)> func) {
    if (!isActiveInHierarchy())
        return;
    func(this);
    for (auto& child : _children) {
        child->traverse(func);
    }
}

void Prim::onAttributeChanged(const Token& name) {
    if (XformTokens::isXformAttribute(name))
        markLocalTransformDirty();
}

void Prim::markLocalTransformDirty() {
    if (_transformComponent)
        _transformComponent->markLocalTransformDirty();
}

void Prim::markWorldTransformDirtyRecursive() {
    if (_transformComponent)
        _transformComponent->markWorldTransformDirtyRecursive();
}

TransformComponent& Prim::getTransformComponentOrThrow() {
    if (!_transformComponent) {
        throw std::runtime_error("Prim '" + _path +
                                 "' has no TransformComponent");
    }
    return *_transformComponent;
}

const TransformComponent& Prim::getTransformComponentOrThrow() const {
    if (!_transformComponent) {
        throw std::runtime_error("Prim '" + _path +
                                 "' has no TransformComponent");
    }
    return *_transformComponent;
}

void Prim::setLocalTranslation(glm::vec3 trans) {
    getTransformComponentOrThrow().setLocalTranslation(trans);
}

void Prim::setLocalScale(glm::vec3 scale) {
    getTransformComponentOrThrow().setLocalScale(scale);
}

void Prim::setLocalRotation(glm::quat quat) {
    getTransformComponentOrThrow().setLocalRotation(quat);
}

void Prim::setLocalMatrix(const glm::mat4& matrix) {
    getTransformComponentOrThrow().setLocalMatrix(matrix);
}

void Prim::setWorldTranslation(glm::vec3 trans) {
    getTransformComponentOrThrow().setWorldTranslation(trans);
}

void Prim::setWorldRotation(glm::quat quat) {
    getTransformComponentOrThrow().setWorldRotation(quat);
}

void Prim::setWorldMatrix(const glm::mat4& matrix) {
    getTransformComponentOrThrow().setWorldMatrix(matrix);
}

glm::mat4 Prim::computeLocalMatrix() {
    return getTransformComponentOrThrow().computeLocalMatrix();
}

glm::mat4 Prim::computeWorldMatrix() {
    return getTransformComponentOrThrow().computeWorldMatrix();
}

glm::mat4 Prim::computeModelMatrix() {
    return getTransformComponentOrThrow().computeModelMatrix();
}

} // namespace Scene
} // namespace KE
