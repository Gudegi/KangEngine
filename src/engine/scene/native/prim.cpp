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
#include "engine/scene/component/articulation_component.hpp"
#include "engine/scene/component/articulation_binding_component.hpp"
#include "engine/scene/component/collision_shape_component.hpp"
#include "engine/scene/component/rigid_body_component.hpp"
#include "engine/scene/scene_backend.hpp"
#include "geometry/primitive_mesh.hpp"
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
    if (_articulationComponent)
        _articulationComponent->detach();
    if (_articulationBindingComponent)
        _articulationBindingComponent->detach();
    if (_collisionShapeComponent)
        _collisionShapeComponent->detach();
    if (_rigidBodyComponent)
        _rigidBodyComponent->detach();
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

std::shared_ptr<ArticulationComponent> Prim::addArticulationComponent() {
    if (_articulationComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has an ArticulationComponent");
    _articulationComponent =
        std::shared_ptr<ArticulationComponent>(
            new ArticulationComponent(this));
    return _articulationComponent;
}

std::shared_ptr<ArticulationComponent>
Prim::getArticulationComponent() const {
    return _articulationComponent;
}

bool Prim::removeArticulationComponent() {
    if (!_articulationComponent)
        return false;
    _articulationComponent->detach();
    _articulationComponent.reset();
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

std::shared_ptr<CollisionShapeComponent> Prim::addCollisionShapeComponent() {
    if (_collisionShapeComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a CollisionShapeComponent");
    _collisionShapeComponent =
        std::shared_ptr<CollisionShapeComponent>(
            new CollisionShapeComponent(this));
    return _collisionShapeComponent;
}

std::shared_ptr<CollisionShapeComponent>
Prim::getCollisionShapeComponent() const {
    return _collisionShapeComponent;
}

bool Prim::removeCollisionShapeComponent() {
    if (!_collisionShapeComponent)
        return false;
    _collisionShapeComponent->detach();
    _collisionShapeComponent.reset();
    return true;
}

std::shared_ptr<RigidBodyComponent> Prim::addRigidBodyComponent() {
    if (_type == PrimType::Root || _type == PrimType::Resource)
        throw std::runtime_error(
            "Prim '" + _path +
            "' cannot add a RigidBodyComponent to a Root or Resource Prim");
    if (_rigidBodyComponent)
        throw std::runtime_error("Prim '" + _path +
                                 "' already has a RigidBodyComponent");
    _rigidBodyComponent =
        std::shared_ptr<RigidBodyComponent>(new RigidBodyComponent(this));
    return _rigidBodyComponent;
}

std::shared_ptr<RigidBodyComponent> Prim::getRigidBodyComponent() const {
    return _rigidBodyComponent;
}

bool Prim::removeRigidBodyComponent() {
    if (!_rigidBodyComponent)
        return false;
    _rigidBodyComponent->detach();
    _rigidBodyComponent.reset();
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
    return Geometry::createCube(scale);
}

MeshData Prim::createPlaneData(float scale) {
    return Geometry::createPlane(scale, UpAxis::Y);
}

MeshData Prim::createPlaneData(float scale, UpAxis upAxis) {
    return Geometry::createPlane(scale, upAxis);
}

MeshData Prim::createSphereData(float radius, int numLongitudes,
                                int numLatitudes) {
    return Geometry::createSphere(radius, numLongitudes, numLatitudes);
}

MeshData Prim::createRectangleData(float xScale, float yScale, float zScale) {
    return Geometry::createBox(xScale, yScale, zScale);
}

MeshData Prim::createCylinderData(float radius, float length, int segments) {
    return Geometry::createCylinder(radius, length, UpAxis::Y, segments);
}

MeshData Prim::createCylinderData(float radius, float length, UpAxis upAxis,
                                  int segments) {
    return Geometry::createCylinder(radius, length, upAxis, segments);
}

MeshData Prim::createArrowData(float baseRadius, float baseHeight,
                               int segments) {
    return Geometry::createArrow(baseRadius, baseHeight, UpAxis::Y, -1.0f,
                                 -1.0f, segments);
}

MeshData Prim::createArrowData(float baseRadius, float baseHeight,
                               UpAxis upAxis, float capRadius, float capHeight,
                               int segments) {
    return Geometry::createArrow(baseRadius, baseHeight, upAxis, capRadius,
                                 capHeight, segments);
}

MeshData Prim::createCapsuleData(float radius, float height, UpAxis upAxis,
                                 int segments) {
    return Geometry::createCapsule(radius, height, upAxis, segments);
}

MeshData Prim::createConeData(float radius, float height, UpAxis upAxis,
                              int segments) {
    return Geometry::createCone(radius, height, upAxis, segments);
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

void Prim::setLocalRotationAxisAngle(glm::vec3 axis, float angleRadians) {
    getTransformComponentOrThrow().setLocalRotationAxisAngle(axis,
                                                            angleRadians);
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

void Prim::setWorldRotationAxisAngle(glm::vec3 axis, float angleRadians) {
    getTransformComponentOrThrow().setWorldRotationAxisAngle(axis,
                                                            angleRadians);
}

void Prim::setWorldMatrix(const glm::mat4& matrix) {
    getTransformComponentOrThrow().setWorldMatrix(matrix);
}

glm::vec3 Prim::getLocalTranslation() {
    return getTransformComponentOrThrow().getLocalTranslation();
}

glm::quat Prim::getLocalRotation() {
    return getTransformComponentOrThrow().getLocalRotation();
}

glm::vec3 Prim::getWorldTranslation() {
    return getTransformComponentOrThrow().getWorldTranslation();
}

glm::quat Prim::getWorldRotation() {
    return getTransformComponentOrThrow().getWorldRotation();
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
