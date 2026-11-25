#include "transform_component.hpp"

#include "engine/scene/native/prim.hpp"
#include "engine/scene/native/xform_token.hpp"

#include <glm/ext/quaternion_transform.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/trigonometric.hpp>

#include <stdexcept>
#include <utility>

namespace KE {
namespace Scene {

namespace {
void decomposeTRS(const glm::mat4& matrix, glm::vec3& translation,
                  glm::quat& rotation, glm::vec3& scale) {
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(matrix, scale, rotation, translation, skew, perspective);
    rotation = glm::normalize(rotation);
}

glm::vec3 matrixTranslation(const glm::mat4& matrix) {
    const float w = matrix[3][3];
    if (glm::epsilonEqual(w, 0.0f, glm::epsilon<float>()))
        throw std::invalid_argument(
            "Cannot extract translation from a matrix with zero homogeneous w");
    // Match decomposition's homogeneous normalization without decomposing TRS.
    return glm::vec3(matrix[3]) / w;
}

} // namespace

TransformComponent::TransformComponent(Prim* owner)
    : ComponentBase(owner, "TransformComponent") {}

void TransformComponent::detach() {
    if (!_owner)
        return;
    detachBase();
}

void TransformComponent::markLocalTransformDirty() {
    requireAttached();
    _localDirty = true;
    if (!_suppressLocalDirtyVersion)
        markChanged();
    markWorldTransformDirtyRecursive(false);
}

void TransformComponent::markWorldTransformDirtyRecursive(
    bool countSelfVersion) {
    requireAttached();
    const bool becameDirty = !_worldDirty;
    _worldDirty = true;
    if (countSelfVersion && becameDirty)
        markChanged();
    for (Prim* child : _owner->getChildren()) {
        if (auto transform = child->getTransformComponent())
            transform->markWorldTransformDirtyRecursive();
    }
}

void TransformComponent::prepareTRS() {
    const auto& attributes = _xformAttributes;
    const auto it = attributes.find(XformTokens::opOrder);
    if (it == attributes.end())
        return;
    static const std::vector<std::string> trsOrder = {
        "xformOp:scale", "xformOp:rotateQuaternion", "xformOp:translate"};
    if (std::get<std::vector<std::string>>(it->second) == trsOrder)
        return;

    // Preserve the evaluated pose when replacing matrix or custom xform ops.
    glm::vec3 translation, scale;
    glm::quat rotation;
    glm::vec3 skew;
    glm::vec4 perspective;
    if (!glm::decompose(computeLocalMatrix(), scale, rotation, translation,
                         skew, perspective) ||
        glm::length(skew) > 1e-5f ||
        glm::length(perspective - glm::vec4(0, 0, 0, 1)) > 1e-5f)
        throw std::invalid_argument(
            "TRS edits require a decomposable affine transform without shear; "
            "use setLocalMatrix for this transform");
    rotation = glm::normalize(rotation);
    setXformAttribute(XformTokens::translate, translation);
    setXformAttribute(XformTokens::rotateQuat, rotation);
    setXformAttribute(XformTokens::scale, scale);
}

void TransformComponent::setXformAttribute(const Token& name,
                                           AttributeValue value) {
    requireAttached();
    _xformAttributes[name] = std::move(value);
    markLocalTransformDirty();
}

void TransformComponent::setXformOpOrder(const std::vector<std::string>& order) {
    setXformAttribute(XformTokens::opOrder, order);
}

void TransformComponent::setLocalTranslation(glm::vec3 translation) {
    requireAttached();
    const auto& attributes = _xformAttributes;
    const auto it = attributes.find(XformTokens::opOrder);
    const auto* order = it == attributes.end() ? nullptr :
        &std::get<std::vector<std::string>>(it->second);
    if (order && order->size() == 1 && order->front() == "xformOp:transform") {
        auto matrix = computeLocalMatrix();
        matrix[3] = glm::vec4(translation, 1.0f);
        setLocalMatrix(matrix);
        return;
    }
    _suppressLocalDirtyVersion = true;
    try {
        prepareTRS();
        setXformAttribute(XformTokens::translate, translation);
        setXformOpOrder(
            {"xformOp:scale", "xformOp:rotateQuaternion", "xformOp:translate"});
    } catch (...) {
        _suppressLocalDirtyVersion = false;
        throw;
    }
    _suppressLocalDirtyVersion = false;
    markChanged();
}

void TransformComponent::setLocalScale(glm::vec3 scale) {
    requireAttached();
    _suppressLocalDirtyVersion = true;
    try {
        prepareTRS();
        setXformAttribute(XformTokens::scale, scale);
        setXformOpOrder(
            {"xformOp:scale", "xformOp:rotateQuaternion", "xformOp:translate"});
    } catch (...) {
        _suppressLocalDirtyVersion = false;
        throw;
    }
    _suppressLocalDirtyVersion = false;
    markChanged();
}

void TransformComponent::setLocalRotation(glm::quat rotation) {
    requireAttached();
    _suppressLocalDirtyVersion = true;
    try {
        prepareTRS();
        setXformAttribute(XformTokens::rotateQuat, glm::normalize(rotation));
        setXformOpOrder(
            {"xformOp:scale", "xformOp:rotateQuaternion", "xformOp:translate"});
    } catch (...) {
        _suppressLocalDirtyVersion = false;
        throw;
    }
    _suppressLocalDirtyVersion = false;
    markChanged();
}

void TransformComponent::setLocalRotationAxisAngle(glm::vec3 axis,
                                                   float angleRadians) {
    if (glm::length(axis) <= 1e-6f)
        throw std::invalid_argument("rotation axis must be non-zero");
    setLocalRotation(glm::angleAxis(angleRadians, glm::normalize(axis)));
}

void TransformComponent::setLocalMatrix(const glm::mat4& matrix) {
    requireAttached();
    _suppressLocalDirtyVersion = true;
    try {
        setXformAttribute(XformTokens::transform, matrix);
        setXformOpOrder({"xformOp:transform"});
    } catch (...) {
        _suppressLocalDirtyVersion = false;
        throw;
    }
    _suppressLocalDirtyVersion = false;
    markChanged();
}

void TransformComponent::setWorldTranslation(glm::vec3 translation) {
    requireAttached();
    const Prim* parent = _owner->getParent();
    const glm::mat4 parentWorld =
        parent ? const_cast<Prim*>(parent)->computeWorldMatrix()
               : glm::mat4(1.0f);
    const glm::vec3 local =
        glm::vec3(glm::inverse(parentWorld) * glm::vec4(translation, 1.0f));
    setLocalTranslation(local);
}

void TransformComponent::setWorldRotation(glm::quat rotation) {
    requireAttached();
    const Prim* parent = _owner->getParent();
    const glm::mat4 parentWorld =
        parent ? const_cast<Prim*>(parent)->computeWorldMatrix()
               : glm::mat4(1.0f);
    glm::vec3 parentTranslation;
    glm::quat parentRotation;
    glm::vec3 parentScale;
    decomposeTRS(parentWorld, parentTranslation, parentRotation, parentScale);
    setLocalRotation(glm::inverse(parentRotation) * glm::normalize(rotation));
}

void TransformComponent::setWorldRotationAxisAngle(glm::vec3 axis,
                                                   float angleRadians) {
    if (glm::length(axis) <= 1e-6f)
        throw std::invalid_argument("rotation axis must be non-zero");
    setWorldRotation(glm::angleAxis(angleRadians, glm::normalize(axis)));
}

void TransformComponent::setWorldMatrix(const glm::mat4& matrix) {
    requireAttached();
    const Prim* parent = _owner->getParent();
    const glm::mat4 parentWorld =
        parent ? const_cast<Prim*>(parent)->computeWorldMatrix()
               : glm::mat4(1.0f);
    const glm::mat4 localMatrix = glm::inverse(parentWorld) * matrix;
    setLocalMatrix(localMatrix);
}

glm::vec3 TransformComponent::getLocalTranslation() {
    return matrixTranslation(computeLocalMatrix());
}

glm::quat TransformComponent::getLocalRotation() {
    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
    decomposeTRS(computeLocalMatrix(), translation, rotation, scale);
    return rotation;
}

glm::vec3 TransformComponent::getWorldTranslation() {
    return matrixTranslation(computeWorldMatrix());
}

glm::quat TransformComponent::getWorldRotation() {
    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
    decomposeTRS(computeWorldMatrix(), translation, rotation, scale);
    return rotation;
}

glm::mat4 TransformComponent::computeLocalMatrix() {
    requireAttached();
    if (_localDirty) {
        glm::mat4 result(1.0f);

        const std::vector<Token>* orderPtr = &XformTokens::defaultOpOrder;
        std::vector<Token> customOrder;

        if (_xformAttributes.count(XformTokens::opOrder)) {
            const auto& strOrder =
                getXformAttribute<std::vector<std::string>>(
                    XformTokens::opOrder);
            customOrder.reserve(strOrder.size());
            for (const auto& s : strOrder) {
                customOrder.emplace_back(s);
            }
            orderPtr = &customOrder;
        }

        for (const auto& opToken : *orderPtr) {
            if (!_xformAttributes.count(opToken))
                continue;

            XformOpType type = XformTokens::getXformOpType(opToken);
            glm::mat4 opMat(1.0f);

            switch (type) {
            case XformOpType::Translate: {
                auto t = getXformAttribute<glm::vec3>(opToken);
                opMat = glm::translate(glm::mat4(1.0f), t);
                break;
            }
            case XformOpType::RotateQuat: {
                auto q = getXformAttribute<glm::quat>(opToken);
                opMat = glm::mat4_cast(q);
                break;
            }
            case XformOpType::RotateXYZ: {
                auto r = getXformAttribute<glm::vec3>(opToken);
                glm::mat4 rot =
                    glm::rotate(glm::mat4(1.0f), glm::radians(r.x), {1, 0, 0});
                rot = glm::rotate(rot, glm::radians(r.y), {0, 1, 0});
                rot = glm::rotate(rot, glm::radians(r.z), {0, 0, 1});
                opMat = rot;
                break;
            }
            case XformOpType::Scale: {
                auto s = getXformAttribute<glm::vec3>(opToken);
                opMat = glm::scale(glm::mat4(1.0f), s);
                break;
            }
            case XformOpType::Matrix: {
                opMat = getXformAttribute<glm::mat4>(opToken);
                break;
            }
            default:
                break;
            }

            // Apply operations in reverse order of the list
            // (Pre-multiplication). If opOrder is {Scale, Rotate, Translate},
            // this results in: Result = Translate * (Rotate * (Scale * I)) =
            // T * R * S. This produces the standard Local-to-Parent transform.
            result = opMat * result;
        }
        _cachedLocalMat = result;
        _localDirty = false;
    }
    return _cachedLocalMat;
}

glm::mat4 TransformComponent::computeWorldMatrix() {
    requireAttached();
    if (_worldDirty) {
        const glm::mat4 local = computeLocalMatrix();
        const Prim* parent = _owner->getParent();
        _cachedWorldMat =
            parent ? const_cast<Prim*>(parent)->computeWorldMatrix() * local
                   : local;
        _worldDirty = false;
    }
    return _cachedWorldMat;
}

glm::mat4 TransformComponent::computeModelMatrix() {
    return computeWorldMatrix();
}

} // namespace Scene
} // namespace KE
