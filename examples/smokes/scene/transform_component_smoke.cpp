// Checks authored xform order and compatibility Attribute access without a renderer.
#include "engine/scene/component/transform_component.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/native/xform_token.hpp"

#include <glm/gtc/epsilon.hpp>
#include <iostream>
#include <stdexcept>

using namespace KE::Scene;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void checkTranslation(Prim& prim, glm::vec3 expected) {
    require(glm::all(glm::epsilonEqual(prim.getLocalTranslation(), expected, 1e-5f)),
            "xform operation order changed translation");
}
} // namespace

int main() {
    Prim prim("ordered", PrimType::Xform);
    auto component = prim.getTransformComponent();
    prim.setAttribute(XformTokens::translate, glm::vec3(1, 2, 3));
    prim.setAttribute(XformTokens::scale, glm::vec3(2, 3, 4));
    prim.setXformOpOrder({"xformOp:translate", "xformOp:scale"});
    checkTranslation(prim, {2, 6, 12});
    component->setXformOpOrder({"xformOp:scale", "xformOp:translate"});
    checkTranslation(prim, {1, 2, 3});

    prim.setAttribute("label", std::string("ordered"));
    const auto snapshot = prim.getAttributes();
    require(snapshot.size() == 4 && snapshot.count(XformTokens::opOrder),
            "inspection omitted authored operations or metadata");
    component->setLocalTranslation({4, 5, 6});
    require(std::get<glm::vec3>(snapshot.at(XformTokens::translate)) == glm::vec3(1, 2, 3),
            "inspection snapshot changed with the source");
    require(prim.getAttribute<glm::vec3>(XformTokens::translate) == glm::vec3(4, 5, 6),
            "Prim and component disagree on the source");
    require(prim.getAttribute<std::string>("label") == "ordered",
            "transform edits changed metadata");
    require(prim.getAttribute<int>("missing", 42) == 42,
            "missing attribute lost its default");
    try {
        prim.getAttribute<int>("missing");
        throw std::runtime_error("missing attribute did not throw");
    } catch (const std::out_of_range&) {
    }

    // Custom order still selects Euler rotation instead of the inactive scale.
    prim.setAttribute(XformTokens::rotateXYZ, glm::vec3(0, 0, 90));
    prim.setXformOpOrder({"xformOp:rotateXYZ", "xformOp:translate"});
    auto matrix = component->computeLocalMatrix();
    require(glm::all(glm::epsilonEqual(glm::vec3(matrix[0]), glm::vec3(0, 1, 0), 1e-5f)),
            "rotateXYZ or inactive operation handling changed");
    component->setLocalScale({2, 3, 4});
    checkTranslation(prim, {4, 5, 6});
    matrix = component->computeLocalMatrix();
    require(glm::all(glm::epsilonEqual(glm::vec3(matrix[0]), glm::vec3(0, 2, 0), 1e-5f)),
            "TRS conversion lost the authored rotation");

    glm::mat4 source(1.0f);
    source[3] = glm::vec4(7, 8, 9, 1);
    prim.setAttribute(XformTokens::transform, source);
    prim.setXformOpOrder({"xformOp:transform"});
    require(component->computeLocalMatrix() == source,
            "matrix Attribute write did not invalidate the cache");
    prim.setXformOpOrder({});
    require(component->computeLocalMatrix() == glm::mat4(1.0f),
            "empty operation order did not produce identity");

    Prim resource("resource", PrimType::Resource);
    require(!resource.hasAttribute(XformTokens::translate),
            "resource unexpectedly has a transform");
    bool rejected = false;
    try {
        resource.setAttribute(XformTokens::translate, glm::vec3(1));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "xform data was stored outside a TransformComponent");
    std::cout << "PASS: TransformComponent source, xform order, and Attribute access\n";
}
