// Checks static scene registration, material values, and instance lifetime.
#include "engine/scene/render/scene_render_bridge.hpp"
#include "engine/scene/native/native_scene.hpp"
#include "engine/scene/scene_resource_manager.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/graphics/material/material.hpp"

#include <iostream>
#include <stdexcept>

using namespace KE;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main() {
    PhongMaterial phong;
    phong.diffuse = {0.2f, 0.4f, 0.6f};
    phong.ambient = {0.05f, 0.1f, 0.15f};
    phong.specular = {0.8f, 0.7f, 0.6f};
    phong.shininess = 64.0f;
    const auto phongInput = Scene::SceneRenderBridge::materialData(phong);
    phong.diffuse = glm::vec3(1.0f);
    require(phongInput.baseColor == glm::vec4(0.2f, 0.4f, 0.6f, 1.0f) &&
                phongInput.ambient == phong.ambient &&
                phongInput.specular == phong.specular &&
                phongInput.shininess == 64.0f && !phongInput.hasNativeTextures,
            "Phong input did not preserve independent material values");

    PBRMaterial pbr;
    pbr.baseColor = {0.4f, 0.3f, 0.2f, 0.7f};
    pbr.metallic = 0.8f;
    pbr.roughness = 0.2f;
    pbr.emissiveColor = {0.1f, 0.3f, 0.5f};
    pbr.emissiveStrength = 2.0f;
    pbr.setUnlit(true);
    pbr.texturePaths["baseColor"] = "deferred.png";
    const auto pbrInput = Scene::SceneRenderBridge::materialData(pbr);
    require(pbrInput.baseColor == pbr.baseColor &&
                pbrInput.metallic == pbr.metallic &&
                pbrInput.roughness == pbr.roughness &&
                pbrInput.emissiveColor == pbr.emissiveColor &&
                pbrInput.emissiveStrength == pbr.emissiveStrength &&
                pbrInput.unlit && pbrInput.hasNativeTextures,
            "PBR input lost factors or deferred texture requirements");
    ColorMaterial color(ColorStyle::Unlit);
    require(Scene::SceneRenderBridge::materialData(color).unlit,
            "legacy unlit style lost its meaning");

    Scene::NativeScene scene;
    Scene::SceneResourceManager resources(&scene);
    auto mesh =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1));
    const auto resource = resources.registerMesh("cube", mesh);
    auto* parent = scene.definePrim("/group", Scene::PrimType::Xform);
    auto* first = scene.definePrim("/group/first", Scene::PrimType::Mesh);
    auto* second = scene.definePrim("/group/second", Scene::PrimType::Mesh);
    for (auto* prim : {first, second}) {
        prim->setMeshData(mesh);
        prim->getMeshComponent()->setResourceHandle(resource);
    }
    parent->setLocalTranslation({10, 0, 0});
    first->setLocalTranslation({1, 2, 3});
    second->setLocalTranslation({4, 5, 6});
    const auto source = Scene::SceneRenderBridge::meshData(*first);
    require(source.mesh.get() == mesh.get() && source.resource == resource,
            "scene mesh resource identity was lost");

    Scene::SceneRenderBridge input;
    input.add(*first);
    input.add(*second);
    input.collect();
    const auto& values = input.instances();
    require(values.transforms.size() == 2 &&
                glm::vec3(values.transforms[0][3]) == glm::vec3(11, 2, 3),
            "parent world transform was not extracted");
    const auto firstId = values.ids[0];
    const auto secondId = values.ids[1];
    require(firstId != secondId && input.resolve(secondId) == second,
            "instance identity is not independent of array position");
    const auto* storage = values.transforms.data();
    input.collect();
    require(storage == values.transforms.data() && firstId == values.ids[0],
            "unchanged collection replaced storage or instance identity");

    first->setVisible(false);
    input.collect();
    require(values.ids.size() == 1 && values.ids[0] == secondId,
            "visibility compaction changed surviving instance identity");
    parent->setActive(false);
    input.collect();
    require(values.ids.empty() && second->isVisible(),
            "active state changed authored visibility");
    parent->setActive(true);
    input.collect();
    second->getSelectionComponent()->setPickable(false);
    input.refreshSelection();
    require(!values.pickable[0], "selection edits required a frame update");

    input.remove(*second);
    scene.removePrim("/group/second");
    input.refreshSelection();
    require(!input.resolve(secondId) && !values.pickable[0],
            "deleted instance still resolves before the next frame");
    auto* replacement =
        scene.definePrim("/group/second", Scene::PrimType::Mesh);
    replacement->setMeshData(mesh);
    input.add(*replacement);
    input.collect();
    require(values.ids.size() == 1 && values.ids[0] != secondId &&
                input.resolve(values.ids[0]) == replacement,
            "recreated path reused a stale instance ID");
    require(mesh->vertices == Scene::Prim::createCubeData(1).vertices,
            "scene extraction changed the source mesh");
    std::cout << "PASS: context-free scene/material input, resource identity, "
                 "transforms, "
                 "visibility, and instance lifetime\n";
}
