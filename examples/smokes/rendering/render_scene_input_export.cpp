// Exports CPU scene data as JSON for the standalone ovrtx probe.
#include "engine/graphics/material/material.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/native/native_scene.hpp"
#include "engine/scene/render/scene_render_bridge.hpp"
#include "engine/scene/scene_resource_manager.hpp"

#include <iostream>
#include <stdexcept>

using namespace KE;

namespace {
void vector(const glm::vec3& value) {
    std::cout << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

void frame(const char* name, const Scene::SceneRenderBridge& input,
           const PBRMaterial& material) {
    const auto values = Scene::SceneRenderBridge::materialData(material);
    std::cout << "{\"name\":\"" << name << "\",\"material\":{\"model\":\"PBR\","
              << "\"base_color\":[" << values.baseColor.r << ','
              << values.baseColor.g << ',' << values.baseColor.b << ','
              << values.baseColor.a << "],\"metallic\":" << values.metallic
              << ",\"roughness\":" << values.roughness
              << ",\"unlit\":" << (values.unlit ? "true" : "false")
              << ",\"has_native_textures\":"
              << (values.hasNativeTextures ? "true" : "false")
              << "},\"instances\":[";
    const auto& instances = input.instances();
    for (size_t i = 0; i < instances.ids.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << "{\"id\":" << instances.ids[i] << ",\"world_matrix\":[";
        // JSON rows describe a column-vector world transform.
        for (int row = 0; row < 4; ++row) {
            if (row)
                std::cout << ',';
            std::cout << '[';
            for (int col = 0; col < 4; ++col) {
                if (col)
                    std::cout << ',';
                std::cout << instances.transforms[i][col][row];
            }
            std::cout << ']';
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}
} // namespace

int main() {
    Scene::NativeScene scene;
    Scene::SceneResourceManager resources(&scene);
    auto mesh =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(1));
    const auto resource = resources.registerMesh("cube", mesh);
    auto* parent = scene.definePrim("/group", Scene::PrimType::Xform);
    parent->setLocalTranslation({0.25f, 0, 0});
    auto create = [&] {
        auto* prim = scene.definePrim("/group/cube", Scene::PrimType::Mesh);
        prim->setMeshData(mesh);
        prim->getMeshComponent()->setResourceHandle(resource);
        prim->setLocalTranslation({-1, 0, 0});
        return prim;
    };
    auto* prim = create();
    const auto source = Scene::SceneRenderBridge::meshData(*prim);
    PBRMaterial material;
    material.baseColor = {0.8f, 0.15f, 0.05f, 1};
    material.roughness = 0.7f;
    Scene::SceneRenderBridge input;
    input.add(*prim);
    input.collect();
    const auto originalId = input.instances().ids[0];

    std::cout << "{\"schema_version\":1,\"mesh\":{\"resource_id\":"
              << source.resource << ",\"points\":[";
    for (size_t i = 0; i < source.mesh->vertices.size(); ++i) {
        if (i)
            std::cout << ',';
        vector(source.mesh->vertices[i]);
    }
    std::cout << "],\"normals\":[";
    for (size_t i = 0; i < source.mesh->normals.size(); ++i) {
        if (i)
            std::cout << ',';
        vector(source.mesh->normals[i]);
    }
    std::cout << "],\"colors\":[";
    for (size_t i = 0; i < source.mesh->colors.size(); ++i) {
        if (i)
            std::cout << ',';
        vector(source.mesh->colors[i]);
    }
    std::cout << "],\"indices\":[";
    for (size_t i = 0; i < source.mesh->indices.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << source.mesh->indices[i];
    }
    std::cout << "]},\"frames\":[";
    frame("initial", input, material);
    prim->setLocalTranslation({0.5f, 0, 0});
    input.collect();
    std::cout << ',';
    frame("moved", input, material);
    material.baseColor = {0.05f, 0.7f, 0.15f, 1};
    std::cout << ',';
    frame("recolored", input, material);
    input.remove(*prim);
    scene.removePrim("/group/cube");
    input.collect();
    std::cout << ',';
    frame("removed", input, material);
    prim = create();
    input.add(*prim);
    input.collect();
    if (input.instances().ids[0] == originalId)
        throw std::runtime_error("recreated instance reused a stale ID");
    std::cout << ',';
    frame("recreated", input, material);
    prim->setLocalTranslation({0.5f, 0, 0});
    input.collect();
    std::cout << ',';
    frame("recreated_moved", input, material);

    // Replace all instances without publishing an empty frame.
    input.remove(*prim);
    scene.removePrim("/group/cube");
    prim = create();
    input.add(*prim);
    input.collect();
    std::cout << ',';
    frame("replaced", input, material);
    std::cout << "]}\n";
}
