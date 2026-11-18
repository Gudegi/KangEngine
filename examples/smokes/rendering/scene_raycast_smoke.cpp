#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/native/native_scene.hpp"
#include "geometry/mesh_bvh.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace KE;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void checkBVH() {
    auto mesh = std::make_shared<Scene::MeshData>();
    std::mt19937 random(42);
    std::uniform_real_distribution<float> value(-4.0f, 4.0f);
    for (unsigned int i = 0; i < 300; ++i) {
        mesh->vertices.push_back({value(random), value(random), value(random)});
        mesh->indices.push_back(i);
    }
    Geometry::MeshBVH bvh(mesh);
    for (int i = 0; i < 500; ++i) {
        const Geometry::Ray ray({value(random), value(random), value(random)},
                                {value(random), value(random), value(random)});
        float expected = std::numeric_limits<float>::infinity();
        bool expectedHit = false;
        for (size_t j = 0; j < mesh->indices.size(); j += 3) {
            glm::vec2 barycentric;
            float distance;
            if (ray.intersectTriangle(mesh->vertices[mesh->indices[j]],
                                      mesh->vertices[mesh->indices[j + 1]],
                                      mesh->vertices[mesh->indices[j + 2]],
                                      barycentric, distance) &&
                distance >= 0.0f && distance < expected) {
                expected = distance;
                expectedHit = true;
            }
        }
        float actual;
        const bool hit = bvh.intersect(ray, actual);
        require(hit == expectedHit,
                "BVH hit differs from brute-force triangles");
        require(!hit || std::abs(actual - expected) < 1e-5f,
                "BVH nearest distance differs from brute force");
    }

    auto invalid = std::make_shared<Scene::MeshData>();
    invalid->vertices = {{0, 0, 0}};
    invalid->indices = {0, 1, 2};
    bool rejected = false;
    try {
        float distance;
        Geometry::MeshBVH(invalid).intersect(Geometry::Ray{}, distance);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    require(rejected, "invalid vertex indices were accepted");
}
} // namespace

int main() {
    checkBVH();
    Window window;
    window.init(64, 64, true);
    Backend::OpenGLDevice device;
    device.initialize();
    Scene::NativeScene scene;
    ColorMaterial material;
    Rasterizer raster(&device);
    raster.setShadowDistance(0);
    const auto view =
        glm::lookAt(glm::vec3(0, 0, 10), glm::vec3(0), glm::vec3(0, 1, 0));
    const auto projection = glm::ortho(-4.0f, 4.0f, -4.0f, 4.0f, 0.1f, 30.0f);
    auto sync = [&] { raster.updateFrameData(view, projection); };
    auto cube =
        std::make_shared<Scene::MeshData>(Scene::Prim::createCubeData(2));
    auto define = [&](const char* path) {
        auto* prim = scene.definePrim(path, Scene::PrimType::Mesh);
        prim->setMeshData(cube);
        return prim;
    };

    // Nearest non-pickable instance must not suppress another in its batch.
    auto* front = define("/front");
    auto* back = define("/back");
    front->setLocalTranslation({0, 0, 4});
    raster.addRenderable(&material, front);
    raster.addRenderable(&material, back);
    sync();
    const Geometry::Ray ray({0, 0, 10}, {0, 0, -1});
    require(raster.raycast(ray).prim == front, "nearest surface was not hit");
    front->getSelectionComponent()->setPickable(false);
    require(raster.raycast(ray).prim == back,
            "non-pickable front hid the back");
    front->setVisible(false);
    back->setVisible(false);
    sync();
    require(!raster.raycast(ray).hit, "hidden batch remained queryable");

    // World distance survives rotated, mirrored, nonuniform transforms.
    auto* transformed = define("/transformed");
    const auto transform =
        glm::translate(glm::mat4(1), glm::vec3(10, 3, 2)) *
        glm::rotate(glm::mat4(1), 0.7f, glm::normalize(glm::vec3(1, 2, 3))) *
        glm::scale(glm::mat4(1), glm::vec3(-2, 0.5f, 3));
    transformed->setLocalMatrix(transform);
    raster.addRenderable(&material, transformed);
    sync();
    const glm::vec3 origin = glm::vec3(transform * glm::vec4(0, 0, 4, 1));
    const glm::vec3 surface = glm::vec3(transform * glm::vec4(0, 0, 1, 1));
    auto hit = raster.raycast({origin, surface - origin});
    require(hit.prim == transformed, "off-camera transformed mesh was missed");
    require(std::abs(hit.distance - 9.0f) < 1e-4f &&
                glm::length(hit.position - surface) < 1e-4f,
            "world distance or position is incorrect under nonuniform scale");
    require(raster.rayPick({origin, surface - origin}).prim == transformed,
            "legacy alias changed query behavior");
    transformed->setLocalMatrix(glm::mat4(1));
    sync();
    hit = raster.raycast({{0, 0, 0}, {1, 0, 0}});
    require(hit.prim == transformed && std::abs(hit.distance - 1) < 1e-5f,
            "ray starting inside bounds missed the exit surface");
    require(!raster.raycast({{0, 0, 2}, {0, 0, 1}}).hit,
            "triangle behind the ray was hit");
    transformed->setLocalMatrix(glm::scale(glm::mat4(1), glm::vec3(0)));
    sync();
    require(!raster.raycast(ray).hit, "singular transform was queried");
    transformed->setVisible(false);

    auto* external = define("/external");
    const auto batch = raster.addRenderable(&material, external,
                                            TransformSource::ExternalBuffer);
    raster.updateRenderableTransforms(
        batch, {glm::translate(glm::mat4(1), glm::vec3(100, 0, 0)),
                glm::translate(glm::mat4(1), glm::vec3(0, 0, 2))});
    sync();
    hit = raster.raycast(ray);
    require(hit.hit && hit.handle == batch && hit.instanceIndex == 1 &&
                hit.prim == nullptr && std::abs(hit.distance - 7) < 1e-5f,
            "CPU external instance identity or surface distance is incorrect");
    external->getSelectionComponent()->setPickable(false);
    require(!raster.raycast(ray).hit,
            "non-pickable external owner was queried");
    external->setVisible(false);

    auto* deform = define("/deform");
    const auto deformHandle = raster.addRenderable(
        &material, deform, TransformSource::ExternalBuffer);
    auto positions = cube->vertices;
    for (auto& position : positions)
        position += glm::vec3(0, 0, 4);
    raster.updateRenderableTransforms(deformHandle, {glm::mat4(1)});
    raster.updateRenderableGeometry(deformHandle, positions, cube->normals);
    sync();
    require(!raster.raycast(ray).hit,
            "deformed mesh returned a source-pose hit");

    Scene::SkinnedMeshData skin;
    skin.mesh = *cube;
    skin.boneIndices.assign(cube->vertices.size(), {0, -1, -1, -1});
    skin.boneWeights.assign(cube->vertices.size(), {1, 0, 0, 0});
    auto* character = scene.definePrim("/character", Scene::PrimType::Mesh);
    character->setMeshData(std::make_shared<Scene::MeshData>(skin.mesh));
    raster.addSkinnedRenderable(&material, character, skin);
    sync();
    require(!raster.raycast(ray).hit, "skinned mesh returned a bind-pose hit");
    require(cube->vertices == Scene::Prim::createCubeData(2).vertices,
            "query or deformation changed the shared source mesh");
    std::cout << "PASS: BVH triangles, world rays, instances and exclusions\n";
}
