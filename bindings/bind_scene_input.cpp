// Exposes evaluated scene data to the Python render adapters.
#include "engine/scene/render_scene_data_collector.hpp"
#include "py_array_view.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace KE::Scene;

void bind_scene_input(py::module& m) {
    py::class_<RenderSceneDataCollector>(m, "_RenderSceneDataCollector")
        .def(py::init<>())
        .def("collect", [](RenderSceneDataCollector& collector,
                           SceneBackend& scene,
                           const SceneResourceManager& resources,
                           const std::string& cameraPath) {
            const auto frame = collector.collect(scene, resources, cameraPath);
            py::dict result;
            py::list objects;
            for (const auto& object : frame.objects) {
                py::dict item, material, textures;
                item["id"] = object.id;
                item["path"] = object.path;
                item["mesh_id"] = object.mesh.resource;
                item["mesh"] =
                    std::const_pointer_cast<MeshData>(object.mesh.mesh);
                item["world"] = floatArrayFromMat4(object.world);
                item["color"] = vec4Values(object.color);
                item["visible"] = object.visible;
                item["double_sided"] = object.doubleSided;
                const auto& value = object.material;
                material["model"] =
                    value.shadingModel == KE::MaterialShadingModel::PBR
                        ? "pbr"
                        : "unsupported";
                material["unlit"] = value.unlit;
                material["has_native_textures"] = value.hasNativeTextures;
                material["base_color"] = vec4Values(value.baseColor);
                material["metallic"] = value.metallic;
                material["roughness"] = value.roughness;
                material["emissive"] =
                    vec3Values(value.emissiveColor * value.emissiveStrength);
                for (const auto& [slot, resource] : object.textures) {
                    const auto& texture = *resource.second;
                    py::dict entry;
                    entry["id"] = resource.first;
                    entry["path"] = texture.path;
                    entry["srgb"] = texture.srgb;
                    entry["flip"] = texture.flip;
                    entry["sampler"] = texture.sampler;
                    textures[py::str(slot)] = entry;
                }
                item["material"] = material;
                item["material_id"] = object.materialResource;
                item["textures"] = textures;
                objects.append(item);
            }
            result["objects"] = objects;
            py::dict camera;
            camera["world"] = floatArrayFromMat4(frame.camera.world);
            camera["vertical_fov"] = frame.camera.verticalFov;
            camera["near"] = frame.camera.nearPlane;
            camera["far"] = frame.camera.farPlane;
            result["camera"] = camera;
            py::list lights;
            for (const auto& light : frame.lights) {
                py::dict item;
                item["direction"] = vec3Values(light.direction);
                item["color"] = vec3Values(light.color);
                item["intensity"] = light.intensity;
                item["ambient"] = vec3Values(light.ambient);
                lights.append(item);
            }
            result["lights"] = lights;
            return result;
        });
}
