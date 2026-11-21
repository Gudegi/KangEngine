// Connects scene updates and Python render adapters to the C++ renderer.
#include "engine/scene/render_scene_data_collector.hpp"
#include "py_array_view.hpp"
#include "py_render_backend.hpp"
#include "engine/scene/types/camera.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <cmath>

namespace py = pybind11;
using namespace KE::Scene;

namespace {
py::dict sceneFrameToDict(const RenderSceneFrame& frame) {
    py::dict result;
    py::list objects;
    for (const auto& object : frame.objects) {
        py::dict item, material, textures;
        item["id"] = object.id;
        item["path"] = object.path;
        item["mesh_id"] = object.mesh.resource;
        item["mesh"] = std::const_pointer_cast<MeshData>(object.mesh.mesh);
        item["world"] = floatArrayFromMat4(object.world);
        item["color"] = vec4Values(object.color);
        item["visible"] = object.visible;
        item["pickable"] = object.pickable;
        item["double_sided"] = object.doubleSided;
        const auto& value = object.material;
        material["model"] = value.shadingModel == KE::MaterialShadingModel::PBR
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
            entry["pixels"] = py::none();
            if (!texture.pixels.empty()) {
                // The view keeps the immutable source alive beyond this frame.
                auto* source = new std::shared_ptr<const TextureSourceResource>(
                    resource.second);
                py::capsule owner(source, [](void* value) {
                    delete static_cast<
                        std::shared_ptr<const TextureSourceResource>*>(value);
                });
                py::array pixels(
                    py::dtype::of<uint8_t>(),
                    {texture.height, texture.width, texture.channels},
                    {py::ssize_t(texture.width) * texture.channels,
                     py::ssize_t(texture.channels), py::ssize_t(1)},
                    texture.pixels.data(), owner);
                pixels.attr("setflags")(false);
                entry["pixels"] = std::move(pixels);
            }
            textures[py::str(slot)] = entry;
        }
        item["material"] = material;
        item["material_id"] = object.materialResource;
        item["textures"] = textures;
        objects.append(item);
    }
    result["objects"] = objects;
    result["incremental"] = frame.incremental;
    result["removed_objects"] = frame.removedObjects;
    py::dict camera;
    camera["world"] = floatArrayFromMat4(frame.camera.world);
    camera["vertical_fov"] = frame.camera.verticalFov;
    camera["near"] = frame.camera.nearPlane;
    camera["far"] = frame.camera.farPlane;
    result["camera"] = camera;
    py::list lights;
    const char* types[] = {"directional", "point", "spot", "rect", "dome"};
    for (const auto& source : frame.lighting.lights) {
        py::dict item;
        item["path"] = source.path;
        item["type"] = types[source.light.index()];
        item["light"] = std::visit(
            [](const auto& light) {
                return py::cast(light, py::return_value_policy::copy);
            },
            source.light);
        item["world"] = floatArrayFromMat4(source.world);
        lights.append(item);
    }
    result["lights"] = frame.lightingChanged ? py::object(lights) : py::none();
    return result;
}

class PythonSceneCollector {
  public:
    py::dict collect(SceneBackend& scene, const SceneResourceManager& resources,
                     const std::string& cameraPath) {
        _pending =
            _updates.prepare(_collector.collect(scene, resources, cameraPath));
        auto result = sceneFrameToDict(_pending);
        // Native resolves paths on its own copy; the baseline retains source
        // identity.
        result["_snapshot"] = py::cast(RenderSceneFrame(_pending));
        return result;
    }
    void accept() { _updates.accept(_pending); }

  private:
    RenderSceneDataCollector _collector;
    RenderSceneUpdateTracker _updates;
    RenderSceneFrame _pending;
};

class PythonRenderBackend final : public KE::RenderBackend {
  public:
    explicit PythonRenderBackend(py::object factory, bool gpuOutput)
        : _factory(std::move(factory)), _gpuOutput(gpuOutput) {}
    ~PythonRenderBackend() override {
        py::gil_scoped_acquire gil;
        if (_adapter) {
            try {
                _adapter.attr("close")();
            } catch (py::error_already_set& error) {
                error.discard_as_unraisable("render backend close");
            }
        }
        _adapter = py::object();
        _factory = py::object();
    }
    KE::RenderBackendCapabilities capabilities() const override {
        KE::RenderBackendCapabilities result;
        result.staticScene = true;
        result.cudaImages = _gpuOutput;
        return result;
    }
    void initialize(int width, int height) override {
        py::gil_scoped_acquire gil;
        _adapter = _factory(width, height);
        _width = width;
        _height = height;
    }
    void resize(int width, int height) override {
        py::gil_scoped_acquire gil;
        _adapter.attr("resize")(width, height);
        _width = width;
        _height = height;
        _image = {};
        _cudaImage = {};
    }
    void syncScene(const RenderSceneFrame& scene) override { _scene = scene; }
    KE::RenderableHandle registerSceneBatch(const RenderMeshData&,
                                            const RenderMaterialSource&,
                                            const RenderInstanceData&,
                                            KE::TransformSource,
                                            const SkinnedMeshData*) override {
        throw std::runtime_error(
            "Python render backend requires scene snapshots");
    }
    void setSceneBatchState(KE::RenderableHandle,
                            const KE::SceneBatchState&) override {
        throw std::runtime_error(
            "Python render backend requires scene snapshots");
    }
    void syncFrame(const glm::mat4& view,
                   const glm::mat4& projection) override {
        if (projection[3][3] != 0 || projection[2][3] != -1 ||
            projection[1][1] <= 0)
            throw std::invalid_argument(
                "Python render backend requires a perspective camera");
        _scene.camera.world = glm::inverse(view);
    }
    void prepareFrame(KE::Camera& camera, KE::UpAxis) override {
        _scene.camera.verticalFov = camera.getFoV();
        _scene.camera.nearPlane = camera.getNearPlane();
        _scene.camera.farPlane = camera.getFarPlane();
    }
    void beginFrame() override {
        _pending = {};
        _cudaImage = {};
    }
    void renderScene(const glm::mat4&, const glm::mat4&, bool) override {
        py::gil_scoped_acquire gil;
        const auto update = _updates.prepare(_scene);
        auto frame = sceneFrameToDict(update);
        if (_gpuOutput) {
            // Pass ovrtx image handles to App; the Python adapter owns the mapping.
            _image = {};
            auto output = _adapter.attr("render_gpu")(frame);
            _cudaImage = {_width, _height, output["device"].cast<int>(),
                          output["array"].cast<uintptr_t>(),
                          output["ready_event"].cast<uintptr_t>()};
            if (!_cudaImage.array)
                throw std::runtime_error("Render adapter returned no CUDA image");
            _updates.accept(update);
            return;
        }
        auto output = _adapter.attr("render")(frame);
        auto pixels = py::array_t<uint8_t, py::array::c_style>::ensure(output);
        if (!pixels || pixels.ndim() != 3 || pixels.shape(0) != _height ||
            pixels.shape(1) != _width || pixels.shape(2) != 3)
            throw std::runtime_error("Render adapter must return contiguous "
                                     "RGB8 [height, width, 3]");
        _pending = {
            _width, _height,
            std::vector<uint8_t>(pixels.data(), pixels.data() + pixels.size())};
        _updates.accept(update);
    }
    void endFrame(const KE::RenderPickHit&, bool) override {
        _image = std::move(_pending);
    }
    KE::RenderImage readImage(bool flipY) override {
        loadGpuReadback();
        if (_image.pixels.empty())
            throw std::runtime_error("No completed scene frame is available");
        if (flipY)
            return _image;
        KE::RenderImage image{_image.width, _image.height,
                              std::vector<uint8_t>(_image.pixels.size())};
        const auto rowSize = static_cast<size_t>(image.width) * 3;
        for (int row = 0; row < image.height; ++row)
            std::copy_n(_image.pixels.data() + row * rowSize, rowSize,
                        image.pixels.data() +
                            (image.height - 1 - row) * rowSize);
        return image;
    }
    KE::RenderImage readImageResized(int width, int height,
                                     bool flipY) override {
        loadGpuReadback();
        if (width == _image.width && height == _image.height)
            return readImage(flipY);
        if (_image.pixels.empty())
            throw std::runtime_error("No completed scene frame is available");
        const auto& source = _image;
        KE::RenderImage image{
            width, height,
            std::vector<uint8_t>(static_cast<size_t>(width) * height * 3)};
        for (int y = 0; y < height; ++y) {
            const float sy =
                std::max(0.0f, (y + 0.5f) * source.height / height - 0.5f);
            const int y0 = static_cast<int>(sy),
                      y1 = std::min(y0 + 1, source.height - 1);
            for (int x = 0; x < width; ++x) {
                const float sx =
                    std::max(0.0f, (x + 0.5f) * source.width / width - 0.5f);
                const int x0 = static_cast<int>(sx),
                          x1 = std::min(x0 + 1, source.width - 1);
                for (int c = 0; c < 3; ++c) {
                    auto sample = [&](int px, int py) {
                        const int row = flipY ? py : source.height - 1 - py;
                        return float(source.pixels[(static_cast<size_t>(row) *
                                                        source.width +
                                                    px) *
                                                       3 +
                                                   c]);
                    };
                    const float top =
                        glm::mix(sample(x0, y0), sample(x1, y0), sx - x0);
                    const float bottom =
                        glm::mix(sample(x0, y1), sample(x1, y1), sx - x0);
                    image.pixels[(static_cast<size_t>(y) * width + x) * 3 + c] =
                        static_cast<uint8_t>(
                            std::round(glm::mix(top, bottom, sy - y0)));
                }
            }
        }
        return image;
    }
    const KE::CudaRenderImage* cudaImage() const override {
        return _cudaImage.array ? &_cudaImage : nullptr;
    }

  private:
    // Download ovrtx output only for explicit CPU reads, once per frame.
    void loadGpuReadback() {
        if (!_gpuOutput || !_image.pixels.empty() || !_cudaImage.array)
            return;
        py::gil_scoped_acquire gil;
        auto pixels = py::array_t<uint8_t, py::array::c_style>::ensure(
            _adapter.attr("read_image")());
        if (!pixels || pixels.ndim() != 3 || pixels.shape(0) != _height ||
            pixels.shape(1) != _width || pixels.shape(2) != 3)
            throw std::runtime_error("Render adapter must return RGB8 readback");
        _image = {_width, _height,
                  std::vector<uint8_t>(pixels.data(), pixels.data() + pixels.size())};
        _cudaImage = {};
    }
    py::object _factory, _adapter;
    bool _gpuOutput = false;
    KE::CudaRenderImage _cudaImage;
    RenderSceneFrame _scene;
    RenderSceneUpdateTracker _updates;
    KE::RenderImage _image, _pending;
    int _width = 0, _height = 0;
};
} // namespace

std::unique_ptr<KE::RenderBackend> makePythonRenderBackend(py::object factory,
                                                        bool gpuOutput) {
    return std::make_unique<PythonRenderBackend>(std::move(factory), gpuOutput);
}

void bind_scene_input(py::module& m) {
    py::class_<RenderSceneFrame>(m, "_RenderSceneFrame");
    py::class_<PythonSceneCollector>(m, "_RenderSceneDataCollector")
        .def(py::init<>())
        .def("collect", &PythonSceneCollector::collect)
        .def("accept", &PythonSceneCollector::accept);
}
