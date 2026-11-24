// Connects scene updates and Python render adapters to the C++ renderer.
#include "engine/scene/render/render_scene_data_collector.hpp"
#include "engine/scene/scene_resource_manager.hpp"
#include "py_array_view.hpp"
#include "py_render_backend.hpp"
#include "engine/scene/types/camera.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <cmath>
#include <climits>

namespace py = pybind11;
using namespace KE::Scene;

namespace {
// Immutable vertex influences shared by independently posed batches.
struct SkinSource {
    std::vector<glm::ivec4> indices;
    std::vector<glm::vec4> weights;
    size_t boneCount = 0;
};

py::array
matrixArray(const std::shared_ptr<const std::vector<glm::mat4>>& values) {
    auto* source = new std::shared_ptr<const std::vector<glm::mat4>>(values);
    py::capsule owner(source, [](void* value) {
        delete static_cast<decltype(source)>(value);
    });
    py::array result(
        py::dtype::of<float>(),
        {py::ssize_t(values->size()), py::ssize_t(4), py::ssize_t(4)},
        {py::ssize_t(sizeof(glm::mat4)), py::ssize_t(sizeof(float)),
         py::ssize_t(sizeof(glm::vec4))},
        values->data(), owner);
    result.attr("setflags")(false);
    return result;
}

py::array geometryArray(const std::shared_ptr<RenderGeometryData>& owner,
                        const std::vector<glm::vec3>& values) {
    py::array result(
        py::dtype::of<float>(), {py::ssize_t(values.size()), py::ssize_t(3)},
        {py::ssize_t(sizeof(glm::vec3)), py::ssize_t(sizeof(float))},
        values.data(), py::cast(owner));
    result.attr("setflags")(false);
    return result;
}

void addMaterialSource(py::dict& item, const RenderMaterialData& value,
                       ResourceHandle materialResource,
                       const RenderTextureSources& sources) {
    py::dict material, textures;
    const char* models[] = {"color", "phong", "pbr", "unsupported"};
    material["model"] = models[static_cast<int>(value.shadingModel)];
    material["style"] = static_cast<int>(value.colorStyle);
    material["specular"] = vec3Values(value.specular);
    material["shininess"] = value.shininess;
    material["unlit"] = value.unlit;
    material["has_native_textures"] = value.hasNativeTextures;
    material["base_color"] = vec4Values(value.baseColor);
    material["metallic"] = value.metallic;
    material["roughness"] = value.roughness;
    material["emissive"] =
        vec3Values(value.emissiveColor * value.emissiveStrength);
    for (const auto& [slot, resource] : sources) {
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
            py::array pixels(py::dtype::of<uint8_t>(),
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
    item["material_id"] = materialResource;
    item["textures"] = textures;
}

py::dict sceneFrameToDict(const RenderSceneFrame& frame) {
    py::dict result;
    py::list objects;
    for (const auto& object : frame.objects) {
        py::dict item;
        item["id"] = object.id;
        item["path"] = object.path;
        item["mesh_id"] = object.mesh.resource;
        item["mesh"] = std::const_pointer_cast<MeshData>(object.mesh.mesh);
        item["geometry"] =
            std::const_pointer_cast<RenderGeometryData>(object.geometry);
        item["world"] = floatArrayFromMat4(object.world);
        item["color"] = vec4Values(object.color);
        item["visible"] = object.visible;
        item["pickable"] = object.pickable;
        item["double_sided"] = object.doubleSided;
        item["alpha_mode"] = static_cast<int>(object.alphaMode);
        item["alpha_cutoff"] = object.alphaCutoff;
        addMaterialSource(item, object.material, object.materialResource,
                          object.textures);
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
        result.cudaImageOutput = _gpuOutput;
        result.cpuInstanceTransformsInput = true;
        result.skinning = true;
        result.texturePixelsInput = true;
        result.screenPicking = _picking;
        result.selectionOutline = _picking;
        return result;
    }
    void initialize(int width, int height) override {
        py::gil_scoped_acquire gil;
        _adapter = _factory(width, height);
        _picking = py::hasattr(_adapter, "pick_result");
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
        _pickRequest.reset();
        _pickResult.reset();
    }
    void syncScene(const RenderSceneFrame& scene) override { _scene = scene; }
    KE::RenderableHandle registerSceneBatch(
        const RenderMeshData& mesh, const RenderMaterialSource& material,
        const RenderInstanceData& owners, KE::TransformSource source,
        const SkinnedMeshData* skin) override {
        if ((source == KE::TransformSource::SceneGraph) != (skin != nullptr))
            throw std::runtime_error(
                "Python batches support static ExternalBuffer or skinned "
                "SceneGraph meshes");
        if (_nextBatch == KE::InvalidHandle)
            throw std::overflow_error("Render batch handles exhausted");
        const auto handle = _nextBatch++;
        Batch batch{mesh, &material, &owners};
        if (skin) {
            batch.skin = skinSource(mesh, *skin);
            batch.joints = std::make_shared<const std::vector<glm::mat4>>(
                batch.skin->boneCount, glm::mat4(1));
        }
        _batches.emplace(handle, std::move(batch));
        return handle;
    }
    void removeSceneBatch(KE::RenderableHandle handle) override {
        const auto mesh = _batches.at(handle).mesh.mesh.get();
        _batches.erase(handle);
        const auto [first, last] = _skins.equal_range(mesh);
        for (auto it = first; it != last;)
            if (it->second.expired())
                it = _skins.erase(it);
            else
                ++it;
    }
    void setSceneBatchState(KE::RenderableHandle handle,
                            const KE::SceneBatchState& state) override {
        if (!state.castsShadow)
            throw std::runtime_error(
                "Python render batches require shadow-casting meshes");
        _batches.at(handle).state = state;
    }
    void
    updateRenderableTransforms(KE::RenderableHandle handle,
                               const std::vector<glm::mat4>& transforms,
                               const std::vector<glm::vec4>* colors) override {
        auto& batch = _batches.at(handle);
        if (batch.skin)
            throw std::runtime_error(
                "Skinned SceneGraph transforms belong to the owner Prim");
        if (colors && !colors->empty())
            throw std::runtime_error(
                "Python render backend does not yet support instance colors");
        batch.transforms =
            std::make_shared<const std::vector<glm::mat4>>(transforms);
        batch.external.reset();
        ++batch.revision;
    }
    void
    setRenderableExternalBuffer(KE::RenderableHandle handle,
                                const KE::ExternalBufferDesc& desc) override {
        auto& batch = _batches.at(handle);
        const auto& view = desc.view;
        if (batch.skin || desc.format != KE::ExternalBufferFormat::Mat4 ||
            !view.isCuda() || view.dtype != KE::Sim::SimDType::Float32)
            throw std::invalid_argument(
                "Python external buffers require CUDA float32 Mat4 instances");
        const int64_t count =
            desc.count ? desc.count : (view.shape.empty() ? 0 : view.shape[0]);
        if (view.shape.size() != 3 || view.shape[1] != 4 ||
            view.shape[2] != 4 || count < 0 || count > view.shape[0] ||
            count > INT_MAX || (count && !view.data) || view.deviceId < 0 ||
            (!view.strides.empty() &&
             view.strides != std::vector<int64_t>{16, 4, 1}) ||
            (desc.strideBytes && desc.strideBytes != sizeof(glm::mat4)))
            throw std::invalid_argument("CUDA instances require contiguous "
                                        "column-major [N,4,4] matrices");
        if (desc.syncPolicy == KE::ExternalSyncPolicy::Fence ||
            (desc.syncPolicy == KE::ExternalSyncPolicy::Event &&
             !view.readyEventHandle))
            throw std::invalid_argument("CUDA instances require a valid event "
                                        "or stream synchronization policy");
        const bool unchanged =
            batch.external &&
            desc.syncPolicy == KE::ExternalSyncPolicy::Versioned &&
            batch.external->syncPolicy == desc.syncPolicy &&
            batch.external->view.data == view.data &&
            batch.external->view.deviceId == view.deviceId &&
            batch.external->view.version == view.version &&
            batch.external->count == count;
        batch.external = desc;
        batch.external->count = static_cast<int>(count);
        if (!batch.transforms->empty())
            batch.transforms = std::make_shared<const std::vector<glm::mat4>>();
        if (!unchanged)
            ++batch.revision;
    }
    void setRenderableColors(KE::RenderableHandle handle,
                             const std::vector<glm::vec4>& colors) override {
        _batches.at(handle);
        if (!colors.empty())
            throw std::runtime_error(
                "Python render backend does not yet support instance colors");
    }
    void updateRenderableSkinningMatrices(
        KE::RenderableHandle handle,
        const std::vector<glm::mat4>& matrices) override {
        auto& batch = _batches.at(handle);
        if (!batch.skin)
            throw std::runtime_error(
                "Skinning updates require a skinned batch");
        if (matrices.size() < batch.skin->boneCount ||
            matrices.size() > MaxSkinningBones)
            throw std::invalid_argument(
                "Skinning matrices must cover the mesh bone palette");
        batch.joints = std::make_shared<const std::vector<glm::mat4>>(
            matrices.begin(), matrices.begin() + batch.skin->boneCount);
        ++batch.skinRevision;
    }
    void
    updateRenderableGeometry(KE::RenderableHandle handle,
                             const std::vector<glm::vec3>& positions,
                             const std::vector<glm::vec3>& normals) override {
        auto& batch = _batches.at(handle);
        if (!batch.skin)
            throw std::runtime_error(
                "ExternalBuffer geometry updates are not yet supported");
        if (positions.size() != batch.mesh.mesh->vertices.size() ||
            (!normals.empty() && normals.size() != positions.size()))
            throw std::invalid_argument(
                "Bind geometry must match the source vertex count");
        auto geometry = std::make_shared<RenderGeometryData>();
        geometry->positions = positions;
        geometry->normals = !normals.empty()
                                ? normals
                                : (batch.geometry ? batch.geometry->normals
                                                  : batch.mesh.mesh->normals);
        batch.geometry = std::move(geometry);
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
    void requestPick(const KE::RenderPickRequest& request) override {
        _pickRequest = request;
        _pickResult.reset();
    }
    std::optional<KE::RenderPickResult> pollPick() override {
        auto result = _pickResult;
        _pickResult.reset();
        return result;
    }
    void renderScene(const glm::mat4&, const glm::mat4&, bool) override {
        py::gil_scoped_acquire gil;
        const auto update = _updates.prepare(_scene);
        auto frame = sceneFrameToDict(update);
        frame["batches"] = batchesToList();
        py::dict selection;
        selection["hit"] = _selection.hit;
        selection["id"] = _selection.sceneInstanceId;
        selection["handle"] = _selection.handle;
        selection["instance"] = _selection.instanceIndex;
        frame["selection"] = selection;
        frame["pick"] = py::none();
        if (_pickRequest) {
            py::dict pick;
            pick["ndc"] =
                py::make_tuple(_pickRequest->ndc.x, _pickRequest->ndc.y);
            frame["pick"] = pick;
        }
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
            acceptPick();
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
        acceptPick();
    }
    void endFrame(const KE::RenderPickHit& selection, bool) override {
        _selection = selection;
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
    void acceptPick() {
        if (!_pickRequest)
            return;
        const auto value = _adapter.attr("pick_result");
        KE::RenderPickResult result;
        result.requestId = _pickRequest->id;
        result.frame = _pickRequest->frame;
        if (!value.is_none()) {
            result.hit.hit = true;
            result.hit.handle = value["handle"].cast<KE::RenderableHandle>();
            result.hit.sceneInstanceId = value["id"].cast<uint64_t>();
            result.hit.instanceIndex = value["instance"].cast<int>();
            result.hit.transformSource =
                value["external"].cast<bool>()
                    ? KE::TransformSource::ExternalBuffer
                    : KE::TransformSource::SceneGraph;
            const auto position =
                value["position"].cast<std::array<float, 3>>();
            result.hit.position = {position[0], position[1], position[2]};
            result.hit.distance = glm::length(
                result.hit.position - glm::vec3(_scene.camera.world[3]));
        }
        _pickResult = result;
        _pickRequest.reset();
    }

    struct Batch {
        RenderMeshData mesh;
        const RenderMaterialSource* material;
        const RenderInstanceData* owners;
        KE::SceneBatchState state;
        std::shared_ptr<const std::vector<glm::mat4>> transforms =
            std::make_shared<const std::vector<glm::mat4>>();
        uint64_t revision = 0;
        std::shared_ptr<SkinSource> skin;
        std::shared_ptr<const std::vector<glm::mat4>> joints;
        std::shared_ptr<RenderGeometryData> geometry;
        uint64_t skinRevision = 0;
        std::optional<KE::ExternalBufferDesc> external;
    };

    std::shared_ptr<SkinSource> skinSource(const RenderMeshData& mesh,
                                           const SkinnedMeshData& skin) {
        if (skin.boneIndices.size() != mesh.mesh->vertices.size() ||
            skin.boneWeights.size() != mesh.mesh->vertices.size())
            throw std::invalid_argument(
                "Skin influences must match the source vertices");
        const auto [first, last] = _skins.equal_range(mesh.mesh.get());
        for (auto it = first; it != last; ++it) {
            auto source = it->second.lock();
            if (source && source->indices == skin.boneIndices &&
                source->weights == skin.boneWeights)
                return source;
        }
        auto source = std::make_shared<SkinSource>();
        source->indices = skin.boneIndices;
        source->weights = skin.boneWeights;
        source->boneCount = std::max(skin.boneNodeIndices.size(),
                                     skin.inverseBindMatrices.size());
        for (size_t vertex = 0; vertex < source->indices.size(); ++vertex) {
            for (int slot = 0; slot < 4; ++slot) {
                const int bone = source->indices[vertex][slot];
                if (bone < -1 || bone >= static_cast<int>(MaxSkinningBones))
                    throw std::invalid_argument(
                        "Skin bone index is outside the supported palette");
                if (!std::isfinite(source->weights[vertex][slot]) ||
                    source->weights[vertex][slot] < 0)
                    throw std::invalid_argument(
                        "Skin weights must be finite and non-negative");
                if (bone >= 0)
                    source->boneCount =
                        std::max(source->boneCount, size_t(bone + 1));
            }
        }
        if (source->boneCount == 0 || source->boneCount > MaxSkinningBones)
            throw std::invalid_argument("Skin requires a valid bone palette");
        _skins.emplace(mesh.mesh.get(), source);
        return source;
    }

    py::list batchesToList() const {
        py::list result;
        for (const auto& [handle, batch] : _batches) {
            py::dict item;
            item["id"] = handle;
            item["path"] = "/Batches/b" + std::to_string(handle);
            item["mesh_id"] = batch.mesh.resource;
            item["mesh"] = std::const_pointer_cast<MeshData>(batch.mesh.mesh);
            item["color"] = vec4Values(glm::vec4(1));
            item["visible"] = !batch.owners->ids.empty();
            item["pickable"] = std::any_of(
                batch.owners->pickable.begin(), batch.owners->pickable.end(),
                [](uint8_t value) { return value != 0; });
            item["instance_ids"] = batch.owners->ids;
            item["double_sided"] = batch.state.doubleSided;
            item["alpha_mode"] = static_cast<int>(batch.state.alphaMode);
            item["alpha_cutoff"] = batch.state.alphaCutoff;
            addMaterialSource(item, batch.material->values,
                              batch.material->resource,
                              batch.material->textures);

            item["transforms"] = matrixArray(batch.transforms);
            item["external"] =
                batch.external ? py::cast(*batch.external) : py::none();
            item["count"] = batch.external ? batch.external->count
                                           : batch.transforms->size();
            item["revision"] = batch.revision;
            item["skin"] = batch.skin;
            if (batch.skin) {
                item["joints"] = matrixArray(batch.joints);
                item["skin_revision"] = batch.skinRevision;
                item["geometry"] = batch.geometry;
                const bool visible = !batch.owners->ids.empty();
                item["world"] = floatArrayFromMat4(
                    visible ? batch.owners->transforms.front() : glm::mat4(1));
                item["color"] = vec4Values(
                    visible ? batch.owners->colors.front() : glm::vec4(1));
            }
            result.append(item);
        }
        return result;
    }

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
    bool _picking = false;
    std::optional<KE::RenderPickRequest> _pickRequest;
    std::optional<KE::RenderPickResult> _pickResult;
    KE::RenderPickHit _selection;
    KE::CudaRenderImage _cudaImage;
    RenderSceneFrame _scene;
    RenderSceneUpdateTracker _updates;
    std::map<KE::RenderableHandle, Batch> _batches;
    std::multimap<const MeshData*, std::weak_ptr<SkinSource>> _skins;
    KE::RenderableHandle _nextBatch = 0;
    KE::RenderImage _image, _pending;
    int _width = 0, _height = 0;
};
} // namespace

std::unique_ptr<KE::RenderBackend> makePythonRenderBackend(py::object factory,
                                                        bool gpuOutput) {
    return std::make_unique<PythonRenderBackend>(std::move(factory), gpuOutput);
}

void bind_scene_input(py::module& m) {
    py::class_<SkinSource, std::shared_ptr<SkinSource>>(m, "_RenderSkinSource")
        .def_readonly("bone_count", &SkinSource::boneCount)
        .def_property_readonly(
            "indices",
            [](std::shared_ptr<SkinSource> source) {
                py::array values(
                    py::dtype::of<int>(),
                    {py::ssize_t(source->indices.size()), py::ssize_t(4)},
                    {py::ssize_t(sizeof(glm::ivec4)), py::ssize_t(sizeof(int))},
                    source->indices.data(), py::cast(source));
                values.attr("setflags")(false);
                return values;
            })
        .def_property_readonly(
            "weights", [](std::shared_ptr<SkinSource> source) {
                py::array values(
                    py::dtype::of<float>(),
                    {py::ssize_t(source->weights.size()), py::ssize_t(4)},
                    {py::ssize_t(sizeof(glm::vec4)),
                     py::ssize_t(sizeof(float))},
                    source->weights.data(), py::cast(source));
                values.attr("setflags")(false);
                return values;
            });
    py::class_<RenderGeometryData, std::shared_ptr<RenderGeometryData>>(
        m, "_RenderGeometryData")
        .def_property_readonly("positions",
                               [](std::shared_ptr<RenderGeometryData> data) {
                                   return geometryArray(data, data->positions);
                               })
        .def_property_readonly("normals",
                               [](std::shared_ptr<RenderGeometryData> data) {
                                   return geometryArray(data, data->normals);
                               });
    py::class_<RenderSceneFrame>(m, "_RenderSceneFrame");
    py::class_<PythonSceneCollector>(m, "_RenderSceneDataCollector")
        .def(py::init<>())
        .def("collect", &PythonSceneCollector::collect)
        .def("accept", &PythonSceneCollector::accept);
}
