///
/// Rendering and graphics-backend Python bindings
///

#include "../src/kangEngine.hpp"
#include "py_array_view.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_render(py::module& m) {
    using namespace KE;

    py::enum_<Backend::BackendType>(m, "BackendType")
        .value("OPENGL", Backend::BackendType::OpenGL)
        .value("VULKAN", Backend::BackendType::Vulkan)
        .value("WEBGPU", Backend::BackendType::WebGPU);

    py::enum_<Backend::ShaderType>(m, "ShaderType")
        .value("VERTEX", Backend::ShaderType::Vertex)
        .value("FRAGMENT", Backend::ShaderType::Fragment)
        .value("GEOMETRY", Backend::ShaderType::Geometry)
        .value("COMPUTE", Backend::ShaderType::Compute);

    py::enum_<TransformSource>(m, "TransformSource")
        .value("SCENE_GRAPH", TransformSource::SceneGraph)
        .value("EXTERNAL_BUFFER", TransformSource::ExternalBuffer);

    py::enum_<ExternalBufferFormat>(m, "ExternalBufferFormat")
        .value("MAT4", ExternalBufferFormat::Mat4)
        .value("POSITION_ROTATION", ExternalBufferFormat::PositionRotation)
        .value("POSITION_ROTATION_SCALE",
               ExternalBufferFormat::PositionRotationScale)
        .value("CUSTOM", ExternalBufferFormat::Custom);

    py::enum_<ExternalSyncPolicy>(m, "ExternalSyncPolicy")
        .value("NONE", ExternalSyncPolicy::None)
        .value("VERSIONED", ExternalSyncPolicy::Versioned)
        .value("FENCE", ExternalSyncPolicy::Fence)
        .value("EVENT", ExternalSyncPolicy::Event);

    py::class_<ExternalBufferDesc>(
        m, "ExternalBufferDesc",
        "Descriptor for renderer buffers whose storage is owned externally.")
        .def(py::init([](const Sim::GpuArrayView& view,
                         ExternalBufferFormat format, int count,
                         int stride_bytes, ExternalSyncPolicy sync_policy) {
                 ExternalBufferDesc config;
                 config.view = view;
                 config.format = format;
                 config.count = count;
                 config.strideBytes = stride_bytes;
                 config.syncPolicy = sync_policy;
                 return config;
             }),
             py::kw_only(),
             py::arg_v("view", Sim::GpuArrayView{}, "GpuArrayView()"),
             py::arg_v("format", ExternalBufferDesc{}.format,
                       "ExternalBufferFormat.MAT4"),
             py::arg("count") = ExternalBufferDesc{}.count,
             py::arg("stride_bytes") = ExternalBufferDesc{}.strideBytes,
             py::arg_v("sync_policy", ExternalBufferDesc{}.syncPolicy,
                       "ExternalSyncPolicy.NONE"),
             "Create an external buffer descriptor from keyword arguments.")
        .def_readwrite("view", &ExternalBufferDesc::view,
                       "External array metadata and ownership view.")
        .def_readwrite("format", &ExternalBufferDesc::format,
                       "Element layout used to decode the external buffer.")
        .def_readwrite("count", &ExternalBufferDesc::count,
                       "Element count; zero derives it from the view shape.")
        .def_readwrite(
            "stride_bytes", &ExternalBufferDesc::strideBytes,
            "Element stride in bytes; zero derives it from the view.")
        .def_readwrite("sync_policy", &ExternalBufferDesc::syncPolicy,
                       "Synchronization policy for external producer updates.")
        .def("__repr__", [](const ExternalBufferDesc& config) {
            return py::str(
                       "ExternalBufferDesc(view={!r}, format={}, count={!r}, "
                       "stride_bytes={!r}, sync_policy={})")
                .attr("format")(config.view, config.format, config.count,
                                config.strideBytes, config.syncPolicy);
        });

    py::class_<Backend::Texture>(
        m, "Texture", "GPU texture object created by a GraphicsDevice.")
        .def("bind", &Backend::Texture::bind, py::arg("slot") = 0,
             "Bind this texture to a texture unit.")
        .def("unbind", &Backend::Texture::unbind,
             "Unbind this texture from the active context.")
        .def("get_width", &Backend::Texture::getWidth,
             "Return the texture width in pixels.")
        .def("get_height", &Backend::Texture::getHeight,
             "Return the texture height in pixels.")
        .def_property_readonly("width", &Backend::Texture::getWidth,
                               "Texture width in pixels.")
        .def_property_readonly("height", &Backend::Texture::getHeight,
                               "Texture height in pixels.");

    py::enum_<Backend::TextureWrap>(
        m, "TextureWrap", "Backend-neutral texture coordinate wrapping mode.")
        .value("REPEAT", Backend::TextureWrap::Repeat)
        .value("CLAMP_TO_EDGE", Backend::TextureWrap::ClampToEdge)
        .value("MIRRORED_REPEAT", Backend::TextureWrap::MirroredRepeat);

    py::enum_<Backend::TextureFilter>(
        m, "TextureFilter", "Backend-neutral texture sampling filter.")
        .value("NEAREST", Backend::TextureFilter::Nearest)
        .value("LINEAR", Backend::TextureFilter::Linear)
        .value("LINEAR_MIPMAP_LINEAR",
               Backend::TextureFilter::LinearMipmapLinear);

    py::class_<Backend::SamplerDesc>(
        m, "SamplerDesc", "Texture sampler settings independent of GL/WebGPU.")
        .def(py::init([](Backend::TextureWrap wrap_u,
                         Backend::TextureWrap wrap_v,
                         Backend::TextureFilter min_filter,
                         Backend::TextureFilter mag_filter) {
                 Backend::SamplerDesc config;
                 config.wrapU = wrap_u;
                 config.wrapV = wrap_v;
                 config.minFilter = min_filter;
                 config.magFilter = mag_filter;
                 return config;
             }),
             py::kw_only(),
             py::arg_v("wrap_u", Backend::SamplerDesc{}.wrapU,
                       "TextureWrap.Repeat"),
             py::arg_v("wrap_v", Backend::SamplerDesc{}.wrapV,
                       "TextureWrap.Repeat"),
             py::arg_v("min_filter", Backend::SamplerDesc{}.minFilter,
                       "TextureFilter.LinearMipmapLinear"),
             py::arg_v("mag_filter", Backend::SamplerDesc{}.magFilter,
                       "TextureFilter.Linear"),
             "Create texture sampler settings from keyword arguments.")
        .def_readwrite("wrap_u", &Backend::SamplerDesc::wrapU,
                       "Texture coordinate wrapping mode on the U axis.")
        .def_readwrite("wrap_v", &Backend::SamplerDesc::wrapV,
                       "Texture coordinate wrapping mode on the V axis.")
        .def_readwrite("min_filter", &Backend::SamplerDesc::minFilter,
                       "Texture minification filter.")
        .def_readwrite("mag_filter", &Backend::SamplerDesc::magFilter,
                       "Texture magnification filter.")
        .def("__repr__", [](const Backend::SamplerDesc& config) {
            return py::str("SamplerDesc(wrap_u={}, wrap_v={}, "
                           "min_filter={}, mag_filter={})")
                .attr("format")(config.wrapU, config.wrapV, config.minFilter,
                                config.magFilter);
        });

    py::enum_<Backend::BufferUsage>(m, "BufferUsage", py::arithmetic())
        .value("NONE", Backend::BufferUsage::None)
        .value("VERTEX", Backend::BufferUsage::Vertex)
        .value("INDEX", Backend::BufferUsage::Index)
        .value("UNIFORM", Backend::BufferUsage::Uniform)
        .value("COPY_SRC", Backend::BufferUsage::CopySrc)
        .value("COPY_DST", Backend::BufferUsage::CopyDst)
        .def("__or__",
             [](Backend::BufferUsage lhs, Backend::BufferUsage rhs) {
                 return lhs | rhs;
             },
             py::is_operator());
    py::enum_<Backend::VertexFormat>(m, "VertexFormat")
        .value("FLOAT32", Backend::VertexFormat::Float32)
        .value("FLOAT32_X2", Backend::VertexFormat::Float32x2)
        .value("FLOAT32_X3", Backend::VertexFormat::Float32x3)
        .value("FLOAT32_X4", Backend::VertexFormat::Float32x4)
        .value("SINT32_X4", Backend::VertexFormat::Sint32x4);
    py::enum_<Backend::VertexStepMode>(m, "VertexStepMode")
        .value("VERTEX", Backend::VertexStepMode::Vertex)
        .value("INSTANCE", Backend::VertexStepMode::Instance);
    py::enum_<Backend::PrimitiveTopology>(m, "PrimitiveTopology")
        .value("POINT_LIST", Backend::PrimitiveTopology::PointList)
        .value("LINE_LIST", Backend::PrimitiveTopology::LineList)
        .value("LINE_STRIP", Backend::PrimitiveTopology::LineStrip)
        .value("TRIANGLE_LIST", Backend::PrimitiveTopology::TriangleList)
        .value("TRIANGLE_STRIP", Backend::PrimitiveTopology::TriangleStrip);
    py::enum_<Backend::CullMode>(m, "CullMode")
        .value("NONE", Backend::CullMode::None)
        .value("FRONT", Backend::CullMode::Front)
        .value("BACK", Backend::CullMode::Back);
    py::enum_<Backend::CompareFunction>(m, "CompareFunction")
        .value("NEVER", Backend::CompareFunction::Never)
        .value("LESS", Backend::CompareFunction::Less)
        .value("LESS_EQUAL", Backend::CompareFunction::LessEqual)
        .value("GREATER", Backend::CompareFunction::Greater)
        .value("GREATER_EQUAL", Backend::CompareFunction::GreaterEqual)
        .value("EQUAL", Backend::CompareFunction::Equal)
        .value("NOT_EQUAL", Backend::CompareFunction::NotEqual)
        .value("ALWAYS", Backend::CompareFunction::Always);
    py::enum_<Backend::IndexFormat>(m, "IndexFormat")
        .value("UINT16", Backend::IndexFormat::Uint16)
        .value("UINT32", Backend::IndexFormat::Uint32);
    py::enum_<SceneHookBlendMode>(m, "SceneHookBlendMode")
        .value("OPAQUE", SceneHookBlendMode::Opaque)
        .value("ALPHA", SceneHookBlendMode::Alpha)
        .value("ADDITIVE", SceneHookBlendMode::Additive);
    py::enum_<RenderHookPhase>(m, "RenderHookPhase")
        .value("AFTER_OPAQUE", RenderHookPhase::AfterOpaque)
        .value("AFTER_TRANSPARENT", RenderHookPhase::AfterTransparent);

    py::enum_<ToneMapMode>(m, "ToneMapMode")
        .value("OFF", ToneMapMode::None)
        .value("REINHARD", ToneMapMode::Reinhard)
        .value("EXPONENTIAL", ToneMapMode::Exponential)
        .value("ACES_NARKOWICZ", ToneMapMode::AcesNarkowicz)
        .value("ACES_FITTED", ToneMapMode::AcesFitted);

    py::enum_<TextureRole>(
        m, "TextureRole",
        "Renderer texture binding roles used by material shaders.")
        .value("BASE_COLOR", TextureRole::BaseColor)
        .value("DIFFUSE", TextureRole::Diffuse)
        .value("NORMAL", TextureRole::Normal)
        .value("METALLIC_ROUGHNESS", TextureRole::MetallicRoughness)
        .value("AMBIENT_OCCLUSION", TextureRole::AmbientOcclusion)
        .value("EMISSIVE", TextureRole::Emissive)
        .value("METALLIC", TextureRole::Metallic)
        .value("ROUGHNESS", TextureRole::Roughness)
        .value("OCCLUSION_ROUGHNESS_METALLIC",
               TextureRole::OcclusionRoughnessMetallic);

    py::enum_<AlphaMode>(
        m, "AlphaMode",
        "How a renderable handles fragment alpha in scene and depth passes.")
        .value("OPAQUE", AlphaMode::Opaque)
        .value("MASK", AlphaMode::Mask)
        .value("BLEND", AlphaMode::Blend);

    py::enum_<TextAlignment>(m, "TextAlignment")
        .value("LEFT", TextAlignment::Left)
        .value("CENTER", TextAlignment::Center)
        .value("RIGHT", TextAlignment::Right);

    py::enum_<TextDepthMode>(m, "TextDepthMode")
        .value("DEPTH_TESTED", TextDepthMode::DepthTested)
        .value("OVERLAY", TextDepthMode::Overlay);

    py::enum_<ScreenAnchor>(m, "ScreenAnchor")
        .value("TOP_LEFT", ScreenAnchor::TopLeft)
        .value("TOP_CENTER", ScreenAnchor::TopCenter)
        .value("TOP_RIGHT", ScreenAnchor::TopRight)
        .value("CENTER_LEFT", ScreenAnchor::CenterLeft)
        .value("CENTER", ScreenAnchor::Center)
        .value("CENTER_RIGHT", ScreenAnchor::CenterRight)
        .value("BOTTOM_LEFT", ScreenAnchor::BottomLeft)
        .value("BOTTOM_CENTER", ScreenAnchor::BottomCenter)
        .value("BOTTOM_RIGHT", ScreenAnchor::BottomRight);

    py::class_<Backend::ShaderStage>(m, "ShaderStage")
        .def(py::init<>())
        .def(py::init<std::string, Backend::ShaderType, std::string>(),
             py::arg("source"), py::arg("stage"),
             py::arg("entry_point") = "main")
        .def_readwrite("source", &Backend::ShaderStage::source)
        .def_readwrite("stage", &Backend::ShaderStage::type)
        .def_readwrite("entry_point", &Backend::ShaderStage::entryPoint);
    py::class_<Backend::ShaderDesc>(m, "ShaderDesc")
        .def(py::init<>())
        .def_readwrite("stages", &Backend::ShaderDesc::stages)
        .def_readwrite("name", &Backend::ShaderDesc::name);
    py::class_<Backend::VertexAttributeDesc>(m, "VertexAttributeDesc")
        .def(py::init<>())
        .def(py::init<Backend::VertexFormat, uint64_t, uint32_t>(),
             py::arg("format"), py::arg("offset"),
             py::arg("shader_location"))
        .def_readwrite("format", &Backend::VertexAttributeDesc::format)
        .def_readwrite("offset", &Backend::VertexAttributeDesc::offset)
        .def_readwrite("shader_location",
                       &Backend::VertexAttributeDesc::shaderLocation);
    py::class_<Backend::VertexBufferLayout>(m, "VertexBufferLayout")
        .def(py::init<>())
        .def_readwrite("array_stride", &Backend::VertexBufferLayout::arrayStride)
        .def_readwrite("step_mode", &Backend::VertexBufferLayout::stepMode)
        .def_readwrite("attributes", &Backend::VertexBufferLayout::attributes);
    py::class_<SceneHookPipelineDesc>(m, "SceneHookPipelineDesc")
        .def(py::init<>())
        .def_readwrite("shader", &SceneHookPipelineDesc::shader)
        .def_readwrite("vertex_buffers", &SceneHookPipelineDesc::vertexBuffers)
        .def_readwrite("topology", &SceneHookPipelineDesc::topology)
        .def_readwrite("cull_mode", &SceneHookPipelineDesc::cullMode)
        .def_readwrite("blend", &SceneHookPipelineDesc::blend)
        .def_readwrite("depth_test", &SceneHookPipelineDesc::depthTest)
        .def_readwrite("depth_write", &SceneHookPipelineDesc::depthWrite)
        .def_readwrite("use_scene_frame_bindings",
                       &SceneHookPipelineDesc::useSceneFrameBindings)
        .def_readwrite("depth_compare", &SceneHookPipelineDesc::depthCompare)
        .def_readwrite("label", &SceneHookPipelineDesc::label);

    py::class_<Backend::Buffer>(m, "Buffer")
        .def_property_readonly("size", &Backend::Buffer::getSize)
        .def_property_readonly("usage", &Backend::Buffer::getUsage)
        .def("set_data",
             [](Backend::Buffer& buffer, py::buffer data, size_t offset) {
                 const py::buffer_info info = data.request();
                 const size_t size =
                     static_cast<size_t>(info.size) * info.itemsize;
                 if (offset + size > buffer.getSize())
                     throw py::value_error("buffer upload exceeds allocation");
                 buffer.setData(info.ptr, size, offset);
             },
             py::arg("data"), py::arg("offset") = 0);
    py::class_<Backend::GraphicsPipeline>(m, "GraphicsPipeline");
    py::class_<Backend::RenderPassEncoder>(m, "RenderPassEncoder")
        .def("set_viewport", &Backend::RenderPassEncoder::setViewport,
             py::arg("x"), py::arg("y"), py::arg("width"),
             py::arg("height"), py::arg("min_depth") = 0.0f,
             py::arg("max_depth") = 1.0f)
        .def("set_pipeline", &Backend::RenderPassEncoder::setPipeline,
             py::arg("pipeline"))
        .def("set_vertex_buffer", &Backend::RenderPassEncoder::setVertexBuffer,
             py::arg("slot"), py::arg("buffer"), py::arg("offset") = 0)
        .def("set_index_buffer", &Backend::RenderPassEncoder::setIndexBuffer,
             py::arg("buffer"), py::arg("format"), py::arg("offset") = 0)
        .def("draw", &Backend::RenderPassEncoder::draw,
             py::arg("vertex_count"), py::arg("instance_count") = 1,
             py::arg("first_vertex") = 0, py::arg("first_instance") = 0)
        .def("draw_indexed", &Backend::RenderPassEncoder::drawIndexed,
             py::arg("index_count"), py::arg("instance_count") = 1,
             py::arg("first_index") = 0, py::arg("base_vertex") = 0,
             py::arg("first_instance") = 0);
    py::class_<RenderHookContext>(m, "RenderHookContext")
        .def_property_readonly(
            "pass_encoder", [](RenderHookContext& context) {
                return &context.pass;
            }, py::return_value_policy::reference)
        .def_readonly("width", &RenderHookContext::width)
        .def_readonly("height", &RenderHookContext::height);

    // Backend::GraphicsDevice
    py::class_<Backend::GraphicsDevice,
               std::shared_ptr<Backend::GraphicsDevice>>(
        m, "GraphicsDevice",
        "Factory for backend graphics resources such as textures and buffers.")
        .def(
            "create_texture",
            [](Backend::GraphicsDevice& device, const std::string& path,
               bool flip, const Backend::SamplerDesc& sampler) {
                return device.createTexture(path, flip, sampler);
            },
            py::arg("path"), py::arg("flip") = false,
            py::arg("sampler") = Backend::SamplerDesc(),
            py::return_value_policy::take_ownership,
            "Load a texture from an image file.")
        .def(
            "create_buffer",
            [](Backend::GraphicsDevice& device, py::buffer data,
               Backend::BufferUsage usage, const std::string& label) {
                const py::buffer_info info = data.request();
                Backend::BufferDesc desc;
                desc.size = static_cast<size_t>(info.size) * info.itemsize;
                desc.usage = usage;
                desc.label = label;
                return device.createBuffer(desc, info.ptr);
            },
            py::arg("data"), py::arg("usage"), py::arg("label") = "",
            "Create a GPU buffer initialized from a contiguous Python buffer.");

    py::class_<Renderer>(
        m, "Renderer",
        "Renderer facade for updating renderable resources and draw settings.")
        .def(
            "device", [](Renderer& self) { return self.device(); },
            py::return_value_policy::reference,
            "Return the graphics device owned by this renderer.")
        .def("create_scene_hook_pipeline",
             &Renderer::createSceneHookPipeline, py::arg("desc"),
             py::return_value_policy::take_ownership,
             "Create a custom graphics pipeline compatible with the scene "
             "render targets.")
        .def(
            "add_render_hook",
            [](Renderer& self, RenderHookPhase phase, py::function callback) {
                return self.addRenderHook(
                    phase, [callback = std::move(callback)](
                               RenderHookContext& context) {
                        py::gil_scoped_acquire acquire;
                        callback(&context);
                    });
            },
            py::arg("phase"), py::arg("callback"),
            "Run a Python command-recording callback at a scene render phase. "
            "Keep all referenced pipelines and buffers alive until removal.")
        .def("remove_render_hook", &Renderer::removeRenderHook,
             py::arg("handle"), "Remove a previously registered render hook.")
        .def("set_point_lights", &Renderer::setPointLights, py::arg("lights"),
             "Store point lights for renderers/shaders that support them.")
        .def("point_lights", &Renderer::pointLights,
             py::return_value_policy::reference_internal,
             "Return stored point lights.")
        .def("set_spot_lights", &Renderer::setSpotLights, py::arg("lights"),
             "Store spot lights for renderers/shaders that support them.")
        .def("spot_lights", &Renderer::spotLights,
             py::return_value_policy::reference_internal,
             "Return stored spot lights.")
        .def("sync_scene_lights", &Renderer::syncSceneLights, py::arg("scene"),
             "Sync renderer lights from /lights scene prims. Intended for "
             "diagnostics and tests.")
        .def(
            "update_renderable_transforms",
            [](Renderer& self, uint32_t handle, const FloatArray& transforms,
               py::object colors) {
                auto transformVec = mat4Array(transforms, "transforms");
                std::vector<glm::vec4> colorVec;
                const std::vector<glm::vec4>* colorPtr = nullptr;
                if (!colors.is_none()) {
                    auto colorArray = colors.cast<FloatArray>();
                    colorVec = vec4Array(colorArray, "colors");
                    if (colorVec.size() != 1 &&
                        colorVec.size() != transformVec.size()) {
                        throw py::value_error(
                            "colors must have length 1 or match transforms");
                    }
                    if (colorVec.size() == 1 && transformVec.size() > 1)
                        colorVec.resize(transformVec.size(), colorVec[0]);
                    colorPtr = &colorVec;
                }
                self.updateRenderableTransforms(handle, transformVec, colorPtr);
            },
            py::arg("handle"), py::arg("transforms"),
            py::arg("colors") = py::none(),
            "Replace instance transforms, optionally with per-instance colors.")
        .def(
            "set_renderable_colors",
            [](Renderer& self, uint32_t handle, const FloatArray& colors) {
                self.setRenderableColors(handle, vec4Array(colors, "colors"));
            },
            py::arg("handle"), py::arg("colors"),
            "Set per-instance colors for a renderable.")
        .def("set_renderable_external_buffer",
             &Renderer::setRenderableExternalBuffer, py::arg("handle"),
             py::arg("descriptor"),
             "Attach an external CPU/GPU transform buffer to a renderable.")
        .def("map_renderable_cuda_transform_buffers",
             &Renderer::mapRenderableCudaTransformBuffers, py::arg("handles"),
             py::arg("count"), py::arg("device_id"),
             py::arg("stream_handle") = 0,
             "Map multiple renderable transform VBOs for direct CUDA writes.")
        .def("unmap_renderable_cuda_transform_buffers",
             &Renderer::unmapRenderableCudaTransformBuffers, py::arg("handles"),
             py::arg("device_id"), py::arg("stream_handle") = 0,
             "Unmap transform VBOs after direct CUDA writes.")
        .def("set_renderable_double_sided", &Renderer::setRenderableDoubleSided,
             py::arg("handle"), py::arg("enabled") = true,
             "Enable or disable double-sided rendering for a renderable.")
        .def("set_renderable_casts_shadow", &Renderer::setRenderableCastsShadow,
             py::arg("handle"), py::arg("enabled") = true,
             "Enable or disable shadow casting for a renderable.")
        .def("set_renderable_alpha_mode", &Renderer::setRenderableAlphaMode,
             py::arg("handle"), py::arg("mode"), py::arg("cutoff") = 0.5f,
             "Select opaque, alpha-mask, or alpha-blend rendering.")
        .def(
            "set_renderable_texture",
            [](Renderer& self, uint32_t handle, Backend::Texture* texture,
               TextureRole role) {
                self.setRenderableTexture(handle, texture, role);
            },
            py::arg("handle"), py::arg("texture"), py::arg("role"),
            "Attach a texture to a renderable using a material texture role.")
        .def(
            "set_renderable_texture",
            [](Renderer& self, uint32_t handle, Backend::Texture* texture,
               int slot) { self.setRenderableTexture(handle, texture, slot); },
            py::arg("handle"), py::arg("texture"), py::arg("slot") = 0,
            "Attach a texture to a renderable using a raw texture slot.")
        .def(
            "update_renderable_geometry",
            [](Renderer& self, uint32_t handle, const FloatArray& positions,
               py::object normals) {
                auto positionVec = vec3Array(positions, "positions");
                std::vector<glm::vec3> normalVec;
                if (!normals.is_none()) {
                    auto normalArray = normals.cast<FloatArray>();
                    normalVec = vec3Array(normalArray, "normals");
                    if (normalVec.size() != positionVec.size()) {
                        throw py::value_error(
                            "normals must match positions length");
                    }
                }
                self.updateRenderableGeometry(handle, positionVec, normalVec);
            },
            py::arg("handle"), py::arg("positions"),
            py::arg("normals") = py::none(),
            "Update dynamic vertex positions and optional normals.")
        .def(
            "update_renderable_skinning_matrices",
            [](Renderer& self, uint32_t handle, const FloatArray& matrices) {
                self.updateRenderableSkinningMatrices(
                    handle, mat4RowMajorArray(matrices, "bone_matrices"));
            },
            py::arg("handle"), py::arg("bone_matrices"),
            "Update bone matrices for a skinned renderable.");


}
