#include "base_panel.hpp"
#define IMVIEWGUIZMO_IMPLEMENTATION
#include "ImViewGuizmo.h"
#include "imgui.h"
#include "engine/core/app/app.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/graphics/renderer/rasterizer.hpp"
#include "engine/scene/component/camera_component.hpp"
#include "engine/scene/component/articulation_component.hpp"
#include "engine/scene/component/articulation_binding_component.hpp"
#include "engine/scene/component/collision_shape_component.hpp"
#include "engine/scene/component/light_component.hpp"
#include "engine/scene/component/material_binding_component.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/component/transform_component.hpp"
#include "engine/scene/native/xform_token.hpp"
#include <IconsFontAwesome7.h>
#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace KE {
namespace {

glm::vec3 toViewGuizmoSpace(const glm::vec3& value, UpAxis upAxis) {
    // ImViewGuizmo is right-handed, Y-up and -Z-forward. Keep Y-up scenes as
    // they are. The cyclic permutation for Z-up preserves handedness and maps
    // all three positive scene axes to positive gizmo axes.
    if (upAxis == UpAxis::Z)
        return {value.y, value.z, value.x};
    return value;
}

glm::vec3 fromViewGuizmoSpace(const glm::vec3& value, UpAxis upAxis) {
    if (upAxis == UpAxis::Z)
        return {value.z, value.x, value.y};
    return value;
}

const char* primTypeLabel(Scene::PrimType type) {
    switch (type) {
    case Scene::PrimType::Root:
        return "Root";
    case Scene::PrimType::Xform:
        return "Xform";
    case Scene::PrimType::Mesh:
        return "Mesh";
    case Scene::PrimType::MeshInstance:
        return "MeshInstance";
    case Scene::PrimType::Camera:
        return "Camera";
    case Scene::PrimType::Light:
        return "Light";
    case Scene::PrimType::Resource:
        return "Resource";
    }
    return "Unknown";
}

const char* lightTypeLabel(Scene::LightType type) {
    switch (type) {
    case Scene::LightType::Directional:
        return "Directional";
    case Scene::LightType::Point:
        return "Point";
    case Scene::LightType::Spot:
        return "Spot";
    }
    return "Unknown";
}

const char* cameraProjectionTypeLabel(Scene::CameraProjectionType type) {
    switch (type) {
    case Scene::CameraProjectionType::Perspective:
        return "Perspective";
    case Scene::CameraProjectionType::Orthographic:
        return "Orthographic";
    }
    return "Unknown";
}

std::vector<Scene::Prim*> collectCameraPrims(Scene::SceneBackend* scene) {
    std::vector<Scene::Prim*> cameras;
    if (!scene || !scene->getRootPrim())
        return cameras;
    scene->getRootPrim()->traverse([&](Scene::Prim* prim) {
        if (prim && prim->getType() == Scene::PrimType::Camera &&
            prim->hasCameraComponent()) {
            cameras.push_back(prim);
        }
    });
    std::sort(cameras.begin(), cameras.end(),
              [](const Scene::Prim* lhs, const Scene::Prim* rhs) {
                  return lhs->getPath() < rhs->getPath();
              });
    return cameras;
}

std::string cameraDisplayName(const Scene::Prim* prim) {
    if (!prim)
        return "None";
    return std::string(ICON_FA_CAMERA " ") + prim->getPath();
}

const char* cameraAspectPresetLabel(int preset) {
    switch (preset) {
    case 0:
        return "Free";
    case 1:
        return "16:9";
    case 2:
        return "4:3";
    case 3:
        return "1:1";
    case 4:
        return "Custom";
    }
    return "Free";
}

float cameraAspectPresetValue(int preset, float customAspect) {
    switch (preset) {
    case 1:
        return 16.0f / 9.0f;
    case 2:
        return 4.0f / 3.0f;
    case 3:
        return 1.0f;
    case 4:
        return std::max(0.01f, customAspect);
    default:
        return 0.0f;
    }
}

const char* cameraCapturePresetLabel(int preset) {
    switch (preset) {
    case 0:
        return "Panel";
    case 1:
        return "FHD";
    case 2:
        return "4K";
    case 3:
        return "Custom";
    }
    return "FHD";
}

ImVec2 cameraCapturePresetSize(int preset, int customWidth, int customHeight,
                               const ImVec2& panelSize) {
    switch (preset) {
    case 0:
        return panelSize;
    case 1:
        return {1920.0f, 1080.0f};
    case 2:
        return {3840.0f, 2160.0f};
    case 3:
        return {static_cast<float>(std::max(1, customWidth)),
                static_cast<float>(std::max(1, customHeight))};
    }
    return {1920.0f, 1080.0f};
}

std::string nextScenePath(Scene::SceneBackend* scene, const std::string& base) {
    if (!scene)
        return base;
    if (!scene->getPrimAtPath(base))
        return base;
    for (int index = 1; index < 10000; ++index) {
        const std::string candidate = base + "_" + std::to_string(index);
        if (!scene->getPrimAtPath(candidate))
            return candidate;
    }
    return base + "_many";
}

bool isEngineOwnedPrim(const Scene::Prim* prim) {
    return prim && prim->getPath() == "/lights/default_directional";
}

bool isResourceNamespacePrim(const Scene::Prim* prim) {
    if (!prim)
        return false;
    const std::string& path = prim->getPath();
    return path == "/.Resources" || path.rfind("/.Resources/", 0) == 0;
}

bool subtreeHasResourceNamespacePrim(Scene::Prim* prim) {
    if (!prim)
        return false;
    bool found = false;
    prim->traverse([&](Scene::Prim* child) {
        if (isResourceNamespacePrim(child))
            found = true;
    });
    return found;
}

bool subtreeHasEngineOwnedPrim(Scene::Prim* prim) {
    if (!prim)
        return false;
    bool found = false;
    prim->traverse([&](Scene::Prim* child) {
        if (isEngineOwnedPrim(child))
            found = true;
    });
    return found;
}

bool subtreeHasExternalPrim(App* app, Scene::Prim* prim) {
    if (!app || !prim)
        return false;
    bool found = false;
    prim->traverse([&](Scene::Prim* child) {
        TransformSource source = TransformSource::SceneGraph;
        if (app->getPrimTransformSource(child, source) &&
            source == TransformSource::ExternalBuffer) {
            found = true;
        }
    });
    return found;
}

Scene::Prim* addLightPrim(App* app, Scene::LightType type) {
    if (!app || !app->getScene())
        return nullptr;

    const char* baseName = "light";
    switch (type) {
    case Scene::LightType::Directional:
        baseName = "directional";
        break;
    case Scene::LightType::Point:
        baseName = "point";
        break;
    case Scene::LightType::Spot:
        baseName = "spot";
        break;
    }

    Scene::Prim* prim = app->getScene()->definePrim(
        nextScenePath(app->getScene(), std::string("/lights/") + baseName),
        Scene::PrimType::Light);
    if (!prim)
        return nullptr;

    switch (type) {
    case Scene::LightType::Directional: {
        Scene::DirectionalLight light = app->getLight();
        prim->setDirectionalLight(light);
        break;
    }
    case Scene::LightType::Point: {
        Scene::PointLight light;
        light.position = glm::vec3(0.0f, 2.0f, 2.0f);
        prim->setPointLight(light);
        break;
    }
    case Scene::LightType::Spot: {
        Scene::SpotLight light;
        light.position = glm::vec3(0.0f, 2.0f, 2.0f);
        light.direction = glm::normalize(glm::vec3(0.0f, -1.0f, -1.0f));
        prim->setSpotLight(light);
        break;
    }
    }

    app->selectPrim(prim);
    return prim;
}

bool setCameraWorldOrientation(Scene::Prim& prim, const glm::vec3& forward,
                               const glm::vec3& up);

Scene::Prim* addCameraPrim(App* app) {
    if (!app || !app->getScene())
        return nullptr;

    Scene::Prim* prim = app->getScene()->definePrim(
        nextScenePath(app->getScene(), "/cameras/camera"),
        Scene::PrimType::Camera);
    if (!prim)
        return nullptr;

    prim->setLocalTranslation(app->getCamera().getCameraPos());
    setCameraWorldOrientation(*prim, app->getCamera().getCameraLookDir(),
                              app->getCamera().getCameraUpDir());
    prim->addCameraComponent();
    app->selectPrim(prim);
    return prim;
}

bool drawLightComponentEditor(Scene::Prim& prim, bool editable) {
    if (prim.getType() != Scene::PrimType::Light)
        return false;

    if (!prim.hasLightComponent())
        prim.addLightComponent();

    auto component = prim.getLightComponent();
    if (!component || !component->isAttached())
        return false;

    bool changed = false;
    ImGui::PushID("LightComponent");
    if (!editable)
        ImGui::BeginDisabled();

    ImGui::TextDisabled("Type");
    ImGui::SameLine();
    ImGui::TextUnformatted(lightTypeLabel(component->type()));

    ImGui::TextDisabled("Source");
    ImGui::SameLine();
    ImGui::TextUnformatted("LightComponent");
    ImGui::TextDisabled("Version");
    ImGui::SameLine();
    ImGui::Text("%llu", static_cast<unsigned long long>(component->version()));

    switch (component->type()) {
    case Scene::LightType::Directional: {
        Scene::DirectionalLight light = component->directionalLight();
        glm::vec3 direction = light.direction;
        glm::vec3 ambient = light.ambient;
        float color[3] = {light.color.r, light.color.g, light.color.b};
        bool lightChanged = false;
        lightChanged |= ImGui::DragFloat3("Direction", &direction.x, 0.02f);
        lightChanged |= ImGui::ColorEdit3("Color", color);
        lightChanged |=
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 5.0f);
        lightChanged |= ImGui::ColorEdit3("Ambient", &ambient.x);
        if (lightChanged) {
            if (glm::length(direction) > 1e-4f)
                light.direction = glm::normalize(direction);
            light.color = glm::vec3(color[0], color[1], color[2]);
            light.ambient = ambient;
            prim.setDirectionalLight(light);
            changed = true;
        }
        break;
    }
    case Scene::LightType::Point: {
        Scene::PointLight light = component->pointLight();
        float color[3] = {light.color.r, light.color.g, light.color.b};
        bool lightChanged = false;
        lightChanged |= ImGui::DragFloat3("Position", &light.position.x, 0.01f);
        lightChanged |= ImGui::ColorEdit3("Color", color);
        lightChanged |=
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 10.0f);
        lightChanged |=
            ImGui::DragFloat("Range", &light.range, 0.05f, 0.0f, FLT_MAX);
        if (lightChanged) {
            light.color = glm::vec3(color[0], color[1], color[2]);
            prim.setPointLight(light);
            changed = true;
        }
        break;
    }
    case Scene::LightType::Spot: {
        Scene::SpotLight light = component->spotLight();
        glm::vec3 direction = light.direction;
        float color[3] = {light.color.r, light.color.g, light.color.b};
        float innerDegrees = glm::degrees(light.innerConeAngle);
        float outerDegrees = glm::degrees(light.outerConeAngle);
        bool lightChanged = false;
        lightChanged |= ImGui::DragFloat3("Position", &light.position.x, 0.01f);
        lightChanged |= ImGui::DragFloat3("Direction", &direction.x, 0.02f);
        lightChanged |= ImGui::ColorEdit3("Color", color);
        lightChanged |=
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 10.0f);
        lightChanged |=
            ImGui::DragFloat("Range", &light.range, 0.05f, 0.0f, FLT_MAX);
        lightChanged |= ImGui::SliderFloat("Inner Cone", &innerDegrees, 0.0f,
                                           179.0f, "%.1f deg");
        lightChanged |= ImGui::SliderFloat("Outer Cone", &outerDegrees, 0.0f,
                                           179.0f, "%.1f deg");
        if (lightChanged) {
            if (glm::length(direction) > 1e-4f)
                light.direction = glm::normalize(direction);
            light.color = glm::vec3(color[0], color[1], color[2]);
            light.innerConeAngle = glm::radians(std::max(0.0f, innerDegrees));
            light.outerConeAngle =
                glm::radians(std::max(innerDegrees, outerDegrees));
            prim.setSpotLight(light);
            changed = true;
        }
        break;
    }
    }

    if (!editable)
        ImGui::EndDisabled();
    ImGui::PopID();
    return changed;
}

bool drawCameraComponentEditor(App* app, Scene::Prim& prim, bool editable) {
    if (prim.getType() != Scene::PrimType::Camera)
        return false;

    if (!prim.hasCameraComponent())
        prim.addCameraComponent();

    auto component = prim.getCameraComponent();
    if (!component || !component->isAttached())
        return false;

    bool changed = false;
    ImGui::PushID("CameraComponent");
    if (!editable)
        ImGui::BeginDisabled();

    ImGui::TextDisabled("Source");
    ImGui::SameLine();
    ImGui::TextUnformatted("CameraComponent");
    ImGui::TextDisabled("Type");
    ImGui::SameLine();
    ImGui::TextUnformatted(
        cameraProjectionTypeLabel(component->projectionType()));
    ImGui::TextDisabled("Version");
    ImGui::SameLine();
    ImGui::Text("%llu", static_cast<unsigned long long>(component->version()));
    if (app) {
        const bool active = app->activeSceneCameraPath() == prim.getPath();
        ImGui::TextDisabled("Scene View");
        ImGui::SameLine();
        ImGui::TextUnformatted(active ? "Camera View / Viewer Source"
                                      : "Editor View");
        if (!active) {
            if (ImGui::Button("Use as Scene Camera"))
                app->setActiveSceneCamera(&prim);
        } else {
            if (ImGui::Button("Clear Scene Camera"))
                app->clearActiveSceneCamera();
        }
    }

    const char* projectionLabels[] = {"Perspective", "Orthographic"};
    int projection = static_cast<int>(component->projectionType());
    float fov = component->verticalFovDegrees();
    float orthoSize = component->orthographicSize();
    float nearPlane = component->nearPlane();
    float farPlane = component->farPlane();

    bool projectionChanged = false;
    projectionChanged |=
        ImGui::Combo("Projection", &projection, projectionLabels,
                     static_cast<int>(std::size(projectionLabels)));
    if (projection ==
        static_cast<int>(Scene::CameraProjectionType::Perspective)) {
        projectionChanged |=
            ImGui::SliderFloat("Vertical FOV", &fov, 1.0f, 179.0f, "%.1f deg");
    } else {
        projectionChanged |= ImGui::DragFloat("Orthographic Size", &orthoSize,
                                              0.05f, 0.001f, FLT_MAX, "%.3f");
    }
    projectionChanged |=
        ImGui::DragFloat("Near Plane", &nearPlane, 0.01f, 0.001f, FLT_MAX);
    projectionChanged |=
        ImGui::DragFloat("Far Plane", &farPlane, 0.1f, 0.001f, FLT_MAX);

    if (projectionChanged) {
        if (projection ==
            static_cast<int>(Scene::CameraProjectionType::Orthographic)) {
            component->setOrthographic(orthoSize, nearPlane, farPlane);
        } else {
            component->setPerspective(fov, nearPlane, farPlane);
        }
        changed = true;
    }

    glm::vec3 position = component->position();
    glm::vec3 forward = component->forward();
    glm::vec3 up = component->up();
    if (ImGui::DragFloat3("Position", &position.x, 0.01f)) {
        prim.setWorldTranslation(position);
        changed = true;
    }
    if (ImGui::DragFloat3("Forward", &forward.x, 0.01f)) {
        changed |= setCameraWorldOrientation(prim, forward, up);
    }
    if (ImGui::DragFloat3("Up", &up.x, 0.01f)) {
        changed |= setCameraWorldOrientation(prim, forward, up);
    }

    if (!editable)
        ImGui::EndDisabled();
    ImGui::PopID();
    return changed;
}

const char* materialTypeLabel(const Material* material) {
    if (!material)
        return "None";
    if (dynamic_cast<const PBRMaterial*>(material))
        return "PBRMaterial";
    if (dynamic_cast<const PhongMaterial*>(material))
        return "PhongMaterial";
    if (dynamic_cast<const VertexColorMaterial*>(material))
        return "VertexColorMaterial";
    return "Material";
}

void drawTextureStatus(const char* label, const Backend::Texture* texture) {
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine();
    if (texture)
        ImGui::TextColored(ImVec4(0.45f, 0.82f, 0.52f, 1.0f), "Bound");
    else
        ImGui::TextDisabled("None");
}

void drawSharedMaterialHint() {
    ImGui::TextWrapped(
        "Editing this material changes every Prim that shares the same "
        "Material*.");
}

void drawMaterialInspector(Scene::Prim& prim) {
    auto binding = prim.getMaterialBindingComponent();
    Material* material = prim.getMaterial();

    ImGui::SeparatorText("Material");
    if (!binding || !binding->isAttached()) {
        ImGui::TextDisabled("No MaterialBindingComponent");
        return;
    }

    ImGui::TextDisabled("Source");
    ImGui::SameLine();
    ImGui::TextUnformatted("MaterialBindingComponent");
    ImGui::TextDisabled("Version");
    ImGui::SameLine();
    ImGui::Text("%llu", static_cast<unsigned long long>(binding->version()));
    ImGui::TextDisabled("Type");
    ImGui::SameLine();
    ImGui::TextUnformatted(materialTypeLabel(material));
    ImGui::TextDisabled("Material*");
    ImGui::SameLine();
    ImGui::Text("%p", static_cast<void*>(material));

    if (!material) {
        ImGui::TextDisabled("No material bound");
        return;
    }

    drawSharedMaterialHint();

    if (auto* phong = dynamic_cast<PhongMaterial*>(material)) {
        ImGui::SeparatorText("Phong Parameters");
        ImGui::ColorEdit3("Ambient", &phong->ambient.x);
        ImGui::ColorEdit3("Diffuse", &phong->diffuse.x);
        ImGui::ColorEdit3("Specular", &phong->specular.x);
        ImGui::DragFloat("Shininess", &phong->shininess, 0.25f, 1.0f, 512.0f);
        drawTextureStatus("Diffuse Map", phong->diffuseMap);
        drawTextureStatus("Specular Map", phong->specularMap);
        drawTextureStatus("Alpha Map", phong->alphaMap);
        drawTextureStatus("Normal Map", phong->normalMap);
    } else if (auto* pbr = dynamic_cast<PBRMaterial*>(material)) {
        ImGui::SeparatorText("PBR Parameters");
        ImGui::ColorEdit4("Base Color", &pbr->baseColor.x);
        ImGui::SliderFloat("Metallic", &pbr->metallic, 0.0f, 1.0f);
        ImGui::SliderFloat("Roughness", &pbr->roughness, 0.02f, 1.0f);
        ImGui::ColorEdit3("Emissive Color", &pbr->emissiveColor.x);
        ImGui::DragFloat("Emissive Strength", &pbr->emissiveStrength, 0.05f,
                         0.0f, 100.0f);
        drawTextureStatus("Base Color Map", pbr->baseColorTexture);
        drawTextureStatus("Normal Map", pbr->normalTexture);
        drawTextureStatus("MetallicRoughness Map",
                          pbr->metallicRoughnessTexture);
        drawTextureStatus("Metallic Map", pbr->metallicTexture);
        drawTextureStatus("Roughness Map", pbr->roughnessTexture);
        drawTextureStatus("AO Map", pbr->aoTexture);
        drawTextureStatus("ORM Map", pbr->ormTexture);
        drawTextureStatus("Emissive Map", pbr->emissiveTexture);
    } else if (dynamic_cast<VertexColorMaterial*>(material)) {
        ImGui::TextWrapped(
            "Compatibility wrapper for legacy shader-only renderables. "
            "Surface color comes from per-instance display/base color.");
    }
}

void drawResourceComponentEditor(App* app, Scene::Prim& prim) {
    auto resource = prim.getResourceComponent();
    if (!resource) {
        ImGui::TextDisabled("No ResourceComponent");
        return;
    }
    ImGui::BeginDisabled();
    ImGui::Text("Component: attached=%s version=%llu",
                resource->isAttached() ? "true" : "false",
                static_cast<unsigned long long>(resource->version()));
    ImGui::Text("Handle: %u", resource->handle());

    const char* kindLabels[] = {"Unknown", "Mesh",          "Material",
                                "Texture", "Shader Source", "Pipeline"};
    int kind = static_cast<int>(resource->type());
    if (ImGui::Combo("Kind", &kind, kindLabels,
                     static_cast<int>(std::size(kindLabels)))) {
        resource->setType(static_cast<Scene::ResourceType>(kind));
    }

    char displayName[256] = {};
    std::snprintf(displayName, sizeof(displayName), "%s",
                  resource->displayName().c_str());
    if (ImGui::InputText("Display Name", displayName, sizeof(displayName)))
        resource->setDisplayName(displayName);

    char uri[512] = {};
    std::snprintf(uri, sizeof(uri), "%s", resource->uri().c_str());
    if (ImGui::InputText("URI", uri, sizeof(uri)))
        resource->setUri(uri);
    ImGui::EndDisabled();

    if (app && resource->handle() != Scene::InvalidResourceHandle) {
        const auto& manager = app->getSceneResourceManager();
        if (const auto* shader = manager.shaderSource(resource->handle())) {
            const char* stage = "Vertex";
            switch (shader->stage) {
            case Backend::ShaderType::Vertex:
                stage = "Vertex";
                break;
            case Backend::ShaderType::Fragment:
                stage = "Fragment";
                break;
            case Backend::ShaderType::Geometry:
                stage = "Geometry";
                break;
            case Backend::ShaderType::Compute:
                stage = "Compute";
                break;
            }
            ImGui::SeparatorText("Shader Source");
            ImGui::Text("Language: %s",
                        shader->language == Scene::ShaderLanguage::WGSL
                            ? "WGSL"
                            : "GLSL");
            ImGui::Text("Stage: %s", stage);
            ImGui::Text("Entry point: %s", shader->entryPoint.c_str());
            if (ImGui::CollapsingHeader("Source"))
                ImGui::TextUnformatted(shader->source.c_str());
        } else if (const auto* pipeline =
                       manager.pipeline(resource->handle())) {
            ImGui::SeparatorText("Pipeline");
            ImGui::Text("Type: %s",
                        pipeline->type == Scene::AuthoredPipelineType::Compute
                            ? "Compute"
                            : "Graphics");
            ImGui::Text("Shader stages: %zu", pipeline->shaderSources.size());
            if (!pipeline->shaderSources.empty() &&
                ImGui::CollapsingHeader(
                    ("Shader Sources (" +
                     std::to_string(pipeline->shaderSources.size()) + ")")
                        .c_str())) {
                for (Scene::ResourceHandle handle : pipeline->shaderSources) {
                    const auto* entry = manager.entry(handle);
                    ImGui::BulletText("%s", entry ? entry->name.c_str()
                                                  : "<missing>");
                }
            }
            if (!pipeline->stateSummary.empty())
                ImGui::TextWrapped("State: %s", pipeline->stateSummary.c_str());
            if (!pipeline->variants.empty() &&
                ImGui::CollapsingHeader(
                    ("Variants (" + std::to_string(pipeline->variants.size()) +
                     ")")
                        .c_str())) {
                for (const std::string& variant : pipeline->variants)
                    ImGui::BulletText("%s", variant.c_str());
            }
        }
    }

    ImGui::SeparatorText("Usage");
    if (!app || resource->handle() == Scene::InvalidResourceHandle) {
        ImGui::TextDisabled("No resource manager handle");
        return;
    }

    const auto& manager = app->getSceneResourceManager();
    const auto& paths = manager.usagePaths(resource->handle());
    ImGui::Text("Usage count: %zu", paths.size());
    if (paths.empty()) {
        ImGui::TextDisabled("Unused");
        return;
    }

    if (ImGui::BeginTable("ResourceUsagePaths", 1,
                          ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersInnerH |
                              ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Used By");
        ImGui::TableHeadersRow();
        for (const std::string& path : paths) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(path.c_str());
        }
        ImGui::EndTable();
    }
}

bool drawAttributeValue(const std::string& name,
                        Scene::AttributeValue& attribute, bool editable) {
    bool changed = false;
    if (!editable)
        ImGui::BeginDisabled();

    std::visit(
        [&](auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, bool>) {
                changed = ImGui::Checkbox("##Value", &value);
            } else if constexpr (std::is_same_v<T, int>) {
                ImGui::SetNextItemWidth(-FLT_MIN);
                changed = ImGui::DragInt("##Value", &value, 1.0f);
            } else if constexpr (std::is_same_v<T, float>) {
                ImGui::SetNextItemWidth(-FLT_MIN);
                changed = ImGui::DragFloat("##Value", &value, 0.01f);
            } else if constexpr (std::is_same_v<T, std::string>) {
                ImGui::TextWrapped("%s", value.c_str());
            } else if constexpr (std::is_same_v<T, glm::vec3>) {
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (name.find("color") != std::string::npos)
                    changed = ImGui::ColorEdit3("##Value", &value.x);
                else
                    changed = ImGui::DragFloat3("##Value", &value.x, 0.01f);
            } else if constexpr (std::is_same_v<T, glm::vec4>) {
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (name.find("color") != std::string::npos)
                    changed = ImGui::ColorEdit4("##Value", &value.x);
                else
                    changed = ImGui::DragFloat4("##Value", &value.x, 0.01f);
            } else if constexpr (std::is_same_v<T, glm::quat>) {
                float components[4] = {value.x, value.y, value.z, value.w};
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::DragFloat4("##Value", components, 0.01f)) {
                    const glm::quat candidate(components[3], components[0],
                                              components[1], components[2]);
                    if (glm::length(candidate) > 1e-6f) {
                        value = glm::normalize(candidate);
                        changed = true;
                    }
                }
            } else if constexpr (std::is_same_v<T, glm::mat4>) {
                for (int row = 0; row < 4; ++row) {
                    ImGui::Text("%.3f  %.3f  %.3f  %.3f", value[0][row],
                                value[1][row], value[2][row], value[3][row]);
                }
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                if (value.empty())
                    ImGui::TextDisabled("(empty)");
                for (const std::string& item : value)
                    ImGui::TextUnformatted(item.c_str());
            }
        },
        attribute);

    if (!editable)
        ImGui::EndDisabled();
    return changed;
}

bool decomposeTransform(const glm::mat4& matrix, glm::vec3& translation,
                        glm::vec3& rotationDegrees, glm::vec3& scale) {
    glm::quat rotation;
    glm::vec3 skew;
    glm::vec4 perspective;
    if (!glm::decompose(matrix, scale, rotation, translation, skew,
                        perspective)) {
        return false;
    }
    rotationDegrees = glm::degrees(glm::eulerAngles(glm::normalize(rotation)));
    return true;
}

bool setCameraWorldOrientation(Scene::Prim& prim, const glm::vec3& forward,
                               const glm::vec3& up) {
    if (glm::length2(forward) < 1.0e-8f || glm::length2(up) < 1.0e-8f)
        return false;

    const glm::vec3 safeForward = glm::normalize(forward);
    const glm::vec3 right = glm::cross(safeForward, glm::normalize(up));
    if (glm::length2(right) < 1.0e-8f)
        return false;

    const glm::vec3 safeUp =
        glm::normalize(glm::cross(glm::normalize(right), safeForward));
    prim.setWorldRotation(glm::quatLookAt(safeForward, safeUp));
    return true;
}

} // namespace

PerformancePanel::PerformancePanel(App* app)
    : Panel("Performance"), _app(app) {}

PerformancePanel::~PerformancePanel() {}

void PerformancePanel::buildPanel() {
    if (!ImGui::Begin(name().c_str(), openPtr())) {
        ImGui::End();
        return;
    }
    ImGui::Text("Performance");
    ImGui::Separator();
    if (_app) {
        bool vsync = _app->getVSync();
        if (ImGui::Checkbox("VSync", &vsync))
            _app->setVSync(vsync);
        ImGui::Text("Displayed FPS: %.1f", _app->getMeasuredRenderFPS());
        ImGui::Text("Frame CPU: %.3f ms", _app->getFrameCPUTimeMs());
        ImGui::Text("Update CPU: %.3f ms", _app->getUpdateCPUTimeMs());
        ImGui::Text("Render CPU: %.3f ms", _app->getRenderCPUTimeMs());
        ImGui::Text("Present: %.3f ms", _app->getPresentCPUTimeMs());
    }
    const float imguiFPS = ImGui::GetIO().Framerate;
    ImGui::Text("ImGui FPS: %.1f (%.3f ms/frame)", imguiFPS,
                imguiFPS > 0.0f ? 1000.0f / imguiFPS : 0.0f);
    ImGui::End();
}

RendererDebugPanel::RendererDebugPanel(App* app)
    : Panel("Renderer Debug"), _app(app) {}

RendererDebugPanel::~RendererDebugPanel() {}

void RendererDebugPanel::buildPanel() {
    if (!_app)
        return;

    if (!ImGui::Begin(name().c_str(), openPtr())) {
        ImGui::End();
        return;
    }

    constexpr ImGuiTreeNodeFlags defaultOpen = ImGuiTreeNodeFlags_DefaultOpen;
    RendererSettings& rendererSettings = _app->getRenderer().settings();

    if (ImGui::CollapsingHeader("Controls", defaultOpen)) {
        ImGui::Checkbox("Wireframe", &_app->_renderWireframe);
        ImGui::SeparatorText("Background");
        ImGui::Checkbox("Show Grid", &rendererSettings.background.showGrid);
        ImGui::ColorEdit4("Grid Color",
                          &rendererSettings.background.gridColor.x);
        ImGui::ColorEdit4("Background Color",
                          &rendererSettings.background.backgroundColor.x);
        ImGui::ColorEdit4("Checker A",
                          &rendererSettings.background.checkerColor1.x);
        ImGui::ColorEdit4("Checker B",
                          &rendererSettings.background.checkerColor2.x);
        const char* groundShadingLabels[] = {"Phong", "PBR"};
        int groundShading =
            static_cast<int>(rendererSettings.background.groundShadingModel);
        if (ImGui::Combo("Ground Shading", &groundShading, groundShadingLabels,
                         2)) {
            rendererSettings.background.groundShadingModel =
                static_cast<GroundShadingModel>(groundShading);
        }
        ImGui::SliderFloat("Ground Metallic",
                           &rendererSettings.background.groundMetallic, 0.0f,
                           1.0f);
        ImGui::SliderFloat("Ground Roughness",
                           &rendererSettings.background.groundRoughness, 0.04f,
                           1.0f);
        ImGui::DragFloat("Grid Scale", &rendererSettings.background.gridScale,
                         0.05f, 0.05f, 20.0f);
        ImGui::DragFloat("Grid Line Width",
                         &rendererSettings.background.gridLineWidth, 0.001f,
                         0.001f, 0.49f);
        ImGui::SliderFloat("Grid Emission",
                           &rendererSettings.background.gridEmission, 0.0f,
                           2.0f);
        ImGui::SeparatorText("Interaction");
        const char* interactionLabels[] = {"Inspect", "Edit", "Force"};
        int interactionMode = static_cast<int>(_app->getInteractionMode());
        if (ImGui::Combo("Interaction Mode", &interactionMode,
                         interactionLabels, 3)) {
            _app->setInteractionMode(
                static_cast<InteractionMode>(interactionMode));
        }
        ImGui::DragFloat("Camera Move Speed", &_app->_cameraMoveSpeed, 0.2f,
                         0.0f, 500.0f, "%.2f");
    }

    if (ImGui::CollapsingHeader("Post Processing", defaultOpen)) {
        ImGui::SliderFloat("Gamma Correction", &rendererSettings.gamma, 0.f,
                           5.f);
        const char* toneMapLabels[] = {"None", "Reinhard Simple", "Exponential",
                                       "ACES Narkowicz", "ACES Hill"};
        int toneMapMode = static_cast<int>(rendererSettings.toneMapMode);
        if (ImGui::Combo("Tone Mapping", &toneMapMode, toneMapLabels, 5)) {
            rendererSettings.toneMapMode =
                static_cast<ToneMapMode>(toneMapMode);
        }
        if (rendererSettings.toneMapMode != ToneMapMode::None) {
            ImGui::SliderFloat("Tone Map Exposure",
                               &rendererSettings.toneMapExposure, 0.f, 5.f);
        }
        ImGui::Checkbox("Bloom", &rendererSettings.bloom.enabled);
        if (rendererSettings.bloom.enabled) {
            ImGui::SliderFloat("Bloom Threshold",
                               &rendererSettings.bloom.threshold, 0.0f, 10.0f);
            ImGui::SliderFloat("Bloom Intensity",
                               &rendererSettings.bloom.intensity, 0.0f, 2.0f);
            ImGui::SliderInt("Bloom Iterations",
                             &rendererSettings.bloom.iterations, 0, 16);
            ImGui::SliderInt("Bloom Downsample",
                             &rendererSettings.bloom.downsample, 1, 8);
        }
    }

    if (ImGui::CollapsingHeader("Selection")) {
        if (SelectionOutlineProcessor* outline =
                _app->getRenderer().selectionOutline()) {
            SelectionOutlineConfig& config = outline->config();
            ImGui::Checkbox("Selection Outline", &config.enabled);
            float outlineColor[4] = {config.color.r, config.color.g,
                                     config.color.b, config.color.a};
            if (ImGui::ColorEdit4("Selection Outline Color", outlineColor)) {
                config.color = glm::vec4(outlineColor[0], outlineColor[1],
                                         outlineColor[2], outlineColor[3]);
            }
            ImGui::SliderFloat("Selection Outline Radius", &config.radius, 1.0f,
                               8.0f, "%.1f px");
        }
    }

    Rasterizer* rasterizer = _app->getRenderer().rasterizer();
    if (!rasterizer) {
        ImGui::End();
        return;
    }

    DirectionalLight light = _app->getRenderer().light();
    glm::vec3 direction = light.direction;
    float color[3] = {light.color.r, light.color.g, light.color.b};
    glm::vec3 ambient = light.ambient;

    if (ImGui::CollapsingHeader("Lighting", defaultOpen)) {
        if (ImGui::DragFloat3("Sun Direction (toward light)", &direction.x,
                              0.02f)) {
            _app->setLightDirection(direction);
        }
        if (ImGui::ColorEdit3("Light Color", color)) {
            _app->setLightColor(glm::vec3(color[0], color[1], color[2]));
        }
        if (ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f,
                               10.0f)) {
            _app->setLightIntensity(light.intensity);
        }
        if (ImGui::ColorEdit3("Ambient", &ambient.x)) {
            _app->setLightAmbient(ambient);
        }
    }

    float distance = rasterizer->getShadowDistance();
    int pcfSamples = rasterizer->getShadowPcfSamples();
    bool useCsm = rasterizer->getUseCsm();
    if (ImGui::CollapsingHeader("Shadows")) {
        if (ImGui::SliderFloat("Shadow Distance (0 disables shadow)", &distance,
                               0.0f, 300.0f)) {
            rasterizer->setShadowDistance(distance);
        }
        if (ImGui::SliderInt("Shadow PCF Samples", &pcfSamples, 1, 16)) {
            rasterizer->setShadowPcfSamples(pcfSamples);
        }
        if (ImGui::Checkbox("Use CSM", &useCsm)) {
            rasterizer->setUseCsm(useCsm);
        }
        if (useCsm) {
            int cascadeCount = rasterizer->getCascadeCount();
            if (ImGui::SliderInt("CSM Cascade Count", &cascadeCount, 1,
                                 Rasterizer::MaxShadowCascades)) {
                rasterizer->setCascadeCount(cascadeCount);
            }
            float cascadeLambda = rasterizer->getCascadeLambda();
            if (ImGui::SliderFloat("CSM Cascade Lambda", &cascadeLambda, 0.0f,
                                   1.0f)) {
                rasterizer->setCascadeLambda(cascadeLambda);
            }
            bool useTightShadowFit = rasterizer->getUseTightShadowFit();
            if (ImGui::Checkbox("Tight Shadow Fit", &useTightShadowFit)) {
                rasterizer->setUseTightShadowFit(useTightShadowFit);
            }
            bool debugCascadeTint = rasterizer->getDebugCsmCascadeTint();
            if (ImGui::Checkbox("Debug CSM Cascade Tint", &debugCascadeTint)) {
                rasterizer->setDebugCsmCascadeTint(debugCascadeTint);
            }
        }
    }

    if (ImGui::CollapsingHeader("Diagnostics")) {
        bool frustumCulling = rasterizer->isFrustumCullingEnabled();
        if (ImGui::Checkbox("Frustum Culling", &frustumCulling)) {
            rasterizer->setFrustumCullingEnabled(frustumCulling);
        }
        bool debugRenderAABB = rasterizer->getDebugRenderAABB();
        if (ImGui::Checkbox("Show Render AABB", &debugRenderAABB)) {
            rasterizer->setDebugRenderAABB(debugRenderAABB);
        }
        if (frustumCulling) {
            // Batch = one instancer/draw group. Instance = one transform inside
            // that batch, culled by its world AABB.
            ImGui::Text("Culled Batches %d / %d",
                        rasterizer->getCullingCulledBatches(),
                        rasterizer->getCullingTotalBatches());
            ImGui::Text("Culled Instances %d / %d",
                        rasterizer->getCullingCulledInstances(),
                        rasterizer->getCullingTotalInstances());
        }
    }

    if (ImGui::CollapsingHeader("Shadow Map Preview")) {
        if (useCsm) {
            const int cascadeCount = rasterizer->getCascadeCount();
            ImGui::Text("CSM Shadow Maps");
            if (ImGui::BeginTable("CSMShadowMapPreview", 2)) {
                for (int i = 0; i < cascadeCount; ++i) {
                    auto* cascadeFbo = rasterizer->getCascadeShadowFbo(i);
                    if (!cascadeFbo)
                        continue;
                    auto* depthTex = cascadeFbo->getDepthTexture();
                    if (!depthTex)
                        continue;
                    ImGui::TableNextColumn();
                    ImGui::Text("Cascade %d %dx%d", i, depthTex->getWidth(),
                                depthTex->getHeight());
                    ImGui::Image(
                        (ImTextureID)(uintptr_t)depthTex->getNativeHandle(),
                        ImVec2(128, 128), ImVec2(0, 1), ImVec2(1, 0));
                }
                ImGui::EndTable();
            }
        } else {
            auto* shadowFbo = rasterizer->getShadowFbo();
            if (shadowFbo) {
                auto* depthTex = shadowFbo->getDepthTexture();
                if (depthTex) {
                    ImGui::Text("Shadow Map %dx%d", depthTex->getWidth(),
                                depthTex->getHeight());
                    ImGui::Image(
                        (ImTextureID)(uintptr_t)depthTex->getNativeHandle(),
                        ImVec2(128, 128), ImVec2(0, 1), ImVec2(1, 0));
                }
            }
        }
    }
    ImGui::End();
}

InspectorPanel::InspectorPanel(App* app) : Panel("Inspector"), _app(app) {}

InspectorPanel::~InspectorPanel() {}

void InspectorPanel::buildPanel() {
    if (!ImGui::Begin(name().c_str(), openPtr())) {
        ImGui::End();
        return;
    }

    if (!_app || !_app->hasSelection()) {
        ImGui::TextDisabled("No selection");
        ImGui::End();
        return;
    }

    const RayPickResult& selection = _app->getSelection();
    Scene::Prim* prim = selection.prim;
    if (!prim) {
        ImGui::SeparatorText("Selection");
        ImGui::TextColored(ImVec4(0.48f, 0.72f, 0.94f, 1.0f),
                           ICON_FA_LOCK "  ExternalBuffer");
        ImGui::Text("Handle: %u", static_cast<unsigned>(selection.handle));
        ImGui::Text("Instance: %d", selection.instanceIndex);
        ImGui::End();
        return;
    }

    TransformSource source = selection.transformSource;
    _app->getPrimTransformSource(prim, source);
    const bool external = source == TransformSource::ExternalBuffer;
    const bool resourceMirror = isResourceNamespacePrim(prim);

    ImGui::SeparatorText("Prim");
    if (ImGui::BeginTable("PrimSummary", 2,
                          ImGuiTableFlags_SizingStretchProp)) {
        const auto property = [](const char* label) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextDisabled("%s", label);
            ImGui::TableSetColumnIndex(1);
        };

        property("Name");
        ImGui::TextUnformatted(prim->getName().c_str());
        property("Path");
        ImGui::TextWrapped("%s", prim->getPath().c_str());
        property("Type");
        ImGui::TextUnformatted(primTypeLabel(prim->getType()));
        property("Source");
        if (external) {
            ImGui::TextColored(ImVec4(0.48f, 0.72f, 0.94f, 1.0f),
                               ICON_FA_LOCK "  ExternalBuffer");
        } else {
            ImGui::TextUnformatted("SceneGraph");
        }
        property("Parent");
        ImGui::TextUnformatted(
            prim->getParent() ? prim->getParent()->getPath().c_str() : "None");
        property("Renderable");
        ImGui::TextUnformatted(prim->isRenderable() ? "Yes" : "No");
        ImGui::EndTable();
    }

    ImGui::SeparatorText("State");
    if (external || resourceMirror)
        ImGui::BeginDisabled();
    bool active = prim->isActive();
    if (ImGui::Checkbox("Active", &active))
        prim->setActive(active);
    bool visible = prim->isVisible();
    if (ImGui::Checkbox("Visible", &visible))
        prim->setVisible(visible);

    const char* policyLabels[] = {"Inherit", "Self", "Parent", "Root",
                                  "Disabled"};
    int policy = static_cast<int>(prim->getManipulationPolicy());
    if (ImGui::Combo("Manipulation", &policy, policyLabels,
                     static_cast<int>(std::size(policyLabels)))) {
        prim->setManipulationPolicy(
            static_cast<Scene::ManipulationPolicy>(policy));
    }
    if (external || resourceMirror)
        ImGui::EndDisabled();
    if (resourceMirror) {
        ImGui::TextDisabled(
            "Resource mirrors are managed by SceneResourceManager.");
    }

    ImGui::SeparatorText("Selection");
    auto selectionComponent = prim->getSelectionComponent();
    if (selectionComponent) {
        ImGui::Text(
            "Component: attached=%s version=%llu",
            selectionComponent->isAttached() ? "true" : "false",
            static_cast<unsigned long long>(selectionComponent->version()));
        if (ImGui::BeginTable("SelectionPolicy", 2,
                              ImGuiTableFlags_SizingStretchProp)) {
            const auto property = [](const char* label) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextDisabled("%s", label);
                ImGui::TableSetColumnIndex(1);
            };
            const auto yesNo = [](bool value) { return value ? "Yes" : "No"; };

            property("Pickable");
            ImGui::TextUnformatted(yesNo(selectionComponent->isPickable()));
            property("Selectable");
            ImGui::TextUnformatted(yesNo(selectionComponent->isSelectable()));
            property("Manipulatable");
            ImGui::TextUnformatted(
                yesNo(selectionComponent->isManipulatable()));
            property("Force Draggable");
            ImGui::TextUnformatted(
                yesNo(selectionComponent->isForceDraggable()));
            property("Interaction Kind");
            ImGui::TextUnformatted(Scene::interactionKindLabel(
                selectionComponent->interactionKind()));
            ImGui::EndTable();
        }
        ImGui::Text("Metadata: env=%d obj=%d body=%d",
                    selectionComponent->envId(), selectionComponent->objId(),
                    selectionComponent->bodyId());
        if (!selectionComponent->userTag().empty())
            ImGui::TextWrapped("Tag: %s",
                               selectionComponent->userTag().c_str());
    } else {
        ImGui::TextDisabled("No SelectionComponent");
    }

    if (auto articulationRoot = prim->getArticulationComponent()) {
        ImGui::SeparatorText("Articulation");
        ImGui::Text(
            "Component: attached=%s version=%llu",
            articulationRoot->isAttached() ? "true" : "false",
            static_cast<unsigned long long>(articulationRoot->version()));
        if (ImGui::BeginTable("ArticulationComponent", 2,
                              ImGuiTableFlags_SizingStretchProp)) {
            const auto property = [](const char* label) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextDisabled("%s", label);
                ImGui::TableSetColumnIndex(1);
            };
            property("Root");
            ImGui::TextWrapped("%s",
                               articulationRoot->rootPath().empty()
                                   ? "<none>"
                                   : articulationRoot->rootPath().c_str());
            property("Asset");
            ImGui::TextWrapped("%s",
                               articulationRoot->assetPath().empty()
                                   ? "<none>"
                                   : articulationRoot->assetPath().c_str());
            property("Mesh Assets");
            ImGui::TextWrapped(
                "%s", articulationRoot->meshAssetBasePath().empty()
                          ? "<none>"
                          : articulationRoot->meshAssetBasePath().c_str());
            property("Bodies");
            ImGui::Text("%d", articulationRoot->bodyCount());
            property("Render Prims");
            ImGui::Text("%d", articulationRoot->renderPrimCount());
            property("Split Visual Geoms");
            ImGui::TextUnformatted(articulationRoot->splitVisualGeoms() ? "Yes"
                                                                        : "No");
            ImGui::EndTable();
        }
    }

    if (auto articulation = prim->getArticulationBindingComponent()) {
        ImGui::SeparatorText("Articulation Binding");
        ImGui::Text("Component: attached=%s version=%llu",
                    articulation->isAttached() ? "true" : "false",
                    static_cast<unsigned long long>(articulation->version()));
        if (ImGui::BeginTable("ArticulationBinding", 2,
                              ImGuiTableFlags_SizingStretchProp)) {
            const auto property = [](const char* label) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextDisabled("%s", label);
                ImGui::TableSetColumnIndex(1);
            };
            property("Role");
            ImGui::TextUnformatted(
                Scene::articulationPrimRoleLabel(articulation->role()));
            property("Body Index");
            ImGui::Text("%d", articulation->bodyIndex());
            property("Body Name");
            ImGui::TextUnformatted(articulation->bodyName().empty()
                                       ? "<none>"
                                       : articulation->bodyName().c_str());
            property("Root");
            ImGui::TextWrapped(
                "%s", articulation->articulationRootPath().empty()
                          ? "<none>"
                          : articulation->articulationRootPath().c_str());
            ImGui::EndTable();
        }
    }

    if (auto collisionShape = prim->getCollisionShapeComponent()) {
        ImGui::SeparatorText("Collision Shape");
        ImGui::Text("Component: attached=%s version=%llu",
                    collisionShape->isAttached() ? "true" : "false",
                    static_cast<unsigned long long>(collisionShape->version()));
        if (ImGui::BeginTable("CollisionShape", 2,
                              ImGuiTableFlags_SizingStretchProp)) {
            const auto property = [](const char* label) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextDisabled("%s", label);
                ImGui::TableSetColumnIndex(1);
            };
            property("Shape");
            ImGui::TextUnformatted(
                Scene::collisionShapeTypeLabel(collisionShape->shapeType()));
            property("Source Geom");
            if (collisionShape->sourceGeomIndex() >= 0)
                ImGui::Text("%d", collisionShape->sourceGeomIndex());
            else
                ImGui::TextUnformatted("Fallback");
            property("Size");
            const glm::vec3& size = collisionShape->size();
            ImGui::Text("%.4f, %.4f, %.4f", size.x, size.y, size.z);
            property("Local Pos");
            const glm::vec3& localPos = collisionShape->localPosition();
            ImGui::Text("%.4f, %.4f, %.4f", localPos.x, localPos.y, localPos.z);
            property("Local Rot");
            const glm::quat& localRot = collisionShape->localRotation();
            ImGui::Text("w %.4f, x %.4f, y %.4f, z %.4f", localRot.w,
                        localRot.x, localRot.y, localRot.z);
            property("From/To");
            ImGui::TextUnformatted(collisionShape->hasFromTo() ? "Yes" : "No");
            if (collisionShape->hasFromTo()) {
                property("From");
                const glm::vec3& from = collisionShape->fromPosition();
                ImGui::Text("%.4f, %.4f, %.4f", from.x, from.y, from.z);
                property("To");
                const glm::vec3& to = collisionShape->toPosition();
                ImGui::Text("%.4f, %.4f, %.4f", to.x, to.y, to.z);
            }
            property("Friction");
            ImGui::Text("static %.4f, dynamic %.4f",
                        collisionShape->staticFriction(),
                        collisionShape->dynamicFriction());
            property("Restitution");
            ImGui::Text("%.4f", collisionShape->restitution());
            property("Condim( Unused )");
            if (collisionShape->condim() >= 0)
                ImGui::Text("%d", collisionShape->condim());
            else
                ImGui::TextUnformatted("<none>");
            property("Margin");
            if (collisionShape->margin() >= 0.0f)
                ImGui::Text("%.4f", collisionShape->margin());
            else
                ImGui::TextUnformatted("<none>");
            ImGui::EndTable();
        }
    }

    ImGui::SeparatorText("Transform");
    auto transform = prim->getTransformComponent();
    if (transform) {
        ImGui::Text("Component: attached=%s version=%llu",
                    transform->isAttached() ? "true" : "false",
                    static_cast<unsigned long long>(transform->version()));
    } else {
        ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1.0f),
                           "Missing TransformComponent");
    }

    if (external) {
        ImGui::TextColored(ImVec4(0.48f, 0.72f, 0.94f, 1.0f),
                           ICON_FA_LOCK "  Owned by ExternalBuffer");
    } else if (!transform) {
        ImGui::TextDisabled("No transform data");
    } else {
        glm::vec3 translation(0.0f);
        glm::vec3 rotationDegrees(0.0f);
        glm::vec3 scale(1.0f);
        if (decomposeTransform(transform->computeWorldMatrix(), translation,
                               rotationDegrees, scale)) {
            ImGui::BeginDisabled();
            ImGui::DragFloat3("Position", &translation.x, 0.01f);
            ImGui::DragFloat3("Rotation(deg)", &rotationDegrees.x, 0.1f, 0.0f,
                              0.0f, "%.2f");
            ImGui::DragFloat3("Scale", &scale.x, 0.01f);
            ImGui::EndDisabled();
        } else {
            ImGui::TextDisabled("Transform decomposition unavailable");
        }
    }

    if (const std::shared_ptr<Scene::MeshData> mesh = prim->resolveMeshData()) {
        ImGui::SeparatorText("Mesh");
        auto meshComponent = prim->getMeshComponent();
        if (meshComponent) {
            ImGui::Text(
                "Component: attached=%s version=%llu",
                meshComponent->isAttached() ? "true" : "false",
                static_cast<unsigned long long>(meshComponent->version()));
            if (meshComponent->resourceHandle() !=
                Scene::InvalidResourceHandle) {
                ImGui::Text("Resource Handle: %u",
                            meshComponent->resourceHandle());
            }
            if (!meshComponent->meshSourcePath().empty())
                ImGui::TextWrapped("Source: %s",
                                   meshComponent->meshSourcePath().c_str());
        }
        ImGui::Text("Vertices: %zu", mesh->vertices.size());
        ImGui::Text("Indices: %zu", mesh->indices.size());
        ImGui::Text("Triangles: %zu", mesh->indices.size() / 3);
    }

    if (prim->hasMaterialBindingComponent())
        drawMaterialInspector(*prim);

    if (prim->getType() == Scene::PrimType::Light) {
        ImGui::SeparatorText("Light");
        drawLightComponentEditor(*prim, !external);
    }

    if (prim->getType() == Scene::PrimType::Camera) {
        ImGui::SeparatorText("Camera");
        drawCameraComponentEditor(_app, *prim, !external);
    }

    if (prim->getType() == Scene::PrimType::Resource) {
        ImGui::SeparatorText("Resource");
        drawResourceComponentEditor(_app, *prim);
    }

    ImGui::SeparatorText("Attributes");
    std::unordered_set<std::string> activeXformOps;
    if (prim->hasAttribute(Scene::XformTokens::opOrder)) {
        const auto& order = prim->getAttribute<std::vector<std::string>>(
            Scene::XformTokens::opOrder);
        activeXformOps.insert(order.begin(), order.end());
    } else {
        for (const Scene::Token& token : Scene::XformTokens::defaultOpOrder)
            activeXformOps.insert(token.str());
    }
    std::vector<std::pair<Scene::Token, Scene::AttributeValue>> attributes;
    attributes.reserve(prim->getAttributes().size());
    for (const auto& [token, value] : prim->getAttributes())
        attributes.emplace_back(token, value);
    std::sort(attributes.begin(), attributes.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first.str() < rhs.first.str();
              });

    if (attributes.empty()) {
        ImGui::TextDisabled("No attributes");
    } else if (ImGui::BeginTable("PrimAttributes", 2,
                                 ImGuiTableFlags_BordersInnerH |
                                     ImGuiTableFlags_Resizable |
                                     ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch,
                                0.42f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch,
                                0.58f);
        ImGui::TableHeadersRow();
        for (auto& [token, value] : attributes) {
            ImGui::PushID(static_cast<int>(token.id()));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            const bool inactiveXform =
                Scene::XformTokens::isXformAttribute(token) &&
                token != Scene::XformTokens::opOrder &&
                activeXformOps.count(token.str()) == 0;
            if (inactiveXform)
                ImGui::TextDisabled("%s  (inactive)", token.str().c_str());
            else
                ImGui::TextWrapped("%s", token.str().c_str());
            ImGui::TableSetColumnIndex(1);
            const bool editable =
                !external && !Scene::XformTokens::isXformAttribute(token);
            if (drawAttributeValue(token.str(), value, editable))
                prim->setAttribute(token, value);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    ImGui::End();
}

MenuBarPanel::MenuBarPanel(App* app) : Panel("Menu Bar"), _app(app) {}

MenuBarPanel::~MenuBarPanel() {}

void MenuBarPanel::buildPanel() {
    if (ImGui::BeginMainMenuBar()) {
        // TODO: implement this
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Scene")) {
            }
            if (ImGui::MenuItem("Open...")) {
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Save")) {
            }
            if (ImGui::MenuItem("Exit")) {
                if (_app)
                    _app->requestClose();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Add")) {
            if (ImGui::MenuItem("Camera"))
                addCameraPrim(_app);
            if (ImGui::BeginMenu("Light")) {
                if (ImGui::MenuItem("Directional"))
                    addLightPrim(_app, Scene::LightType::Directional);
                if (ImGui::MenuItem("Point"))
                    addLightPrim(_app, Scene::LightType::Point);
                if (ImGui::MenuItem("Spot"))
                    addLightPrim(_app, Scene::LightType::Spot);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Settings")) {
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            if (_app) {
                PanelManager& panels = _app->_panelManager;
                UILayoutMode layoutMode = panels.getLayoutMode();
                if (ImGui::BeginMenu("Layout Mode")) {
                    if (ImGui::MenuItem("Viewer", nullptr,
                                        layoutMode == UILayoutMode::Viewer)) {
                        panels.setLayoutMode(UILayoutMode::Viewer);
                    }
                    if (ImGui::MenuItem("Editor", nullptr,
                                        layoutMode == UILayoutMode::Editor)) {
                        panels.setLayoutMode(UILayoutMode::Editor);
                    }
                    if (ImGui::MenuItem("Overlay", nullptr,
                                        layoutMode == UILayoutMode::Overlay)) {
                        panels.setLayoutMode(UILayoutMode::Overlay);
                    }
                    ImGui::EndMenu();
                }

                ImGui::Separator();

                bool sceneOpen = panels.isPanelOpen(PanelManager::PANEL_SCENE);
                bool rendererDebugOpen =
                    panels.isPanelOpen(PanelManager::PANEL_RENDERER_DEBUG);
                bool performanceOpen =
                    panels.isPanelOpen(PanelManager::PANEL_PERFORMANCE);
                bool inspectorOpen =
                    panels.isPanelOpen(PanelManager::PANEL_INSPECTOR);
                bool viewportOpen =
                    panels.isPanelOpen(PanelManager::PANEL_VIEWPORT);
                bool cameraViewOpen =
                    panels.isPanelOpen(PanelManager::PANEL_CAMERA_VIEW);

                if (layoutMode == UILayoutMode::Editor &&
                    ImGui::MenuItem("Viewport", nullptr, viewportOpen))
                    panels.setPanelOpen(PanelManager::PANEL_VIEWPORT,
                                        !viewportOpen);
                if (layoutMode == UILayoutMode::Editor &&
                    ImGui::MenuItem("Camera View", nullptr, cameraViewOpen))
                    panels.setPanelOpen(PanelManager::PANEL_CAMERA_VIEW,
                                        !cameraViewOpen);
                if (ImGui::MenuItem("Scene", nullptr, sceneOpen))
                    panels.setPanelOpen(PanelManager::PANEL_SCENE, !sceneOpen);
                if (ImGui::MenuItem("Renderer Debug", nullptr,
                                    rendererDebugOpen))
                    panels.setPanelOpen(PanelManager::PANEL_RENDERER_DEBUG,
                                        !rendererDebugOpen);
                if (ImGui::MenuItem("Performance", nullptr, performanceOpen))
                    panels.setPanelOpen(PanelManager::PANEL_PERFORMANCE,
                                        !performanceOpen);
                if (ImGui::MenuItem("Inspector", nullptr, inspectorOpen))
                    panels.setPanelOpen(PanelManager::PANEL_INSPECTOR,
                                        !inspectorOpen);

                ImGui::Separator();
                if (ImGui::MenuItem("Reset Dock Layout"))
                    panels.resetLayout();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            const bool shortcutHelpOpen = _app && _app->_showShortcutHelp;
            if (ImGui::MenuItem("Keyboard & Mouse Help", "?", shortcutHelpOpen,
                                _app != nullptr)) {
                _app->_showShortcutHelp = !shortcutHelpOpen;
            }
            ImGui::EndMenu();
        }
        /*
        if (ImGui::Button("Play")) {}
        ImGui::SameLine();
        if (ImGui::Button("Pause")) {}
        */

        char fpsText[32];
        std::snprintf(fpsText, sizeof(fpsText), "FPS: %d",
                      int(ImGui::GetIO().Framerate));
        const ImGuiStyle& style = ImGui::GetStyle();
        const float textWidth = ImGui::CalcTextSize(fpsText).x;
        const float rightPadding = style.FramePadding.x;
        const float buttonWidth = ImGui::GetFrameHeight();
        const float buttonSpacing = style.ItemSpacing.x;
        const float fpsSpacing = style.ItemSpacing.x * 2.0f;
        const float controlsWidth =
            buttonWidth * 3.0f + buttonSpacing * 2.0f + fpsSpacing + textWidth;
        const float cursorX =
            ImGui::GetWindowWidth() - controlsWidth - rightPadding;
        if (cursorX > ImGui::GetCursorPosX()) {
            ImGui::SetCursorPosX(cursorX);
        }

        if (_app) {
            PanelManager& panels = _app->_panelManager;
            auto layoutButton = [&](const char* label, const char* tooltip,
                                    UILayoutMode mode) {
                const bool selected = panels.getLayoutMode() == mode;
                const ImGuiCol buttonColor =
                    selected ? ImGuiCol_HeaderHovered : ImGuiCol_MenuBarBg;
                ImGui::PushStyleColor(ImGuiCol_Button,
                                      ImGui::GetStyleColorVec4(buttonColor));
                if (ImGui::Button(label, ImVec2(buttonWidth, 0.0f)))
                    panels.setLayoutMode(mode);
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                    ImGui::SetTooltip("%s", tooltip);
            };

            layoutButton(ICON_FA_EYE "##ViewerLayout", "Viewer Mode",
                         UILayoutMode::Viewer);
            ImGui::SameLine(0.0f, buttonSpacing);
            layoutButton(ICON_FA_PEN_TO_SQUARE "##EditorLayout", "Editor Mode",
                         UILayoutMode::Editor);
            ImGui::SameLine(0.0f, buttonSpacing);
            layoutButton(ICON_FA_LAYER_GROUP "##OverlayLayout", "Overlay Mode",
                         UILayoutMode::Overlay);
            ImGui::SameLine(0.0f, fpsSpacing);
        }
        ImGui::TextDisabled("%s", fpsText);
        ImGui::Spacing();
        // if (ImGui::SmallButton("somthing")) {}

        ImGui::EndMainMenuBar();
    }
}

ViewportPanel::ViewportPanel(App* app, Camera* camera, std::string name,
                             std::string cameraLabel)
    : Panel(std::move(name)), _app(app), _camera(camera),
      _cameraLabel(std::move(cameraLabel)) {
    setOpen(false);
}

ViewportPanel::~ViewportPanel() {}

void ViewportPanel::setCameraLabel(std::string cameraLabel) {
    _cameraLabel = std::move(cameraLabel);
}

void ViewportPanel::buildPanel() {
    _hovered = false;
    _focused = false;
    _viewGuizmoCapturesMouse = false;
    if (!ImGui::Begin(name().c_str(), openPtr())) {
        ImGui::End();
        return;
    }

    _contentMin = ImGui::GetCursorScreenPos();
    _contentSize = ImGui::GetContentRegionAvail();
    _focused = ImGui::IsWindowFocused();

    const ImVec2 max(_contentMin.x + _contentSize.x,
                     _contentMin.y + _contentSize.y);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(_contentMin, max,
                            ImGui::GetColorU32(ImGuiCol_WindowBg));
    drawList->AddRect(_contentMin, max, ImGui::GetColorU32(ImGuiCol_Border));

    Backend::Texture* texture = _app ? _app->getPresentedTexture() : nullptr;
    _imageMin = _contentMin;
    _imageSize = _contentSize;
    if (texture && texture->getWidth() > 0 && texture->getHeight() > 0 &&
        _contentSize.x > 1.0f && _contentSize.y > 1.0f) {
        const float textureAspect = static_cast<float>(texture->getWidth()) /
                                    static_cast<float>(texture->getHeight());
        const float panelAspect = _contentSize.x / _contentSize.y;
        if (panelAspect > textureAspect) {
            _imageSize.y = _contentSize.y;
            _imageSize.x = _imageSize.y * textureAspect;
            _imageMin.x += (_contentSize.x - _imageSize.x) * 0.5f;
        } else {
            _imageSize.x = _contentSize.x;
            _imageSize.y = _imageSize.x / textureAspect;
            _imageMin.y += (_contentSize.y - _imageSize.y) * 0.5f;
        }

        ImGui::SetCursorScreenPos(_imageMin);
        ImGui::Image((ImTextureID)(uintptr_t)texture->getNativeHandle(),
                     _imageSize, ImVec2(0, 1), ImVec2(1, 0));
    }

    ImGui::SetCursorScreenPos(_contentMin);
    const ImVec2 buttonSize(std::max(_contentSize.x, 1.0f),
                            std::max(_contentSize.y, 1.0f));
    ImGui::Dummy(buttonSize);

    if (_app) {
        const ImGuiStyle& style = ImGui::GetStyle();
        constexpr int toolboxSlots = 5;
        const float margin = style.ItemSpacing.x;
        const float toolButtonSize =
            ImGui::GetFrameHeight() + style.FramePadding.x * 2.0f;
        const ImVec2 toolboxSize(toolButtonSize + style.WindowPadding.x * 2.0f,
                                 toolButtonSize * toolboxSlots +
                                     style.ItemSpacing.y * (toolboxSlots - 1) +
                                     style.WindowPadding.y * 2.0f);
        const ImVec2 toolboxPos(_imageMin.x + margin, _imageMin.y + margin);
        const bool toolboxFits =
            toolboxSize.x + margin * 2.0f <= _imageSize.x &&
            toolboxSize.y + margin * 2.0f <= _imageSize.y;

        if (toolboxFits) {
            ImGui::SetCursorScreenPos(toolboxPos);
            ImGui::PushStyleColor(ImGuiCol_ChildBg,
                                  ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
            if (ImGui::BeginChild("ViewportToolbox", toolboxSize,
                                  ImGuiChildFlags_Borders,
                                  ImGuiWindowFlags_NoScrollbar |
                                      ImGuiWindowFlags_NoScrollWithMouse)) {
                auto modeButton = [&](const char* label, const char* tooltip,
                                      InteractionMode mode) {
                    const bool selected = _app->getInteractionMode() == mode;
                    const ImGuiCol buttonColor =
                        selected ? ImGuiCol_HeaderHovered : ImGuiCol_ChildBg;
                    ImGui::PushStyleColor(
                        ImGuiCol_Button, ImGui::GetStyleColorVec4(buttonColor));
                    if (ImGui::Button(label,
                                      ImVec2(toolButtonSize, toolButtonSize))) {
                        _app->setInteractionMode(mode);
                    }
                    ImGui::PopStyleColor();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                        ImGui::SetTooltip("%s", tooltip);
                };

                modeButton(ICON_FA_ARROW_POINTER "##InspectMode", "Inspect",
                           InteractionMode::Inspect);
                modeButton(ICON_FA_UP_DOWN_LEFT_RIGHT "##EditMode",
                           "Edit Transform", InteractionMode::Edit);
                modeButton(ICON_FA_HAND_FIST "##ForceMode", "Apply Force",
                           InteractionMode::Force);
            }
            ImGui::EndChild();
            ImGui::PopStyleColor();

            const float labelX =
                toolboxPos.x + toolboxSize.x + style.ItemSpacing.x;
            if (labelX + ImGui::CalcTextSize(_cameraLabel.c_str()).x + margin <=
                _imageMin.x + _imageSize.x) {
                ImGui::SetCursorScreenPos(
                    ImVec2(labelX, toolboxPos.y + style.FramePadding.y));
                ImGui::TextDisabled("%s", _cameraLabel.c_str());
            }
        }

        constexpr float viewGuizmoScale = 0.5f;
        ImViewGuizmo::Style& guizmoStyle = ImViewGuizmo::GetStyle();
        guizmoStyle.scale = viewGuizmoScale;

        // Rotate() names this argument `position`, but the implementation uses
        // it as the gizmo center. Include the axis-handle radius in the bounds.
        const float guizmoExtent =
            (128.0f + guizmoStyle.circleRadius) * guizmoStyle.scale;
        const float guizmoMargin = style.ItemSpacing.x;
        if (_camera && _imageSize.x >= 2.0f * (guizmoExtent + guizmoMargin) &&
            _imageSize.y >= 2.0f * (guizmoExtent + guizmoMargin)) {
            const ImVec2 imageMax(_imageMin.x + _imageSize.x,
                                  _imageMin.y + _imageSize.y);
            const ImVec2 guizmoCenter(imageMax.x - guizmoExtent - guizmoMargin,
                                      _imageMin.y + guizmoExtent +
                                          guizmoMargin);

            Camera& camera = *_camera;
            const UpAxis upAxis = camera.getUpAxis();
            if (upAxis == UpAxis::Z) {
                // Gizmo X/Y/Z correspond to scene Y/Z/X after the cyclic
                // conversion above. Remap labels and conventional axis colors
                // so the widget still describes scene-space axes.
                guizmoStyle.axisLabels[0] = "Y";
                guizmoStyle.axisLabels[1] = "-Y";
                guizmoStyle.axisLabels[2] = "Z";
                guizmoStyle.axisLabels[3] = "-Z";
                guizmoStyle.axisLabels[4] = "X";
                guizmoStyle.axisLabels[5] = "-X";
                guizmoStyle.axisColors[0] = IM_COL32(140, 206, 40, 255);
                guizmoStyle.axisColors[1] = IM_COL32(49, 155, 249, 255);
                guizmoStyle.axisColors[2] = IM_COL32(233, 62, 85, 255);
            } else {
                guizmoStyle.axisLabels[0] = "X";
                guizmoStyle.axisLabels[1] = "-X";
                guizmoStyle.axisLabels[2] = "Y";
                guizmoStyle.axisLabels[3] = "-Y";
                guizmoStyle.axisLabels[4] = "Z";
                guizmoStyle.axisLabels[5] = "-Z";
                guizmoStyle.axisColors[0] = IM_COL32(233, 62, 85, 255);
                guizmoStyle.axisColors[1] = IM_COL32(140, 206, 40, 255);
                guizmoStyle.axisColors[2] = IM_COL32(49, 155, 249, 255);
            }
            glm::vec3 cameraPosition =
                toViewGuizmoSpace(camera.getCameraPos(), upAxis);
            const glm::vec3 pivot =
                toViewGuizmoSpace(camera.getTargetPos(), upAxis);
            const glm::vec3 lookDirection = pivot - cameraPosition;

            if (glm::length2(lookDirection) > 1.0e-8f) {
                const glm::vec3 cameraUp =
                    toViewGuizmoSpace(camera.getCameraUpDir(), upAxis);
                glm::quat cameraRotation = glm::quatLookAt(
                    glm::normalize(lookDirection), glm::normalize(cameraUp));

                ImViewGuizmo::BeginFrame();
                ImGui::GetWindowDrawList()->PushClipRect(_imageMin, imageMax,
                                                         true);
                const bool cameraModified = ImViewGuizmo::Rotate(
                    cameraPosition, cameraRotation, pivot, guizmoCenter);
                ImGui::GetWindowDrawList()->PopClipRect();

                _viewGuizmoCapturesMouse =
                    ImViewGuizmo::IsUsing() || ImViewGuizmo::IsOver();
                if (cameraModified) {
                    // This Camera is target-based, so keeping the target fixed
                    // and applying the new orbit position is sufficient. Its
                    // view matrix, pole and azimuth are refreshed by the
                    // setter.
                    camera.setCameraPos(
                        fromViewGuizmoSpace(cameraPosition, upAxis));
                }
            }
        }
    }

    const ImVec2 imageMax(_imageMin.x + _imageSize.x,
                          _imageMin.y + _imageSize.y);
    _hovered = ImGui::IsWindowHovered() &&
               ImGui::IsMouseHoveringRect(_imageMin, imageMax);
    if (_app && _camera)
        _app->renderSelectionGizmo(*_camera, _imageMin, _imageSize,
                                   ImGui::GetWindowDrawList());
    ImGui::End();
}

CameraViewPanel::CameraViewPanel(App* app, std::string name)
    : Panel(std::move(name)), _app(app) {
    setOpen(false);
}

CameraViewPanel::~CameraViewPanel() {}

void CameraViewPanel::buildPanel() {
    if (!ImGui::Begin(name().c_str(), openPtr())) {
        ImGui::End();
        return;
    }

    if (!_app || !_app->getScene()) {
        ImGui::TextDisabled("No scene");
        ImGui::End();
        return;
    }

    std::vector<Scene::Prim*> cameras = collectCameraPrims(_app->getScene());
    Scene::Prim* activeCamera = nullptr;
    for (Scene::Prim* camera : cameras) {
        if (camera && camera->getPath() == _app->activeSceneCameraPath()) {
            activeCamera = camera;
            break;
        }
    }

    const std::string currentLabel = activeCamera
                                         ? cameraDisplayName(activeCamera)
                                         : "No active scene camera";
    ImGui::SetNextItemWidth(std::min(360.0f, ImGui::GetContentRegionAvail().x));
    if (ImGui::BeginCombo("Camera", currentLabel.c_str())) {
        for (Scene::Prim* camera : cameras) {
            const bool selected = camera == activeCamera;
            const std::string label = cameraDisplayName(camera);
            if (ImGui::Selectable(label.c_str(), selected)) {
                _app->setActiveSceneCamera(camera);
                activeCamera = camera;
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    if (cameras.empty()) {
        ImGui::TextDisabled("No CameraComponent in scene");
        ImGui::End();
        return;
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::BeginCombo("Aspect", cameraAspectPresetLabel(_aspectPreset))) {
        for (int preset = 0; preset <= 4; ++preset) {
            const bool selected = preset == _aspectPreset;
            if (ImGui::Selectable(cameraAspectPresetLabel(preset), selected))
                _aspectPreset = preset;
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if (_aspectPreset == 4) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        ImGui::DragFloat("Custom", &_customAspect, 0.01f, 0.01f, 100.0f,
                         "%.2f");
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(115.0f);
    if (ImGui::BeginCombo("Capture",
                          cameraCapturePresetLabel(_capturePreset))) {
        for (int preset = 0; preset <= 3; ++preset) {
            const bool selected = preset == _capturePreset;
            if (ImGui::Selectable(cameraCapturePresetLabel(preset), selected))
                _capturePreset = preset;
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if (_capturePreset == 3) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::InputInt("W", &_customCaptureWidth, 0, 0);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(90.0f);
        ImGui::InputInt("H", &_customCaptureHeight, 0, 0);
        _customCaptureWidth = std::max(1, _customCaptureWidth);
        _customCaptureHeight = std::max(1, _customCaptureHeight);
    }

    ImGui::SameLine();
    const bool screenshotRequested =
        ImGui::Button(ICON_FA_CAMERA "##CameraViewScreenshot");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Save Camera View screenshot");

    if (!activeCamera) {
        ImGui::TextDisabled("Select a camera to preview");
        ImGui::End();
        return;
    }

    const ImVec2 contentMin = ImGui::GetCursorScreenPos();
    const ImVec2 contentSize = ImGui::GetContentRegionAvail();
    ImVec2 imageMin = contentMin;
    ImVec2 imageSize = contentSize;
    const float aspectOverride =
        cameraAspectPresetValue(_aspectPreset, _customAspect);
    if (aspectOverride > 0.0f && contentSize.x > 1.0f && contentSize.y > 1.0f) {
        const float availableAspect = contentSize.x / contentSize.y;
        if (availableAspect > aspectOverride) {
            imageSize.y = contentSize.y;
            imageSize.x = contentSize.y * aspectOverride;
            imageMin.x += (contentSize.x - imageSize.x) * 0.5f;
        } else {
            imageSize.x = contentSize.x;
            imageSize.y = contentSize.x / aspectOverride;
            imageMin.y += (contentSize.y - imageSize.y) * 0.5f;
        }
    }

    const int width = std::max(1, static_cast<int>(imageSize.x));
    const int height = std::max(1, static_cast<int>(imageSize.y));
    Backend::Texture* texture =
        _app->renderActiveSceneCameraPreview(width, height, aspectOverride);
    if (!texture || texture->getWidth() <= 0 || texture->getHeight() <= 0) {
        ImGui::TextDisabled("Camera preview unavailable");
        ImGui::End();
        return;
    }
    if (screenshotRequested) {
        const ImVec2 captureSize =
            cameraCapturePresetSize(_capturePreset, _customCaptureWidth,
                                    _customCaptureHeight, imageSize);
        const int captureWidth = std::max(1, static_cast<int>(captureSize.x));
        const int captureHeight = std::max(1, static_cast<int>(captureSize.y));
        const float captureAspect = aspectOverride > 0.0f
                                        ? aspectOverride
                                        : static_cast<float>(captureWidth) /
                                              static_cast<float>(captureHeight);
        const bool saved = _app->writeActiveSceneCameraPreviewPNG(
            captureWidth, captureHeight, captureAspect);
        _lastSaveStatus = saved ? "Saved Camera View screenshot"
                                : "Failed to save screenshot";
    }

    _imageMin = imageMin;
    _imageSize = imageSize;
    ImGui::SetCursorScreenPos(_imageMin);
    ImGui::Image((ImTextureID)(uintptr_t)texture->getNativeHandle(), _imageSize,
                 ImVec2(0, 1), ImVec2(1, 0));

    ImGui::SetCursorScreenPos(
        ImVec2(_imageMin.x + ImGui::GetStyle().ItemSpacing.x,
               _imageMin.y + ImGui::GetStyle().ItemSpacing.y));
    ImGui::TextDisabled("%s", activeCamera->getPath().c_str());
    if (!_lastSaveStatus.empty()) {
        ImGui::SetCursorScreenPos(
            ImVec2(_imageMin.x + ImGui::GetStyle().ItemSpacing.x,
                   _imageMin.y + ImGui::GetStyle().ItemSpacing.y +
                       ImGui::GetTextLineHeightWithSpacing()));
        ImGui::TextDisabled("%s", _lastSaveStatus.c_str());
    }

    ImGui::End();
}

ScenePanel::ScenePanel(App* app) : Panel("Scene"), _app(app) {}

ScenePanel::~ScenePanel() {}

void ScenePanel::buildPanel() {
    if (!ImGui::Begin(name().c_str(), openPtr())) {
        ImGui::End();
        return;
    }
    if (auto* root = _app->getScene()->getRootPrim()) {
        auto drawPrimTree = [&](auto& self, Scene::Prim* prim) -> void {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::PushID(prim);
            bool visible = prim->isVisible();
            const bool resourceMirror = isResourceNamespacePrim(prim);
            if (resourceMirror)
                ImGui::BeginDisabled();
            if (ImGui::Checkbox("##Visible", &visible))
                prim->setVisible(visible);
            if (resourceMirror)
                ImGui::EndDisabled();
            ImGui::SameLine();

            const bool activeInHierarchy = prim->isActiveInHierarchy();
            const bool visibleInHierarchy = prim->isVisibleInHierarchy();
            const bool disabled = !activeInHierarchy || !visibleInHierarchy;
            TransformSource transformSource = TransformSource::SceneGraph;
            const bool external =
                _app->getPrimTransformSource(prim, transformSource) &&
                transformSource == TransformSource::ExternalBuffer;
            bool unusedResource = false;
            bool actualResourcePrim = false;
            if (prim->getType() == Scene::PrimType::Resource) {
                if (auto resource = prim->getResourceComponent()) {
                    actualResourcePrim = true;
                    const auto handle = resource->handle();
                    unusedResource =
                        handle != Scene::InvalidResourceHandle &&
                        _app->getSceneResourceManager().usageCount(handle) == 0;
                }
            }
            const bool resourceFolderMirror =
                resourceMirror && !actualResourcePrim;
            const bool customTextColor =
                disabled || external || resourceFolderMirror || unusedResource;
            if (customTextColor) {
                const ImVec4 textColor =
                    (disabled || unusedResource || resourceFolderMirror)
                        ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled)
                        : ImVec4(0.48f, 0.72f, 0.94f, 1.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, textColor);
            }

            const auto& children = prim->getChildren();
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                                       ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                       ImGuiTreeNodeFlags_SpanAvailWidth;
            if (_app->isPrimSelected(prim))
                flags |= ImGuiTreeNodeFlags_Selected;
            if (children.empty())
                flags |= ImGuiTreeNodeFlags_Leaf |
                         ImGuiTreeNodeFlags_NoTreePushOnOpen;

            const bool open =
                external
                    ? ImGui::TreeNodeEx("##Prim", flags, ICON_FA_LOCK "  %s",
                                        prim->getName().c_str())
                    : ImGui::TreeNodeEx("##Prim", flags, "%s",
                                        prim->getName().c_str());
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left))
                _app->selectPrim(prim);
            if (ImGui::BeginPopupContextItem("PrimContextMenu")) {
                const bool rootPrim = prim->getPath() == "/";
                const bool engineOwned = isEngineOwnedPrim(prim);
                const bool subtreeHasEngineOwned =
                    subtreeHasEngineOwnedPrim(prim);
                const bool resourceNamespace = isResourceNamespacePrim(prim);
                const bool subtreeHasResourceNamespace =
                    subtreeHasResourceNamespacePrim(prim);
                const bool subtreeHasExternal =
                    subtreeHasExternalPrim(_app, prim);
                const bool canDelete =
                    !rootPrim && !engineOwned && !subtreeHasEngineOwned &&
                    !resourceNamespace && !subtreeHasResourceNamespace &&
                    !subtreeHasExternal;
                if (!canDelete)
                    ImGui::BeginDisabled();
                if (ImGui::MenuItem("Delete...")) {
                    _pendingDeletePath = prim->getPath();
                    _deletePopupRequested = true;
                }
                if (!canDelete)
                    ImGui::EndDisabled();
                if (rootPrim)
                    ImGui::TextDisabled("Root prim cannot be deleted.");
                else if (engineOwned || subtreeHasEngineOwned)
                    ImGui::TextDisabled(
                        "Subtree contains engine-owned default light.");
                else if (resourceNamespace || subtreeHasResourceNamespace)
                    ImGui::TextDisabled("Resource mirrors are managed by "
                                        "SceneResourceManager.");
                else if (subtreeHasExternal)
                    ImGui::TextDisabled(
                        "Subtree contains ExternalBuffer prims.");
                ImGui::EndPopup();
            }
            if (external &&
                ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::SetTooltip("External Buffer (read-only transform)");
            }
            if (unusedResource &&
                ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::SetTooltip("Unused resource");
            }

            ImGui::TableSetColumnIndex(1);
            ImGui::TextDisabled("%s", primTypeLabel(prim->getType()));

            if (customTextColor)
                ImGui::PopStyleColor();

            if (open && !children.empty()) {
                for (auto* child : children)
                    self(self, child);
                ImGui::TreePop();
            }
            ImGui::PopID();
        };
        if (ImGui::BeginTable("ScenePrimTree", 2,
                              ImGuiTableFlags_Resizable |
                                  ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Scene",
                                    ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed,
                                    96.0f);
            ImGui::TableHeadersRow();
            for (auto* child : root->getChildren())
                drawPrimTree(drawPrimTree, child);
            ImGui::EndTable();
        }
    }

    if (_deletePopupRequested) {
        ImGui::OpenPopup("Delete Prim");
        _deletePopupRequested = false;
    }
    if (ImGui::BeginPopupModal("Delete Prim", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Delete this prim and all children?");
        ImGui::Spacing();
        ImGui::TextWrapped("%s", _pendingDeletePath.c_str());
        ImGui::Spacing();
        if (ImGui::Button("Cancel")) {
            _pendingDeletePath.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::Button("Delete")) {
            if (_app && !_pendingDeletePath.empty())
                _app->removePrim(_pendingDeletePath);
            _pendingDeletePath.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleColor();
        ImGui::EndPopup();
    }
    ImGui::End();
}

} // namespace KE
