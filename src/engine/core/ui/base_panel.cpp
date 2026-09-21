#include "base_panel.hpp"
#define IMVIEWGUIZMO_IMPLEMENTATION
#include "ImViewGuizmo.h"
#include "imgui.h"
#include <implot.h>
#include "engine/core/app/app.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/material/material.hpp"
#include "engine/graphics/material/colors.hpp"
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
#include <array>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
#include <map>
#include <limits>
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

void PerformancePanel::buildTimingPlot() {
    if (!ImGui::CollapsingHeader("Performance Profiler",
                                 ImGuiTreeNodeFlags_DefaultOpen))
        return;
    if (!_profileTargetInitialized) {
        // Timing can be configured after the panel is constructed.
        const float hz = _app->getRenderHz();
        _profileTargetFps = std::isfinite(hz) && hz > 0.0f ? hz : 0.0f;
        _profileTargetInitialized = true;
    }
    constexpr size_t capacity = RendererProfiler::HistoryCapacity;
    struct Series {
        const char* label;
        ColorType color;
        const char* tooltip;
        std::array<double, capacity> values;
        double sum = 0;
        size_t valid = 0, latest = 0;
    };
    std::array<Series, 5> series{
        {{"Total", ColorType::GOLD,
          "Interval between frame starts, including pacing and present."},
         {"CPU", ColorType::SKY_BLUE,
          "CPU frame wall time, including present; excludes pacing outside the "
          "frame."},
         {"GPU", ColorType::ORCHID,
          "Latest available GPU frame interval; includes idle gaps, excludes "
          "present/CUDA. Results arrive several frames later."},
         {"Update", ColorType::LIME_GREEN,
          "Sum of pre-update, fixed-update and pre-render callbacks."},
         {"Present", ColorType::ORANGE,
          "CPU time in buffer swap. This is not a measurement of GPU-only "
          "waiting."}}};
    for (auto& s : series)
        s.values.fill(std::numeric_limits<double>::quiet_NaN());
    std::array<double, capacity> frames{};
    const auto history = _app->getRenderer().frameProfileHistory();
    size_t count = 0;
    if (!history.empty()) {
        const auto& latest = history.back();
        for (const auto& frame : history) {
            if (frame->captureId != latest->captureId ||
                latest->frameIndex - frame->frameIndex >= capacity ||
                count == capacity)
                continue;
            frames[count] =
                -static_cast<double>(latest->frameIndex - frame->frameIndex);
            const auto interval = frame->metadata.find("frame_interval_ms");
            if (interval != frame->metadata.end()) {
                double value = 0;
                const auto& text = interval->second;
                const auto parsed = std::from_chars(
                    text.data(), text.data() + text.size(), value);
                if (parsed.ec == std::errc{} &&
                    parsed.ptr == text.data() + text.size() &&
                    std::isfinite(value) && value > 0)
                    series[0].values[count] = value;
            }
            double update = 0;
            bool preUpdate = false, preRender = false, updateComplete = true;
            for (const auto& sample : frame->samples) {
                const bool ready = sample.available() && sample.durationMs &&
                                   std::isfinite(*sample.durationMs);
                if (sample.domain == Backend::ProfileTimingDomain::Cpu) {
                    if (sample.path == "frame" && ready)
                        series[1].values[count] = *sample.durationMs;
                    if (sample.path == "present" && ready)
                        series[4].values[count] = *sample.durationMs;
                    if (sample.path == "simulation/pre_update" ||
                        sample.path == "simulation/fixed_update" ||
                        sample.path == "simulation/pre_render") {
                        preUpdate |= sample.path == "simulation/pre_update";
                        preRender |= sample.path == "simulation/pre_render";
                        if (ready)
                            update += *sample.durationMs;
                        else
                            updateComplete = false;
                    }
                } else if (sample.path == "frame" && ready) {
                    series[2].values[count] = *sample.durationMs;
                }
            }
            if (preUpdate && preRender && updateComplete &&
                !frame->droppedSamples)
                series[3].values[count] = update;
            for (auto& s : series) {
                if (std::isfinite(s.values[count])) {
                    s.sum += s.values[count];
                    ++s.valid;
                    s.latest = count;
                }
            }
            ++count;
        }
    }
    const auto color = [](ColorType type) {
        const auto& c = ColorLibrary::get(type);
        return ImVec4(c.r, c.g, c.b, c.a);
    };
    const bool sideBySide =
        ImGui::GetContentRegionAvail().x >= ImGui::GetFontSize() * 32;
    const bool layout =
        sideBySide && ImGui::BeginTable("Timing layout", 2,
                                        ImGuiTableFlags_SizingStretchProp);
    if (layout) {
        ImGui::TableSetupColumn("Summary", ImGuiTableColumnFlags_WidthStretch,
                                0.57f);
        ImGui::TableSetupColumn("History", ImGuiTableColumnFlags_WidthStretch,
                                0.43f);
        ImGui::TableNextColumn();
    }
    if (ImGui::BeginTable("Timing values", 2,
                          ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Timing", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Current / average");
        for (const auto& s : series) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(color(s.color), "%s", s.label);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", s.tooltip);
            ImGui::TableNextColumn();
            if (s.valid) {
                ImGui::TextColored(color(s.color), "%.2f ms (Avg: %.2f ms)",
                                   s.values[s.latest], s.sum / s.valid);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "%zu valid samples; latest is %.0f frames behind. %s",
                        s.valid, -frames[s.latest], s.tooltip);
            } else {
                ImGui::TextDisabled("N/A (Avg: N/A)");
            }
        }
        ImGui::EndTable();
    }
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 4);
    ImGui::DragFloat("Target FPS", &_profileTargetFps, 1.0f, 0.0f, 1000.0f,
                     "%.2f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Initially uses render Hz. Visual budget only; zero "
                          "disables the target.");
    const double targetMs =
        _profileTargetFps > 0 ? 1000.0 / _profileTargetFps : 0.0;
    ImGui::SameLine();
    if (targetMs > 0)
        ImGui::TextColored(color(ColorType::SALMON), "%.2f ms", targetMs);
    else
        ImGui::TextDisabled("Unlimited");
    if (layout)
        ImGui::TableNextColumn();
    ImPlot::PushStyleVar(ImPlotStyleVar_FitPadding, ImVec2(0.0f, 0.2f));
    if (ImPlot::BeginPlot("##Frame timings",
                          ImVec2(-1, ImGui::GetTextLineHeightWithSpacing() * 8),
                          ImPlotFlags_NoLegend | ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes("Recent frames", "ms", ImPlotAxisFlags_NoInitialFit,
                          ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, -static_cast<double>(capacity - 1),
                                0, ImPlotCond_Always);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0, DBL_MAX);
        for (const auto& s : series) {
            ImPlot::SetNextLineStyle(color(s.color));
            // NaNs leave gaps, while asynchronous revisions retain the original
            // frame position.
            ImPlot::PlotLine(s.label, frames.data(), s.values.data(),
                             static_cast<int>(count));
        }
        if (targetMs > 0) {
            const double x[] = {-static_cast<double>(capacity - 1), 0};
            const double y[] = {targetMs, targetMs};
            ImPlot::SetNextLineStyle(color(ColorType::SALMON), 2);
            ImPlot::PlotLine("Target", x, y, 2);
        }
        ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
    if (layout)
        ImGui::EndTable();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled(
        "Recent 240 frames; timings overlap. GPU results are delayed.");
    ImGui::PopTextWrapPos();
}

void PerformancePanel::buildMemoryUsage(const char* label,
                                        const std::string& key,
                                        std::optional<uint64_t> system,
                                        std::optional<uint64_t> process,
                                        std::optional<uint64_t> capacity,
                                        const char* tooltip) {
    ImGui::PushID(key.c_str());
    ImGui::SeparatorText(label);
    auto& peaks = _memoryPeaks[key];
    const auto row = [&](const char* name, std::optional<uint64_t> used,
                         std::optional<uint64_t>& peak, ColorType type) {
        constexpr double gib = 1024.0 * 1024.0 * 1024.0;
        if (used)
            peak = std::max(peak.value_or(0), *used);
        ImGui::TextUnformatted(name);
        ImGui::SameLine();
        if (used && capacity)
            ImGui::Text("%.2f / %.2f GiB", *used / gib, *capacity / gib);
        else
            ImGui::TextDisabled("N/A");
        ImGui::SameLine();
        if (peak)
            ImGui::Text("(Peak: %.2f GiB)", *peak / gib);
        const auto& color = ColorLibrary::get(type);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              ImVec4(color.r, color.g, color.b, color.a));
        const float fraction = used && capacity && *capacity
                                   ? static_cast<float>(std::clamp(
                                         double(*used) / *capacity, 0.0, 1.0))
                                   : 0.0f;
        ImGui::ProgressBar(fraction, ImVec2(-1, ImGui::GetFontSize() * 0.45f),
                           "");
        ImGui::PopStyleColor();
        if (peak && capacity && *capacity) {
            const auto lo = ImGui::GetItemRectMin();
            const auto hi = ImGui::GetItemRectMax();
            const float x =
                lo.x + (hi.x - lo.x) *
                           static_cast<float>(
                               std::clamp(double(*peak) / *capacity, 0.0, 1.0));
            const auto& marker = ColorLibrary::get(ColorType::ORANGE);
            ImGui::GetWindowDrawList()->AddLine(
                ImVec2(x, lo.y), ImVec2(x, hi.y),
                ImGui::ColorConvertFloat4ToU32(
                    ImVec4(marker.r, marker.g, marker.b, marker.a)),
                2);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "%s\nUsed / physical capacity, not a process budget.\n"
                "Orange marker: peak observed while this panel is "
                "sampling.\n%s",
                name, tooltip);
    };
    row("System", system, peaks.system, ColorType::SKY_BLUE);
    row("This process", process, peaks.process, ColorType::LIME_GREEN);
    ImGui::PopID();
}

void PerformancePanel::buildResourceUsage() {
    if (!ImGui::CollapsingHeader("Resources", ImGuiTreeNodeFlags_DefaultOpen))
        return;
    auto& renderer = _app->getRenderer();
    _resourceMonitor.requestSample();
    const auto usage = _resourceMonitor.snapshot();
    if (!usage.sequence) {
        ImGui::TextDisabled("Waiting for resource measurements...");
    } else {
        const double age =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          usage.sampledAt)
                .count();
        if (age > 2.5)
            ImGui::TextDisabled("Refresh pending (%.1f s old)", age);
        if (ImGui::BeginTable("CPU usage", 3,
                              ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Resource",
                                    ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("System");
            ImGui::TableSetupColumn("This process");
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("CPU");
            ImGui::TableNextColumn();
            if (usage.systemCpuPercent)
                ImGui::Text("%.2f%%", *usage.systemCpuPercent);
            else
                ImGui::TextDisabled("N/A");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Total host CPU usage. Sampled about once per "
                    "second; needs two samples.\n%s",
                    usage.status.c_str());
            ImGui::TableNextColumn();
            if (usage.processCpuPercent && usage.logicalCpuCount)
                ImGui::Text("%.2f%%",
                            *usage.processCpuPercent / usage.logicalCpuCount);
            else
                ImGui::TextDisabled("N/A");
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(
                    "This PID only, normalized to total host CPU capacity.");
                if (usage.processCpuPercent)
                    ImGui::Text("Per core: %.2f%% (100%% = one logical CPU)",
                                *usage.processCpuPercent);
                ImGui::TextUnformatted(
                    "Sampled about once per second; needs two samples.");
                if (!usage.status.empty())
                    ImGui::TextWrapped("%s", usage.status.c_str());
                ImGui::EndTooltip();
            }
            ImGui::EndTable();
        }
        if (usage.gpus.empty()) {
            ImGui::TextDisabled("GPU / VRAM: N/A");
            if (ImGui::IsItemHovered() && !usage.gpuStatus.empty())
                ImGui::SetTooltip("%s", usage.gpuStatus.c_str());
        }
        for (const auto& gpu : usage.gpus) {
            const auto label = "VRAM " + std::to_string(gpu.index);
            const auto key =
                "gpu/" +
                (gpu.uuid.empty() ? std::to_string(gpu.index) : gpu.uuid);
            std::string note =
                gpu.memoryIncludesReserved
                    ? "Legacy device counter includes driver reservations. "
                    : "Device counter excludes driver reservations. ";
            note += gpu.processMemoryStatus;
            if (!gpu.status.empty())
                note += "\n" + gpu.status;
            buildMemoryUsage(label.c_str(), key, gpu.memoryUsedBytes,
                             gpu.processMemoryBytes, gpu.memoryTotalBytes,
                             note.c_str());
            if (gpu.utilizationPercent)
                ImGui::Text("GPU %u total activity: %.2f%%", gpu.index,
                            double(*gpu.utilizationPercent));
            else
                ImGui::TextDisabled("GPU %u total activity: N/A", gpu.index);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s; includes all applications.\n%s",
                                  gpu.name.c_str(), gpu.status.c_str());
        }
        const auto ramUsed =
            usage.ramTotalBytes && usage.ramAvailableBytes
                ? std::optional<uint64_t>(*usage.ramTotalBytes -
                                          *usage.ramAvailableBytes)
                : std::nullopt;
        const std::string ramNote =
            "System = total - available; process = RSS (may include shared "
            "pages). Host RAM is not a container budget.\n" +
            usage.status;
        buildMemoryUsage("RAM Usage", "ram", ramUsed, usage.processRssBytes,
                         usage.ramTotalBytes, ramNote.c_str());
    }

    if (ImGui::TreeNode("Hardware details")) {
        if (const auto* device = renderer.device()) {
            const auto backend = device->getBackendType();
            ImGui::Text("Render backend: %s",
                        backend == Backend::BackendType::OpenGL   ? "OpenGL"
                        : backend == Backend::BackendType::WebGPU ? "WebGPU"
                                                                  : "Vulkan");
            const auto& metadata = device->profileContext()->deviceMetadata();
            auto deviceText = [&](const char* label, const char* key) {
                const auto found = metadata.find(key);
                ImGui::TextWrapped(
                    "%s: %s", label,
                    found == metadata.end() ? "N/A" : found->second.c_str());
            };
            deviceText("Render device", "gpu");
            deviceText("Vendor", "vendor");
            deviceText("Driver / GL version", "driver_gl_version");
        }
        ImGui::TextWrapped("OS: %s",
                           usage.os.empty() ? "N/A" : usage.os.c_str());
        ImGui::TextWrapped(
            "CPU: %s", usage.cpuModel.empty() ? "N/A" : usage.cpuModel.c_str());
        if (usage.logicalCpuCount)
            ImGui::Text("Logical CPUs: %u", usage.logicalCpuCount);
        if (!usage.gpuDriver.empty())
            ImGui::Text("NVIDIA driver: %s", usage.gpuDriver.c_str());
        for (const auto& gpu : usage.gpus)
            ImGui::TextWrapped("GPU %u: %s", gpu.index, gpu.name.c_str());
        ImGui::TextWrapped("GPU indices above are not automatically "
                           "matched to the render device.");
        ImGui::TreePop();
    }
}

void PerformancePanel::buildPanel() {
    if (!ImGui::Begin(name().c_str(), openPtr())) {
        ImGui::End();
        return;
    }
    if (_app) {
        bool vsync = _app->getVSync();
        if (ImGui::Checkbox("VSync", &vsync))
            _app->setVSync(vsync);
        ImGui::SameLine();
        ImGui::Text("Displayed FPS: %.2f", _app->getMeasuredRenderFPS());
        auto& renderer = _app->getRenderer();
        bool enabled = renderer.profilerEnabled();
        if (ImGui::Checkbox("Renderer Profiler", &enabled))
            renderer.setProfilerEnabled(enabled);
        if (renderer.latestFrameProfile())
            buildTimingPlot();
        else
            ImGui::TextDisabled(
                "Enable Renderer Profiler to collect frame timings.");
        buildResourceUsage();
        if (const auto frame = renderer.latestFrameProfile();
            frame && ImGui::CollapsingHeader("Profiler details")) {
            ImGui::Text("Capture %llu / frame %llu",
                        (unsigned long long)frame->captureId,
                        (unsigned long long)frame->frameIndex);
            ImGui::TextDisabled("RHI counters exclude ImGui");
            const auto& c = frame->counters;
            ImGui::Text("Draws: %llu  Instances: %llu  Triangles: %llu",
                        (unsigned long long)c.drawCalls,
                        (unsigned long long)c.instances,
                        (unsigned long long)c.triangles);
            ImGui::Text("Upload: %llu bytes  Allocations: %llu",
                        (unsigned long long)(c.bufferUploadBytes +
                                             c.textureUploadBytes),
                        (unsigned long long)c.bufferAllocations);
            if (ImGui::TreeNode("Scope statistics")) {
                ImGui::Combo("Statistic", &_profileStatistic,
                             "Mean\0Median\0p95\0Max\0");
                ImGui::SliderInt("Recent frames", &_profileWindow, 1, 240);
                const auto summary = renderer.profileSummary(_profileWindow);
                ImGui::Text("Capture %llu: %llu frames",
                            (unsigned long long)summary.captureId,
                            (unsigned long long)summary.frameCount);
                ImGui::TextDisabled("Per-frame inclusive sums; "
                                    "absent/incomplete scopes excluded");
                if (ImGui::BeginTable("Scope summary", 5,
                                      ImGuiTableFlags_Borders |
                                          ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Scope");
                    ImGui::TableSetupColumn("ms");
                    ImGui::TableSetupColumn("Ready");
                    ImGui::TableSetupColumn("Pending");
                    ImGui::TableSetupColumn("Unavailable");
                    ImGui::TableHeadersRow();
                    for (const auto& scope : summary.scopes) {
                        const auto value =
                            _profileStatistic == 1   ? scope.medianMs
                            : _profileStatistic == 2 ? scope.p95Ms
                            : _profileStatistic == 3 ? scope.maxMs
                                                     : scope.meanMs;
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("%s %s",
                                    scope.domain ==
                                            Backend::ProfileTimingDomain::Cpu
                                        ? "CPU"
                                        : "GPU",
                                    scope.path.c_str());
                        ImGui::TableNextColumn();
                        if (value)
                            ImGui::Text("%.3f", *value);
                        else
                            ImGui::TextUnformatted("--");
                        ImGui::TableNextColumn();
                        ImGui::Text("%llu",
                                    (unsigned long long)scope.readyFrames);
                        ImGui::TableNextColumn();
                        ImGui::Text("%llu",
                                    (unsigned long long)scope.pendingFrames);
                        ImGui::TableNextColumn();
                        ImGui::Text(
                            "%llu",
                            (unsigned long long)scope.unavailableFrames);
                    }
                    ImGui::EndTable();
                }
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("CPU scopes (inclusive)")) {
                ImGui::TextDisabled("Latest frame: total per name / calls; "
                                    "expand for individual calls");
                ImGui::TextDisabled(
                    "Inclusive times overlap; do not sum different scopes");
                std::map<std::string,
                         std::vector<const Backend::ProfileSample*>>
                    groups;
                for (const auto& sample : frame->samples)
                    if (sample.domain == Backend::ProfileTimingDomain::Cpu)
                        groups[sample.path].push_back(&sample);
                for (const auto& [path, samples] : groups) {
                    double totalMs = 0;
                    bool complete = frame->droppedSamples == 0;
                    for (const auto* sample : samples) {
                        if (sample->available() && sample->durationMs)
                            totalMs += *sample->durationMs;
                        else
                            complete = false;
                    }
                    // The path, rather than changing timings/counts, owns the
                    // ImGui ID so expansion survives subsequent frames.
                    const bool expanded =
                        complete ? ImGui::TreeNodeEx(
                                       path.c_str(), ImGuiTreeNodeFlags_None,
                                       "%s: %.3f ms (%zu calls)", path.c_str(),
                                       totalMs, samples.size())
                                 : ImGui::TreeNodeEx(
                                       path.c_str(), ImGuiTreeNodeFlags_None,
                                       "%s: total unavailable (%zu calls)",
                                       path.c_str(), samples.size());
                    if (!expanded)
                        continue;
                    if (frame->droppedSamples)
                        ImGui::TextDisabled(
                            "Capture overflow: some calls may be missing");
                    for (const auto* sample : samples) {
                        const char* parent = "root";
                        if (sample->parentSampleId &&
                            *sample->parentSampleId < frame->samples.size())
                            parent = frame->samples[*sample->parentSampleId]
                                         .path.c_str();
                        if (sample->available() && sample->durationMs)
                            ImGui::Text("#%llu: %.3f ms (in %s)",
                                        (unsigned long long)sample->sampleId,
                                        *sample->durationMs, parent);
                        else
                            ImGui::TextDisabled(
                                "#%llu: %s (in %s)",
                                (unsigned long long)sample->sampleId,
                                sample->status ==
                                        Backend::ProfileSampleStatus::Pending
                                    ? "pending"
                                    : "unavailable",
                                parent);
                    }
                    ImGui::TreePop();
                }
                ImGui::TreePop();
            }
            if (!renderer.profilerCapabilities().passTimestamps) {
                ImGui::TextDisabled("GPU pass timing unavailable");
            } else if (ImGui::TreeNode("GPU timings (inclusive)")) {
                const auto history = renderer.frameProfileHistory();
                auto found = std::find_if(
                    history.rbegin(), history.rend(), [&](const auto& f) {
                        return f->captureId == frame->captureId &&
                               f->finalized &&
                               std::any_of(
                                   f->samples.begin(), f->samples.end(),
                                   [](const auto& sample) {
                                       return sample.domain ==
                                                  Backend::ProfileTimingDomain::
                                                      Gpu &&
                                              sample.available();
                                   });
                    });
                if (found == history.rend())
                    ImGui::TextUnformatted("Waiting for GPU results");
                else {
                    ImGui::Text("Measured frame %llu",
                                (unsigned long long)(*found)->frameIndex);
                    if ((*found)->gpuLatencyFrames)
                        ImGui::Text("Result delay: %u frames",
                                    *(*found)->gpuLatencyFrames);
                    else
                        ImGui::TextDisabled("Some GPU scopes unavailable");
                    for (const auto& sample : (*found)->samples)
                        if (sample.domain ==
                                Backend::ProfileTimingDomain::Gpu &&
                            sample.available())
                            ImGui::Text("%s: %.3f ms", sample.path.c_str(),
                                        *sample.durationMs);
                }
                if (renderer.profilerCapabilities().externalTimestamps)
                    ImGui::TextDisabled(
                        "Frame: GL interval before swap, includes idle gaps");
                else
                    ImGui::TextDisabled(
                        "Whole-frame/UI/native timing unavailable");
                ImGui::TextDisabled(
                    "Excludes present/CUDA timing; do not sum scopes");
                ImGui::TreePop();
            }
        }
    }
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
