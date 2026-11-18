#pragma once

// Material models and CPU values used by Native shading and render adapters.

#include <glm/glm.hpp>

namespace KE {

enum class MaterialShadingModel {
    Color,
    Phong,
    PBR,
    Custom,
    VertexColor = Color,
};

enum class ColorStyle {
    Untextured,
    Textured,
    Checkerboard,
    DebugChecker,
    Unlit,
};

namespace Scene {

// Material values in linear RGB, independent of GPU bindings.
struct RenderMaterialData {
    MaterialShadingModel shadingModel = MaterialShadingModel::Custom;
    ColorStyle colorStyle = ColorStyle::Untextured;
    bool unlit = false;
    // Textures still require the Native adapter; values alone are incomplete.
    bool hasNativeTextures = false;
    glm::vec4 baseColor{1.0f};
    glm::vec3 ambient{0.1f};
    glm::vec3 specular{0.5f};
    float shininess = 32.0f;
    float metallic = 0.0f;
    float roughness = 0.5f;
    glm::vec3 emissiveColor{0.0f};
    float emissiveStrength = 0.0f;
};

} // namespace Scene
} // namespace KE
