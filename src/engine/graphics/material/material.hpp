#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/backend/graphics_factory.hpp"
#include "engine/graphics/material/pbrMaterials.hpp"
#include "engine/graphics/material/phongMaterials.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"

#include <glm/glm.hpp>

namespace KE {

enum class MaterialShadingModel {
    VertexColor,
    Phong,
    PBR,
    Custom,
};

enum class VertexColorStyle {
    Untextured,
    Textured,
    Checkerboard,
    DebugChecker,
};

class Material {
  public:
    virtual ~Material() = default;
    virtual bool hasNormalMap() const { return false; }
    // Texture sampled by depth-only passes when AlphaMode::Mask is active.
    virtual Backend::Texture* alphaTexture() const { return nullptr; }
    virtual bool alphaTextureUsesRedChannel() const { return false; }
    virtual MaterialShadingModel shadingModel() const {
        return MaterialShadingModel::Custom;
    }
    virtual VertexColorStyle vertexColorStyle() const {
        return VertexColorStyle::Untextured;
    }
};

class VertexColorMaterial : public Material {
  private:
    VertexColorStyle _style = VertexColorStyle::Untextured;

  public:
    explicit VertexColorMaterial(
        VertexColorStyle style = VertexColorStyle::Untextured)
        : _style(style) {}
    MaterialShadingModel shadingModel() const override {
        return MaterialShadingModel::VertexColor;
    }
    VertexColorStyle vertexColorStyle() const override { return _style; }
    void setStyle(VertexColorStyle style) { _style = style; }
};

class PhongMaterial : public Material {
  public:
    PhongMaterial() = default;

    // Properties
    glm::vec3 ambient = glm::vec3(0.1f);
    glm::vec3 diffuse = glm::vec3(1.0f);
    glm::vec3 specular = glm::vec3(0.5f);
    float shininess = 32.0f;

    // Textures
    Backend::Texture* diffuseMap = nullptr;
    Backend::Texture* specularMap = nullptr;
    Backend::Texture* alphaMap = nullptr;
    Backend::Texture* normalMap = nullptr;

    PhongMaterial* setAmbient(glm::vec3 v) {
        ambient = v;
        return this;
    }
    PhongMaterial* setDiffuse(glm::vec3 v) {
        diffuse = v;
        return this;
    }
    PhongMaterial* setSpecular(glm::vec3 v) {
        specular = v;
        return this;
    }
    PhongMaterial* setShininess(float v) {
        shininess = v;
        return this;
    }
    PhongMaterial* setDiffuseMap(Backend::Texture* texture) {
        diffuseMap = texture;
        return this;
    }
    PhongMaterial* setSpecularMap(Backend::Texture* texture) {
        specularMap = texture;
        return this;
    }
    PhongMaterial* setAlphaMap(Backend::Texture* texture) {
        alphaMap = texture;
        return this;
    }
    PhongMaterial* setNormalMap(Backend::Texture* texture) {
        normalMap = texture;
        return this;
    }

    bool hasNormalMap() const override { return normalMap != nullptr; }
    MaterialShadingModel shadingModel() const override {
        return MaterialShadingModel::Phong;
    }
    Backend::Texture* alphaTexture() const override {
        return alphaMap ? alphaMap : diffuseMap;
    }
    bool alphaTextureUsesRedChannel() const override {
        // for an alpha mask only texture(normally stored in red channel)
        return alphaMap != nullptr;
    }

    void loadFromPreset(PhongMaterialType type) {
        auto props = PhongMaterialLibrary::get(type);
        ambient =
            glm::vec3(props.ambient[0], props.ambient[1], props.ambient[2]);
        diffuse =
            glm::vec3(props.diffuse[0], props.diffuse[1], props.diffuse[2]);
        specular =
            glm::vec3(props.specular[0], props.specular[1], props.specular[2]);
        shininess = props.shininess * 128.0f;
    }

    static PhongMaterial* createFromPreset(PhongMaterialType type) {
        auto material = new PhongMaterial();
        material->loadFromPreset(type);
        return material;
    }
};

class PBRMaterial : public Material {
  public:
    PBRMaterial() = default;

    glm::vec4 baseColor = glm::vec4(1.0f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    glm::vec3 emissiveColor = glm::vec3(0.0f);
    float emissiveStrength = 0.0f;

    Backend::Texture* baseColorTexture = nullptr;
    Backend::Texture* normalTexture = nullptr;
    Backend::Texture* metallicRoughnessTexture = nullptr;
    Backend::Texture* metallicTexture = nullptr;
    Backend::Texture* roughnessTexture = nullptr;
    Backend::Texture* aoTexture = nullptr;
    Backend::Texture* ormTexture = nullptr;
    Backend::Texture* emissiveTexture = nullptr;

    bool hasNormalMap() const override { return normalTexture != nullptr; }
    MaterialShadingModel shadingModel() const override {
        return MaterialShadingModel::PBR;
    }
    Backend::Texture* alphaTexture() const override { return baseColorTexture; }

    PBRMaterial* setBaseColor(glm::vec4 v) {
        baseColor = v;
        return this;
    }
    PBRMaterial* setMetallic(float v) {
        metallic = v;
        return this;
    }
    PBRMaterial* setRoughness(float v) {
        roughness = v;
        return this;
    }
    PBRMaterial* setEmissiveColor(glm::vec3 v) {
        emissiveColor = v;
        return this;
    }
    PBRMaterial* setEmissiveStrength(float v) {
        emissiveStrength = v;
        return this;
    }
    PBRMaterial* setBaseColorTexture(Backend::Texture* texture) {
        baseColorTexture = texture;
        return this;
    }
    PBRMaterial* setNormalTexture(Backend::Texture* texture) {
        normalTexture = texture;
        return this;
    }
    PBRMaterial* setMetallicRoughnessTexture(Backend::Texture* texture) {
        metallicRoughnessTexture = texture;
        return this;
    }
    PBRMaterial* setMetallicTexture(Backend::Texture* texture) {
        metallicTexture = texture;
        return this;
    }
    PBRMaterial* setRoughnessTexture(Backend::Texture* texture) {
        roughnessTexture = texture;
        return this;
    }
    PBRMaterial* setAoTexture(Backend::Texture* texture) {
        aoTexture = texture;
        return this;
    }
    PBRMaterial* setOrmTexture(Backend::Texture* texture) {
        ormTexture = texture;
        return this;
    }
    PBRMaterial* setEmissiveTexture(Backend::Texture* texture) {
        emissiveTexture = texture;
        return this;
    }

    void loadFromPreset(PBRMaterialType type) {
        const auto& props = PBRMaterialLibrary::get(type);
        baseColor = glm::vec4(props.baseColor[0], props.baseColor[1],
                              props.baseColor[2], props.baseColor[3]);
        metallic = props.metallic;
        roughness = props.roughness;
        emissiveColor =
            glm::vec3(props.emissiveColor[0], props.emissiveColor[1],
                      props.emissiveColor[2]);
        emissiveStrength = props.emissiveStrength;
    }

    static PBRMaterial* createFromPreset(PBRMaterialType type) {
        auto material = new PBRMaterial();
        material->loadFromPreset(type);
        return material;
    }
};

} // namespace KE

#endif
