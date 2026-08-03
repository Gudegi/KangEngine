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
};

class Material {
  protected:
    Backend::Shader* _shader = nullptr;

  public:
    virtual ~Material() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;
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
    virtual Backend::Shader* getShader() const { return _shader; }
    virtual void setShader(Backend::Shader* shader) { _shader = shader; }
};

// Compatibility material for legacy shader-only renderables.
//
// It intentionally owns no surface parameters: common.fs/commonTex.fs consume
// per-instance vColor and any texture slots that MeshInstancer binds. This lets
// the scene/render path treat shader-only objects as material-backed without
// changing their visual behavior.
class VertexColorMaterial : public Material {
  private:
    VertexColorStyle _style = VertexColorStyle::Untextured;

  public:
    explicit VertexColorMaterial(Backend::Shader* shader = nullptr) {
        setShader(shader);
        if (!shader)
            return;
        const std::string& name = shader->getName();
        if (name.find("shaders/checkerboard.fs") != std::string::npos)
            _style = VertexColorStyle::Checkerboard;
        else if (name.find("shaders/commonTex.fs") != std::string::npos)
            _style = VertexColorStyle::Textured;
    }

    void bind() override {
        if (_shader)
            _shader->use();
    }

    void unbind() override {}
    MaterialShadingModel shadingModel() const override {
        return MaterialShadingModel::VertexColor;
    }
    VertexColorStyle vertexColorStyle() const override { return _style; }
    void setStyle(VertexColorStyle style) { _style = style; }
};

class PhongMaterial : public Material {
  public:
    PhongMaterial() = default;
    explicit PhongMaterial(Backend::Shader* shader) { setShader(shader); }

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

    void bind() override {
        if (!_shader)
            return;
        _shader->use();
        _shader->setVec3("material.ambient", ambient);
        _shader->setVec3("material.diffuse", diffuse);
        _shader->setVec3("material.specular", specular);
        _shader->setFloat("material.shininess", shininess);

        if (diffuseMap) {
            diffuseMap->bind(RendererTextureSlot::Diffuse);
            _shader->setInt("uTexture", RendererTextureSlot::Diffuse);
            _shader->setInt("material.diffuseMap",
                            RendererTextureSlot::Diffuse);
        }
        _shader->setInt("useDiffuseMap", diffuseMap ? 1 : 0);
        if (specularMap) {
            specularMap->bind(RendererTextureSlot::Specular);
            _shader->setInt("specularMap", RendererTextureSlot::Specular);
            _shader->setInt("material.specularMap",
                            RendererTextureSlot::Specular);
        }
        _shader->setInt("useSpecularMap", specularMap ? 1 : 0);
        if (alphaMap) {
            alphaMap->bind(RendererTextureSlot::Alpha);
            _shader->setInt("alphaMap", RendererTextureSlot::Alpha);
        }
        _shader->setInt("useAlphaMap", alphaMap ? 1 : 0);
        if (normalMap) {
            normalMap->bind(RendererTextureSlot::Normal);
            _shader->setInt("normalMap", RendererTextureSlot::Normal);
        }
        _shader->setInt("useNormalMap", normalMap ? 1 : 0);
    }

    void unbind() override {}

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

    static PhongMaterial* createFromPreset(PhongMaterialType type,
                                           Backend::Shader* shader) {
        auto material = new PhongMaterial();
        material->setShader(shader);
        material->loadFromPreset(type);
        return material;
    }
};

class PBRMaterial : public Material {
  public:
    PBRMaterial() = default;
    explicit PBRMaterial(Backend::Shader* shader) { setShader(shader); }

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

    void bind() override {
        if (!_shader)
            return;
        _shader->use();
        _shader->setVec4("uBaseColorFactor", baseColor);
        _shader->setFloat("uMetallicFactor", metallic);
        _shader->setFloat("uRoughnessFactor", roughness);
        _shader->setVec3("uEmissiveColor", emissiveColor);
        _shader->setFloat("uEmissiveStrength", emissiveStrength);

        bindTexture("uBaseColorMap", "useBaseColorMap", baseColorTexture,
                    RendererTextureSlot::BaseColor);
        bindTexture("normalMap", "useNormalMap", normalTexture,
                    RendererTextureSlot::Normal);
        bindTexture("uMetallicRoughnessMap", "useMetallicRoughnessMap",
                    metallicRoughnessTexture,
                    RendererTextureSlot::MetallicRoughness);
        bindTexture("uMetallicMap", "useMetallicMap", metallicTexture,
                    RendererTextureSlot::Metallic);
        bindTexture("uRoughnessMap", "useRoughnessMap", roughnessTexture,
                    RendererTextureSlot::Roughness);
        bindTexture("uAoMap", "useAoMap", aoTexture,
                    RendererTextureSlot::AmbientOcclusion);
        bindTexture("uOrmMap", "useOrmMap", ormTexture,
                    RendererTextureSlot::OcclusionRoughnessMetallic);
        bindTexture("uEmissiveMap", "useEmissiveMap", emissiveTexture,
                    RendererTextureSlot::Emissive);
    }

    void unbind() override {}

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

    static PBRMaterial* createFromPreset(PBRMaterialType type,
                                         Backend::Shader* shader) {
        auto material = new PBRMaterial();
        material->setShader(shader);
        material->loadFromPreset(type);
        return material;
    }

  private:
    void bindTexture(const char* samplerName, const char* flagName,
                     Backend::Texture* texture, int slot) {
        if (texture) {
            texture->bind(slot);
            _shader->setInt(samplerName, slot);
        }
        _shader->setInt(flagName, texture ? 1 : 0);
    }
};

} // namespace KE

#endif
