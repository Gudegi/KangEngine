#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_

#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/graphics/backend/graphics_factory.hpp"
#include "engine/graphics/material/pbrMaterials.hpp"
#include "engine/graphics/material/phongMaterials.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"

#include <glm/glm.hpp>

namespace KE {

class Material {
  protected:
    Backend::Shader* _shader = nullptr;

  public:
    virtual ~Material() = default;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual bool hasNormalMap() const { return false; }
    virtual Backend::Shader* getShader() const { return _shader; }
    virtual void setShader(Backend::Shader* shader) { _shader = shader; }
};

class PhongMaterial : public Material {
  public:
    // Properties
    glm::vec3 ambient = glm::vec3(0.1f);
    glm::vec3 diffuse = glm::vec3(1.0f);
    glm::vec3 specular = glm::vec3(0.5f);
    float shininess = 32.0f;

    // Textures
    Backend::Texture* diffuseMap = nullptr;
    Backend::Texture* specularMap = nullptr;
    Backend::Texture* normalMap = nullptr;

    PhongMaterial* setAmbient(glm::vec3 v) {
        ambient = v;
        return this;
    }
    PhongMaterial* setDiffuse(glm::vec3 v) {
        diffuse = v;
        return this;
    }

    bool hasNormalMap() const override { return normalMap != nullptr; }

    void bind() override {
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
    glm::vec4 baseColor = glm::vec4(1.0f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    glm::vec3 emissiveColor = glm::vec3(0.0f);
    float emissiveStrength = 0.0f;

    Backend::Texture* baseColorTexture = nullptr;
    Backend::Texture* normalTexture = nullptr;
    Backend::Texture* metallicRoughnessTexture = nullptr;
    Backend::Texture* aoTexture = nullptr;
    Backend::Texture* emissiveTexture = nullptr;

    bool hasNormalMap() const override { return normalTexture != nullptr; }

    void bind() override {
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
        bindTexture("uAoMap", "useAoMap", aoTexture,
                    RendererTextureSlot::AmbientOcclusion);
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
