#ifndef _MATERIAL_HPP_
#define _MATERIAL_HPP_

#include <glm/glm.hpp>

#include "backend/base/graphics_device.hpp"
#include "backend/graphics_factory.hpp"
#include "material/phongMaterials.hpp"

namespace KE
{

class Material {
protected:
    Backend::Shader* _shader;
    //Backend::Texture* _texture;
public:
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual Backend::Shader* getShader() const { return _shader; }
    virtual void setShader(Backend::Shader* shader) { _shader = shader; }
}; // class Material

class PhongMaterial: public Material {
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

    // Setters 
    PhongMaterial* setAmbient(glm::vec3 v) { ambient = v; return this; }
    PhongMaterial* setDiffuse(glm::vec3 v) { diffuse = v; return this; }

    void bind() override {
        _shader->use();
        _shader->setVec3("material.ambient", ambient);
        _shader->setVec3("material.diffuse", diffuse);
        _shader->setVec3("material.specular", specular);
        _shader->setFloat("material.shininess", shininess);

        if (diffuseMap) {
            diffuseMap->bind(0);
            _shader->setInt("material.diffuseMap", 0);
        }
    }

    void unbind() override {
        // TODO: Implement unbind logic if needed
    }

    void loadFromPreset(PhongMaterialType type) {
        auto props = PhongMaterialLibrary::get(type);
        ambient = glm::vec3(props.ambient[0], props.ambient[1], props.ambient[2]);
        diffuse = glm::vec3(props.diffuse[0], props.diffuse[1], props.diffuse[2]);
        specular = glm::vec3(props.specular[0], props.specular[1], props.specular[2]);
        shininess = props.shininess * 128.0f;
    }

    // Factory method for creating material from preset
    static PhongMaterial* createFromPreset(PhongMaterialType type, Backend::Shader* shader) {
        auto material = new PhongMaterial();
        material->setShader(shader);
        material->loadFromPreset(type);
        return material;
    }
};


} //namespace KE


#endif