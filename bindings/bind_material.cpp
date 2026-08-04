///
/// Material and color Python bindings
///

#include "../src/kangEngine.hpp"
#include "../src/engine/graphics/material/colors.hpp"
#include "../src/engine/graphics/material/material.hpp"
#include "engine/graphics/material/phongMaterials.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace KE;

void bind_material(py::module& m) {
    py::enum_<PhongMaterialType>(m, "PhongMaterialType",
                                 "Built-in blin-phong based material presets.")
        .value("EMERALD", PhongMaterialType::EMERALD)
        .value("JADE", PhongMaterialType::JADE)
        .value("OBSIDIAN", PhongMaterialType::OBSIDIAN)
        .value("PEARL", PhongMaterialType::PEARL)
        .value("RUBY", PhongMaterialType::RUBY)
        .value("TURQUOISE", PhongMaterialType::TURQUOISE)
        .value("BRASS", PhongMaterialType::BRASS)
        .value("BRONZE", PhongMaterialType::BRONZE)
        .value("CHROME", PhongMaterialType::CHROME)
        .value("COPPER", PhongMaterialType::COPPER)
        .value("GOLD", PhongMaterialType::GOLD)
        .value("SILVER", PhongMaterialType::SILVER)
        .value("BLACK_PLASTIC", PhongMaterialType::BLACK_PLASTIC)
        .value("CYAN_PLASTIC", PhongMaterialType::CYAN_PLASTIC)
        .value("GREEN_PLASTIC", PhongMaterialType::GREEN_PLASTIC)
        .value("RED_PLASTIC", PhongMaterialType::RED_PLASTIC)
        .value("WHITE_PLASTIC", PhongMaterialType::WHITE_PLASTIC)
        .value("YELLOW_PLASTIC", PhongMaterialType::YELLOW_PLASTIC)
        .value("BLACK_RUBBER", PhongMaterialType::BLACK_RUBBER)
        .value("CYAN_RUBBER", PhongMaterialType::CYAN_RUBBER)
        .value("GREEN_RUBBER", PhongMaterialType::GREEN_RUBBER)
        .value("RED_RUBBER", PhongMaterialType::RED_RUBBER)
        .value("WHITE_RUBBER", PhongMaterialType::WHITE_RUBBER)
        .value("YELLOW_RUBBER", PhongMaterialType::YELLOW_RUBBER);

    py::enum_<PBRMaterialType>(m, "PBRMaterialType",
                               "Built-in physically based material presets.")
        .value("GRAY_CARD", PBRMaterialType::GRAY_CARD)
        .value("WHITE_PLASTIC", PBRMaterialType::WHITE_PLASTIC)
        .value("BLACK_PLASTIC", PBRMaterialType::BLACK_PLASTIC)
        .value("BLACK_RUBBER", PBRMaterialType::BLACK_RUBBER)
        .value("CHARCOAL", PBRMaterialType::CHARCOAL)
        .value("CARROT", PBRMaterialType::CARROT)
        .value("CONCRETE", PBRMaterialType::CONCRETE)
        .value("RED_BRICK", PBRMaterialType::RED_BRICK)
        .value("ALUMINUM", PBRMaterialType::ALUMINUM)
        .value("CHROME", PBRMaterialType::CHROME)
        .value("COPPER", PBRMaterialType::COPPER)
        .value("GOLD", PBRMaterialType::GOLD)
        .value("EMISSIVE_BLUE", PBRMaterialType::EMISSIVE_BLUE);

    // Materials & Colors
    py::class_<Color>(m, "Color")
        .def(py::init<>())
        .def_readwrite("r", &Color::r)
        .def_readwrite("g", &Color::g)
        .def_readwrite("b", &Color::b)
        .def_readwrite("a", &Color::a);

    py::enum_<ColorType>(m, "ColorType")
        .value("WHITE", ColorType::WHITE)
        .value("BLACK", ColorType::BLACK)
        .value("RED", ColorType::RED)
        .value("GREEN", ColorType::GREEN)
        .value("BLUE", ColorType::BLUE)
        .value("YELLOW", ColorType::YELLOW)
        .value("CYAN", ColorType::CYAN)
        .value("MAGENTA", ColorType::MAGENTA)
        .value("ORANGE", ColorType::ORANGE)
        .value("PURPLE", ColorType::PURPLE)
        .value("PINK", ColorType::PINK)
        .value("BROWN", ColorType::BROWN)
        .value("GRAY", ColorType::GRAY)
        .value("LIGHT_GRAY", ColorType::LIGHT_GRAY)
        .value("DARK_GRAY", ColorType::DARK_GRAY)
        .value("CORAL", ColorType::CORAL)
        .value("SALMON", ColorType::SALMON)
        .value("TOMATO", ColorType::TOMATO)
        .value("CRIMSON", ColorType::CRIMSON)
        .value("FOREST_GREEN", ColorType::FOREST_GREEN)
        .value("LIME_GREEN", ColorType::LIME_GREEN)
        .value("SEA_GREEN", ColorType::SEA_GREEN)
        .value("TEAL", ColorType::TEAL)
        .value("NAVY", ColorType::NAVY)
        .value("SKY_BLUE", ColorType::SKY_BLUE)
        .value("STEEL_BLUE", ColorType::STEEL_BLUE)
        .value("ROYAL_BLUE", ColorType::ROYAL_BLUE)
        .value("INDIGO", ColorType::INDIGO)
        .value("VIOLET", ColorType::VIOLET)
        .value("ORCHID", ColorType::ORCHID)
        .value("GOLD", ColorType::GOLD)
        .value("KHAKI", ColorType::KHAKI)
        .value("OLIVE", ColorType::OLIVE)
        .value("MAROON", ColorType::MAROON)
        .value("BEIGE", ColorType::BEIGE)
        .value("IVORY", ColorType::IVORY)
        .value("MINT", ColorType::MINT)
        .value("LAVENDER", ColorType::LAVENDER)
        .value("SLATE_GRAY", ColorType::SLATE_GRAY)
        .value("PASTEL_RED", ColorType::PASTEL_RED)
        .value("PASTEL_ORANGE", ColorType::PASTEL_ORANGE)
        .value("PASTEL_YELLOW", ColorType::PASTEL_YELLOW)
        .value("PASTEL_GREEN", ColorType::PASTEL_GREEN)
        .value("PASTEL_MINT", ColorType::PASTEL_MINT)
        .value("PASTEL_CYAN", ColorType::PASTEL_CYAN)
        .value("PASTEL_BLUE", ColorType::PASTEL_BLUE)
        .value("PASTEL_PURPLE", ColorType::PASTEL_PURPLE)
        .value("PASTEL_PINK", ColorType::PASTEL_PINK)
        .value("PASTEL_ROSE", ColorType::PASTEL_ROSE)
        .value("PASTEL_PEACH", ColorType::PASTEL_PEACH)
        .value("PASTEL_LAVENDER", ColorType::PASTEL_LAVENDER)
        .value("PASTEL_LILAC", ColorType::PASTEL_LILAC)
        .value("PASTEL_CORAL", ColorType::PASTEL_CORAL)
        .value("PASTEL_CREAM", ColorType::PASTEL_CREAM)
        .value("PASTEL_SKY", ColorType::PASTEL_SKY)
        .value("DARK_GREEN", ColorType::DARK_GREEN);

    py::class_<ColorLibrary>(m, "ColorLibrary")
        .def_static("get", &ColorLibrary::get, py::arg("type"));

    py::class_<Material>(m, "Material", "Base class for renderer materials.");

    py::enum_<VertexColorStyle>(m, "VertexColorStyle")
        .value("UNTEXTURED", VertexColorStyle::Untextured)
        .value("TEXTURED", VertexColorStyle::Textured)
        .value("CHECKERBOARD", VertexColorStyle::Checkerboard)
        .value("DEBUG_CHECKER", VertexColorStyle::DebugChecker);

    py::class_<VertexColorMaterial, Material>(
        m, "VertexColorMaterial",
        "Vertex/display-color material with a built-in rendering style.")
        .def(py::init<VertexColorStyle>(),
             py::arg("style") = VertexColorStyle::Untextured)
        .def("set_style", &VertexColorMaterial::setStyle, py::arg("style"))
        .def_property_readonly("style", &VertexColorMaterial::vertexColorStyle);

    py::class_<PhongMaterial, Material>(
        m, "PhongMaterial",
        "Classic Blinn-Phong material with diffuse/specular factors and "
        "optional textures.")
        .def(py::init<>(), "Create a Phong material with default factors.")
        .def(
            "load_from_preset",
            [](PhongMaterial& self, PhongMaterialType type) -> PhongMaterial& {
                self.loadFromPreset(type);
                return self;
            },
            py::arg("type"), py::return_value_policy::reference_internal,
            "Load material factors from a built-in Phong preset and return "
            "this material.")
        .def("set_ambient", &PhongMaterial::setAmbient, py::arg("ambient"),
             py::return_value_policy::reference_internal)
        .def("set_diffuse", &PhongMaterial::setDiffuse, py::arg("diffuse"),
             py::return_value_policy::reference_internal)
        .def("set_specular", &PhongMaterial::setSpecular, py::arg("specular"),
             py::return_value_policy::reference_internal)
        .def("set_shininess", &PhongMaterial::setShininess,
             py::arg("shininess"), py::return_value_policy::reference_internal)
        .def("set_diffuse_map", &PhongMaterial::setDiffuseMap,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_specular_map", &PhongMaterial::setSpecularMap,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_alpha_map", &PhongMaterial::setAlphaMap, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_normal_map", &PhongMaterial::setNormalMap, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def_readwrite("ambient", &PhongMaterial::ambient,
                       "Ambient RGB factor.")
        .def_readwrite("diffuse", &PhongMaterial::diffuse,
                       "Diffuse RGB factor.")
        .def_readwrite("specular", &PhongMaterial::specular,
                       "Specular RGB factor.")
        .def_readwrite("shininess", &PhongMaterial::shininess,
                       "Specular highlight exponent.")
        .def_readwrite("diffuse_map", &PhongMaterial::diffuseMap,
                       "Optional diffuse texture.")
        .def_readwrite("specular_map", &PhongMaterial::specularMap,
                       "Optional specular texture.")
        .def_readwrite("alpha_map", &PhongMaterial::alphaMap,
                       "Optional alpha mask texture, such as OBJ map_d.")
        .def_readwrite("normal_map", &PhongMaterial::normalMap,
                       "Optional tangent-space normal map.");

    py::class_<PBRMaterial, Material>(
        m, "PBRMaterial",
        "Physically based material using metallic-roughness parameters.")
        .def(py::init<>(), "Create a PBR material with default factors.")
        .def(
            "load_from_preset",
            [](PBRMaterial& self, PBRMaterialType type) -> PBRMaterial& {
                self.loadFromPreset(type);
                return self;
            },
            py::arg("type"), py::return_value_policy::reference_internal,
            "Load material factors from a built-in PBR preset and return "
            "this material.")
        .def("set_base_color", &PBRMaterial::setBaseColor,
             py::arg("base_color"), py::return_value_policy::reference_internal)
        .def("set_metallic", &PBRMaterial::setMetallic, py::arg("metallic"),
             py::return_value_policy::reference_internal)
        .def("set_roughness", &PBRMaterial::setRoughness, py::arg("roughness"),
             py::return_value_policy::reference_internal)
        .def("set_emissive_color", &PBRMaterial::setEmissiveColor,
             py::arg("emissive_color"),
             py::return_value_policy::reference_internal)
        .def("set_emissive_strength", &PBRMaterial::setEmissiveStrength,
             py::arg("emissive_strength"),
             py::return_value_policy::reference_internal)
        .def("set_base_color_texture", &PBRMaterial::setBaseColorTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_normal_texture", &PBRMaterial::setNormalTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_metallic_roughness_texture",
             &PBRMaterial::setMetallicRoughnessTexture, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_metallic_texture", &PBRMaterial::setMetallicTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_roughness_texture", &PBRMaterial::setRoughnessTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def("set_ao_texture", &PBRMaterial::setAoTexture, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_orm_texture", &PBRMaterial::setOrmTexture, py::arg("texture"),
             py::return_value_policy::reference_internal)
        .def("set_emissive_texture", &PBRMaterial::setEmissiveTexture,
             py::arg("texture"), py::return_value_policy::reference_internal)
        .def_readwrite("base_color", &PBRMaterial::baseColor,
                       "Base color factor as RGBA.")
        .def_readwrite("metallic", &PBRMaterial::metallic,
                       "Metallic factor in the range 0..1.")
        .def_readwrite("roughness", &PBRMaterial::roughness,
                       "Roughness factor in the range 0..1.")
        .def_readwrite("emissive_color", &PBRMaterial::emissiveColor,
                       "Emissive RGB color factor.")
        .def_readwrite("emissive_strength", &PBRMaterial::emissiveStrength,
                       "Multiplier for emissive_color.")
        .def_readwrite("base_color_texture", &PBRMaterial::baseColorTexture,
                       "Optional base color texture.")
        .def_readwrite("normal_texture", &PBRMaterial::normalTexture,
                       "Optional tangent-space normal texture.")
        .def_readwrite("metallic_roughness_texture",
                       &PBRMaterial::metallicRoughnessTexture,
                       "Optional packed metallic-roughness texture.")
        .def_readwrite("metallic_texture", &PBRMaterial::metallicTexture,
                       "Optional single-channel metallic texture.")
        .def_readwrite("roughness_texture", &PBRMaterial::roughnessTexture,
                       "Optional single-channel roughness texture.")
        .def_readwrite("ao_texture", &PBRMaterial::aoTexture,
                       "Optional ambient occlusion texture.")
        .def_readwrite("orm_texture", &PBRMaterial::ormTexture,
                       "Optional packed occlusion-roughness-metallic texture.")
        .def_readwrite("emissive_texture", &PBRMaterial::emissiveTexture,
                       "Optional emissive texture.");

}
