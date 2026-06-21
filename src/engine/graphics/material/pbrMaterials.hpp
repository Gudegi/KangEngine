// pbrMaterials.hpp
// Reference values are curated from public PBR material tables such as
// physicallybased.info. Roughness values are artist-friendly defaults for the
// KangEngine forward PBR preview shader.
#ifndef _PBRMATERIALS_HPP_
#define _PBRMATERIALS_HPP_

#include <array>
#include <cstddef>

namespace KE {

struct PBRMaterialProperties {
    std::array<float, 4> baseColor;
    float metallic;
    float roughness;
    std::array<float, 3> emissiveColor;
    float emissiveStrength;
};

enum class PBRMaterialType : size_t {
    GRAY_CARD = 0,
    WHITE_PLASTIC,
    BLACK_PLASTIC,
    BLACK_RUBBER,
    CHARCOAL,
    CARROT,
    CONCRETE,
    RED_BRICK,
    ALUMINUM,
    CHROME,
    COPPER,
    GOLD,
    EMISSIVE_BLUE,
    COUNT
};

class PBRMaterialLibrary {
  private:
    static constexpr std::array<PBRMaterialProperties,
                                static_cast<size_t>(PBRMaterialType::COUNT)>
        materials = {{
            // GRAY_CARD
            {{0.18f, 0.18f, 0.18f, 1.0f},
             0.0f,
             0.55f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // WHITE_PLASTIC
            {{0.82f, 0.82f, 0.78f, 1.0f},
             0.0f,
             0.36f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // BLACK_PLASTIC
            {{0.01f, 0.01f, 0.01f, 1.0f},
             0.0f,
             0.42f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // BLACK_RUBBER
            {{0.02f, 0.02f, 0.018f, 1.0f},
             0.0f,
             0.82f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // CHARCOAL
            {{0.020f, 0.020f, 0.020f, 1.0f},
             0.0f,
             0.90f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // CARROT
            {{0.713f, 0.170f, 0.026f, 1.0f},
             0.0f,
             0.50f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // CONCRETE
            {{0.51f, 0.50f, 0.47f, 1.0f},
             0.0f,
             0.78f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // RED_BRICK
            {{0.48f, 0.16f, 0.10f, 1.0f},
             0.0f,
             0.72f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // ALUMINUM
            {{0.91f, 0.92f, 0.92f, 1.0f},
             1.0f,
             0.28f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // CHROME
            {{0.65f, 0.68f, 0.70f, 1.0f},
             1.0f,
             0.08f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // COPPER
            {{0.93f, 0.62f, 0.52f, 1.0f},
             1.0f,
             0.23f,
             {0.0f, 0.0f, 0.0f},
             0.0f},
            // GOLD
            {{1.0f, 0.77f, 0.31f, 1.0f}, 1.0f, 0.22f, {0.0f, 0.0f, 0.0f}, 0.0f},
            // EMISSIVE_BLUE
            {{0.18f, 0.42f, 1.0f, 1.0f},
             0.0f,
             0.35f,
             {0.20f, 0.70f, 1.0f},
             3.0f},
        }};

    static constexpr const char*
        names[static_cast<size_t>(PBRMaterialType::COUNT)] = {
            "Gray Card",    "White Plastic", "Black Plastic", "Black Rubber",
            "Charcoal",     "Carrot",        "Concrete",      "Red Brick",
            "Aluminum",     "Chrome",        "Copper",        "Gold",
            "Emissive Blue"};

  public:
    static constexpr const PBRMaterialProperties& get(PBRMaterialType type) {
        return materials[static_cast<size_t>(type)];
    }

    static constexpr const char* getName(PBRMaterialType type) {
        return names[static_cast<size_t>(type)];
    }

    static constexpr size_t getCount() {
        return static_cast<size_t>(PBRMaterialType::COUNT);
    }
};

} // namespace KE

#endif
