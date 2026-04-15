// colors.hpp
// Popular colors and pastel tone color library
#ifndef _COLORS_HPP_
#define _COLORS_HPP_

#include <array>

namespace KE {

struct Color {
    union {
        std::array<float, 4> rgba;
        struct {
            float r, g, b, a;
        };
    };
};

enum class ColorType : size_t {
    // === Basic Colors ===
    WHITE = 0,
    BLACK,
    RED,
    GREEN,
    BLUE,
    YELLOW,
    CYAN,
    MAGENTA,
    ORANGE,
    PURPLE,
    PINK,
    BROWN,
    GRAY,
    LIGHT_GRAY,
    DARK_GRAY,

    // === Popular Web Colors ===
    CORAL,
    SALMON,
    TOMATO,
    CRIMSON,
    FOREST_GREEN,
    LIME_GREEN,
    SEA_GREEN,
    TEAL,
    NAVY,
    SKY_BLUE,
    STEEL_BLUE,
    ROYAL_BLUE,
    INDIGO,
    VIOLET,
    ORCHID,
    GOLD,
    KHAKI,
    OLIVE,
    MAROON,
    BEIGE,
    IVORY,
    MINT,
    LAVENDER,
    SLATE_GRAY,

    // === Pastel Colors ===
    PASTEL_RED,
    PASTEL_ORANGE,
    PASTEL_YELLOW,
    PASTEL_GREEN,
    PASTEL_MINT,
    PASTEL_CYAN,
    PASTEL_BLUE,
    PASTEL_PURPLE,
    PASTEL_PINK,
    PASTEL_ROSE,
    PASTEL_PEACH,
    PASTEL_LAVENDER,
    PASTEL_LILAC,
    PASTEL_CORAL,
    PASTEL_CREAM,
    PASTEL_SKY,

    COUNT
};

class ColorLibrary {
  private:
    static constexpr std::array<Color, static_cast<size_t>(ColorType::COUNT)>
        colors = {{
            // === Basic Colors ===
            // WHITE
            {{{1.0f, 1.0f, 1.0f, 1.0f}}},
            // BLACK
            {{{0.0f, 0.0f, 0.0f, 1.0f}}},
            // RED
            {{{1.0f, 0.0f, 0.0f, 1.0f}}},
            // GREEN
            {{{0.0f, 1.0f, 0.0f, 1.0f}}},
            // BLUE
            {{{0.0f, 0.0f, 1.0f, 1.0f}}},
            // YELLOW
            {{{1.0f, 1.0f, 0.0f, 1.0f}}},
            // CYAN
            {{{0.0f, 1.0f, 1.0f, 1.0f}}},
            // MAGENTA
            {{{1.0f, 0.0f, 1.0f, 1.0f}}},
            // ORANGE
            {{{1.0f, 0.647f, 0.0f, 1.0f}}},
            // PURPLE
            {{{0.502f, 0.0f, 0.502f, 1.0f}}},
            // PINK
            {{{1.0f, 0.753f, 0.796f, 1.0f}}},
            // BROWN
            {{{0.647f, 0.165f, 0.165f, 1.0f}}},
            // GRAY
            {{{0.502f, 0.502f, 0.502f, 1.0f}}},
            // LIGHT_GRAY
            {{{0.827f, 0.827f, 0.827f, 1.0f}}},
            // DARK_GRAY
            {{{0.333f, 0.333f, 0.333f, 1.0f}}},

            // === Popular Web Colors ===
            // CORAL
            {{{1.0f, 0.498f, 0.314f, 1.0f}}},
            // SALMON
            {{{0.980f, 0.502f, 0.447f, 1.0f}}},
            // TOMATO
            {{{1.0f, 0.388f, 0.278f, 1.0f}}},
            // CRIMSON
            {{{0.863f, 0.078f, 0.235f, 1.0f}}},
            // FOREST_GREEN
            {{{0.133f, 0.545f, 0.133f, 1.0f}}},
            // LIME_GREEN
            {{{0.196f, 0.804f, 0.196f, 1.0f}}},
            // SEA_GREEN
            {{{0.180f, 0.545f, 0.341f, 1.0f}}},
            // TEAL
            {{{0.0f, 0.502f, 0.502f, 1.0f}}},
            // NAVY
            {{{0.0f, 0.0f, 0.502f, 1.0f}}},
            // SKY_BLUE
            {{{0.529f, 0.808f, 0.922f, 1.0f}}},
            // STEEL_BLUE
            {{{0.275f, 0.510f, 0.706f, 1.0f}}},
            // ROYAL_BLUE
            {{{0.255f, 0.412f, 0.882f, 1.0f}}},
            // INDIGO
            {{{0.294f, 0.0f, 0.510f, 1.0f}}},
            // VIOLET
            {{{0.933f, 0.510f, 0.933f, 1.0f}}},
            // ORCHID
            {{{0.855f, 0.439f, 0.839f, 1.0f}}},
            // GOLD
            {{{1.0f, 0.843f, 0.0f, 1.0f}}},
            // KHAKI
            {{{0.941f, 0.902f, 0.549f, 1.0f}}},
            // OLIVE
            {{{0.502f, 0.502f, 0.0f, 1.0f}}},
            // MAROON
            {{{0.502f, 0.0f, 0.0f, 1.0f}}},
            // BEIGE
            {{{0.961f, 0.961f, 0.863f, 1.0f}}},
            // IVORY
            {{{1.0f, 1.0f, 0.941f, 1.0f}}},
            // MINT
            {{{0.624f, 1.0f, 0.624f, 1.0f}}},
            // LAVENDER
            {{{0.902f, 0.902f, 0.980f, 1.0f}}},
            // SLATE_GRAY
            {{{0.439f, 0.502f, 0.565f, 1.0f}}},

            // === Pastel Colors ===
            // PASTEL_RED
            {{{1.0f, 0.702f, 0.702f, 1.0f}}},
            // PASTEL_ORANGE
            {{{1.0f, 0.800f, 0.600f, 1.0f}}},
            // PASTEL_YELLOW
            {{{1.0f, 1.0f, 0.702f, 1.0f}}},
            // PASTEL_GREEN
            {{{0.702f, 1.0f, 0.702f, 1.0f}}},
            // PASTEL_MINT
            {{{0.702f, 1.0f, 0.867f, 1.0f}}},
            // PASTEL_CYAN
            {{{0.702f, 1.0f, 1.0f, 1.0f}}},
            // PASTEL_BLUE
            {{{0.702f, 0.800f, 1.0f, 1.0f}}},
            // PASTEL_PURPLE
            {{{0.800f, 0.702f, 1.0f, 1.0f}}},
            // PASTEL_PINK
            {{{1.0f, 0.753f, 0.863f, 1.0f}}},
            // PASTEL_ROSE
            {{{1.0f, 0.800f, 0.820f, 1.0f}}},
            // PASTEL_PEACH
            {{{1.0f, 0.855f, 0.725f, 1.0f}}},
            // PASTEL_LAVENDER
            {{{0.867f, 0.800f, 1.0f, 1.0f}}},
            // PASTEL_LILAC
            {{{0.855f, 0.725f, 0.925f, 1.0f}}},
            // PASTEL_CORAL
            {{{1.0f, 0.745f, 0.675f, 1.0f}}},
            // PASTEL_CREAM
            {{{1.0f, 0.992f, 0.878f, 1.0f}}},
            // PASTEL_SKY
            {{{0.686f, 0.867f, 1.0f, 1.0f}}},
        }};

    static constexpr const char* names[static_cast<size_t>(
        ColorType::COUNT)] = {
        // Basic Colors
        "White", "Black", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta",
        "Orange", "Purple", "Pink", "Brown", "Gray", "Light Gray", "Dark Gray",
        // Popular Web Colors
        "Coral", "Salmon", "Tomato", "Crimson", "Forest Green", "Lime Green",
        "Sea Green", "Teal", "Navy", "Sky Blue", "Steel Blue", "Royal Blue",
        "Indigo", "Violet", "Orchid", "Gold", "Khaki", "Olive", "Maroon",
        "Beige", "Ivory", "Mint", "Lavender", "Slate Gray",
        // Pastel Colors
        "Pastel Red", "Pastel Orange", "Pastel Yellow", "Pastel Green",
        "Pastel Mint", "Pastel Cyan", "Pastel Blue", "Pastel Purple",
        "Pastel Pink", "Pastel Rose", "Pastel Peach", "Pastel Lavender",
        "Pastel Lilac", "Pastel Coral", "Pastel Cream", "Pastel Sky"};

  public:
    // usage: KE::ColorLibrary::get(KE::ColorType::PASTEL_PINK)
    static constexpr const Color& get(ColorType type) {
        return colors[static_cast<size_t>(type)];
    }

    static constexpr const char* getName(ColorType type) {
        return names[static_cast<size_t>(type)];
    }

    static constexpr size_t getCount() {
        return static_cast<size_t>(ColorType::COUNT);
    }

    // Helper to get rgba array directly
    static constexpr const std::array<float, 4>& getRGBA(ColorType type) {
        return colors[static_cast<size_t>(type)].rgba;
    }

    // Helper to create color from hex (e.g., 0xFF5733FF for RGBA)
    static constexpr Color fromHex(uint32_t hex) {
        return {{static_cast<float>((hex >> 24) & 0xFF) / 255.0f,
                 static_cast<float>((hex >> 16) & 0xFF) / 255.0f,
                 static_cast<float>((hex >> 8) & 0xFF) / 255.0f,
                 static_cast<float>(hex & 0xFF) / 255.0f}};
    }

    // Helper to create color from RGB values (0-255)
    static constexpr Color fromRGB(uint8_t r, uint8_t g, uint8_t b,
                                   uint8_t a = 255) {
        return {{static_cast<float>(r) / 255.0f, static_cast<float>(g) / 255.0f,
                 static_cast<float>(b) / 255.0f,
                 static_cast<float>(a) / 255.0f}};
    }
};

} // namespace KE

#endif
