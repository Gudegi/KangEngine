#version 410 core
in vec2 vTexCoord;

uniform sampler2D ke_g3_b0;
layout(std140) uniform ke_g3_b2 {
    vec4 alphaParams;
};

void main() {
    // Match the color-pass cutout so transparent atlas regions do not cast a
    // solid rectangular shadow.
    vec4 texel = texture(ke_g3_b0, vTexCoord);
    float alpha = alphaParams.y != 0.0 ? texel.r : texel.a;
    if (alpha < alphaParams.x) {
        discard;
    }
}
