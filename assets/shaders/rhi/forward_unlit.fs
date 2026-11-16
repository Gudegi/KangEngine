#version 410 core
layout(location = 0) out vec4 outColor;
in vec4 vColor;
in vec2 TexCoord;

layout(std140) uniform ke_g3_b0 {
    vec4 baseColor;
    // x: alpha mode, y: cutoff, z: separate red-channel opacity.
    vec4 alphaParams;
};
uniform sampler2D ke_g3_b1;
uniform sampler2D ke_g3_b2;

void main() {
    vec4 color = baseColor * vColor * texture(ke_g3_b1, TexCoord);
    if (alphaParams.z > 0.5)
        color.a = baseColor.a * vColor.a * texture(ke_g3_b2, TexCoord).r;
    if (int(alphaParams.x + 0.5) == 1 && color.a < alphaParams.y)
        discard;
    outColor = color;
}
