#version 410 core
layout(location = 0) out vec4 outColor;
in vec2 vUv;
in vec4 vColor;

uniform sampler2D ke_g3_b0;

void main() {
    float coverage = texture(ke_g3_b0, vUv).r;
    if (coverage <= 0.001)
        discard;
    outColor = vec4(vColor.rgb, vColor.a * coverage);
}
