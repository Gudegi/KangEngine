#version 410 core
in vec2 vUv;
in vec4 vColor;
out vec4 FragColor;

uniform sampler2D uFontAtlas;

void main() {
    float coverage = texture(uFontAtlas, vUv).r;
    if (coverage <= 0.001)
        discard;
    FragColor = vec4(vColor.rgb, vColor.a * coverage);
}
