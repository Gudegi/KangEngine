#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec4 vColor;

uniform vec4 checkerColor1;
uniform vec4 checkerColor2;

layout(std140) uniform lightUBO {
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
};

float checker(vec2 uv) {
    return mod(floor(uv.x) + floor(uv.y), 2.0);
}

void main() {
    float t = checker(TexCoord);
    vec4 col = mix(checkerColor1, checkerColor2, t);
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    vec3 light = ambient.rgb + diff * lightColor.rgb;
    FragColor = col * vec4(light, 1.0);
}