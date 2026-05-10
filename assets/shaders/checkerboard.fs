#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec2 TexCoord;
in vec4 vColor;

uniform vec4 checkerColor1;
uniform vec4 checkerColor2;

layout(std140) uniform lightUBO {
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
};
layout(std140) uniform shadowUBO {
    mat4 lightSpaceMatrix;
    vec4 shadowParams;
    vec4 shadowInfo; // x: ortho half-size, y: map width, z: PCF samples
};

uniform sampler2D shadowMap;

#import "shadow_common.glsl"

float checker(vec2 uv) {
    return mod(floor(uv.x) + floor(uv.y), 2.0);
}

void main() {
    float t = checker(TexCoord);
    vec4 col = mix(checkerColor1, checkerColor2, t);
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    float shadow = ShadowCalculation();
    vec3 light = ambient.rgb + (1.0 - shadow) * diff * lightColor.rgb;
    FragColor = col * vec4(light, 1.0);
}
