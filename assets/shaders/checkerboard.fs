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
    mat4 lightSpaceMatrices[4];
    vec4 cascadeSplits;
    vec4 cascadeOrthoHalfSizes;
    vec4 cascadeMapSizes;
    vec4 shadowParams;
    vec4 shadowInfo; // x: PCF samples, y: cascade count, z: use CSM
};

uniform sampler2D shadowMap0;
uniform sampler2D shadowMap1;
uniform sampler2D shadowMap2;
uniform sampler2D shadowMap3;
uniform int debugCsmCascadeTint;

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
    FragColor = vec4(col.rgb * light * ShadowCascadeTint(), col.a);
}
