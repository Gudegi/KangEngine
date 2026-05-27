#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec4 vColor;

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

void main() {
    vec3 N = normalize(Normal);
    if(!gl_FrontFacing)
        N = -N;
    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor.rgb * vColor.rgb;
    vec3 V = normalize(-FragPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32) * 0.3;
    vec3 specular = spec * lightColor.rgb;

    float shadow = ShadowCalculation();
    vec3 lit = ambient.rgb * vColor.rgb + (1.0 - shadow) * (diffuse + specular);
    FragColor = vec4(lit * ShadowCascadeTint(), vColor.a);
}
