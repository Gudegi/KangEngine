#version 410 core
// TexturedVertexColorMaterial path.
// This shader combines the base-color texture with per-instance vColor and
// optional normal mapping.
layout(location = 0) out vec4 outColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec3 Tangent;
in float TangentHandedness;
in vec2 TexCoord;
in vec4 vColor;

layout(std140) uniform ke_g0_b1 {
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
};
layout(std140) uniform ke_g0_b2 {
    mat4 lightSpaceMatrices[4];
    vec4 cascadeSplits;
    vec4 cascadeOrthoHalfSizes;
    vec4 cascadeMapSizes;
    vec4 shadowParams;
    vec4 shadowInfo;
};
uniform sampler2D ke_g1_b0;
uniform sampler2D ke_g1_b1;
uniform sampler2D ke_g1_b2;
uniform sampler2D ke_g1_b3;
layout(std140) uniform ke_g1_b5 {
    vec4 rhiShadowSamplingParams;
};

uniform sampler2D ke_g3_b0;
uniform sampler2D ke_g3_b1;
layout(std140) uniform ke_g3_b3 {
    vec4 texturedParams; // x: normal map, y: alpha mode, z: cutoff
};

#define shadowMap0 ke_g1_b0
#define shadowMap1 ke_g1_b1
#define shadowMap2 ke_g1_b2
#define shadowMap3 ke_g1_b3
#define debugCsmCascadeTint int(rhiShadowSamplingParams.x + 0.5)
#import "../shadow_common.glsl"

void main() {
    vec4 texColor = texture(ke_g3_b0, TexCoord) * vColor;
    if (int(texturedParams.y + 0.5) == 1 && texColor.a < texturedParams.z)
        discard;

    vec3 N = normalize(Normal);
    vec3 T = normalize(Tangent - dot(Tangent, N) * N);
    vec3 B = normalize(cross(N, T) * TangentHandedness);
    if (texturedParams.x > 0.5) {
        vec3 sampledNormal = texture(ke_g3_b1, TexCoord).xyz * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * sampledNormal);
    }
    if (!gl_FrontFacing)
        N = -N;

    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor.rgb * texColor.rgb;
    vec3 V = normalize(-FragPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.3;
    float shadow = ShadowCalculation();
    vec3 lit = ambient.rgb * texColor.rgb +
               (1.0 - shadow) * (diffuse + spec * lightColor.rgb);
    outColor = vec4(lit * ShadowCascadeTint(), texColor.a);
}
