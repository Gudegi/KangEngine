#version 410 core
layout(location = 0) out vec4 outColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
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

layout(std140) uniform ke_g3_b0 {
    vec4 checkerColor1;
    vec4 checkerColor2;
    vec4 gridColor;
    vec4 checkerParams; // x: show grid
};

#define shadowMap0 ke_g1_b0
#define shadowMap1 ke_g1_b1
#define shadowMap2 ke_g1_b2
#define shadowMap3 ke_g1_b3
#define debugCsmCascadeTint int(rhiShadowSamplingParams.x + 0.5)

#import "../shadow_common.glsl"

float checkerAA(vec2 uv) {
    vec2 fw = max(fwidth(uv), vec2(1e-6));
    vec2 grid = abs(fract(uv - 0.5) - 0.5) / fw;
    vec2 line = clamp(grid, 0.0, 1.0);
    float raw = mod(floor(uv.x) + floor(uv.y), 2.0);
    return mix(0.5, raw, min(line.x, line.y));
}

float gridAA(vec2 uv) {
    vec2 coord = uv;
    vec2 grid = abs(fract(coord - 0.5) - 0.5);
    vec2 width = max(fwidth(coord), vec2(1e-6));
    vec2 line = smoothstep(vec2(0.005) + width,
                           vec2(0.005) - width, grid);
    return clamp(line.x + line.y, 0.0, 1.0);
}

void main() {
    vec4 checker = mix(checkerColor1, checkerColor2, checkerAA(TexCoord));
    vec3 N = normalize(Normal);
    if (!gl_FrontFacing)
        N = -N;
    vec3 L = normalize(lightDir.xyz);
    float diffuse = max(dot(N, L), 0.0);
    float shadow = ShadowCalculation();
    vec3 lighting = ambient.rgb +
                    (1.0 - shadow) * diffuse * lightColor.rgb;
    vec4 color = vec4(checker.rgb * lighting * ShadowCascadeTint(), checker.a);
    if (checkerParams.x > 0.5)
        color = mix(color, gridColor, gridAA(TexCoord));
    outColor = color;
}
