#version 410 core
// Standard ground forward path. BackgroundSettings owns both the checker/grid
// appearance and the PBR surface factors exposed by the Background panel.
layout(location = 0) out vec4 outColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec2 TexCoord;
in vec4 vColor;

const int MAX_POINT_LIGHTS = 4;
const int MAX_SPOT_LIGHTS = 2;

layout(std140) uniform ke_g0_b1 {
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
    vec4 pointPositionRange[MAX_POINT_LIGHTS];
    vec4 pointColorIntensity[MAX_POINT_LIGHTS];
    vec4 spotPositionRange[MAX_SPOT_LIGHTS];
    vec4 spotDirectionInner[MAX_SPOT_LIGHTS];
    vec4 spotColorOuter[MAX_SPOT_LIGHTS];
    ivec4 lightCounts;
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
    vec4 shadowSamplingParams;
};

layout(std140) uniform ke_g3_b0 {
    vec4 checkerColor1;
    vec4 checkerColor2;
    vec4 gridColor;
    vec4 groundFactors; // show grid, metallic, roughness, grid scale
    vec4 gridFactors;   // line width, emission strength
};

#define shadowMap0 ke_g1_b0
#define shadowMap1 ke_g1_b1
#define shadowMap2 ke_g1_b2
#define shadowMap3 ke_g1_b3
#define debugCsmCascadeTint int(shadowSamplingParams.x + 0.5)
#import "../shadow_common.glsl"

float checkerAA(vec2 uv) {
    vec2 fw = max(fwidth(uv), vec2(1e-6));
    vec2 grid = abs(fract(uv - 0.5) - 0.5) / fw;
    vec2 line = clamp(grid, 0.0, 1.0);
    float raw = mod(floor(uv.x) + floor(uv.y), 2.0);
    return mix(0.5, raw, min(line.x, line.y));
}

float gridAA(vec2 uv, float lineWidth) {
    vec2 grid = abs(fract(uv - 0.5) - 0.5);
    vec2 width = max(fwidth(uv), vec2(1e-6));
    vec2 line = smoothstep(vec2(lineWidth) + width,
                           vec2(lineWidth) - width, grid);
    return clamp(line.x + line.y, 0.0, 1.0);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a2 = pow(roughness, 4.0);
    float nh = max(dot(N, H), 0.0);
    float d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 0.000001);
}

float geometrySchlickGGX(float nv, float roughness) {
    float k = pow(roughness + 1.0, 2.0) / 8.0;
    return nv / max(nv * (1.0 - k) + k, 0.000001);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 evaluateLight(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo,
                   float metallic, float roughness) {
    vec3 H = normalize(V + L);
    float nv = max(dot(N, V), 0.0), nl = max(dot(N, L), 0.0);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0),
                            mix(vec3(0.04), albedo, metallic));
    float G = geometrySchlickGGX(nv, roughness) *
              geometrySchlickGGX(nl, roughness);
    vec3 specular = distributionGGX(N, H, roughness) * G * F /
                    max(4.0 * nv * nl, 0.000001);
    vec3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    return (diffuse + specular) * radiance * nl;
}

float attenuation(float distanceToLight, float range) {
    float f = clamp(1.0 - distanceToLight / max(range, 0.0001), 0.0, 1.0);
    return f * f / max(distanceToLight * distanceToLight, 1.0);
}

void main() {
    vec2 uv = TexCoord * max(groundFactors.w, 0.0001);
    float gridMask = groundFactors.x > 0.5
        ? gridAA(uv, clamp(gridFactors.x, 0.0001, 0.49)) : 0.0;
    vec4 base = mix(checkerColor1, checkerColor2, checkerAA(uv)) * vColor;
    base = mix(base, gridColor * vColor, gridMask);

    vec3 N = normalize(Normal);
    if (!gl_FrontFacing)
        N = -N;
    vec3 V = normalize(-FragPos);
    vec3 albedo = max(base.rgb, vec3(0.0));
    float metallic = clamp(groundFactors.y, 0.0, 1.0);
    float roughness = clamp(groundFactors.z, 0.04, 1.0);
    float shadow = ShadowCalculation();
    vec3 direct = (1.0 - shadow) * evaluateLight(
        N, V, normalize(lightDir.xyz), lightColor.rgb,
        albedo, metallic, roughness);
    for (int i = 0; i < min(lightCounts.x, MAX_POINT_LIGHTS); ++i) {
        vec3 d = pointPositionRange[i].xyz - FragPos;
        float len = length(d);
        direct += evaluateLight(N, V, d / max(len, 0.0001),
            pointColorIntensity[i].rgb *
                attenuation(len, pointPositionRange[i].w),
            albedo, metallic, roughness);
    }
    for (int i = 0; i < min(lightCounts.y, MAX_SPOT_LIGHTS); ++i) {
        vec3 d = spotPositionRange[i].xyz - FragPos;
        float len = length(d);
        vec3 L = d / max(len, 0.0001);
        float cone = smoothstep(spotColorOuter[i].w, spotDirectionInner[i].w,
                                dot(normalize(-L),
                                    normalize(spotDirectionInner[i].xyz)));
        direct += evaluateLight(N, V, L, spotColorOuter[i].rgb * cone *
            attenuation(len, spotPositionRange[i].w),
            albedo, metallic, roughness);
    }
    vec3 gridEmission = gridColor.rgb * gridFactors.y * gridMask;
    outColor = vec4((ambient.rgb * albedo + direct + gridEmission) *
                    ShadowCascadeTint(), base.a);
}
