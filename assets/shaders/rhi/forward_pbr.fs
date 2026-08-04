#version 410 core
// PBRMaterial forward path.
// Material factors/textures provide shared appearance, while vColor is the
// per-instance multiplier so color overrides do not break instancing.
layout(location = 0) out vec4 outColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec3 Tangent;
in float TangentHandedness;
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
    vec4 shadowInfo; // x: PCF samples, y: cascade count, z: use CSM
};
uniform sampler2D ke_g1_b0;
uniform sampler2D ke_g1_b1;
uniform sampler2D ke_g1_b2;
uniform sampler2D ke_g1_b3;
layout(std140) uniform ke_g1_b5 {
    vec4 shadowSamplingParams;
};

layout(std140) uniform ke_g3_b0 {
    vec4 pbrBaseColor;
    vec4 pbrFactors;       // metallic, roughness, emissive strength, alpha cutoff
    vec4 pbrEmissiveAlpha; // rgb emissive, w alpha mode
    vec4 pbrTextureFlags0; // base, normal, metallic-roughness, metallic
    vec4 pbrTextureFlags1; // roughness, ao, orm, emissive
};
uniform sampler2D ke_g3_b1;
uniform sampler2D ke_g3_b2;
uniform sampler2D ke_g3_b3;
uniform sampler2D ke_g3_b4;
uniform sampler2D ke_g3_b5;
uniform sampler2D ke_g3_b6;
uniform sampler2D ke_g3_b7;
uniform sampler2D ke_g3_b8;

#define shadowMap0 ke_g1_b0
#define shadowMap1 ke_g1_b1
#define shadowMap2 ke_g1_b2
#define shadowMap3 ke_g1_b3
#define debugCsmCascadeTint int(shadowSamplingParams.x + 0.5)
#import "../shadow_common.glsl"

vec3 getViewNormal() {
    vec3 N = normalize(Normal);
    if (pbrTextureFlags0.y > 0.5) {
        vec3 T = normalize(Tangent - dot(Tangent, N) * N);
        vec3 B = normalize(cross(N, T) * TangentHandedness);
        N = normalize(mat3(T, B, N) *
                      (texture(ke_g3_b2, TexCoord).xyz * 2.0 - 1.0));
    }
    if (!gl_FrontFacing)
        N = -N;
    return N;
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
    vec4 base = pbrBaseColor * vColor;
    if (pbrTextureFlags0.x > 0.5)
        base *= texture(ke_g3_b1, TexCoord);
    if (int(pbrEmissiveAlpha.w + 0.5) == 1 && base.a < pbrFactors.w)
        discard;
    float metallic = clamp(pbrFactors.x, 0.0, 1.0);
    float roughness = clamp(pbrFactors.y, 0.04, 1.0);
    float ao = 1.0;
    if (pbrTextureFlags1.z > 0.5) {
        vec3 orm = texture(ke_g3_b7, TexCoord).rgb;
        ao = orm.r;
        roughness = clamp(roughness * orm.g, 0.04, 1.0);
        metallic = clamp(metallic * orm.b, 0.0, 1.0);
    }
    if (pbrTextureFlags0.z > 0.5) {
        vec4 mr = texture(ke_g3_b3, TexCoord);
        roughness = clamp(roughness * mr.g, 0.04, 1.0);
        metallic = clamp(metallic * mr.b, 0.0, 1.0);
    }
    if (pbrTextureFlags0.w > 0.5)
        metallic *= texture(ke_g3_b4, TexCoord).r;
    if (pbrTextureFlags1.x > 0.5)
        roughness = clamp(
            roughness * texture(ke_g3_b5, TexCoord).r, 0.04, 1.0);
    if (pbrTextureFlags1.y > 0.5)
        ao *= texture(ke_g3_b6, TexCoord).r;
    vec3 emissive = pbrEmissiveAlpha.rgb * pbrFactors.z;
    if (pbrTextureFlags1.w > 0.5)
        emissive *= texture(ke_g3_b8, TexCoord).rgb;

    vec3 N = getViewNormal();
    vec3 V = normalize(-FragPos);
    vec3 albedo = max(base.rgb, vec3(0.0));
    float shadow = ShadowCalculation();
    vec3 direct = (1.0 - shadow) * evaluateLight(N, V, normalize(lightDir.xyz),
        lightColor.rgb, albedo, metallic, roughness);
    for (int i = 0; i < min(lightCounts.x, MAX_POINT_LIGHTS); ++i) {
        vec3 d = pointPositionRange[i].xyz - FragPos;
        float len = length(d);
        direct += evaluateLight(N, V, d / max(len, 0.0001),
            pointColorIntensity[i].rgb * attenuation(len, pointPositionRange[i].w),
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
            attenuation(len, spotPositionRange[i].w), albedo, metallic, roughness);
    }
    outColor = vec4((ambient.rgb * albedo * ao + direct + emissive) *
                    ShadowCascadeTint(), base.a);
}
