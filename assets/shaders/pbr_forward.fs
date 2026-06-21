#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec3 Tangent;
in float TangentHandedness;
in vec2 TexCoord;
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

uniform vec4 uBaseColorFactor;
uniform float uMetallicFactor;
uniform float uRoughnessFactor;
uniform vec3 uEmissiveColor;
uniform float uEmissiveStrength;

uniform sampler2D uBaseColorMap;
uniform sampler2D normalMap;
uniform sampler2D uMetallicRoughnessMap;
uniform sampler2D uAoMap;
uniform sampler2D uEmissiveMap;
uniform int useBaseColorMap;
uniform int useNormalMap;
uniform int useMetallicRoughnessMap;
uniform int useAoMap;
uniform int useEmissiveMap;

uniform sampler2D shadowMap0;
uniform sampler2D shadowMap1;
uniform sampler2D shadowMap2;
uniform sampler2D shadowMap3;
uniform int debugCsmCascadeTint;

#import "shadow_common.glsl"

vec3 getViewNormal() {
    vec3 N = normalize(Normal);
    vec3 T = normalize(Tangent - dot(Tangent, N) * N);
    vec3 B = normalize(cross(N, T) * TangentHandedness);
    if(useNormalMap != 0) {
        vec3 sampled = texture(normalMap, TexCoord).xyz * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * sampled);
    }
    if(!gl_FrontFacing)
        N = -N;
    return N;
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 0.000001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 0.000001);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) *
           geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec4 baseColor = uBaseColorFactor * vColor;
    if(useBaseColorMap != 0)
        baseColor *= texture(uBaseColorMap, TexCoord);

    float metallic = clamp(uMetallicFactor, 0.0, 1.0);
    float roughness = clamp(uRoughnessFactor, 0.04, 1.0);
    if(useMetallicRoughnessMap != 0) {
        vec4 mr = texture(uMetallicRoughnessMap, TexCoord);
        roughness = clamp(roughness * mr.g, 0.04, 1.0);
        metallic = clamp(metallic * mr.b, 0.0, 1.0);
    }

    float ao = useAoMap != 0 ? texture(uAoMap, TexCoord).r : 1.0;
    vec3 emissive = uEmissiveColor * uEmissiveStrength;
    if(useEmissiveMap != 0)
        emissive *= texture(uEmissiveMap, TexCoord).rgb;

    vec3 albedo = max(baseColor.rgb, vec3(0.0));
    vec3 N = getViewNormal();
    vec3 V = normalize(-FragPos);
    vec3 L = normalize(lightDir.xyz);
    vec3 H = normalize(V + L);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (NDF * G * F) / max(4.0 * NdotV * NdotL, 0.000001);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    float shadow = ShadowCalculation();
    vec3 direct = (diffuse + specular) * lightColor.rgb * NdotL;
    vec3 ambientColor = ambient.rgb * albedo * ao;
    vec3 color = ambientColor + (1.0 - shadow) * direct + emissive;

    FragColor = vec4(color * ShadowCascadeTint(), baseColor.a);
}
