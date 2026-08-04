#version 410 core
// PhongMaterial path.
// Material parameters provide the shared default surface color, while vColor
// is the per-instance multiplier so color overrides do not break instancing.
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
    vec4 materialAmbientShininess;
    vec4 materialDiffuseAlphaCutoff;
    vec4 materialSpecularAlphaMode;
    vec4 materialTextureFlags;
};
uniform sampler2D ke_g3_b1;
uniform sampler2D ke_g3_b2;
uniform sampler2D ke_g3_b3;
uniform sampler2D ke_g3_b4;

#define shadowMap0 ke_g1_b0
#define shadowMap1 ke_g1_b1
#define shadowMap2 ke_g1_b2
#define shadowMap3 ke_g1_b3
#define debugCsmCascadeTint int(shadowSamplingParams.x + 0.5)
#import "../shadow_common.glsl"

void main() {
    bool useDiffuse = materialTextureFlags.x > 0.5;
    bool useSpecular = materialTextureFlags.y > 0.5;
    bool useAlpha = materialTextureFlags.z > 0.5;
    bool useNormal = materialTextureFlags.w > 0.5;
    vec4 diffuseTexel = useDiffuse ? texture(ke_g3_b1, TexCoord) : vec4(1.0);
    float alphaTexel = useAlpha ? texture(ke_g3_b3, TexCoord).r : diffuseTexel.a;
    float alpha = alphaTexel * vColor.a;
    int alphaMode = int(materialSpecularAlphaMode.w + 0.5);
    if (alphaMode == 1 && alpha < materialDiffuseAlphaCutoff.w)
        discard;

    vec3 N = normalize(Normal);
    if (useNormal) {
        vec3 T = normalize(Tangent - dot(Tangent, N) * N);
        vec3 B = normalize(cross(N, T) * TangentHandedness);
        N = normalize(mat3(T, B, N) *
                      (texture(ke_g3_b4, TexCoord).xyz * 2.0 - 1.0));
    }
    if (!gl_FrontFacing)
        N = -N;
    vec3 L = normalize(lightDir.xyz);
    vec3 V = normalize(-FragPos);
    vec3 H = normalize(L + V);
    vec3 baseDiffuse = materialDiffuseAlphaCutoff.rgb * diffuseTexel.rgb * vColor.rgb;
    vec3 baseAmbient = materialAmbientShininess.rgb * diffuseTexel.rgb * vColor.rgb;
    vec3 baseSpecular = materialSpecularAlphaMode.rgb;
    if (useSpecular)
        baseSpecular *= texture(ke_g3_b2, TexCoord).rgb;
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0),
                     max(materialAmbientShininess.w, 1.0));
    float shadow = ShadowCalculation();
    vec3 lit = ambient.rgb * baseAmbient +
               (1.0 - shadow) *
                   (diff * lightColor.rgb * baseDiffuse +
                    spec * lightColor.rgb * baseSpecular);
    outColor = vec4(lit * ShadowCascadeTint(), alpha);
}
