#version 410 core
// PhongMaterial path.
// Material uniforms provide the shared default surface color, while vColor is
// the per-instance multiplier so color overrides do not break instancing.
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

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

uniform Material material;
// 0=opaque, 1=mask, 2=blend. For non-textured Phong, mask uses instance alpha.
uniform int uAlphaMode;
uniform float uAlphaCutoff;
uniform sampler2D uTexture;
uniform sampler2D normalMap;
uniform sampler2D specularMap;
uniform int useDiffuseMap;
uniform int useNormalMap;
uniform int useSpecularMap;
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

void main() {
    vec4 diffuseTexel = useDiffuseMap != 0
        ? texture(uTexture, TexCoord)
        : vec4(1.0);
    float alpha = diffuseTexel.a * vColor.a;
    if(uAlphaMode == 1 && alpha < uAlphaCutoff)
        discard;

    vec3 N = getViewNormal();

    vec3 L = normalize(lightDir.xyz);
    vec3 V = normalize(-FragPos);
    vec3 H = normalize(L + V);

    // Material color is the shared default. vColor is the per-instance
    // multiplier so differently colored instances can stay in the same batch.
    vec3 baseDiffuse = material.diffuse * diffuseTexel.rgb * vColor.rgb;
    vec3 baseAmbient = material.ambient * diffuseTexel.rgb * vColor.rgb;
    vec3 baseSpecular = material.specular;
    if(useSpecularMap != 0)
        baseSpecular *= texture(specularMap, TexCoord).rgb;

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), max(material.shininess, 1.0));

    vec3 diffuse = diff * lightColor.rgb * baseDiffuse;
    vec3 specular = spec * lightColor.rgb * baseSpecular;

    float shadow = ShadowCalculation();
    vec3 lit = ambient.rgb * baseAmbient + (1.0 - shadow) * (diffuse + specular);
    FragColor = vec4(lit * ShadowCascadeTint(), alpha);
}
