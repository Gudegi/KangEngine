#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec3 WorldPos;
in vec3 WorldNormal;
in vec2 TexCoord;
in vec4 vColor;

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

uniform sampler2D uTexture;
uniform sampler2D shadowMap;

#import "shadow_common.glsl"

void main() {
    vec4 texColor = texture(uTexture, TexCoord) * vColor;
    vec3 N = normalize(Normal);
    if(!gl_FrontFacing)
        N = -N;
    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor.rgb * texColor.rgb;
    vec3 V = normalize(-FragPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32) * 0.3;
    vec3 specular = spec * lightColor.rgb;

    float shadow = ShadowCalculation();
    FragColor = vec4(ambient.rgb * texColor.rgb + (1.0 - shadow) * (diffuse + specular), texColor.a);
}
