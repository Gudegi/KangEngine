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
    mat4 lightSpaceMatrix;
    vec4 shadowParams;
    vec4 shadowInfo; // x: shadow_extents, y: depthmap width,height, zw: unused
};

uniform sampler2D shadowMap;

const float PI = 3.14159265359;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 poissonDisk[16] = vec2[](vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760), vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464), vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379), vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420), vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188), vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590), vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790));

float ShadowCalculation() {
    float shadowExtents = shadowInfo.x;
    float shadowMapWH = shadowInfo.y;
    if(shadowExtents <= 0.0)
        return 0.0;

    vec3 sunDirection = shadowParams.xyz;
    float shadowRadius = shadowParams.w;
    vec3 normal = normalize(WorldNormal);
    if(!gl_FrontFacing)
        normal = -normal;

    float worldTexel = (shadowExtents * 2.0) / shadowMapWH;
    float normalBias = 2.0 * worldTexel;

    vec4 lightSpacePos = lightSpaceMatrix * vec4(WorldPos + normal * normalBias, 1.0);
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.z > 1.0)
        return 0.0;

    float fragDepth = projCoords.z;

    float fade = 1.0;
    float margin = 0.15;
    fade *= smoothstep(0.0, margin, projCoords.x);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.x);
    fade *= smoothstep(0.0, margin, projCoords.y);
    fade *= smoothstep(0.0, margin, 1.0 - projCoords.y);

    float NdotL = max(dot(normal, normalize(sunDirection)), 0.0);
    float depthBias = mix(0.0003, 0.00002, NdotL);
    float biasedDepth = fragDepth - depthBias;

    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    float angle = rand(gl_FragCoord.xy) * 2.0 * PI;
    float s = sin(angle), c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);
    for(int i = 0; i < 8; i++) {
        vec2 offset = rot * poissonDisk[i];
        float pcfDepth = texture(shadowMap, projCoords.xy + offset * shadowRadius * texelSize).r;
        if(pcfDepth < biasedDepth)
            shadow += 1.0;
    }
    return (shadow / 8.0) * fade;
}

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
    FragColor = col * vec4(light, 1.0);
}