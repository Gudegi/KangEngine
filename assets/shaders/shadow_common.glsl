const float PI = 3.14159265359;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 poissonDisk[16] = vec2[](vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760), vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464), vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379), vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420), vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188), vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590), vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790));

int SelectShadowCascade() {
    int cascadeCount = clamp(int(shadowInfo.y + 0.5), 1, 4);
    if(shadowInfo.z <= 0.5 || cascadeCount <= 1) // shadowInfo.z : whether use CSM
        return 0;

    float viewDepth = max(0.0, -FragPos.z);
    int cascadeIndex = 0;
    for(int i = 0; i < 3; ++i) {
        if(i >= cascadeCount - 1)
            break;
        if(viewDepth > cascadeSplits[i])
            cascadeIndex = i + 1;
    }
    return cascadeIndex;
}

float SampleShadowDepth(int cascadeIndex, vec2 uv) {
    if(cascadeIndex == 0)
        return texture(shadowMap0, uv).r;
    if(cascadeIndex == 1)
        return texture(shadowMap1, uv).r;
    if(cascadeIndex == 2)
        return texture(shadowMap2, uv).r;
    return texture(shadowMap3, uv).r;
}

vec3 ShadowCascadeTint() {
    if(debugCsmCascadeTint == 0 || shadowInfo.z <= 0.5)
        return vec3(1.0);

    int cascadeIndex = SelectShadowCascade();
    if(cascadeIndex == 0)
        return vec3(1.0, 0.25, 0.25);
    if(cascadeIndex == 1)
        return vec3(0.25, 1.0, 0.25);
    if(cascadeIndex == 2)
        return vec3(0.25, 0.45, 1.0);
    return vec3(1.0, 0.35, 1.0);
}

float ShadowCalculation() {
    int cascadeIndex = SelectShadowCascade();
    float shadowOrthoHalfSize = cascadeOrthoHalfSizes[cascadeIndex];
    float shadowMapWH = cascadeMapSizes[cascadeIndex];
    int pcfSamples = clamp(int(shadowInfo.x + 0.5), 1, 16);
    if(shadowOrthoHalfSize <= 0.0)
        return 0.0;

    vec3 sunDirection = shadowParams.xyz;
    float shadowRadius = shadowParams.w;
    vec3 normal = normalize(WorldNormal);
    if(!gl_FrontFacing)
        normal = -normal;

    float worldTexel = (shadowOrthoHalfSize * 2.0) / shadowMapWH;
    float normalBias = 2.0 * worldTexel;

    vec4 lightSpacePos = lightSpaceMatrices[cascadeIndex] * vec4(WorldPos + normal * normalBias, 1.0);
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0)
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
    vec2 texelSize = vec2(1.0 / shadowMapWH);
    float angle = rand(gl_FragCoord.xy) * 2.0 * PI;
    float s = sin(angle), c = cos(angle);
    mat2 rot = mat2(c, -s, s, c);
    for(int i = 0; i < 16; i++) {
        if(i >= pcfSamples)
            break;
        vec2 offset = rot * poissonDisk[i];
        float pcfDepth = SampleShadowDepth(cascadeIndex, projCoords.xy + offset * shadowRadius * texelSize);
        if(pcfDepth < biasedDepth)
            shadow += 1.0;
    }
    return (shadow / float(pcfSamples)) * fade;
}
