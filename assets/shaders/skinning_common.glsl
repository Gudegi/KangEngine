// Keep this in sync with KE::Scene::MaxSkinningBones.
const int MAX_BONES = 128;
uniform mat4 uBoneMatrices[MAX_BONES];

mat4 skinMatrix(ivec4 boneIndices, vec4 boneWeights) {
    mat4 skin = mat4(0.0);
    float weightSum = 0.0;
    if (boneIndices.x >= 0) {
        skin += boneWeights.x * uBoneMatrices[boneIndices.x];
        weightSum += boneWeights.x;
    }
    if (boneIndices.y >= 0) {
        skin += boneWeights.y * uBoneMatrices[boneIndices.y];
        weightSum += boneWeights.y;
    }
    if (boneIndices.z >= 0) {
        skin += boneWeights.z * uBoneMatrices[boneIndices.z];
        weightSum += boneWeights.z;
    }
    if (boneIndices.w >= 0) {
        skin += boneWeights.w * uBoneMatrices[boneIndices.w];
        weightSum += boneWeights.w;
    }
    if (weightSum <= 0.0) {
        skin = mat4(1.0);
    }
    return skin;
}
