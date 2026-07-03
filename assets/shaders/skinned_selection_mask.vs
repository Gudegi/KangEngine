#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;
layout(location = 8) in ivec4 aBoneIndices;
layout(location = 9) in vec4 aBoneWeights;

#import "skinning_common.glsl"

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

out vec2 TexCoord;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    mat4 skin = skinMatrix(aBoneIndices, aBoneWeights);
    gl_Position = projection * view * model * skin * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
