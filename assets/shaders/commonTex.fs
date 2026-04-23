#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;

layout(std140) uniform lightUBO {
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambient;
};

uniform sampler2D uTexture;

void main() {
    vec4 texColor = texture(uTexture, TexCoord);
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
    FragColor = vec4(ambient.rgb * texColor.rgb + diffuse + specular, texColor.a);
}