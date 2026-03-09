#include "kangEngine.hpp"
#include "material/colors.hpp"
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Scene;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

static const char* phongVs = R"(
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
out vec3 Normal;
out vec3 FragPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMat;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(view * model * vec4(aPos, 1.0));
    Normal = normalMat * aNormal;
}
)";

static const char* phongFs = R"(
#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
uniform vec4 objColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 camPos;
void main() {
    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * lightColor;
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor;
    float specularStrength = 0.5;
    vec3 V = normalize(camPos - FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    FragColor = objColor * vec4(ambient + diffuse + specular, 1.0);
}
)";

static const char* groundVs = R"(
#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMat;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    FragPos = vec3(view * model * vec4(aPos, 1.0));
    Normal = normalMat * aNormal;
}
)";

static const char* groundFs = R"(
#version 410 core
out vec4 FragColor;
in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 camPos;
float checker(vec2 uv) {
    return mod(floor(uv.x * 8.0) + floor(uv.y * 8.0), 2.0);
}
void main() {
    float t = checker(TexCoord);
    vec4 col = mix(vec4(0.85, 0.85, 0.85, 1.0), vec4(0.55, 0.75, 0.55, 1.0), t);
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 light = 0.2 * lightColor + diff * lightColor;
    FragColor = col * vec4(light, 1.0);
}
)";

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class PrimShowcaseApp : public App {
  public:
    std::unique_ptr<Backend::Shader> phongShader;
    std::unique_ptr<Backend::Shader> groundShader;

    glm::vec3 lightPos = {2.f, -2.f, 6.f};
    float lightColor[3] = {1.f, 1.f, 1.f};

    struct ShapeEntry {
        Prim* prim;
        std::string label;
    };
    std::vector<ShapeEntry> entries;

    // ---------------------------------------------------------------------------
    void setup() override {
        phongShader = getGraphicsDevice()->createShader(phongVs, phongFs);
        groundShader = getGraphicsDevice()->createShader(groundVs, groundFs);

        phongShader->use();
        phongShader->setVec3("lightColor", 1.f, 1.f, 1.f);

        // Ground
        auto* ground = getScene()->definePrim("/ground", PrimType::Mesh);
        ground->setMeshData(
            std::make_shared<MeshData>(Prim::createPlaneData(20.f, UpAxis::Z)));
        addShape(groundShader.get(), ground);

        // Helper: define, register, add
        auto add = [&](const std::string& path, MeshData meshData,
                       glm::vec3 pos, glm::vec4 color,
                       const std::string& label) {
            auto* prim = getScene()->definePrim(path, PrimType::Mesh);
            prim->setMeshData(std::make_shared<MeshData>(std::move(meshData)));
            prim->addTranslateOp(pos);
            prim->setDisplayColorAlpha(color);
            addShape(phongShader.get(), prim);
            entries.push_back({prim, label});
        };

        // Row y=0 — basic shapes
        add("/shapes/square", Prim::createSquareData(1.f), {-6.f, 0.f, 0.01f},
            {0.95f, 0.35f, 0.35f, 1.f}, "Square");

        add("/shapes/sphere", Prim::createSphereData(0.7f, 32, 16),
            {-2.f, 0.f, 0.7f}, {0.35f, 0.65f, 0.95f, 1.f}, "Sphere");

        add("/shapes/rectangle", Prim::createRectangleData(1.2f, 0.7f, 0.5f),
            {2.f, 0.f, 0.5f}, {0.95f, 0.80f, 0.20f, 1.f}, "Rectangle");

        // Row y=0 — planes (Y / Z / X)
        add("/shapes/plane_y", Prim::createPlaneData(1.2f, UpAxis::Y),
            {6.f, 0.f, 0.01f}, {0.95f, 0.70f, 0.70f, 1.f}, "Plane (Y-up)");

        add("/shapes/plane_z", Prim::createPlaneData(1.2f, UpAxis::Z),
            {8.f, 0.f, 0.01f}, {0.40f, 0.85f, 0.55f, 1.f}, "Plane (Z-up)");

        add("/shapes/plane_x", Prim::createPlaneData(1.2f, UpAxis::X),
            {10.f, 0.f, 0.6f}, {0.55f, 0.55f, 0.95f, 1.f}, "Plane (X-up)");

        // Row y=5 — cylinders (Y / Z / X)
        add("/shapes/cylinder_y", Prim::createCylinderData(0.35f, 1.2f),
            {-4.f, 5.f, 0.35f}, {0.70f, 0.35f, 0.90f, 1.f},
            "Cylinder (Y-up, lying)");

        add("/shapes/cylinder_z",
            Prim::createCylinderData(0.35f, 1.2f, UpAxis::Z), {0.f, 5.f, 0.6f},
            {0.25f, 0.85f, 0.85f, 1.f}, "Cylinder (Z-up, standing)");

        add("/shapes/cylinder_x",
            Prim::createCylinderData(0.35f, 1.2f, UpAxis::X), {4.f, 5.f, 0.35f},
            {0.90f, 0.55f, 0.25f, 1.f}, "Cylinder (X-up, sideways)");

        // Row y=10 — arrows (Y / Z / X)
        add("/shapes/arrow_y", Prim::createArrowData(0.15f, 1.4f),
            {-4.f, 10.f, 0.f}, {0.95f, 0.55f, 0.15f, 1.f},
            "Arrow (Y-up, horizontal)");

        add("/shapes/arrow_z", Prim::createArrowData(0.15f, 1.4f, UpAxis::Z),
            {0.f, 10.f, 0.f}, {0.55f, 0.95f, 0.15f, 1.f},
            "Arrow (Z-up, pointing up)");

        add("/shapes/arrow_x", Prim::createArrowData(0.15f, 1.4f, UpAxis::X),
            {4.f, 10.f, 0.f}, {0.15f, 0.55f, 0.95f, 1.f},
            "Arrow (X-up, sideways)");

        add("/shapes/arrow_custom",
            Prim::createArrowData(0.20f, 1.0f, UpAxis::Z, 0.35f, 0.5f),
            {8.f, 10.f, 0.f}, {0.95f, 0.35f, 0.75f, 1.f},
            "Arrow (Z-up, fat cap)");

        // Row y=15 — capsules (Y / Z / X)
        add("/shapes/capsule_y", Prim::createCapsuleData(0.35f, 1.0f),
            {-4.f, 15.f, 0.35f}, {0.60f, 0.90f, 0.50f, 1.f},
            "Capsule (Y-up, lying)");

        add("/shapes/capsule_z",
            Prim::createCapsuleData(0.35f, 1.0f, UpAxis::Z), {0.f, 15.f, 0.85f},
            {0.40f, 0.60f, 0.95f, 1.f}, "Capsule (Z-up, standing)");

        add("/shapes/capsule_x",
            Prim::createCapsuleData(0.35f, 1.0f, UpAxis::X), {4.f, 15.f, 0.35f},
            {0.95f, 0.55f, 0.30f, 1.f}, "Capsule (X-up, sideways)");

        // Row y=15 — cones (Y / Z / X)
        add("/shapes/cone_y", Prim::createConeData(0.40f, 1.2f),
            {-8.f, 15.f, 0.f}, {0.95f, 0.80f, 0.25f, 1.f},
            "Cone (Y-up, horizontal)");

        add("/shapes/cone_z", Prim::createConeData(0.40f, 1.2f, UpAxis::Z),
            {8.f, 15.f, 0.f}, {0.30f, 0.90f, 0.70f, 1.f},
            "Cone (Z-up, pointing up)");

        // Row y=20 — points and lines
        {
            std::vector<glm::vec3> pts = {
                {-1.f, 0.f, 0.f},   {0.f, 0.f, 0.f},   {1.f, 0.f, 0.f},
                {-0.5f, 0.f, 0.8f}, {0.5f, 0.f, 0.8f},
            };
            add("/shapes/points", Prim::createPointsData(pts, 0.12f, 12),
                {-4.f, 20.f, 0.f}, {0.95f, 0.35f, 0.55f, 1.f},
                "Points (5 spheres)");
        }
        {
            std::vector<glm::vec3> verts = {
                {0.f, 0.f, 0.f},   {1.f, 0.f, 0.f}, {1.f, 0.f, 0.f},
                {1.f, 0.f, 1.f},   {1.f, 0.f, 1.f}, {0.f, 0.f, 1.f},
                {0.f, 0.f, 1.f},   {0.f, 0.f, 0.f}, {0.5f, 0.f, 0.5f},
                {0.5f, 0.f, 1.2f},
            };
            std::vector<unsigned int> lineIdx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            add("/shapes/lines",
                Prim::createLinesData(verts, lineIdx, 0.04f, 8),
                {2.f, 20.f, 0.f}, {0.35f, 0.75f, 0.95f, 1.f},
                "Lines (5 capsule segments)");
        }

        checkError();
    }

    // ---------------------------------------------------------------------------
    void render() override {
        auto view = getViewMatrix();
        glm::vec3 lightViewPos = glm::vec3(view * glm::vec4(lightPos, 1.f));

        phongShader->use();
        phongShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
        phongShader->setVec3("lightPos", lightViewPos);
        phongShader->setVec3("camPos", glm::vec3(0.f));

        groundShader->use();
        groundShader->setVec3("lightColor", lightColor[0], lightColor[1],
                              lightColor[2]);
        groundShader->setVec3("lightPos", lightViewPos);
        groundShader->setVec3("camPos", glm::vec3(0.f));

        // ImGui panel
        ImGui::Begin("Prim Showcase");
        ImGui::Text("All Prim::create*() functions");
        ImGui::Separator();
        for (auto& e : entries) {
            auto pos = e.prim->getAttribute<glm::vec3>("xformOp:translate");
            auto col =
                e.prim->getAttribute<glm::vec4>("primvars:displaycolorAlpha");
            ImGui::ColorButton("##col", ImVec4(col.r, col.g, col.b, col.a),
                               ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14));
            ImGui::SameLine();
            ImGui::Text("%-30s  (%.1f, %.1f, %.1f)", e.label.c_str(), pos.x,
                        pos.y, pos.z);
        }
        ImGui::Separator();
        ImGui::DragFloat3("Light Pos", &lightPos.x, 0.1f);
        ImGui::ColorEdit3("Light Color", lightColor);
        ImGui::End();

        checkError();
    }
};

// ---------------------------------------------------------------------------
int main() {
    PrimShowcaseApp app;
    app.initialize(1920, 1080, false, UpAxis::Z);
    app.start();
    return 0;
}
