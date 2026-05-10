#include "engine/scene/debug_draw.hpp"
#include "kangEngine.hpp"
#include "engine/graphics/material/colors.hpp"
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

using namespace KE;
using namespace KE::Scene;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class PrimShowcaseApp : public App {
  public:
    std::unique_ptr<Backend::Shader> phongShader;
    std::unique_ptr<Backend::Shader> groundShader;

    struct ShapeEntry {
        Prim* prim;
        std::string label;
    };
    std::vector<ShapeEntry> entries;

    glm::vec3 defaultSunDirection() const {
        return glm::normalize(upPos(0.35f, 0.45f, 0.82f));
    }

    // ---------------------------------------------------------------------------
    void setup() override {
        setLight(DirectionalLight{defaultSunDirection(), glm::vec3(1.0f), 0.75f,
                                  glm::vec3(0.10f)});

        auto commonVSPath = KE::getAssetPath("shaders/common.vs");
        auto commonFSPath = KE::getAssetPath("shaders/common.fs");
        auto groundFSPath = KE::getAssetPath("shaders/checkerboard.fs");

        phongShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                                commonFSPath);
        groundShader = getGraphicsDevice()->createShaderFromFile(commonVSPath,
                                                                 groundFSPath);

        phongShader->setUniformBlockBinding("cameraUBO", 0);
        phongShader->setUniformBlockBinding("lightUBO", 1);
        phongShader->setUniformBlockBinding("shadowUBO", 2);

        groundShader->setUniformBlockBinding("cameraUBO", 0);
        groundShader->setUniformBlockBinding("lightUBO", 1);
        groundShader->setUniformBlockBinding("shadowUBO", 2);

        auto white = ColorLibrary::get(KE::ColorType::WHITE);
        auto pG = ColorLibrary::get(KE::ColorType::PASTEL_GREEN);
        groundShader->use();
        groundShader->setVec4("checkerColor1",
                              glm::vec4(white.r, white.g, white.b, white.a));
        groundShader->setVec4("checkerColor2",
                              glm::vec4(pG.r, pG.g, pG.b, pG.a));

        // Ground
        {
            MeshPrimDesc desc;
            desc.shader = groundShader.get();
            desc.path = "/ground";
            desc.meshData = Prim::createPlaneData(20.f, _upAxis);
            addMeshPrim(std::move(desc));
        }

        // Helper: define, register, add
        auto add = [&](const std::string& path, MeshData meshData,
                       glm::vec3 pos, glm::vec4 color, const std::string& label,
                       glm::quat ori = {1.0f, 0.0f, 0.0f, 0.0f},
                       bool doubleSided = false) {
            MeshPrimDesc desc;
            desc.shader = phongShader.get();
            desc.path = path;
            desc.meshData = std::move(meshData);
            desc.position = pos;
            desc.orientation = ori;
            desc.color = color;
            desc.doubleSided = doubleSided;
            auto result = addMeshPrim(std::move(desc));
            if (result.prim)
                entries.push_back({result.prim, label});
        };

        auto prims =
            Prim::defineCoordinateAxes(getScene(), "/shapes/coordinate_axes",
                                       1.8f, 0.04f, 16, upPos(0.f, 0.f, 0.02f));
        for (auto* p : prims)
            addShape(phongShader.get(), p);
        if (!prims.empty())
            entries.push_back({prims[0], "Scene Axes (arrow prims)"});

        {
            std::vector<glm::vec3> starts = {
                upPos(-6.5f, -6.f, 0.05f),
                upPos(-6.f, -6.f, 0.05f),
                upPos(-5.5f, -6.f, 0.05f),
            };
            std::vector<glm::vec3> ends = {
                upPos(-6.2f, -6.f, 1.1f),
                upPos(-5.7f, -6.f, 0.7f),
                upPos(-5.2f, -6.f, 1.4f),
            };
            std::vector<glm::vec4> colors = {
                {1.0f, 0.25f, 0.20f, 1.0f},
                {0.20f, 0.90f, 0.35f, 1.0f},
                {0.25f, 0.45f, 1.0f, 1.0f},
            };
            DebugDraw::logLines(this, phongShader.get(), "/debug/sample_lines",
                                starts, ends, colors, 0.02f, 8);
        }

        {
            constexpr int spokeCount = 16;
            const glm::vec3 center = {-3.f, -6.f, 0.08f};
            const float radius = 1.1f;
            std::vector<glm::vec3> starts;
            std::vector<glm::vec3> ends;
            std::vector<glm::vec4> colors;
            starts.reserve(spokeCount);
            ends.reserve(spokeCount);
            colors.reserve(spokeCount);

            for (int i = 0; i < spokeCount; ++i) {
                float t =
                    static_cast<float>(i) / static_cast<float>(spokeCount);
                float angle = t * glm::two_pi<float>();
                glm::vec3 end =
                    center + glm::vec3(glm::cos(angle) * radius, 0.0f,
                                       glm::sin(angle) * radius);
                starts.push_back(upPos(center.x, center.y, center.z));
                ends.push_back(upPos(end.x, end.y, end.z));
                colors.push_back({1.0f - t, 0.35f + 0.4f * t, t, 1.0f});
            }

            DebugDraw::logLines(this, phongShader.get(),
                                "/debug/radial_log_lines", starts, ends, colors,
                                0.012f, 8);
        }

        {
            constexpr int arrowCount = 12;
            const glm::vec3 center = {0.0f, -6.f, 0.08f};
            const float radius = 1.1f;
            std::vector<glm::vec3> starts;
            std::vector<glm::vec3> ends;
            std::vector<glm::vec4> colors;
            starts.reserve(arrowCount);
            ends.reserve(arrowCount);
            colors.reserve(arrowCount);

            for (int i = 0; i < arrowCount; ++i) {
                float t =
                    static_cast<float>(i) / static_cast<float>(arrowCount);
                float angle = t * glm::two_pi<float>();
                glm::vec3 end =
                    center + glm::vec3(glm::cos(angle) * radius,
                                       glm::sin(angle) * radius, 0.0f);
                starts.push_back(upPos(center.x, center.y, center.z));
                ends.push_back(upPos(end.x, end.y, end.z));
                colors.push_back({t, 1.0f - t, 0.25f + 0.5f * t, 1.0f});
            }

            DebugDraw::logArrows(this, phongShader.get(),
                                 "/debug/radial_log_arrows", starts, ends,
                                 colors, 0.035f, 12);
        }

        // Row y=0 — basic shapes
        glm::quat squareOri = {0.9238795, 0.3826834, 0, 0};
        glm::vec3 squarePos = {-6.f, -3.f, 1.01f};
        add("/shapes/square", Prim::createSquareData(1.f),
            upPos(squarePos.x, squarePos.y, squarePos.z),
            {0.95f, 0.35f, 0.35f, 1.f}, "Square",
            upQuat(squareOri.w, squareOri.x, squareOri.y, squareOri.z));
        DebugDraw::logCoordinateAxes(
            this, phongShader.get(), "/debug/square_axis",
            upPos(squarePos.x, squarePos.y, squarePos.z),
            upQuat(squareOri.w, squareOri.x, squareOri.y, squareOri.z), 1.8f,
            0.01f);

        add("/shapes/sphere", Prim::createSphereData(0.7f, 32, 16),
            upPos(-3.f, -3.f, 0.7f), {0.35f, 0.65f, 0.95f, 1.f}, "Sphere");

        add("/shapes/rectangle", Prim::createRectangleData(1.2f, 0.7f, 0.5f),
            upPos(0.f, -3.f, 0.5f), {0.95f, 0.80f, 0.20f, 1.f}, "Rectangle");

        // Row y=0 — planes (Y / Z / X)
        add("/shapes/plane_y", Prim::createPlaneData(1.2f, UpAxis::Y),
            upPos(3.f, -3.f, 0.01f), {0.95f, 0.70f, 0.70f, 1.f}, "Plane (Y-up)",
            {1.0f, 0.0f, 0.0f, 0.0f}, true);

        add("/shapes/plane_z", Prim::createPlaneData(1.2f, UpAxis::Z),
            upPos(6.f, -3.f, 0.01f), {0.40f, 0.85f, 0.55f, 1.f}, "Plane (Z-up)",
            {1.0f, 0.0f, 0.0f, 0.0f}, true);

        add("/shapes/plane_x", Prim::createPlaneData(1.2f, UpAxis::X),
            upPos(-6.f, 0.f, 0.6f), {0.55f, 0.55f, 0.95f, 1.f}, "Plane (X-up)",
            {1.0f, 0.0f, 0.0f, 0.0f}, true);

        // Row y=5 — cylinders (Y / Z / X)
        add("/shapes/cylinder_y", Prim::createCylinderData(0.35f, 1.2f),
            upPos(-3.f, 0.f, 0.35f), {0.70f, 0.35f, 0.90f, 1.f},
            "Cylinder (Y-up, lying)");

        add("/shapes/cylinder_z",
            Prim::createCylinderData(0.35f, 1.2f, UpAxis::Z),
            upPos(3.f, 0.f, 0.6f), {0.25f, 0.85f, 0.85f, 1.f},
            "Cylinder (Z-up, standing)");

        add("/shapes/cylinder_x",
            Prim::createCylinderData(0.35f, 1.2f, UpAxis::X),
            upPos(6.f, 0.f, 0.35f), {0.90f, 0.55f, 0.25f, 1.f},
            "Cylinder (X-up, sideways)");

        // Row y=10 — arrows (Y / Z / X)
        add("/shapes/arrow_y", Prim::createArrowData(0.15f, 1.4f),
            upPos(-6.f, 3.f, 0.f), {0.95f, 0.55f, 0.15f, 1.f},
            "Arrow (Y-up, horizontal)");

        add("/shapes/arrow_z", Prim::createArrowData(0.15f, 1.4f, UpAxis::Z),
            upPos(-3.f, 3.f, 0.f), {0.55f, 0.95f, 0.15f, 1.f},
            "Arrow (Z-up, pointing up)");

        add("/shapes/arrow_x", Prim::createArrowData(0.15f, 1.4f, UpAxis::X),
            upPos(0.f, 3.f, 0.f), {0.15f, 0.55f, 0.95f, 1.f},
            "Arrow (X-up, sideways)");

        add("/shapes/arrow_custom",
            Prim::createArrowData(0.20f, 1.0f, UpAxis::Z, 0.35f, 0.5f),
            upPos(3.f, 3.f, 0.f), {0.95f, 0.35f, 0.75f, 1.f},
            "Arrow (Z-up, fat cap)");

        // Row y=15 — capsules (Y / Z / X)
        add("/shapes/capsule_y", Prim::createCapsuleData(0.35f, 1.0f),
            upPos(6.f, 3.f, 0.35f), {0.60f, 0.90f, 0.50f, 1.f},
            "Capsule (Y-up, lying)");

        add("/shapes/capsule_z",
            Prim::createCapsuleData(0.35f, 1.0f, UpAxis::Z),
            upPos(-6.f, 6.f, 0.85f), {0.40f, 0.60f, 0.95f, 1.f},
            "Capsule (Z-up, standing)");

        add("/shapes/capsule_x",
            Prim::createCapsuleData(0.35f, 1.0f, UpAxis::X),
            upPos(-3.f, 6.f, 0.35f), {0.95f, 0.55f, 0.30f, 1.f},
            "Capsule (X-up, sideways)");

        // Row y=15 — cones (Y / Z / X)
        add("/shapes/cone_y", Prim::createConeData(0.40f, 1.2f),
            upPos(0.f, 6.f, 0.f), {0.95f, 0.80f, 0.25f, 1.f},
            "Cone (Y-up, horizontal)");

        add("/shapes/cone_z", Prim::createConeData(0.40f, 1.2f, UpAxis::Z),
            upPos(3.f, 6.f, 0.f), {0.30f, 0.90f, 0.70f, 1.f},
            "Cone (Z-up, pointing up)");

        // Row y=20 — points and lines (instanced)
        {
            std::vector<glm::vec3> pts = {
                upPos(5.4f, 5.6f, 0.f),  upPos(6.0f, 5.6f, 0.f),
                upPos(6.6f, 5.6f, 0.f),  upPos(5.7f, 6.2f, 0.8f),
                upPos(6.3f, 6.2f, 0.8f),
            };
            auto prims =
                Prim::definePoints(getScene(), "/shapes/points", pts, 0.12f,
                                   {0.95f, 0.35f, 0.55f, 1.f}, 12);
            for (auto* p : prims)
                addShape(phongShader.get(), p);
            if (!prims.empty())
                entries.push_back({prims[0], "Points (5 spheres, instanced)"});
        }
        {
            std::vector<glm::vec3> verts = {
                upPos(4.8f, 4.2f, 0.f),  upPos(6.0f, 4.2f, 0.f),
                upPos(6.0f, 4.2f, 0.f),  upPos(6.0f, 4.2f, 1.f),
                upPos(6.0f, 4.2f, 1.f),  upPos(4.8f, 4.2f, 1.f),
                upPos(4.8f, 4.2f, 1.f),  upPos(4.8f, 4.2f, 0.f),
                upPos(5.4f, 4.2f, 0.5f), upPos(5.4f, 4.2f, 1.2f),
            };
            std::vector<unsigned int> lineIdx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            auto prims =
                Prim::defineLines(getScene(), "/shapes/lines", verts, lineIdx,
                                  0.04f, {0.35f, 0.75f, 0.95f, 1.f}, 8);
            for (auto* p : prims)
                addShape(phongShader.get(), p);
            if (!prims.empty())
                entries.push_back({prims[0], "Lines (5 capsules, instanced)"});
        }

        checkError();
    }

    // ---------------------------------------------------------------------------
    void render() override {
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
