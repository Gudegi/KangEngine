#include "kangEngine.hpp"
#include "physics/physics.hpp"
#include <memory>
#include <vector>

using namespace KE;
using namespace physx;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

static const char* stlVs = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;
layout(location = 7) in vec4 aInstanceColor;

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

out vec3 Normal;
out vec3 FragPos;
out vec4 vColor;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(view * model * vec4(aPos, 1.0));
    Normal = mat3(view * model) * aNormal;
    vColor = aInstanceColor;
}
)";

static const char* stlFs = R"(
#version 410 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec4 vColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
void main() {
    float ambientStrength = 0.15;
    vec3 ambient = ambientStrength * lightColor;
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightPos - FragPos);
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * lightColor;
    float specularStrength = 0.5;
    vec3 V = normalize(-FragPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    FragColor = vec4(vColor.rgb * (ambient + diffuse + specular), vColor.a);
}
)";

static const char* groundVs = R"(
#version 410 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
// column vectors for transform matrix
layout(location = 3) in vec4 aInstanceTransform0;
layout(location = 4) in vec4 aInstanceTransform1;
layout(location = 5) in vec4 aInstanceTransform2;
layout(location = 6) in vec4 aInstanceTransform3;

layout(std140) uniform cameraUBO {
    mat4 view;
    mat4 projection;
};

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

void main() {
    mat4 model = mat4(aInstanceTransform0, aInstanceTransform1,
                      aInstanceTransform2, aInstanceTransform3);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    FragPos = vec3(view * model * vec4(aPos, 1.0));
    Normal = mat3(view * model) * aNormal;
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
// scissorFilter: prevents inter-arm collisions (same word2 → kill)
// ---------------------------------------------------------------------------

static PxFilterFlags scissorFilter(PxFilterObjectAttributes, PxFilterData fd0,
                                   PxFilterObjectAttributes, PxFilterData fd1,
                                   PxPairFlags& pairFlags, const void*, PxU32) {
    if (fd0.word2 != 0 && fd0.word2 == fd1.word2)
        return PxFilterFlag::eKILL;
    pairFlags |= PxPairFlag::eCONTACT_DEFAULT;
    return PxFilterFlag::eDEFAULT;
}

static PhysicsConfig makeScissorConfig() {
    PhysicsConfig cfg;
    cfg.filterShader = scissorFilter;
    cfg.solverType = PxSolverType::eTGS;
    return cfg;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class ScissorLiftApp : public App {
  public:
    std::unique_ptr<Backend::Shader> cubeShader;
    std::unique_ptr<Backend::Shader> planeShader;

    glm::vec3 lightPos = {3.f, 6.f, 4.f};
    float lightColor[3] = {1.f, 1.f, 1.f};

    PhysicsWorld physics{makeScissorConfig()};

    PxArticulationReducedCoordinate* gArticulation = nullptr;
    PxArticulationJointReducedCoordinate* gDriveJoint = nullptr;
    bool gClosing = true;
    bool paused = false;
    bool spaceWasPressed = false;

    struct LinkVisual {
        PxArticulationLink* link;
        Scene::Prim* prim;
    };
    std::vector<LinkVisual> linkVisuals;
    std::vector<std::pair<PxRigidDynamic*, Scene::Prim*>> boxVisuals;
    int linkIdx = 0;
    int boxIdx = 0;

    // -----------------------------------------------------------------------
    void initialize(int w, int h, Backend::BackendType bt) {
        App::initialize(w, h, false, UpAxis::Y, bt);
    }

    // -----------------------------------------------------------------------
    // Helper: create a Prim with a box mesh and register for rendering
    Scene::Prim* makeBoxPrim(const std::string& path, PxVec3 halfExt,
                             glm::vec4 color) {
        auto mesh = Scene::Prim::createRectangleData(
            halfExt.x * 2.f, halfExt.y * 2.f, halfExt.z * 2.f);
        auto meshPtr = std::make_shared<Scene::MeshData>(std::move(mesh));
        auto* prim = getScene()->definePrim(path, Scene::PrimType::Mesh);
        prim->setMeshData(meshPtr);
        prim->setAttribute("primvars:displaycolorAlpha", color);
        addShape(cubeShader.get(), prim);
        return prim;
    }

    void trackLink(PxArticulationLink* link, PxVec3 halfExt, glm::vec4 color) {
        auto* prim = makeBoxPrim("/lift/link_" + std::to_string(linkIdx++),
                                 halfExt, color);
        linkVisuals.push_back({link, prim});
    }

    // -----------------------------------------------------------------------
    // Scissor lift construction — faithful port of SnippetArticulation.cpp
    void createScissorLift() {
        auto* px = physics.getPhysics();
        auto* mat = physics.getMaterial();
        auto* pxScene = physics.getScene();

        const glm::vec4 baseColor = {0.35f, 0.35f, 0.35f, 1.f};
        const glm::vec4 armColor = {0.30f, 0.45f, 0.70f, 1.f};
        const glm::vec4 topColor = {0.80f, 0.50f, 0.20f, 1.f};

        const PxReal runnerLength = 2.f;
        const PxReal placementDistance = 1.8f;
        const PxReal cosAng = placementDistance / runnerLength;
        const PxReal angle = PxAcos(cosAng);
        const PxReal sinAng = PxSin(angle);

        const PxQuat leftRot(-angle, PxVec3(1.f, 0.f, 0.f));
        const PxQuat rightRot(angle, PxVec3(1.f, 0.f, 0.f));

        // (1) Base
        PxArticulationLink* base = gArticulation->createLink(
            nullptr, PxTransform(PxVec3(0.f, 0.25f, 0.f)));
        PxRigidActorExt::createExclusiveShape(
            *base, PxBoxGeometry(0.5f, 0.25f, 1.5f), *mat);
        PxRigidBodyExt::updateMassAndInertia(*base, 3.f);
        trackLink(base, PxVec3(0.5f, 0.25f, 1.5f), baseColor);

        gArticulation->setSolverIterationCounts(32);

        // Root links
        PxArticulationLink* leftRoot = gArticulation->createLink(
            base, PxTransform(PxVec3(0.f, 0.55f, -0.9f)));
        PxRigidActorExt::createExclusiveShape(
            *leftRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *mat);
        PxRigidBodyExt::updateMassAndInertia(*leftRoot, 1.f);
        trackLink(leftRoot, PxVec3(0.5f, 0.05f, 0.05f), baseColor);

        PxArticulationLink* rightRoot = gArticulation->createLink(
            base, PxTransform(PxVec3(0.f, 0.55f, 0.9f)));
        PxRigidActorExt::createExclusiveShape(
            *rightRoot, PxBoxGeometry(0.5f, 0.05f, 0.05f), *mat);
        PxRigidBodyExt::updateMassAndInertia(*rightRoot, 1.f);
        trackLink(rightRoot, PxVec3(0.5f, 0.05f, 0.05f), baseColor);

        // Fixed joint for leftRoot
        PxArticulationJointReducedCoordinate* joint;
        joint = static_cast<PxArticulationJointReducedCoordinate*>(
            leftRoot->getInboundJoint());
        joint->setJointType(PxArticulationJointType::eFIX);
        joint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, -0.9f)));
        joint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));

        // Prismatic drive joint for rightRoot (animates the lift)
        gDriveJoint = static_cast<PxArticulationJointReducedCoordinate*>(
            rightRoot->getInboundJoint());
        gDriveJoint->setJointType(PxArticulationJointType::ePRISMATIC);
        gDriveJoint->setMotion(PxArticulationAxis::eZ,
                               PxArticulationMotion::eLIMITED);
        gDriveJoint->setLimit(PxArticulationAxis::eZ, -1.4f, 0.2f);
        gDriveJoint->setDrive(PxArticulationAxis::eZ, 100000.f, 0.f,
                              PX_MAX_F32);
        gDriveJoint->setParentPose(PxTransform(PxVec3(0.f, 0.25f, 0.9f)));
        gDriveJoint->setChildPose(PxTransform(PxVec3(0.f, -0.05f, 0.f)));

        const PxU32 linkHeight = 3;

        // --- Scissor frame loop (shared by both sides) ---
        auto buildScissorArm = [&](PxVec3 xOffset,
                                   PxArticulationLink*& currLeft,
                                   PxArticulationLink*& currRight,
                                   PxQuat& leftParentRot,
                                   PxQuat& rightParentRot) {
            for (PxU32 i = 0; i < linkHeight; ++i) {
                const PxVec3 pos =
                    xOffset + PxVec3(0.f, 0.55f + 0.1f * (1 + i), 0.f);

                PxArticulationLink* leftLink = gArticulation->createLink(
                    currLeft,
                    PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f),
                                leftRot));
                PxRigidActorExt::createExclusiveShape(
                    *leftLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *mat);
                PxRigidBodyExt::updateMassAndInertia(*leftLink, 1.f);
                trackLink(leftLink, PxVec3(0.05f, 0.05f, 1.f), armColor);

                const PxVec3 leftAnchor =
                    pos + PxVec3(0.f, sinAng * (2 * i), -0.9f);
                joint = static_cast<PxArticulationJointReducedCoordinate*>(
                    leftLink->getInboundJoint());
                joint->setJointType(PxArticulationJointType::eREVOLUTE);
                joint->setParentPose(PxTransform(
                    currLeft->getGlobalPose().transformInv(leftAnchor),
                    leftParentRot));
                joint->setChildPose(
                    PxTransform(PxVec3(0.f, 0.f, -1.f), rightRot));
                joint->setMotion(PxArticulationAxis::eTWIST,
                                 PxArticulationMotion::eLIMITED);
                joint->setLimit(PxArticulationAxis::eTWIST, -PxPi, angle);
                leftParentRot = leftRot;

                PxArticulationLink* rightLink = gArticulation->createLink(
                    currRight,
                    PxTransform(pos + PxVec3(0.f, sinAng * (2 * i + 1), 0.f),
                                rightRot));
                PxRigidActorExt::createExclusiveShape(
                    *rightLink, PxBoxGeometry(0.05f, 0.05f, 1.f), *mat);
                PxRigidBodyExt::updateMassAndInertia(*rightLink, 1.f);
                trackLink(rightLink, PxVec3(0.05f, 0.05f, 1.f), armColor);

                const PxVec3 rightAnchor =
                    pos + PxVec3(0.f, sinAng * (2 * i), 0.9f);
                joint = static_cast<PxArticulationJointReducedCoordinate*>(
                    rightLink->getInboundJoint());
                joint->setJointType(PxArticulationJointType::eREVOLUTE);
                joint->setParentPose(PxTransform(
                    currRight->getGlobalPose().transformInv(rightAnchor),
                    rightParentRot));
                joint->setChildPose(
                    PxTransform(PxVec3(0.f, 0.f, 1.f), leftRot));
                joint->setMotion(PxArticulationAxis::eTWIST,
                                 PxArticulationMotion::eLIMITED);
                joint->setLimit(PxArticulationAxis::eTWIST, -angle, PxPi);
                rightParentRot = rightRot;

                // D6 joint connecting the two crossing arms at their pivot
                PxD6Joint* d6 =
                    PxD6JointCreate(*px, leftLink, PxTransform(PxIdentity),
                                    rightLink, PxTransform(PxIdentity));
                d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
                d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
                d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

                currLeft = rightLink;
                currRight = leftLink;
            }
        };

        // Right-side arm (x = +0.5)
        PxArticulationLink* currLeft = leftRoot;
        PxArticulationLink* currRight = rightRoot;
        PxQuat leftParentRot(PxIdentity), rightParentRot(PxIdentity);
        buildScissorArm(PxVec3(0.5f, 0.f, 0.f), currLeft, currRight,
                        leftParentRot, rightParentRot);

        // Top caps
        PxArticulationLink* leftTop = gArticulation->createLink(
            currLeft, currLeft->getGlobalPose().transform(PxTransform(
                          PxVec3(-0.5f, 0.f, -1.f), leftParentRot)));
        PxRigidActorExt::createExclusiveShape(
            *leftTop, PxBoxGeometry(0.5f, 0.05f, 0.05f), *mat);
        PxRigidBodyExt::updateMassAndInertia(*leftTop, 1.f);
        trackLink(leftTop, PxVec3(0.5f, 0.05f, 0.05f), topColor);

        PxArticulationLink* rightTop = gArticulation->createLink(
            currRight, currRight->getGlobalPose().transform(PxTransform(
                           PxVec3(-0.5f, 0.f, 1.f), rightParentRot)));
        PxRigidActorExt::createExclusiveShape(
            *rightTop, PxBoxGeometry(0.5f, 0.05f, 0.05f), *mat);
        PxRigidBodyExt::updateMassAndInertia(*rightTop, 1.f);
        trackLink(rightTop, PxVec3(0.5f, 0.05f, 0.05f), topColor);

        joint = static_cast<PxArticulationJointReducedCoordinate*>(
            leftTop->getInboundJoint());
        joint->setParentPose(
            PxTransform(PxVec3(0.f, 0.f, -1.f),
                        currLeft->getGlobalPose().q.getConjugate()));
        joint->setChildPose(PxTransform(
            PxVec3(0.5f, 0.f, 0.f), leftTop->getGlobalPose().q.getConjugate()));
        joint->setJointType(PxArticulationJointType::eREVOLUTE);
        joint->setMotion(PxArticulationAxis::eTWIST,
                         PxArticulationMotion::eFREE);

        joint = static_cast<PxArticulationJointReducedCoordinate*>(
            rightTop->getInboundJoint());
        joint->setParentPose(
            PxTransform(PxVec3(0.f, 0.f, 1.f),
                        currRight->getGlobalPose().q.getConjugate()));
        joint->setChildPose(
            PxTransform(PxVec3(0.5f, 0.f, 0.f),
                        rightTop->getGlobalPose().q.getConjugate()));
        joint->setJointType(PxArticulationJointType::eREVOLUTE);
        joint->setMotion(PxArticulationAxis::eTWIST,
                         PxArticulationMotion::eFREE);

        // Left-side arm (x = -0.5)
        currLeft = leftRoot;
        currRight = rightRoot;
        leftParentRot = PxQuat(PxIdentity);
        rightParentRot = PxQuat(PxIdentity);
        buildScissorArm(PxVec3(-0.5f, 0.f, 0.f), currLeft, currRight,
                        leftParentRot, rightParentRot);

        // Cross-side D6 joints linking both frames to the top caps
        PxD6Joint* d6;
        d6 = PxD6JointCreate(*px, currLeft, PxTransform(PxVec3(0.f, 0.f, -1.f)),
                             leftTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
        d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
        d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
        d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

        d6 = PxD6JointCreate(*px, currRight, PxTransform(PxVec3(0.f, 0.f, 1.f)),
                             rightTop, PxTransform(PxVec3(-0.5f, 0.f, 0.f)));
        d6->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
        d6->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
        d6->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

        // Top platform
        const PxTransform topPose(
            PxVec3(0.f, leftTop->getGlobalPose().p.y + 0.15f, 0.f));
        PxArticulationLink* top = gArticulation->createLink(leftTop, topPose);
        PxRigidActorExt::createExclusiveShape(
            *top, PxBoxGeometry(0.5f, 0.1f, 1.5f), *mat);
        PxRigidBodyExt::updateMassAndInertia(*top, 1.f);
        trackLink(top, PxVec3(0.5f, 0.1f, 1.5f), topColor);

        joint = static_cast<PxArticulationJointReducedCoordinate*>(
            top->getInboundJoint());
        joint->setJointType(PxArticulationJointType::eFIX);
        joint->setParentPose(PxTransform(PxVec3(0.f, 0.f, 0.f)));
        joint->setChildPose(PxTransform(PxVec3(0.f, -0.15f, -0.9f)));

        // Add to scene, set damping and collision filter on all links
        pxScene->addArticulation(*gArticulation);

        for (PxU32 i = 0; i < gArticulation->getNbLinks(); ++i) {
            PxArticulationLink* link;
            gArticulation->getLinks(&link, 1, i);
            link->setLinearDamping(0.2f);
            link->setAngularDamping(0.2f);
            link->setMaxAngularVelocity(20.f);
            link->setMaxLinearVelocity(100.f);
            if (link != top) {
                for (PxU32 b = 0; b < link->getNbShapes(); ++b) {
                    PxShape* shape;
                    link->getShapes(&shape, 1, b);
                    shape->setSimulationFilterData(PxFilterData(0, 0, 1, 0));
                }
            }
        }

        // Dynamic boxes stacked on top of the lift
        struct BoxInit {
            PxVec3 pos;
            glm::vec4 color;
        };
        const BoxInit boxes[] = {
            {{-0.25f, 5.f, 0.5f}, {0.85f, 0.25f, 0.20f, 1.f}},
            {{0.25f, 5.f, 0.5f}, {0.20f, 0.60f, 0.90f, 1.f}},
            {{-0.25f, 4.5f, 0.5f}, {0.90f, 0.80f, 0.20f, 1.f}},
            {{0.25f, 4.5f, 0.5f}, {0.30f, 0.85f, 0.35f, 1.f}},
            {{-0.25f, 5.f, 0.f}, {0.70f, 0.20f, 0.80f, 1.f}},
            {{0.25f, 5.f, 0.f}, {0.20f, 0.80f, 0.70f, 1.f}},
            {{-0.25f, 4.5f, 0.f}, {1.00f, 0.50f, 0.00f, 1.f}},
            {{0.25f, 4.5f, 0.f}, {0.50f, 0.30f, 0.70f, 1.f}},
        };
        const PxVec3 boxHalf(0.25f);
        for (auto& b : boxes) {
            PxRigidDynamic* box = px->createRigidDynamic(PxTransform(b.pos));
            PxShape* shape = PxRigidActorExt::createExclusiveShape(
                *box, PxBoxGeometry(boxHalf), *mat);
            shape->setContactOffset(0.2f);
            PxRigidBodyExt::updateMassAndInertia(*box, 0.5f);
            pxScene->addActor(*box);

            auto* prim = makeBoxPrim("/boxes/box_" + std::to_string(boxIdx++),
                                     boxHalf, b.color);
            boxVisuals.push_back({box, prim});
        }
    }

    // -----------------------------------------------------------------------
    void setup() override {
        cubeShader = getGraphicsDevice()->createShader(stlVs, stlFs);
        planeShader = getGraphicsDevice()->createShader(groundVs, groundFs);

        cubeShader->use();
        cubeShader->setUniformBlockBinding("cameraUBO", 0);

        planeShader->use();
        planeShader->setUniformBlockBinding("cameraUBO", 0);

        // Ground
        physics.addDefaultGround();
        auto* planePrim =
            getScene()->definePrim("/ground", Scene::PrimType::Mesh);
        planePrim->setMeshData(std::make_shared<Scene::MeshData>(
            Scene::Prim::createPlaneData(30.f)));
        addShape(planeShader.get(), planePrim);

        // Articulation
        gArticulation =
            physics.getPhysics()->createArticulationReducedCoordinate();
        createScissorLift();

        checkError();
    }

    // -----------------------------------------------------------------------
    void preRender() override {
        // Lighting uniforms
        auto view = getViewMatrix();
        glm::vec3 lightViewPos = glm::vec3(view * glm::vec4(lightPos, 1.f));

        cubeShader->use();
        cubeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                            lightColor[2]);
        cubeShader->setVec3("lightPos", lightViewPos);

        planeShader->use();
        planeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
        planeShader->setVec3("lightPos", lightViewPos);

        // Spacebar: pause/resume
        bool spaceDown = glfwGetKey(getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasPressed)
            paused = !paused;
        spaceWasPressed = spaceDown;

        if (!paused) {
            // Animate drive joint (oscillate open/close)
            const float dt = 1.f / 60.f;
            PxReal driveVal =
                gDriveJoint->getDriveTarget(PxArticulationAxis::eZ);
            if (gClosing && driveVal < -1.2f)
                gClosing = false;
            else if (!gClosing && driveVal > 0.f)
                gClosing = true;
            driveVal += gClosing ? -dt * 0.25f : dt * 0.25f;
            gDriveJoint->setDriveTarget(PxArticulationAxis::eZ, driveVal);

            physics.step();
        }

        // Sync articulation link transforms → Prims
        for (auto& [link, prim] : linkVisuals) {
            auto p = link->getGlobalPose();
            prim->setAttribute("xformOp:translate", pxToGlm(p.p));
            prim->setAttribute("xformOp:rotateQuaternion", pxToGlm(p.q));
        }

        // Sync dynamic box transforms → Prims
        for (auto& [box, prim] : boxVisuals) {
            auto p = box->getGlobalPose();
            prim->setAttribute("xformOp:translate", pxToGlm(p.p));
            prim->setAttribute("xformOp:rotateQuaternion", pxToGlm(p.q));
        }

        // Ground plane lighting
        planeShader->use();
        planeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
        planeShader->setVec3("lightPos", lightViewPos);
        planeShader->setMat3("normalMat",
                             glm::mat3(glm::transpose(glm::inverse(view))));

        checkError();
    }

    // -----------------------------------------------------------------------
    void render() override {
        KE_TRACE_FUNCTION();
        ImGui::Begin("Scissor Lift");
        ImGui::Text("PhysX Articulation — Scissor Lift");
        ImGui::Separator();
        PxReal driveVal = gDriveJoint->getDriveTarget(PxArticulationAxis::eZ);
        ImGui::Text("Drive: %.3f  [%s]", driveVal,
                    gClosing ? "closing" : "opening");
        ImGui::Text("Links: %u | Boxes: %zu", gArticulation->getNbLinks(),
                    boxVisuals.size());
        ImGui::Spacing();
        ImGui::Text("Space: %s", paused ? "PAUSED" : "running");
        ImGui::Text("WASD / mouse: camera");
        ImGui::End();

        auto view = this->getViewMatrix();
        planeShader->use();
        glm::mat4 groundModel = glm::mat4(1.0f);
        glm::mat3 normalMat = glm::mat3(transpose(inverse(view * groundModel)));
        planeShader->setMat3("normalMat", normalMat);
        planeShader->setVec3("lightPos",
                             glm::vec3(view * glm::vec4(lightPos, 1.0f)));
        planeShader->setVec3("lightColor", lightColor[0], lightColor[1],
                             lightColor[2]);
    }

    void postRender() override {}
};

// ---------------------------------------------------------------------------
int main() {
    ScissorLiftApp app;
    app.initialize(1920, 1080, Backend::BackendType::OpenGL);
    app.start();
    return 0;
}
