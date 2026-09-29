#include "physics/d6_joint.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace KE;

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    auto world = std::make_unique<PhysicsWorld>(PhysicsConfig{});
    auto* actor = world->createDynamicBox(glm::vec3(.1f), glm::vec3(0, 0, 3));
    auto body = std::make_shared<JointBody>(*world, actor);
    D6JointConfig config;
    auto survivor = D6Joint::create(*world, nullptr, body, config);
    PxConstraint* constraint = nullptr;
    actor->getConstraints(&constraint, 1);
    check(constraint && (constraint->getFlags() & PxConstraintFlag::eDISABLE_CONSTRAINT),
          "initial disabled state not applied");
    actor->putToSleep();
    survivor->setEnabled(false);
    check(actor->isSleeping(), "unchanged disabled state woke actor");
    survivor->setEnabled(true);
    check(!actor->isSleeping(), "activation did not wake actor");
    check(!(constraint->getFlags() & PxConstraintFlag::eDISABLE_CONSTRAINT),
          "activation did not enable constraint");
    actor->putToSleep();
    survivor->setEnabled(true);
    check(actor->isSleeping(), "unchanged enabled state woke actor");
    survivor->setEnabled(false);
    check(!actor->isSleeping(), "deactivation did not wake actor");
    const JointFrame first = {1, 2, 3, 0, 0, 0, 1};
    const JointFrame second = {4, 5, 6, 0, 0, 0, 1};
    survivor->setFrame0(first);
    check(survivor->frame0() == first && survivor->frame1() == config.frame1,
          "frame0 setter changed the other frame");
    survivor->setFrame1(second);
    check(survivor->frame0() == first && survivor->frame1() == second,
          "frame1 setter changed the other frame");
    bool badFrameRejected = false;
    try { survivor->setFrame0(JointFrame{}); }
    catch (const std::invalid_argument&) { badFrameRejected = true; }
    check(badFrameRejected && survivor->frame0() == first, "invalid frame accepted");
    badFrameRejected = false;
    try { survivor->setFrame1(JointFrame{}); }
    catch (const std::invalid_argument&) { badFrameRejected = true; }
    check(badFrameRejected && survivor->frame1() == second, "invalid frame accepted");
    std::vector<std::shared_ptr<D6Joint>> releasedHandles;
    for (int i = 0; i < 10000; ++i) {
        auto joint = D6Joint::create(*world, nullptr, body, config);
        check(world->numTrackedD6Joints() == 2, "registration accumulated");
        if (i % 2 == 0) {
            joint->release();
            joint->release();
            releasedHandles.push_back(joint);
        }
        // Odd iterations exercise destruction without explicit release.
        joint.reset();
        check(world->numTrackedD6Joints() == 1, "unregistration failed");
    }
    check(survivor->valid(), "unrelated live joint invalidated");
    survivor->release();
    bool releasedRejected = false;
    try { survivor->setEnabled(false); }
    catch (const std::runtime_error&) { releasedRejected = true; }
    check(releasedRejected, "no-op update bypassed released handle validation");
    survivor.reset();
    check(world->numTrackedD6Joints() == 0, "survivor still registered");

    auto invalid = D6Joint::create(*world, nullptr, body, config);
    auto invalidAtShutdown = D6Joint::create(*world, nullptr, body, config);
    actor->release();
    check(!invalid->valid(), "deleted endpoint not invalidated");
    bool rejected = false;
    try { invalid->setEnabled(true); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "invalid endpoint accepted update");
    invalid->release();
    invalid->release();
    check(world->numTrackedD6Joints() == 1, "invalid joint did not unregister");

    auto* other = world->createDynamicBox(glm::vec3(.1f), glm::vec3(2, 0, 3));
    auto otherBody = std::make_shared<JointBody>(*world, other);
    std::vector<std::shared_ptr<D6Joint>> live;
    for (int i = 0; i < 128; ++i)
        live.push_back(D6Joint::create(*world, nullptr, otherBody, config));
    check(world->numTrackedD6Joints() == 129, "unexpected shutdown registry size");
    // Keep handles alive beyond world destruction to test shutdown unregistering.
    world.reset();
    check(!otherBody->valid(), "world endpoint still valid");
    check(!invalidAtShutdown->valid(), "invalid joint survived shutdown");
    invalidAtShutdown->release();
    for (auto& joint : live) {
        check(!joint->valid(), "joint survived shutdown");
        joint->release();
        joint->release();
    }
    releasedHandles.clear();
    std::cout << "D6 registry PASS: 10000 cycles, retained released handles, "
                 "endpoint deletion, 129 shutdown entries\n";
}
