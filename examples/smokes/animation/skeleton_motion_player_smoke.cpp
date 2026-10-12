#include "animation/skeleton_motion_player.hpp"

#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace KE::Animation;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(float actual, float expected, float tolerance,
          const std::string& message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            message + ": actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected));
}

void samePose(const SkeletonState& actual, const SkeletonState& expected) {
    require((actual.rootTranslation() - expected.rootTranslation()).norm() <
                2e-5f,
            "root pose discontinuity");
    for (int joint = 0; joint < actual.numJoints(); ++joint) {
        near(std::abs(actual.rotation(joint).dot(expected.rotation(joint))),
             1.0f, 2e-5f, "joint rotation discontinuity");
    }
}

std::shared_ptr<SkeletonTree> makeTree() {
    return std::make_shared<SkeletonTree>(
        std::vector<std::string>{"root", "spine", "head", "shoulder", "hand"},
        std::vector<int>{-1, 0, 1, 1, 3},
        std::vector<Eigen::Vector3f>{
            {0, 0, 0}, {0, 0, .4f}, {0, 0, .3f}, {0, .2f, .2f}, {0, .3f, 0}},
        std::vector<Eigen::Quaternionf>(5, Eigen::Quaternionf::Identity()),
        std::vector<int>(5, 1));
}

SkeletonMotion makeClip(const std::shared_ptr<SkeletonTree>& tree, float height,
                        float jointAngle, bool moving = false,
                        bool flipSigns = false, float fps = 60.0f) {
    constexpr int frameCount = 181;
    std::vector<float> roots;
    std::vector<float> rotations;
    roots.reserve(frameCount * 3);
    rotations.reserve(frameCount * tree->numJoints() * 4);
    for (int frame = 0; frame < frameCount; ++frame) {
        const float time = frame / fps;
        roots.insert(roots.end(), {moving ? time : 0.0f, 0.0f, height});
        for (int joint = 0; joint < tree->numJoints(); ++joint) {
            const float angle =
                joint == 0 ? 0.0f : jointAngle + (moving ? .3f * time : 0);
            Eigen::Quaternionf rotation(
                Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitX()));
            if (flipSigns && frame % 2)
                rotation.coeffs() *= -1;
            rotations.insert(rotations.end(), {rotation.w(), rotation.x(),
                                               rotation.y(), rotation.z()});
        }
    }
    return SkeletonMotion(tree, fps, "player_smoke", roots, rotations);
}

void testTransitionAndDecay() {
    const auto tree = makeTree();
    SkeletonMotionPlayer player({makeClip(tree, 1, 0), makeClip(tree, 2, .8f)},
                                2, 0, .2f);
    const auto before = player.state();
    player.transition(1, player.duration());
    samePose(player.state(), before);
    near(player.rootLinearVelocity().norm(), 0, 1e-6f, "initial root speed");
    const auto* rotationStorage = player.state().rotations().data();
    double elapsed = 0;
    const double rate = 2 * std::log(2.0) / .2;
    for (double deltaTime : {.025, .07, .2, .8}) {
        elapsed += deltaTime;
        player.advance(deltaTime);
        const double decay = std::exp(-rate * elapsed);
        near(player.rootPosition().z(),
             static_cast<float>(2 - (1 + rate * elapsed) * decay), 2e-5f,
             "height offset decay after clip endpoint");
        near(player.rootLinearVelocity().z(),
             static_cast<float>(rate * rate * elapsed * decay), 2e-5f,
             "height offset speed after clip endpoint");
        require(player.time() == player.duration(),
                "clip time must stay clamped");
        require(player.state().rotations().data() == rotationStorage,
                "advance must reuse pose rotation storage");
    }
    player.advance(2);
    const auto target = makeClip(tree, 2, .8f).frame(180);
    samePose(player.state(), target);
}

void testSharedMotionStorage() {
    SkeletonMotion source = makeClip(makeTree(), 1, .2f, true);
    SkeletonMotion copied = source;
    SkeletonMotion assigned;
    assigned = source;
    const auto* roots = source.rootTranslationsFlat().data();
    const auto* rotations = source.localRotationsWxyzFlat().data();
    require(copied.rootTranslationsFlat().data() == roots &&
                assigned.rootTranslationsFlat().data() == roots &&
                copied.localRotationsWxyzFlat().data() == rotations &&
                assigned.localRotationsWxyzFlat().data() == rotations,
            "motion copy construction/assignment must share frame arrays");

    SkeletonMotion rebuilt(source.skeletonTreePtr(), source.fps(), "rebuilt",
                            source.rootTranslationsFlat(),
                            source.localRotationsWxyzFlat());
    require(rebuilt.rootTranslationsFlat().data() != roots &&
                rebuilt.localRotationsWxyzFlat().data() != rotations,
            "a newly constructed motion must own its input arrays");
    const auto expected = source.sample(.25f, false);
    SkeletonMotionPlayer first({source}), second({source});
    source = SkeletonMotion();
    assigned = SkeletonMotion();
    require(source.numFrames() == 0 && source.duration() == 0 &&
                source.rootTranslationsFlat().empty() &&
                source.localRotationsWxyzFlat().empty(),
            "default motion array access must remain valid");
    samePose(copied.sample(.25f, false), expected);
    SkeletonMotion moved = std::move(copied);
    require(copied.numFrames() == 0 && copied.rootTranslationsFlat().empty(),
            "moved-from motion array access must remain valid");
    require(moved.rootTranslationsFlat().data() == roots,
            "move must retain the same source allocation");
    moved = SkeletonMotion();

    first.advance(.25);
    samePose(first.state(), expected);
    require(second.time() == 0, "players must have independent clocks");
    require(first.state().rotations().data() != second.state().rotations().data(),
            "players must own independent output buffers");
    const auto firstPose = first.state();
    second.resetPlaced(0, .4, Eigen::Quaternionf::Identity(),
                       Eigen::Vector3f(10, 20, 0));
    samePose(first.state(), firstPose);
}

void testSharedFrameValidation() {
    const auto source = makeClip(makeTree(), 1, .2f, true);
    std::vector<std::future<SkeletonState>> outputs;
    for (int i = 0; i < 8; ++i) {
        outputs.push_back(std::async(std::launch::async, [source] {
            SkeletonMotionPlayer player({source});
            player.advance(.25);
            return player.state();
        }));
    }
    const auto expected = source.sample(.25f, false);
    for (auto& output : outputs)
        samePose(output.get(), expected);

    auto roots = source.rootTranslationsFlat();
    roots[3] = std::numeric_limits<float>::quiet_NaN();
    const SkeletonMotion invalid(source.skeletonTreePtr(), source.fps(), "invalid",
                                 roots, source.localRotationsWxyzFlat());
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool rejected = false;
        try {
            SkeletonMotionPlayer player({invalid});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "failed validation must never be cached as successful");
    }
}

void testInterruptedTransitionsAndSigns() {
    const auto tree = makeTree();
    const auto makePlayer = [&](bool signs) {
        return SkeletonMotionPlayer({makeClip(tree, 1, .2f, true, signs),
                                     makeClip(tree, 1.6f, -.7f, true, signs)},
                                    2, 0, .15f);
    };
    auto player = makePlayer(false);
    auto signedPlayer = makePlayer(true);
    player.advance(.317);
    signedPlayer.advance(.317);
    for (int iteration = 0; iteration < 4; ++iteration) {
        const auto before = player.state();
        const auto velocity = player.rootLinearVelocity();
        const auto angularVelocities = player.localAngularVelocities();
        const int next = (iteration + 1) % 2;
        const double time = .4 + .1 * iteration;
        player.transition(next, time);
        signedPlayer.transition(next, time);
        samePose(player.state(), before);
        require((player.rootLinearVelocity() - velocity).norm() < 2e-5f,
                "interrupted transition root velocity continuity");
        for (int joint = 0; joint < tree->numJoints(); ++joint) {
            require((player.localAngularVelocities()[joint] -
                     angularVelocities[joint])
                            .norm() < 2e-5f,
                    "interrupted transition joint velocity continuity");
        }
        player.advance(.027);
        signedPlayer.advance(.027);
        samePose(player.state(), signedPlayer.state());
    }
}

void testRawPlaybackAndValidation() {
    const auto tree = makeTree();
    const auto clip = makeClip(tree, 1, .2f, true, false, 29.97f);
    SkeletonMotionPlayer player({clip}, 2, 0, 0);
    player.advance(1000);
    samePose(player.state(), clip.frame(clip.numFrames() - 1));
    require(player.time() ==
                (clip.numFrames() - 1) / static_cast<double>(clip.fps()),
            "nonintegral fps endpoint precision");
    near(player.rootLinearVelocity().norm(), 0, 1e-6f, "clamped speed");
    bool rejected = false;
    try {
        player.advance(-1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "negative delta time must be rejected");
    rejected = false;
    try {
        SkeletonMotionPlayer invalid({}, 2, 0, .1f);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty motion set must be rejected");
}

void testReusableSampling() {
    const auto tree = makeTree();
    const auto clip = makeClip(tree, 1, .2f, true, false, 29.97f);
    auto state = clip.frame(0);
    const auto* rotationStorage = state.rotations().data();
    for (bool loop : {false, true}) {
        for (float time :
             {-.4f, 0.0f, .137f, clip.duration(), clip.duration() + .33f}) {
            clip.sampleInto(time, state, loop);
            samePose(state, clip.sample(time, loop));
            require(state.rotations().data() == rotationStorage,
                    "sampleInto must reuse state rotation storage");
        }
    }
    const auto otherTree = std::make_shared<SkeletonTree>(
        std::vector<std::string>{"root"}, std::vector<int>{-1},
        std::vector<Eigen::Vector3f>{{0, 0, 0}},
        std::vector<Eigen::Quaternionf>{Eigen::Quaternionf::Identity()},
        std::vector<int>{1});
    auto incompatibleState = SkeletonState::zeroPose(otherTree);
    bool rejected = false;
    try {
        clip.sampleInto(.1f, incompatibleState);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "sampleInto must reject an incompatible output skeleton");
}

void testRoundedDurationAndExactHalfTurn() {
    const auto tree = makeTree();
    std::vector<float> rotations(2 * tree->numJoints() * 4, 0.0f);
    for (int joint = 0; joint < tree->numJoints(); ++joint) {
        rotations[joint * 4] = 1;
        const int component = joint == 0 ? 3 : 1;
        rotations[(tree->numJoints() + joint) * 4 + component] = 1;
    }
    const std::vector<float> roots{0, 0, 1, .2f, 0, 1};
    const SkeletonMotion positive(tree, 30, "pi_positive", roots, rotations);
    SkeletonMotionPlayer rounded({positive});
    require(positive.duration() > rounded.duration(),
            "duration regression needs upward float rounding");
    rounded.reset(0, positive.duration());
    require(rounded.time() == rounded.duration(),
            "reset must clamp rounded source duration");
    rounded.transition(0, positive.duration());
    require(rounded.time() == rounded.duration(),
            "transition must clamp rounded source duration");

    for (int firstSign : {1, -1}) {
        for (int lastSign : {1, -1}) {
            auto signedRotations = rotations;
            for (int joint = 0; joint < tree->numJoints(); ++joint) {
                signedRotations[joint * 4] *= firstSign;
                for (int component = 0; component < 4; ++component)
                    signedRotations[(tree->numJoints() + joint) * 4 +
                                    component] *= lastSign;
            }
            const SkeletonMotion signedClip(tree, 30, "pi_signed", roots,
                                            signedRotations);
            SkeletonMotionPlayer baseline({positive});
            SkeletonMotionPlayer signedPlayer({signedClip});
            baseline.advance(.25 / 30);
            signedPlayer.advance(.25 / 30);
            samePose(baseline.state(), signedPlayer.state());
            samePose(positive.sample(.25f / 30, false),
                     signedClip.sample(.25f / 30, false));
            for (int joint = 0; joint < tree->numJoints(); ++joint) {
                require(
                    (baseline.localAngularVelocities()[joint] -
                     signedPlayer.localAngularVelocities()[joint])
                            .norm() < 2e-5f,
                    "exact-pi angular velocities must ignore quaternion sign");
            }
            baseline.reset(0, .25 / 30);
            signedPlayer.reset(0, .25 / 30);
            samePose(baseline.state(), signedPlayer.state());
        }
    }
}

} // namespace

int main() {
    try {
        testTransitionAndDecay();
        testSharedMotionStorage();
        testSharedFrameValidation();
        testInterruptedTransitionsAndSigns();
        testRawPlaybackAndValidation();
        testReusableSampling();
        testRoundedDurationAndExactHalfTurn();
        std::cout << "SkeletonMotionPlayer smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SkeletonMotionPlayer smoke failed: " << error.what()
                  << '\n';
        return 1;
    }
}
