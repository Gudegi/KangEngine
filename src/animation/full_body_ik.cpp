#include "full_body_ik.hpp"

#include <Eigen/Geometry>
#include <LBFGS.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace KE::Animation {
namespace {

using Vector = Eigen::VectorXd;

struct AxisControl {
    int joint;
    Eigen::Vector3f axis;
};

struct Workspace {
    std::vector<Eigen::Matrix3d> local;
    std::vector<Eigen::Matrix3d> global;
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector3d> predicted;
};

void forwardKinematics(const SkeletonTree& tree, const Eigen::Vector3d& root,
                       Workspace& work) {
    work.global[0] = work.local[0];
    work.positions[0] = root;
    for (int joint = 1; joint < tree.numJoints(); ++joint) {
        const int parent = tree.parentIndex(joint);
        work.global[joint] = work.global[parent] * work.local[joint];
        work.positions[joint] =
            work.positions[parent] +
            work.global[parent] * tree.localTranslation(joint).cast<double>();
    }
}

double evaluate(const SkeletonTree& tree,
                const std::vector<Eigen::Matrix3d>& baseLocal,
                const Vector& seedDofs,
                const std::vector<Eigen::Vector3d>& targets,
                const std::vector<int>& effectors,
                const std::vector<Eigen::Vector3f>& offsets,
                const std::vector<AxisControl>& controls,
                const std::vector<unsigned char>& ancestors, const Vector& x,
                Vector* gradient, Workspace& work, double poseDamping) {
    work.local = baseLocal;
    for (size_t i = 0; i < controls.size(); ++i) {
        const auto& control = controls[i];
        const Eigen::Vector3d axis = control.axis.cast<double>().normalized();
        work.local[control.joint] =
            Eigen::AngleAxisd(x[3 + static_cast<int>(i)], axis)
                .toRotationMatrix() *
            work.local[control.joint];
    }
    forwardKinematics(tree, x.head<3>(), work);

    double value = 0.0;
    for (size_t i = 0; i < effectors.size(); ++i) {
        work.predicted[i] =
            work.positions[effectors[i]] +
            work.global[effectors[i]] * offsets[i].cast<double>();
        value += 0.5 * (work.predicted[i] - targets[i]).squaredNorm();
    }
    if (!gradient)
        return value;

    gradient->setZero(x.size());
    for (size_t i = 0; i < effectors.size(); ++i)
        gradient->head<3>() += work.predicted[i] - targets[i];

    for (size_t c = 0; c < controls.size(); ++c) {
        const auto& control = controls[c];
        const int parent = tree.parentIndex(control.joint);
        // Local rotations are composed as Dn ... D1 * Rseed. A parameter's
        // instantaneous axis must therefore include every later delta on the
        // same joint. Using only the bind-space axis makes the objective and
        // analytic gradient disagree for multi-axis joints.
        Eigen::Matrix3d laterDeltas = Eigen::Matrix3d::Identity();
        for (size_t later = c + 1; later < controls.size(); ++later) {
            if (controls[later].joint != control.joint)
                continue;
            const Eigen::Vector3d axis =
                controls[later].axis.cast<double>().normalized();
            laterDeltas =
                Eigen::AngleAxisd(x[3 + static_cast<int>(later)], axis)
                    .toRotationMatrix() *
                laterDeltas;
        }
        Eigen::Vector3d worldAxis = laterDeltas * control.axis.cast<double>();
        if (parent >= 0)
            worldAxis = work.global[parent] * worldAxis;
        worldAxis.normalize();
        const Eigen::Vector3d& pivot = work.positions[control.joint];
        for (size_t e = 0; e < effectors.size(); ++e) {
            if (!ancestors[c * effectors.size() + e])
                continue;
            const Eigen::Vector3d derivative =
                worldAxis.cross(work.predicted[e] - pivot);
            (*gradient)[3 + static_cast<int>(c)] +=
                (work.predicted[e] - targets[e]).dot(derivative);
        }
    }
    const Vector displacement = x - seedDofs;
    value += poseDamping * displacement.squaredNorm();
    gradient->noalias() += (2.0 * poseDamping) * displacement;
    return value;
}

} // namespace

static FullBodyIKBatchResult
solveFullBodyIKBatchImpl(const SkeletonMotion& motion,
                         const std::vector<float>& targetPositions,
                         const std::vector<IKEffector>& effectors,
                         const std::vector<IKJointControl>& jointControls,
                         const FullBodyIKConfig& config, bool parallel) {
    const auto& tree = motion.skeletonTree();
    const int frames = motion.numFrames();
    const int joints = motion.numJoints();
    const size_t effectorCount = effectors.size();
    if (effectorCount == 0 ||
        targetPositions.size() !=
            static_cast<size_t>(frames) * effectorCount * 3)
        throw std::invalid_argument("Invalid full-body IK target dimensions");
    if (config.maxIterations < 0 || config.historySize <= 0 ||
        config.maxLineSearch <= 0 || config.gradientTolerance <= 0.0 ||
        config.relativeTolerance < 0.0 || config.poseDamping < 0.0 ||
        config.maxThreads <= 0)
        throw std::invalid_argument("Invalid full-body IK solver config");
    for (const auto& effector : effectors)
        if (effector.joint < 0 || effector.joint >= joints)
            throw std::out_of_range("IK effector joint is out of range");
    std::vector<int> effectorJoints;
    std::vector<Eigen::Vector3f> effectorOffsets;
    effectorJoints.reserve(effectorCount);
    effectorOffsets.reserve(effectorCount);
    for (const auto& effector : effectors) {
        effectorJoints.push_back(effector.joint);
        effectorOffsets.push_back(effector.offset);
    }
    std::vector<AxisControl> controls;
    std::vector<size_t> controlOffsets;
    controlOffsets.reserve(jointControls.size() + 1);
    controlOffsets.push_back(0);
    std::vector<unsigned char> controlledJoints(static_cast<size_t>(joints), 0);
    for (const auto& control : jointControls) {
        if (control.joint < 0 || control.joint >= joints)
            throw std::out_of_range("IK control joint is out of range");
        if (controlledJoints[static_cast<size_t>(control.joint)])
            throw std::invalid_argument(
                "IK control joint appears more than once");
        if (control.axes.empty() || control.axes.size() > 3)
            throw std::invalid_argument(
                "IK joint control requires one to three axes");
        controlledJoints[static_cast<size_t>(control.joint)] = 1;
        for (const auto& axis : control.axes) {
            if (!axis.allFinite() || axis.squaredNorm() < 1e-12f)
                throw std::invalid_argument("Invalid IK control axis");
            controls.push_back({control.joint, axis});
        }
        controlOffsets.push_back(controls.size());
    }

    std::vector<unsigned char> ancestors(controls.size() * effectorCount, 0);
    for (size_t c = 0; c < controls.size(); ++c) {
        for (size_t e = 0; e < effectorCount; ++e) {
            for (int joint = effectors[e].joint; joint >= 0;
                 joint = tree.parentIndex(joint)) {
                if (joint == controls[c].joint) {
                    ancestors[c * effectorCount + e] = 1;
                    break;
                }
            }
        }
    }

    std::vector<float> roots = motion.rootTranslationsFlat();
    std::vector<float> rotations = motion.localRotationsWxyzFlat();
    FullBodyIKBatchResult result;
    result.bodyPositions.resize(static_cast<size_t>(frames) * joints * 3);
    result.finalErrors.resize(static_cast<size_t>(frames) * effectorCount);
    result.iterations.resize(static_cast<size_t>(frames));
    std::atomic<int> nextFrame{0};
    auto worker = [&]() {
        Workspace work{std::vector<Eigen::Matrix3d>(joints),
                       std::vector<Eigen::Matrix3d>(joints),
                       std::vector<Eigen::Vector3d>(joints),
                       std::vector<Eigen::Vector3d>(effectorCount)};
        std::vector<Eigen::Matrix3d> baseLocal(joints);
        std::vector<Eigen::Vector3d> frameTargets(effectorCount);
        Vector seedDofs(3 + static_cast<int>(controls.size()));
        for (;;) {
            const int frame = nextFrame.fetch_add(1, std::memory_order_relaxed);
            if (frame >= frames)
                break;
            for (int joint = 0; joint < joints; ++joint) {
                const size_t q =
                    (static_cast<size_t>(frame) * joints + joint) * 4;
                baseLocal[joint] =
                    Eigen::Quaterniond(rotations[q], rotations[q + 1],
                                       rotations[q + 2], rotations[q + 3])
                        .normalized()
                        .toRotationMatrix();
            }
            const size_t rootBase = static_cast<size_t>(frame) * 3;
            const Eigen::Vector3d seedRoot(roots[rootBase], roots[rootBase + 1],
                                           roots[rootBase + 2]);
            seedDofs.setZero();
            seedDofs.head<3>() = seedRoot;
            // Convert each controlled local rotation into the same scalar DOF
            // layout consumed by FK. Three orthonormal axes use an exact Euler
            // decomposition; hinge joints preserve any residual rotation in a
            // constant base transform.
            for (size_t group = 0; group < jointControls.size(); ++group) {
                const size_t begin = controlOffsets[group];
                const size_t end = controlOffsets[group + 1];
                const int joint = jointControls[group].joint;
                const size_t count = end - begin;
                if (count == 3) {
                    Eigen::Matrix3d basis;
                    for (int axis = 0; axis < 3; ++axis)
                        basis.col(axis) = controls[begin + axis]
                                              .axis.cast<double>()
                                              .normalized();
                    if (std::abs(basis.determinant() - 1.0) < 1e-5) {
                        const Eigen::Matrix3d inBasis =
                            basis.transpose() * baseLocal[joint] * basis;
                        const Eigen::Vector3d zyx =
                            inBasis.canonicalEulerAngles(2, 1, 0);
                        seedDofs[3 + static_cast<int>(begin)] = zyx[2];
                        seedDofs[3 + static_cast<int>(begin + 1)] = zyx[1];
                        seedDofs[3 + static_cast<int>(begin + 2)] = zyx[0];
                        baseLocal[joint].setIdentity();
                    }
                } else if (count == 1) {
                    const Eigen::Vector3d axis =
                        controls[begin].axis.cast<double>().normalized();
                    Eigen::Quaterniond q(baseLocal[joint]);
                    q.normalize();
                    const double angle =
                        2.0 * std::atan2(q.vec().dot(axis), q.w());
                    seedDofs[3 + static_cast<int>(begin)] = angle;
                    baseLocal[joint] =
                        Eigen::AngleAxisd(-angle, axis).toRotationMatrix() *
                        baseLocal[joint];
                }
            }
            for (size_t e = 0; e < effectorCount; ++e) {
                const size_t base =
                    (static_cast<size_t>(frame) * effectorCount + e) * 3;
                frameTargets[e] = Eigen::Vector3d(targetPositions[base],
                                                  targetPositions[base + 1],
                                                  targetPositions[base + 2]);
            }
            struct Objective {
                const SkeletonTree& tree;
                const std::vector<Eigen::Matrix3d>& baseLocal;
                const Vector& seedDofs;
                const std::vector<Eigen::Vector3d>& frameTargets;
                const std::vector<int>& effectorJoints;
                const std::vector<Eigen::Vector3f>& effectorOffsets;
                const std::vector<AxisControl>& controls;
                const std::vector<unsigned char>& ancestors;
                Workspace& work;
                double poseDamping;
                double operator()(const Vector& x, Vector& gradient) {
                    return evaluate(tree, baseLocal, seedDofs, frameTargets,
                                    effectorJoints, effectorOffsets, controls,
                                    ancestors, x, &gradient, work, poseDamping);
                }
            } objective{
                tree,           baseLocal,         seedDofs, frameTargets,
                effectorJoints, effectorOffsets,   controls, ancestors,
                work,           config.poseDamping};
            LBFGSpp::LBFGSParam<double> parameters;
            parameters.m = config.historySize;
            parameters.epsilon = config.gradientTolerance;
            parameters.epsilon_rel = config.relativeTolerance;
            parameters.max_iterations = config.maxIterations;
            parameters.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING;
            parameters.max_linesearch = config.maxLineSearch;
            LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBacktracking>
                solver(parameters);
            Vector solution = seedDofs;
            double objectiveValue = 0.0;
            try {
                result.iterations[static_cast<size_t>(frame)] =
                    solver.minimize(objective, solution, objectiveValue);
            } catch (const std::runtime_error&) {
                // Keep the last valid iterate when a difficult frame exhausts
                // its line search, matching libLBFGS's recoverable termination
                // model.
                result.iterations[static_cast<size_t>(frame)] = -1;
            }
            evaluate(tree, baseLocal, seedDofs, frameTargets, effectorJoints,
                     effectorOffsets, controls, ancestors, solution, nullptr,
                     work, config.poseDamping);
            const Eigen::Vector3d solvedRoot = solution.head<3>();
            roots[rootBase] = static_cast<float>(solvedRoot.x());
            roots[rootBase + 1] = static_cast<float>(solvedRoot.y());
            roots[rootBase + 2] = static_cast<float>(solvedRoot.z());
            for (int joint = 0; joint < joints; ++joint) {
                Eigen::Quaterniond q(work.local[joint]);
                q.normalize();
                const size_t base =
                    (static_cast<size_t>(frame) * joints + joint) * 4;
                rotations[base] = static_cast<float>(q.w());
                rotations[base + 1] = static_cast<float>(q.x());
                rotations[base + 2] = static_cast<float>(q.y());
                rotations[base + 3] = static_cast<float>(q.z());
                const size_t p =
                    (static_cast<size_t>(frame) * joints + joint) * 3;
                result.bodyPositions[p] =
                    static_cast<float>(work.positions[joint].x());
                result.bodyPositions[p + 1] =
                    static_cast<float>(work.positions[joint].y());
                result.bodyPositions[p + 2] =
                    static_cast<float>(work.positions[joint].z());
            }
            for (size_t e = 0; e < effectorCount; ++e)
                result.finalErrors[static_cast<size_t>(frame) * effectorCount +
                                   e] =
                    static_cast<float>(
                        (work.predicted[e] - frameTargets[e]).norm());
        }
    };
    if (parallel) {
        const unsigned available =
            std::max(1u, std::thread::hardware_concurrency());
        const int threadCount = std::min(
            frames, static_cast<int>(std::min(
                        available, static_cast<unsigned>(config.maxThreads))));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(threadCount));
        for (int i = 0; i < threadCount; ++i)
            workers.emplace_back(worker);
        for (auto& thread : workers)
            thread.join();
    } else {
        worker();
    }
    result.motion = SkeletonMotion(motion.skeletonTreePtr(), motion.fps(),
                                   motion.motionName(), std::move(roots),
                                   std::move(rotations));
    return result;
}

FullBodyIKResult solveFullBodyIK(
    const SkeletonState& state,
    const std::vector<Eigen::Vector3f>& targetPositions,
    const std::vector<IKEffector>& effectors,
    const std::vector<IKJointControl>& controls,
    const FullBodyIKConfig& config) {
    if (!state.isLocal())
        throw std::invalid_argument("Full-body IK requires local rotations");
    std::vector<float> roots{
        state.rootTranslation().x(), state.rootTranslation().y(),
        state.rootTranslation().z()};
    std::vector<float> rotations;
    rotations.reserve(static_cast<size_t>(state.numJoints()) * 4);
    for (const auto& rotation : state.rotations()) {
        const Eigen::Quaternionf q = rotation.normalized();
        rotations.insert(rotations.end(), {q.w(), q.x(), q.y(), q.z()});
    }
    std::vector<float> targets;
    targets.reserve(targetPositions.size() * 3);
    for (const auto& target : targetPositions)
        targets.insert(targets.end(), {target.x(), target.y(), target.z()});
    SkeletonMotion motion(state.skeletonTreePtr(), 1.0f, "IK Frame",
                          std::move(roots), std::move(rotations));
    FullBodyIKBatchResult solved = solveFullBodyIKBatchImpl(
        motion, targets, effectors, controls, config, false);
    return {solved.motion.frame(0), std::move(solved.bodyPositions),
            std::move(solved.finalErrors), solved.iterations[0]};
}

FullBodyIKBatchResult solveFullBodyIKBatch(
    const SkeletonMotion& motion, const std::vector<float>& targetPositions,
    const std::vector<IKEffector>& effectors,
    const std::vector<IKJointControl>& controls,
    const FullBodyIKConfig& config) {
    return solveFullBodyIKBatchImpl(motion, targetPositions, effectors, controls,
                                    config, true);
}

} // namespace KE::Animation
