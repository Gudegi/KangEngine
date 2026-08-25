#ifndef _ARTICULATION_MOTION_HPP_
#define _ARTICULATION_MOTION_HPP_

#include "asset/articulation_desc.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

namespace KE {
namespace Animation {

enum class ArticulationCoordinateType {
    Fixed,
    Revolute,
    Prismatic,
    Spherical,
    Free,
};

struct ArticulationCoordinateBlock {
    int bodyIndex = -1;
    int parentBodyIndex = -1;
    int jointIndexInBody = -1;
    std::string bodyName;
    std::string jointName;
    ArticulationCoordinateType type = ArticulationCoordinateType::Fixed;
    Eigen::Vector3f axis = Eigen::Vector3f::Zero();
    Eigen::Vector3f referenceTranslation = Eigen::Vector3f::Zero();
    Eigen::Quaternionf referenceRotation = Eigen::Quaternionf::Identity();
    float lowerLimit = 0.0f;
    float upperLimit = 0.0f;
    int qOffset = 0;
    int qSize = 0;
    int qdOffset = 0;
    int qdSize = 0;
};

class ArticulationCoordinateLayout {
  public:
    // Build q/qd offsets while preserving the source articulation order.
    static ArticulationCoordinateLayout
    fromData(const Asset::ArticulationDesc& data, bool freeRoot = false);

    // Return the total configuration coordinate count.
    int nq() const { return _nq; }
    // Return the total tangent velocity coordinate count.
    int nv() const { return _nv; }
    // Return whether the layout starts with a synthetic free root.
    bool hasFreeRoot() const { return _hasFreeRoot; }
    // Return the BFS/DFS order inherited from the source articulation.
    const std::string& traversalOrder() const { return _traversalOrder; }
    // Return the deterministic compatibility signature for this layout.
    const std::string& modelSignature() const { return _modelSignature; }
    // Return coordinate blocks in source articulation order.
    const std::vector<ArticulationCoordinateBlock>& blocks() const {
        return _blocks;
    }
    // Return the skeleton topology associated with this layout.
    const std::shared_ptr<const SkeletonTree>& skeletonTreePtr() const {
        return _skeletonTree;
    }

  private:
    int _nq = 0;
    int _nv = 0;
    bool _hasFreeRoot = false;
    std::string _traversalOrder;
    std::string _modelSignature;
    std::shared_ptr<const SkeletonTree> _skeletonTree;
    std::vector<ArticulationCoordinateBlock> _blocks;
};

class ArticulationMotion {
  public:
    ArticulationMotion() = default;
    ArticulationMotion(ArticulationCoordinateLayout layout, int numFrames,
                       float fps, std::string motionName, std::vector<float> q,
                       std::vector<float> qd);

    // Return the number of sampled frames.
    int numFrames() const { return _numFrames; }
    // Return the clip frame rate in Hz.
    float fps() const { return _fps; }
    // Return the clip duration in seconds.
    float duration() const;
    // Return the motion name.
    const std::string& motionName() const { return _motionName; }
    // Return the coordinate layout owned by this motion.
    const ArticulationCoordinateLayout& layout() const { return _layout; }
    // Return mutable frame-major q storage.
    std::vector<float>& qFlat() { return _q; }
    // Return immutable frame-major q storage.
    const std::vector<float>& qFlat() const { return _q; }
    // Return mutable frame-major qd storage.
    std::vector<float>& qdFlat() { return _qd; }
    // Return immutable frame-major qd storage.
    const std::vector<float>& qdFlat() const { return _qd; }

  private:
    ArticulationCoordinateLayout _layout;
    int _numFrames = 0;
    float _fps = 30.0f;
    std::string _motionName;
    std::vector<float> _q;
    std::vector<float> _qd;
};

} // namespace Animation
} // namespace KE

#endif
