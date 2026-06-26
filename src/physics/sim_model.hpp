#ifndef _SIM_MODEL_HPP_
#define _SIM_MODEL_HPP_

#include "engine/graphics/renderer/renderer_types.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cassert>
#include <string>
#include <vector>

namespace KE {

// Immutable-ish topology for high-throughput simulation visuals.
// This is intentionally separate from Scene Prim data: many-env simulation can
// render from compact body/shape/renderable arrays without creating one Prim
// per body.
struct SimModel {
    int numBodies = 0;
    std::vector<int> shapeBody;
    std::vector<glm::vec3> shapeLocalPos;
    std::vector<glm::quat> shapeLocalRot;
    std::vector<RenderableHandle> shapeRenderables;
    std::vector<int> objectBodyStart;
    std::vector<int> objectBodyCount;
    std::vector<std::string> bodyNames;
    std::vector<std::string> shapeNames;
    std::vector<std::string> objectNames;

    void setBodyRenderables(const std::vector<RenderableHandle>& handles) {
        numBodies = static_cast<int>(handles.size());
        shapeBody.clear();
        shapeLocalPos.clear();
        shapeLocalRot.clear();
        shapeRenderables.clear();
        objectBodyStart.clear();
        objectBodyCount.clear();
        shapeNames.clear();
        for (int i = 0; i < static_cast<int>(handles.size()); ++i)
            addShape(i, handles[static_cast<size_t>(i)]);
        addObjectBoundary(0, numBodies);
        assert(isValid() && "SimModel setBodyRenderables produced invalid data");
    }

    int addShape(int bodyId, RenderableHandle renderable,
                 const glm::vec3& localPos = glm::vec3(0.0f),
                 const glm::quat& localRot =
                     glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                 const std::string& name = {}) {
        assert(bodyId >= 0 && "SimModel shape body index must be non-negative");
        if (bodyId >= numBodies)
            numBodies = bodyId + 1;
        shapeBody.push_back(bodyId);
        shapeLocalPos.push_back(localPos);
        shapeLocalRot.push_back(localRot);
        shapeRenderables.push_back(renderable);
        shapeNames.push_back(name);
        return shapeCount() - 1;
    }

    int addObjectBoundary(int bodyStart, int bodyCount,
                          const std::string& name = {}) {
        assert(bodyStart >= 0 && bodyCount >= 0 &&
               "SimModel object boundary must be non-negative");
        objectBodyStart.push_back(bodyStart);
        objectBodyCount.push_back(bodyCount);
        objectNames.push_back(name);
        return static_cast<int>(objectBodyStart.size()) - 1;
    }

    int bodyCount() const {
        return numBodies;
    }

    int shapeCount() const { return static_cast<int>(shapeRenderables.size()); }

    bool isValid() const {
        const size_t count = shapeRenderables.size();
        if (shapeBody.size() != count || shapeLocalPos.size() != count ||
            shapeLocalRot.size() != count || shapeNames.size() != count)
            return false;
        for (int bodyId : shapeBody) {
            if (bodyId < 0 || bodyId >= numBodies)
                return false;
        }
        if (objectBodyStart.size() != objectBodyCount.size() ||
            objectBodyStart.size() != objectNames.size())
            return false;
        for (size_t i = 0; i < objectBodyStart.size(); ++i) {
            const int start = objectBodyStart[i];
            const int countBodies = objectBodyCount[i];
            if (start < 0 || countBodies < 0 || start + countBodies > numBodies)
                return false;
        }
        return true;
    }
};

// Mutable runtime state, Newton-style.  bodyPos/bodyRot are flat arrays indexed
// by env/body and can later become CPU views, GPU buffers, or tensor views.
struct SimState {
    int numEnvs = 0;
    int numBodies = 0;
    std::vector<glm::vec3> bodyPos;
    std::vector<glm::quat> bodyRot;

    void resize(int envCount, int bodyCount) {
        assert(envCount >= 0 && bodyCount >= 0 &&
               "SimState dimensions must be non-negative");
        numEnvs = envCount;
        numBodies = bodyCount;
        const size_t count = static_cast<size_t>(numEnvs * numBodies);
        bodyPos.resize(count);
        bodyRot.resize(count);
    }

    int bodyIndex(int envId, int bodyId) const {
        assert(envId >= 0 && envId < numEnvs &&
               "SimState env index out of range");
        assert(bodyId >= 0 && bodyId < numBodies &&
               "SimState body index out of range");
        return envId * numBodies + bodyId;
    }

    void setBodyTransform(int envId, int bodyId, const glm::vec3& pos,
                          const glm::quat& rot) {
        const int index = bodyIndex(envId, bodyId);
        bodyPos[static_cast<size_t>(index)] = pos;
        bodyRot[static_cast<size_t>(index)] = rot;
    }

    glm::mat4 bodyMatrix(int envId, int bodyId) const {
        const int index = bodyIndex(envId, bodyId);
        return glm::translate(glm::mat4(1.0f),
                              bodyPos[static_cast<size_t>(index)]) *
               glm::mat4_cast(bodyRot[static_cast<size_t>(index)]);
    }
};

// Converts SimState body transforms into per-renderable instance transform
// arrays.  Renderer upload still happens outside this class for now.
class SimVisualBatch {
  public:
    void setModel(const SimModel* model) {
        if (!model) {
            _shapeBody.clear();
            _shapeLocalPos.clear();
            _shapeLocalRot.clear();
            _renderables.clear();
            return;
        }
        _shapeBody = model->shapeBody;
        _shapeLocalPos = model->shapeLocalPos;
        _shapeLocalRot = model->shapeLocalRot;
        _renderables = model->shapeRenderables;
        assert(model->isValid() && "SimVisualBatch received invalid SimModel");
    }

    void prepareFromState(const SimState& state) {
        if (_renderables.empty()) {
            assert(state.numBodies == 0 &&
                   "SimVisualBatch requires a SimModel before prepareFromState");
            _renderableTransforms.clear();
            return;
        }

        const int shapeCount = static_cast<int>(_renderables.size());
        assert(_shapeBody.size() == _renderables.size() &&
               _shapeLocalPos.size() == _renderables.size() &&
               _shapeLocalRot.size() == _renderables.size() &&
               "SimVisualBatch model arrays must have matching shape counts");
        _renderableTransforms.resize(static_cast<size_t>(shapeCount));

        for (int shapeId = 0; shapeId < shapeCount; ++shapeId) {
            const int bodyId = _shapeBody[static_cast<size_t>(shapeId)];
            assert(bodyId >= 0 && bodyId < state.numBodies &&
                   "SimVisualBatch shape body index is outside SimState");
            const glm::mat4 local =
                glm::translate(glm::mat4(1.0f),
                               _shapeLocalPos[static_cast<size_t>(shapeId)]) *
                glm::mat4_cast(_shapeLocalRot[static_cast<size_t>(shapeId)]);
            auto& transforms =
                _renderableTransforms[static_cast<size_t>(shapeId)];
            transforms.resize(static_cast<size_t>(state.numEnvs));
            for (int envId = 0; envId < state.numEnvs; ++envId)
                transforms[static_cast<size_t>(envId)] =
                    state.bodyMatrix(envId, bodyId) * local;
        }
    }

    int renderableCount() const {
        return static_cast<int>(_renderableTransforms.size());
    }

    RenderableHandle renderable(int shapeId) const {
        if (_renderables.empty())
            return InvalidHandle;
        return _renderables[static_cast<size_t>(shapeId)];
    }

    const std::vector<glm::mat4>& transforms(int shapeId) const {
        return _renderableTransforms[static_cast<size_t>(shapeId)];
    }

  private:
    std::vector<int> _shapeBody;
    std::vector<glm::vec3> _shapeLocalPos;
    std::vector<glm::quat> _shapeLocalRot;
    std::vector<RenderableHandle> _renderables;
    std::vector<std::vector<glm::mat4>> _renderableTransforms;
};

} // namespace KE

#endif
