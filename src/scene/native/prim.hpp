///
/// Scene Prim (USD-like hierarchy)
///

#ifndef _SCENE_PRIM_HPP_
#define _SCENE_PRIM_HPP_

#include <glm/fwd.hpp>
#include <glm/gtc/quaternion.hpp>
#include <variant>
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include "../scene_backend.hpp"
#include "token.hpp"
#include "utils/types.hpp"

namespace KE {

namespace Scene {

using AttributeValue =
    std::variant<bool, int, float, std::string, glm::vec3, glm::vec4, glm::mat4,
                 glm::quat, std::vector<std::string>>;

// Prim 타입
enum class PrimType {
    Root,   // "/" 루트
    Xform,  // Transform (그룹)
    Mesh,   // Mesh 데이터
    Camera, // Camera
    Light   // Light
};

// Scene Graph Node (USD Prim처럼)
class Prim {
  private:
    std::string _name; // "Cube"
    std::string _path; // "/World/Cube"
    PrimType _type;
    Prim* _parent; // 부모 (소유하지 않음)
    std::vector<std::unique_ptr<Prim>> _children; // 자식들
    std::map<std::string, Prim*> _childrenMap;    // 빠른 탐색용

    // 데이터 (타입에 따라 다름)
    std::shared_ptr<MeshData> _meshData;
    std::unordered_map<Token, AttributeValue, Token::Hash> _Attributes;

    // cached TODO: refactoring
    bool _isDirty = true; // true -> compute
    glm::mat4 _cachedModelMat = glm::mat4(1.0f);

  public:
    Prim(const std::string& name, PrimType type, Prim* parent = nullptr);
    ~Prim() = default;

    // 이동만 허용 (unique_ptr 멤버 때문에 복사 불가)
    Prim(Prim&&) = default;
    Prim& operator=(Prim&&) = default;
    Prim(const Prim&) = delete;
    Prim& operator=(const Prim&) = delete;

    // 계층 구조
    Prim* addChild(const std::string& name, PrimType type);
    Prim* getChild(const std::string& name) const;
    Prim* getPrimAtPath(const std::string& path); // "/World/Cube" 경로로 탐색
    std::vector<Prim*> getChildren() const;

    // Getters
    const std::string& getName() const { return _name; }
    const std::string& getPath() const { return _path; }
    PrimType getType() const { return _type; }
    Prim* getParent() const { return _parent; }

    // Mesh 데이터 (type == Mesh일 때만)
    void setMeshData(std::shared_ptr<MeshData> data);
    std::shared_ptr<MeshData> getMeshData() const;

    static MeshData createCubeData(float scale);
    static MeshData createSquareData(float scale); // Deprecated.
    static MeshData createPlaneData(float scale);  // Y-up by default
    static MeshData createPlaneData(float scale, UpAxis upAxis);
    static MeshData createSphereData(float radius, int numLongitudes,
                                     int numLatitudes);
    static MeshData createRectangleData(float xScale, float yScale,
                                        float zScale);
    static MeshData createCylinderData(float radius, float length,
                                       int segments = 32);
    static MeshData createCylinderData(float radius, float length,
                                       UpAxis upAxis, int segments = 32);
    static MeshData createArrowData(float baseRadius, float baseHeight,
                                    int segments = 32);
    static MeshData createArrowData(float baseRadius, float baseHeight,
                                    UpAxis upAxis, float capRadius = -1.0f,
                                    float capHeight = -1.0f, int segments = 32);
    static MeshData createCapsuleData(float radius, float height,
                                      UpAxis upAxis = UpAxis::Y,
                                      int segments = 32);
    static MeshData createConeData(float radius, float height,
                                   UpAxis upAxis = UpAxis::Y,
                                   int segments = 32);
    // Points: each point rendered as a small sphere (radius = pointRadius)
    // TODO: instanced render
    static MeshData createPointsData(const std::vector<glm::vec3>& points,
                                     float pointRadius = 0.01f,
                                     int segments = 8);
    // Lines: each segment rendered as a capsule (radius = lineRadius)
    // TODO: instanced render
    static MeshData createLinesData(const std::vector<glm::vec3>& vertices,
                                    const std::vector<unsigned int>& indices,
                                    float lineRadius = 0.005f,
                                    int segments = 8);

    // 순회
    void traverse(std::function<void(Prim*)> func);

    template <typename T> void setAttribute(const Token& name, const T& value) {
        _Attributes[name] = value;
        this->onDirtyAttributeChanged();
    }

    template <typename T>
    void setAttribute(const std::string& name, const T& value) {
        setAttribute(Token(name), value);
    }

    template <typename T> T getAttribute(const Token& name) const {
        return std::get<T>(_Attributes.at(name));
    }

    template <typename T> T getAttribute(const std::string& name) const {
        return getAttribute<T>(Token(name));
    }

    template <typename T>
    T getAttribute(const Token& name, const T& defaultValue) const {
        auto it = _Attributes.find(name);
        if (it == _Attributes.end())
            return defaultValue;
        return std::get<T>(it->second);
    }

    template <typename T>
    T getAttribute(const std::string& name, const T& defaultValue) const {
        return getAttribute<T>(Token(name), defaultValue);
    }

    bool hasAttribute(const Token& name) const {
        return _Attributes.count(name) > 0;
    }

    bool hasAttribute(const std::string& name) const {
        return hasAttribute(Token(name));
    }

    void setXformOpOrder(const std::vector<std::string>& order) {
        this->setAttribute("xformOpOrder", order);
    }

    // TODO: maybe impractical saving vec3 rgb color and vec4 rgba color
    void setDisplayColor(glm::vec3 color) {
        this->setAttribute("primvars:displaycolor", color);
    }

    std::optional<glm::vec3> getDisplayColor() {
        if (hasAttribute("primvars:displaycolor")) {
            return getAttribute<glm::vec3>("primvars:displaycolor");
        }
        return std::nullopt;
    }

    void setDisplayColorAlpha(glm::vec4 color) {
        this->setAttribute("primvars:displaycolorAlpha", color);
    }

    std::optional<glm::vec4> getDisplayColorAlpha() {
        if (hasAttribute("primvars:displaycolorAlpha")) {
            return getAttribute<glm::vec4>("primvars:displaycolorAlpha");
        }
        return std::nullopt;
    }

    void addTranslateOp(glm::vec3 trans) {
        this->setAttribute("xformOp:translate", trans);
    }

    void addScaleOp(glm::vec3 scale) {
        this->setAttribute("xformOp:scale", scale);
    }

    void addRotateQuaternionOp(glm::quat quat) {
        this->setAttribute("xformOp:rotateQuaternion", quat);
    }

    glm::mat4 computeModelMatrix();

    void onDirtyAttributeChanged() {
        _isDirty = true;
        // Note: Children are not dirty because _cachedModelMat represents
        // the Local transform, which is independent of the parent.
    }
};

} // namespace Scene
} // namespace KE

#endif
