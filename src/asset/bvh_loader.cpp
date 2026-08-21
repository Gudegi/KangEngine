#include "asset/bvh_loader.hpp"

#include <fmt/core.h>

#include <Eigen/Geometry>

#include <cctype>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace KE {
namespace Asset {

using Animation::SkeletonMotion;
using Animation::SkeletonTree;

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG_TO_RAD = PI / 180.0f;

enum class BVHChannel {
    Xposition,
    Yposition,
    Zposition,
    Xrotation,
    Yrotation,
    Zrotation,
};

struct BVHJoint {
    std::string name;
    int parentIndex = -1;
    Eigen::Vector3f offset = Eigen::Vector3f::Zero();
    std::vector<BVHChannel> channels;
};

struct BVHParsedData {
    std::vector<BVHJoint> joints;
    int frameCount = 0;
    float frameTime = 0.0f;
    std::vector<float> frameValues;
    ImportDiagnostics diagnostics;
};

class TokenStream {
  public:
    explicit TokenStream(std::string text) : _text(std::move(text)) {}

    bool empty() {
        skipWhitespace();
        return _pos >= _text.size();
    }

    std::string next() {
        skipWhitespace();
        if (_pos >= _text.size())
            throw std::runtime_error("BVH: unexpected end of file");

        if (_text[_pos] == '{' || _text[_pos] == '}') {
            return std::string(1, _text[_pos++]);
        }

        const size_t start = _pos;
        while (_pos < _text.size() && !std::isspace(byte(_text[_pos])) &&
               _text[_pos] != '{' && _text[_pos] != '}') {
            ++_pos;
        }
        return _text.substr(start, _pos - start);
    }

    std::string peek() {
        const size_t saved = _pos;
        std::string out = next();
        _pos = saved;
        return out;
    }

    void expect(const std::string& token) {
        const std::string got = next();
        if (got != token) {
            throw std::runtime_error(
                fmt::format("BVH: expected '{}', got '{}'", token, got));
        }
    }

    int nextInt(const std::string& label) {
        const std::string token = next();
        try {
            size_t used = 0;
            int value = std::stoi(token, &used);
            if (used != token.size())
                throw std::invalid_argument("trailing characters");
            return value;
        } catch (const std::exception&) {
            throw std::runtime_error(fmt::format(
                "BVH: expected integer for {}, got '{}'", label, token));
        }
    }

    float nextFloat(const std::string& label) {
        const std::string token = next();
        try {
            size_t used = 0;
            float value = std::stof(token, &used);
            if (used != token.size())
                throw std::invalid_argument("trailing characters");
            return value;
        } catch (const std::exception&) {
            throw std::runtime_error(fmt::format(
                "BVH: expected float for {}, got '{}'", label, token));
        }
    }

  private:
    static unsigned char byte(char c) { return static_cast<unsigned char>(c); }

    void skipWhitespace() {
        while (_pos < _text.size() && std::isspace(byte(_text[_pos])))
            ++_pos;
    }

    std::string _text;
    size_t _pos = 0;
};

std::string readTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error(
            fmt::format("Failed to open BVH file: {}", path));

    file.seekg(0, std::ios::end);
    const std::streampos size = file.tellg();
    if (size <= 0)
        throw std::runtime_error(fmt::format("BVH file is empty: {}", path));

    std::string text(static_cast<size_t>(size), '\0');
    file.seekg(0, std::ios::beg);
    if (!file.read(text.data(), size)) {
        throw std::runtime_error(
            fmt::format("Failed to read BVH file: {}", path));
    }
    return text;
}

BVHChannel parseChannel(const std::string& token) {
    if (token == "Xposition")
        return BVHChannel::Xposition;
    if (token == "Yposition")
        return BVHChannel::Yposition;
    if (token == "Zposition")
        return BVHChannel::Zposition;
    if (token == "Xrotation")
        return BVHChannel::Xrotation;
    if (token == "Yrotation")
        return BVHChannel::Yrotation;
    if (token == "Zrotation")
        return BVHChannel::Zrotation;
    throw std::runtime_error(
        fmt::format("BVH: unsupported channel '{}'", token));
}

Eigen::Quaternionf axisRotation(BVHChannel channel, float degrees) {
    const float radians = degrees * DEG_TO_RAD;
    switch (channel) {
    case BVHChannel::Xrotation:
        return Eigen::Quaternionf(
            Eigen::AngleAxisf(radians, Eigen::Vector3f::UnitX()));
    case BVHChannel::Yrotation:
        return Eigen::Quaternionf(
            Eigen::AngleAxisf(radians, Eigen::Vector3f::UnitY()));
    case BVHChannel::Zrotation:
        return Eigen::Quaternionf(
            Eigen::AngleAxisf(radians, Eigen::Vector3f::UnitZ()));
    default:
        return Eigen::Quaternionf::Identity();
    }
}

void parseJointBlock(TokenStream& tokens, BVHParsedData& out,
                     const std::string& name, int parentIndex, float scale);

void parseEndSite(TokenStream& tokens) {
    tokens.expect("{");
    tokens.expect("OFFSET");
    tokens.nextFloat("End Site OFFSET x");
    tokens.nextFloat("End Site OFFSET y");
    tokens.nextFloat("End Site OFFSET z");
    tokens.expect("}");
}

void parseJointBlock(TokenStream& tokens, BVHParsedData& out,
                     const std::string& name, int parentIndex, float scale) {
    const int jointIndex = static_cast<int>(out.joints.size());
    out.joints.push_back({name, parentIndex, Eigen::Vector3f::Zero(), {}});

    tokens.expect("{");
    bool sawOffset = false;
    bool sawChannels = false;
    while (true) {
        const std::string token = tokens.next();
        if (token == "}") {
            break;
        } else if (token == "OFFSET") {
            const float x = tokens.nextFloat("OFFSET x") * scale;
            const float y = tokens.nextFloat("OFFSET y") * scale;
            const float z = tokens.nextFloat("OFFSET z") * scale;
            out.joints[static_cast<size_t>(jointIndex)].offset =
                Eigen::Vector3f(x, y, z);
            sawOffset = true;
        } else if (token == "CHANNELS") {
            const int count = tokens.nextInt("CHANNELS count");
            auto& channels =
                out.joints[static_cast<size_t>(jointIndex)].channels;
            channels.reserve(static_cast<size_t>(count));
            for (int i = 0; i < count; ++i)
                channels.push_back(parseChannel(tokens.next()));
            sawChannels = true;
        } else if (token == "JOINT") {
            parseJointBlock(tokens, out, tokens.next(), jointIndex, scale);
        } else if (token == "End") {
            tokens.expect("Site");
            parseEndSite(tokens);
        } else {
            throw std::runtime_error(
                fmt::format("BVH: unexpected hierarchy token '{}'", token));
        }
    }

    if (!sawOffset) {
        out.diagnostics.warnings.push_back(
            fmt::format("joint '{}' has no OFFSET; using zero", name));
    }
    if (!sawChannels) {
        out.diagnostics.warnings.push_back(
            fmt::format("joint '{}' has no CHANNELS", name));
    }
}

int totalChannelCount(const std::vector<BVHJoint>& joints) {
    int count = 0;
    for (const BVHJoint& joint : joints)
        count += static_cast<int>(joint.channels.size());
    return count;
}

BVHParsedData parseBVH(const std::string& path, float scale) {
    TokenStream tokens(readTextFile(path));
    BVHParsedData out;

    tokens.expect("HIERARCHY");
    tokens.expect("ROOT");
    parseJointBlock(tokens, out, tokens.next(), -1, scale);

    tokens.expect("MOTION");
    tokens.expect("Frames:");
    out.frameCount = tokens.nextInt("Frames");
    tokens.expect("Frame");
    tokens.expect("Time:");
    out.frameTime = tokens.nextFloat("Frame Time");
    if (out.frameCount <= 0)
        throw std::runtime_error("BVH: frame count must be positive");
    if (out.frameTime <= 0.0f)
        throw std::runtime_error("BVH: frame time must be positive");

    const int channels = totalChannelCount(out.joints);
    if (channels <= 0)
        throw std::runtime_error("BVH: hierarchy has no animated channels");

    out.frameValues.reserve(static_cast<size_t>(out.frameCount) *
                            static_cast<size_t>(channels));
    for (int f = 0; f < out.frameCount; ++f) {
        for (int c = 0; c < channels; ++c) {
            if (tokens.empty()) {
                throw std::runtime_error(
                    fmt::format("BVH: missing channel data at frame {}", f));
            }
            out.frameValues.push_back(tokens.nextFloat("motion channel"));
        }
    }

    if (!tokens.empty()) {
        out.diagnostics.warnings.push_back(
            "BVH has extra tokens after expected motion data");
    }

    return out;
}

void validateArmatureStructure(const BVHParsedData& data) {
    if (data.joints.size() < 2)
        throw std::runtime_error("BVH: armature joint has no skeleton child");
    int childCount = 0;
    for (size_t i = 1; i < data.joints.size(); ++i)
        childCount += data.joints[i].parentIndex == 0 ? 1 : 0;
    if (childCount != 1 || data.joints[1].parentIndex != 0) {
        throw std::runtime_error(
            "BVH: armature joint must have exactly one direct skeleton child");
    }
}

SkeletonTree makeSkeletonTree(const BVHParsedData& data,
                              bool hasArmatureJoint) {
    if (hasArmatureJoint)
        validateArmatureStructure(data);
    std::vector<std::string> nodeNames;
    std::vector<int> parentIndices;
    std::vector<Eigen::Vector3f> localTranslations;
    std::vector<Eigen::Quaternionf> localRotations;
    std::vector<int> numJointsInBody;

    nodeNames.reserve(data.joints.size());
    parentIndices.reserve(data.joints.size());
    localTranslations.reserve(data.joints.size());
    localRotations.reserve(data.joints.size());
    numJointsInBody.reserve(data.joints.size());

    const size_t first = hasArmatureJoint ? 1 : 0;
    for (size_t i = first; i < data.joints.size(); ++i) {
        const BVHJoint& joint = data.joints[i];
        nodeNames.push_back(joint.name);
        parentIndices.push_back(hasArmatureJoint
                                    ? (joint.parentIndex == 0
                                           ? -1
                                           : joint.parentIndex - 1)
                                    : joint.parentIndex);
        localTranslations.push_back(hasArmatureJoint && i == 1
                                        ? Eigen::Vector3f::Zero()
                                        : joint.offset);
        localRotations.push_back(Eigen::Quaternionf::Identity());
        numJointsInBody.push_back(joint.channels.empty() ? 0 : 1);
    }

    return SkeletonTree(std::move(nodeNames), std::move(parentIndices),
                        std::move(localTranslations), std::move(localRotations),
                        std::move(numJointsInBody));
}

SkeletonMotion makeMotion(const std::string& path, BVHParsedData& data,
                          float scale, bool hasArmatureJoint) {
    auto tree = std::make_shared<SkeletonTree>(
        makeSkeletonTree(data, hasArmatureJoint));
    const int sourceJoints = static_cast<int>(data.joints.size());
    const int joints = sourceJoints - (hasArmatureJoint ? 1 : 0);
    const float fps = 1.0f / data.frameTime;

    std::vector<float> rootTranslations;
    rootTranslations.reserve(static_cast<size_t>(data.frameCount) * 3);

    std::vector<float> localRotationsWxyz;
    localRotationsWxyz.reserve(static_cast<size_t>(data.frameCount) *
                               static_cast<size_t>(joints) * 4);

    size_t valueOffset = 0;
    for (int f = 0; f < data.frameCount; ++f) {
        Eigen::Vector3f rootTranslation = Eigen::Vector3f::Zero();
        std::vector<Eigen::Quaternionf> frameRotations(
            static_cast<size_t>(joints), Eigen::Quaternionf::Identity());

        for (int j = 0; j < sourceJoints; ++j) {
            Eigen::Quaternionf localRotation = Eigen::Quaternionf::Identity();
            Eigen::Vector3f jointPosition = Eigen::Vector3f::Zero();
            bool hasPositionChannel = false;
            bool hasNonRootPosition = false;

            for (BVHChannel channel :
                 data.joints[static_cast<size_t>(j)].channels) {
                const float raw = data.frameValues[valueOffset++];
                switch (channel) {
                case BVHChannel::Xposition:
                    jointPosition.x() = raw * scale;
                    hasPositionChannel = true;
                    if (j != 0)
                        hasNonRootPosition = true;
                    break;
                case BVHChannel::Yposition:
                    jointPosition.y() = raw * scale;
                    hasPositionChannel = true;
                    if (j != 0)
                        hasNonRootPosition = true;
                    break;
                case BVHChannel::Zposition:
                    jointPosition.z() = raw * scale;
                    hasPositionChannel = true;
                    if (j != 0)
                        hasNonRootPosition = true;
                    break;
                case BVHChannel::Xrotation:
                case BVHChannel::Yrotation:
                case BVHChannel::Zrotation:
                    localRotation = localRotation * axisRotation(channel, raw);
                    break;
                }
            }

            if (hasArmatureJoint && j == 0) {
                if (jointPosition.norm() > 1e-5f ||
                    localRotation.angularDistance(
                        Eigen::Quaternionf::Identity()) > 1e-5f) {
                    throw std::runtime_error(fmt::format(
                        "BVH: armature joint '{}' must have identity animation",
                        data.joints[0].name));
                }
            } else if ((!hasArmatureJoint && j == 0) ||
                       (hasArmatureJoint && j == 1)) {
                rootTranslation =
                    hasPositionChannel
                        ? jointPosition
                        : data.joints[static_cast<size_t>(j)].offset;
            } else if (hasNonRootPosition && f == 0) {
                data.diagnostics.warnings.push_back(fmt::format(
                    "joint '{}' has animated position channels; "
                    "SkeletonMotion stores only root translation, so "
                    "non-root position animation is ignored",
                    data.joints[static_cast<size_t>(j)].name));
            }

            if (localRotation.norm() <= 1e-6f)
                localRotation = Eigen::Quaternionf::Identity();
            else
                localRotation.normalize();
            if (!hasArmatureJoint || j > 0) {
                const int outputJoint = j - (hasArmatureJoint ? 1 : 0);
                frameRotations[static_cast<size_t>(outputJoint)] = localRotation;
            }
        }

        rootTranslations.push_back(rootTranslation.x());
        rootTranslations.push_back(rootTranslation.y());
        rootTranslations.push_back(rootTranslation.z());

        for (const Eigen::Quaternionf& q : frameRotations) {
            localRotationsWxyz.push_back(q.w());
            localRotationsWxyz.push_back(q.x());
            localRotationsWxyz.push_back(q.y());
            localRotationsWxyz.push_back(q.z());
        }
    }

    const std::string motionName = path;
    return SkeletonMotion(std::move(tree), fps, motionName,
                          std::move(rootTranslations),
                          std::move(localRotationsWxyz));
}

} // namespace

SkeletonTree BVHLoader::loadSkeleton(const std::string& bvhPath, float scale,
                                     bool hasArmatureJoint) {
    return makeSkeletonTree(parseBVH(bvhPath, scale), hasArmatureJoint);
}

SkeletonMotion BVHLoader::loadMotion(const std::string& bvhPath, float scale,
                                     bool hasArmatureJoint) {
    BVHParsedData data = parseBVH(bvhPath, scale);
    SkeletonMotion motion = makeMotion(bvhPath, data, scale, hasArmatureJoint);
    data.diagnostics.printWarnings("BVHLoader " + bvhPath);
    return motion;
}

BVHImportResult BVHLoader::parse(const std::string& bvhPath, float scale,
                                 bool hasArmatureJoint) {
    BVHParsedData data = parseBVH(bvhPath, scale);
    BVHImportResult result;
    result.frameCount = data.frameCount;
    result.frameTime = data.frameTime;
    result.frameRate = 1.0f / data.frameTime;
    result.motion = makeMotion(bvhPath, data, scale, hasArmatureJoint);
    result.diagnostics = std::move(data.diagnostics);
    result.diagnostics.printWarnings("BVHLoader " + bvhPath);
    return result;
}

} // namespace Asset
} // namespace KE
