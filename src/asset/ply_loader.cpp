#include "asset/mesh_loader.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace KE::Asset {
namespace {

// PLY 1.0 format: https://www.paulbourke.net/dataformats/ply/
enum class Scalar { Int8, Uint8, Int16, Uint16, Int32, Uint32, Float32, Float64 };

struct ScalarInfo {
    size_t size;
    double min;
    double max;
    bool integer;
};

ScalarInfo scalarInfo(Scalar type) {
    switch (type) {
    case Scalar::Int8: return {1, -128, 127, true};
    case Scalar::Uint8: return {1, 0, 255, true};
    case Scalar::Int16: return {2, -32768, 32767, true};
    case Scalar::Uint16: return {2, 0, 65535, true};
    case Scalar::Int32: return {4, -2147483648.0, 2147483647.0, true};
    case Scalar::Uint32: return {4, 0, 4294967295.0, true};
    case Scalar::Float32:
        return {4, -std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max(), false};
    case Scalar::Float64:
        return {8, -std::numeric_limits<double>::max(),
                 std::numeric_limits<double>::max(), false};
    }
    throw std::logic_error("Unknown PLY scalar type");
}

struct Property {
    Scalar type;
    std::optional<Scalar> countType;
    int attribute = -1;
    bool faceIndices = false;
    bool textureIndices = false;
    // UV fields: tx, tn, u, v.
    int textureAttribute = -1;
};

struct Element {
    std::string name;
    size_t count;
    std::vector<Property> properties;
};

struct TextureFace {
    unsigned int texture = 0;
    unsigned int face = 0;
    size_t begin = 0;
    size_t end = 0;
};

class PlyReader {
  public:
    explicit PlyReader(const std::string& path)
        : _input(path, std::ios::binary), _context("PLY '" + path + "': ") {
        if (!_input)
            fail("cannot open file");
        _input.imbue(std::locale::classic());
        _input.seekg(0, std::ios::end);
        _end = _input.tellg();
        _input.seekg(0);
    }

    Scene::MeshData load(bool colorsSrgb) {
        const auto elements = readHeader();
        size_t vertexCount = 0;
        bool hasVertices = false;
        size_t faceCount = 0;
        bool hasFaces = false;
        const Element* textureVertices = nullptr;
        const Element* textureFaces = nullptr;
        std::array<bool, 11> attributes{};
        for (const auto& element : elements) {
            size_t minRecordBytes = 0;
            bool hasFaceIndices = false;
            for (const auto& property : element.properties) {
                minRecordBytes += _ascii ? 1
                    : scalarInfo(property.countType.value_or(property.type)).size;
                hasFaceIndices |= property.faceIndices;
            }
            if (element.count && (!minRecordBytes ||
                element.count > remainingBytes() / minRecordBytes))
                fail("element count exceeds the available payload");
            if (element.name == "face" && element.count && !hasFaceIndices)
                fail("face vertex_indices property is required");
            if (element.name == "face") {
                if (hasFaces)
                    fail("duplicate face element");
                hasFaces = true;
                faceCount = element.count;
            }
            if (element.name == "multi_texture_vertex" || element.name == "multi_texture_face") {
                auto& target = element.name == "multi_texture_vertex" ? textureVertices : textureFaces;
                if (target)
                    fail("duplicate texture element");
                target = &element;
                std::array<bool, 4> fields{};
                bool hasTextureIndices = false;
                for (const auto& property : element.properties) {
                    hasTextureIndices |= property.textureIndices;
                    const int attr = property.textureAttribute;
                    if (attr < 0)
                        continue;
                    if (property.countType || fields[attr] ||
                        (attr < 2 && !scalarInfo(property.type).integer))
                        fail("texture attributes must be unique scalars with integer tx/tn");
                    fields[attr] = true;
                }
                if (!fields[0] || (element.name == "multi_texture_vertex"
                    ? !fields[2] || !fields[3] : !fields[1] || !hasTextureIndices))
                    fail("incomplete texture element");
            }
            if (element.name != "vertex")
                continue;
            if (hasVertices)
                fail("duplicate vertex element");
            hasVertices = true;
            vertexCount = element.count;
            if (vertexCount > std::numeric_limits<unsigned int>::max())
                fail("too many vertices for MeshData indices");
            for (const auto& property : element.properties) {
                if (property.attribute < 0)
                    continue;
                if (property.countType || attributes[property.attribute])
                    fail("vertex attributes must be unique scalar properties");
                attributes[property.attribute] = true;
            }
        }
        if (!hasVertices || !attributes[0] || !attributes[1] || !attributes[2])
            fail("vertex x, y and z properties are required");
        if (bool(textureVertices) != bool(textureFaces))
            fail("texture vertices and faces must both be present");
        if (textureVertices && (textureVertices->count > std::numeric_limits<unsigned int>::max() ||
            !hasFaces || textureFaces->count != faceCount))
            fail("texture data must cover each face and use 32-bit UV indices");
        const auto complete = [&](int first, int count) {
            int present = 0;
            for (int i = first; i < first + count; ++i)
                present += attributes[i];
            if (present != 0 && present != count)
                fail("incomplete vertex normals, colors or UVs");
            return present != 0;
        };

        Scene::MeshData mesh;
        mesh.vertices.resize(vertexCount);
        if (complete(3, 3))
            mesh.normals.resize(vertexCount);
        if (complete(6, 3))
            mesh.colors.resize(vertexCount);
        if (complete(9, 2))
            mesh.uvs.resize(vertexCount);

        std::vector<glm::vec2> textureUvs(textureVertices ? textureVertices->count : 0);
        std::vector<unsigned int> textureIds(textureUvs.size());
        std::vector<TextureFace> textureFaceData;
        std::vector<unsigned int> textureCornerIds;
        std::vector<size_t> faceOffsets(textureFaces ? faceCount + 1 : 0);
        if (textureFaces)
            textureFaceData.reserve(textureFaces->count);

        for (const auto& element : elements) {
            for (size_t row = 0; row < element.count; ++row) {
                TextureFace textureFace;
                textureFace.begin = textureCornerIds.size();
                if (textureFaces && element.name == "face")
                    faceOffsets[row] = mesh.indices.size();
                for (const auto& property : element.properties) {
                    if (property.countType) {
                        const double size = readScalar(*property.countType);
                        const size_t itemBytes = _ascii ? 1 : scalarInfo(property.type).size;
                        if (size < 0 || size > remainingBytes() / itemBytes)
                            fail("invalid or truncated list length");
                        const auto count = static_cast<size_t>(size);
                        const bool isFace = property.faceIndices || property.textureIndices;
                        if (isFace && count < 3)
                            fail("faces must have at least three vertices");
                        unsigned int first = 0, previous = 0;
                        for (size_t i = 0; i < count; ++i) {
                            const double value = readScalar(property.type);
                            if (!isFace)
                                continue;
                            const size_t limit = property.faceIndices ? vertexCount : textureUvs.size();
                            if (value < 0 || value >= limit)
                                fail("face vertex index out of range");
                            const auto index = static_cast<unsigned int>(value);
                            auto& indices = property.faceIndices ? mesh.indices : textureCornerIds;
                            // Fan triangulation requires convex faces.
                            if (i >= 2)
                                indices.insert(indices.end(), {first, previous, index});
                            else if (i == 0)
                                first = index;
                            previous = index;
                        }
                    } else {
                        double value = readScalar(property.type);
                        if (property.textureAttribute >= 0) {
                            const int attr = property.textureAttribute;
                            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
                                fail("texture attributes must be finite float values");
                            if (attr < 2 && (value < 0 || value > std::numeric_limits<unsigned int>::max()))
                                fail("texture and face identifiers must be unsigned 32-bit values");
                            if (element.name == "multi_texture_vertex") {
                                if (attr == 0)
                                    textureIds[row] = static_cast<unsigned int>(value);
                                else
                                    textureUvs[row][attr - 2] = static_cast<float>(value);
                            } else if (attr == 0) {
                                textureFace.texture = static_cast<unsigned int>(value);
                            } else {
                                textureFace.face = static_cast<unsigned int>(value);
                            }
                            continue;
                        }
                        const int attr = property.attribute;
                        if (element.name != "vertex" || attr < 0)
                            continue;
                        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
                            fail("vertex attributes must be finite float values");
                        if (attr >= 6 && attr < 9) {
                            const auto info = scalarInfo(property.type);
                            if (info.integer)
                                value /= info.max;
                            if (value < 0 || value > 1)
                                fail("vertex RGB must be in the declared integer range or [0, 1]");
                            if (colorsSrgb)
                                value = value <= 0.04045 ? value / 12.92
                                    : std::pow((value + 0.055) / 1.055, 2.4);
                        }
                        if (attr < 3)
                            mesh.vertices[row][attr] = static_cast<float>(value);
                        else if (attr < 6)
                            mesh.normals[row][attr - 3] = static_cast<float>(value);
                        else if (attr < 9)
                            mesh.colors[row][attr - 6] = static_cast<float>(value);
                        else
                            mesh.uvs[row][attr - 9] = static_cast<float>(value);
                    }
                }
                if (element.name == "multi_texture_face") {
                    textureFace.end = textureCornerIds.size();
                    textureFaceData.push_back(textureFace);
                }
            }
            if (textureFaces && element.name == "face")
                faceOffsets.back() = mesh.indices.size();
        }
        if (textureFaces && faceCount)
            mesh = applyTextureUvs(mesh, textureUvs, textureIds, textureFaceData,
                                   textureCornerIds, faceOffsets);
        // Points have no surface normals to generate.
        if (!mesh.indices.empty())
            mesh.fillMissingAttributes();
        return mesh;
    }

  private:
    Scene::MeshData applyTextureUvs(
        const Scene::MeshData& source, const std::vector<glm::vec2>& uvs,
        const std::vector<unsigned int>& textureIds, const std::vector<TextureFace>& faces,
        const std::vector<unsigned int>& corners, const std::vector<size_t>& faceOffsets) const {
        Scene::MeshData mesh;
        mesh.vertices.reserve(source.vertices.size());
        mesh.uvs.reserve(source.vertices.size());
        if (!source.normals.empty())
            mesh.normals.reserve(source.vertices.size());
        if (!source.colors.empty())
            mesh.colors.reserve(source.vertices.size());
        mesh.indices.resize(source.indices.size());
        std::vector<bool> seenFaces(faces.size(), false);
        std::unordered_map<uint64_t, unsigned int> vertices;
        vertices.reserve(source.vertices.size());

        for (const auto& face : faces) {
            if (face.face >= faces.size() || seenFaces[face.face])
                fail("tn must identify each source face exactly once");
            seenFaces[face.face] = true;
            if (face.texture != faces.front().texture)
                fail("multiple textures require material subsets; only one texture is supported");
            const size_t begin = faceOffsets[face.face];
            if (face.end - face.begin != faceOffsets[face.face + 1] - begin)
                fail("texture face and source face corner counts differ");

            for (size_t i = face.begin; i < face.end; ++i) {
                const auto uvIdx = corners[i];
                if (textureIds[uvIdx] != face.texture)
                    fail("texture face and UV texture identifiers differ");
                const size_t corner = begin + i - face.begin;
                const auto vertexIdx = source.indices[corner];
                // Split vertices at UV seams.
                const uint64_t key = (uint64_t(vertexIdx) << 32) | uvIdx;
                if (mesh.vertices.size() >= std::numeric_limits<unsigned int>::max())
                    fail("too many vertices after splitting UV seams");
                auto [it, inserted] = vertices.emplace(key, static_cast<unsigned int>(mesh.vertices.size()));
                if (inserted) {
                    mesh.vertices.push_back(source.vertices[vertexIdx]);
                    // Convert image-top V to the OBJ UV convention.
                    mesh.uvs.emplace_back(uvs[uvIdx].x, 1.0f - uvs[uvIdx].y);
                    if (!source.normals.empty())
                        mesh.normals.push_back(source.normals[vertexIdx]);
                    if (!source.colors.empty())
                        mesh.colors.push_back(source.colors[vertexIdx]);
                }
                mesh.indices[corner] = it->second;
            }
        }
        return mesh;
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(_context + message);
    }

    size_t remainingBytes() {
        const auto position = _input.tellg();
        return position >= 0 && position <= _end ? static_cast<size_t>(_end - position) : 0;
    }

    Scalar readType(const std::string& name) const {
        if (name == "char" || name == "int8") return Scalar::Int8;
        if (name == "uchar" || name == "uint8") return Scalar::Uint8;
        if (name == "short" || name == "int16") return Scalar::Int16;
        if (name == "ushort" || name == "uint16") return Scalar::Uint16;
        if (name == "int" || name == "int32") return Scalar::Int32;
        if (name == "uint" || name == "uint32") return Scalar::Uint32;
        if (name == "float" || name == "float32") return Scalar::Float32;
        if (name == "double" || name == "float64") return Scalar::Float64;
        fail("unsupported property type: " + name);
    }

    std::vector<Element> readHeader() {
        std::string line;
        if (!std::getline(_input, line) || (line != "ply" && line != "ply\r"))
            fail("missing ply signature");
        std::vector<Element> elements;
        bool hasFormat = false;
        while (std::getline(_input, line)) {
            std::istringstream words(line);
            std::string keyword;
            words >> keyword;
            if (keyword == "comment" || keyword == "obj_info")
                continue;
            if (keyword == "end_header") {
                if (!hasFormat) fail("missing format declaration");
                return elements;
            }
            if (keyword == "format") {
                std::string format, version;
                if (hasFormat || !(words >> format >> version) || version != "1.0")
                    fail("expected one PLY 1.0 format declaration");
                if (format != "ascii" && format != "binary_little_endian" && format != "binary_big_endian")
                    fail("unsupported format: " + format);
                _ascii = format == "ascii";
                const uint16_t one = 1;
                const bool hostLittle = *reinterpret_cast<const unsigned char*>(&one) == 1;
                _swapBytes = hostLittle != (format == "binary_little_endian");
                hasFormat = true;
            } else if (keyword == "element") {
                Element element;
                long long count;
                if (!(words >> element.name >> count) || count < 0 ||
                    static_cast<unsigned long long>(count) > std::numeric_limits<size_t>::max())
                    fail("invalid element declaration");
                element.count = static_cast<size_t>(count);
                elements.push_back(std::move(element));
            } else if (keyword == "property") {
                if (elements.empty()) fail("property without an element");
                std::string type, name;
                if (!(words >> type >> name)) fail("invalid property declaration");
                Property property;
                if (type == "list") {
                    property.countType = readType(name);
                    if (!scalarInfo(*property.countType).integer || !(words >> type >> name))
                        fail("list count must have an integer type");
                }
                property.type = readType(type);
                auto& element = elements.back();
                if (element.name == "vertex") {
                    constexpr std::array<const char*, 11> names{
                        "x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "u", "v"};
                    if (name == "s" || name == "texture_u") name = "u";
                    if (name == "t" || name == "texture_v") name = "v";
                    for (size_t i = 0; i < names.size(); ++i)
                        if (name == names[i]) property.attribute = static_cast<int>(i);
                } else if (element.name == "face" && (name == "vertex_indices" || name == "vertex_index")) {
                    if (!property.countType || !scalarInfo(property.type).integer)
                        fail("face vertex indices must be an integer list");
                    for (const auto& existing : element.properties)
                        if (existing.faceIndices) fail("duplicate face vertex indices");
                    property.faceIndices = true;
                } else if (element.name == "multi_texture_vertex") {
                    if (name == "tx") property.textureAttribute = 0;
                    else if (name == "u") property.textureAttribute = 2;
                    else if (name == "v") property.textureAttribute = 3;
                } else if (element.name == "multi_texture_face") {
                    if (name == "tx") property.textureAttribute = 0;
                    else if (name == "tn") property.textureAttribute = 1;
                    else if (name == "texture_vertex_indices") {
                        if (!property.countType || !scalarInfo(property.type).integer)
                            fail("texture vertex indices must be an integer list");
                        for (const auto& existing : element.properties)
                            if (existing.textureIndices)
                                fail("duplicate texture vertex indices");
                        property.textureIndices = true;
                    }
                }
                element.properties.push_back(property);
            } else {
                fail("unsupported header directive: " + keyword);
            }
        }
        fail("missing end_header");
    }

    template <typename T> double readBinary() {
        std::array<char, sizeof(T)> bytes;
        if (!_input.read(bytes.data(), bytes.size())) fail("truncated binary payload");
        if (_swapBytes) std::reverse(bytes.begin(), bytes.end());
        T value;
        std::memcpy(&value, bytes.data(), sizeof(T));
        return static_cast<double>(value);
    }

    double readScalar(Scalar type) {
        if (_ascii) {
            if (!(_input >> _token)) fail("truncated ASCII payload");
            // Parse numbers independently of the process locale.
            const char* begin = _token.data();
            const char* end = begin + _token.size();
            if (begin != end && *begin == '+')
                ++begin;
            double value = 0;
            const auto parsed = std::from_chars(begin, end, value);
            const auto info = scalarInfo(type);
            if (parsed.ec != std::errc{} || parsed.ptr != end ||
                value < info.min || value > info.max ||
                (info.integer && (!std::isfinite(value) || std::floor(value) != value)))
                fail("invalid ASCII scalar: " + _token);
            return value;
        }
        switch (type) {
        case Scalar::Int8: return readBinary<int8_t>();
        case Scalar::Uint8: return readBinary<uint8_t>();
        case Scalar::Int16: return readBinary<int16_t>();
        case Scalar::Uint16: return readBinary<uint16_t>();
        case Scalar::Int32: return readBinary<int32_t>();
        case Scalar::Uint32: return readBinary<uint32_t>();
        case Scalar::Float32: return readBinary<float>();
        case Scalar::Float64: return readBinary<double>();
        }
        fail("unsupported scalar type");
    }

    std::ifstream _input;
    std::string _context;
    std::string _token;
    std::streamoff _end = 0;
    bool _ascii = true;
    bool _swapBytes = false;
};

} // namespace

Scene::MeshData loadPly(const std::string& path, bool colorsSrgb) {
    return PlyReader(path).load(colorsSrgb);
}

} // namespace KE::Asset
