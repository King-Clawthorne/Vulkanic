// ObjModel.cpp — minimal Wavefront .obj loader.
//
// The parser walks the file line-by-line, accumulates v / vn records into
// flat arrays, and emits a triangulated, GPU-ready vertex buffer for each
// face. Polygons with more than three vertices are fan-triangulated; faces
// without explicit normals get a synthesized face normal so the closest-hit
// shader always has a usable shading normal.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "ObjModel.h"

#include <windows.h>

#include <array>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
// One vertex reference in a face record. OBJ uses 1-based indices and
// allows negative indices (relative to the end of the array), so the raw
// integer is preserved here and resolved later by ResolveObjIndex().
struct ObjIndex
{
    int position = 0;
    int normal = 0;
};

// Normalize, but fall back to +Y when the vector is degenerate. Used both
// for parsed normals and for synthesized face normals on degenerate
// triangles, where any unit vector is preferable to producing NaNs that
// would later poison the BVH and shading code.
Vec3 NormalizeSafe(const Vec3& value)
{
    const float length = Length(value);
    if (length <= 1.0e-6f)
    {
        return {0.0f, 1.0f, 0.0f};
    }
    return value * (1.0f / length);
}

// Translate a raw OBJ index into a zero-based array position.
// OBJ semantics: positive = 1-based from the start, negative = relative
// from the end (-1 = last element), 0 is illegal.
int ResolveObjIndex(int index, size_t count, const std::string& context)
{
    if (index > 0)
    {
        const int resolved = index - 1;
        if (resolved >= static_cast<int>(count))
        {
            throw std::runtime_error(context + " references an index outside the source data.");
        }
        return resolved;
    }

    if (index < 0)
    {
        const int resolved = static_cast<int>(count) + index;
        if (resolved < 0 || resolved >= static_cast<int>(count))
        {
            throw std::runtime_error(context + " references an index outside the source data.");
        }
        return resolved;
    }

    throw std::runtime_error(context + " uses OBJ index 0, which is invalid.");
}

// Parse an integer from a string_view without copying it into a std::string.
// std::from_chars guarantees the entire token must be consumed for success,
// so trailing garbage in a face token is rejected here.
int ParseObjInteger(std::string_view token, const std::string& context)
{
    int value = 0;
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size())
    {
        throw std::runtime_error(context + " contains an invalid integer index.");
    }
    return value;
}

// Parse a single face vertex reference of the form "v", "v/vt", "v//vn",
// or "v/vt/vn". Texture coordinates are accepted syntactically but ignored
// because the path tracer only consumes positions and normals.
ObjIndex ParseFaceIndex(std::string_view token, const std::string& context)
{
    ObjIndex result{};

    const size_t firstSlash = token.find('/');
    if (firstSlash == std::string_view::npos)
    {
        result.position = ParseObjInteger(token, context);
        return result;
    }

    result.position = ParseObjInteger(token.substr(0, firstSlash), context);

    const size_t secondSlash = token.find('/', firstSlash + 1);
    if (secondSlash == std::string_view::npos)
    {
        return result;
    }

    const std::string_view normalToken = token.substr(secondSlash + 1);
    if (!normalToken.empty())
    {
        result.normal = ParseObjInteger(normalToken, context);
    }

    return result;
}

// Pack a (position, normal) pair into the GPU-aligned ModelVertex layout
// expected by the ray-tracing pipeline. position.w is forced to 1 so the
// shader can multiply by an instance transform without a separate splat.
ModelVertex MakeModelVertex(const Vec3& position, const Vec3& normal)
{
    ModelVertex vertex{};
    vertex.position[0] = position.x;
    vertex.position[1] = position.y;
    vertex.position[2] = position.z;
    vertex.position[3] = 1.0f;
    vertex.normal[0] = normal.x;
    vertex.normal[1] = normal.y;
    vertex.normal[2] = normal.z;
    return vertex;
}
} // namespace

// Search a small, fixed list of locations for a model file. The order is
// chosen so that an installed deployment (Models/ next to the exe) takes
// precedence over the source-tree layout (Models/ next to the build dir).
std::filesystem::path ResolveModelFilePath(const std::string& fileName)
{
    const std::filesystem::path filePath(fileName);
    if (filePath.is_absolute() && std::filesystem::exists(filePath))
    {
        return filePath;
    }

    WCHAR exePath[MAX_PATH]{};
    const DWORD pathLen = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (pathLen == 0 || pathLen == MAX_PATH)
    {
        throw std::runtime_error("Failed to resolve executable path.");
    }

    const auto exeDir = std::filesystem::path(exePath).parent_path();
    const std::array<std::filesystem::path, 5> candidatePaths = {
        exeDir / "Models" / filePath,
        exeDir.parent_path() / "Models" / filePath,
        std::filesystem::current_path() / "Models" / filePath,
        exeDir / filePath,
        std::filesystem::current_path() / filePath,
    };

    for (const auto& candidate : candidatePaths)
    {
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }
    }

    return {};
}

// Stream the file line-by-line and build an indexed triangle mesh.
//
// The function intentionally produces a 1:1 vertex-to-index mapping rather
// than welding shared vertices: the path tracer's BLAS build is fast even
// for redundant geometry, and skipping a hash-based dedup keeps the loader
// short and predictable.
ObjModel LoadObjModel(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath);
    if (!file)
    {
        throw std::runtime_error("Failed to open OBJ file: " + filePath.string());
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    ObjModel model{};

    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(file, line))
    {
        ++lineNumber;
        std::stringstream lineStream(line);
        std::string keyword;
        lineStream >> keyword;
        if (keyword.empty() || keyword[0] == '#')
        {
            continue;
        }

        if (keyword == "v")
        {
            Vec3 position{};
            if (!(lineStream >> position.x >> position.y >> position.z))
            {
                throw std::runtime_error("Invalid vertex position in OBJ file at line " + std::to_string(lineNumber) + ".");
            }
            positions.push_back(position);
            continue;
        }

        if (keyword == "vn")
        {
            Vec3 normal{};
            if (!(lineStream >> normal.x >> normal.y >> normal.z))
            {
                throw std::runtime_error("Invalid vertex normal in OBJ file at line " + std::to_string(lineNumber) + ".");
            }
            normals.push_back(NormalizeSafe(normal));
            continue;
        }

        if (keyword != "f")
        {
            continue;
        }

        std::vector<ObjIndex> faceIndices;
        std::string token;
        while (lineStream >> token)
        {
            faceIndices.push_back(ParseFaceIndex(token, "Face at line " + std::to_string(lineNumber)));
        }

        if (faceIndices.size() < 3)
        {
            throw std::runtime_error("OBJ faces must contain at least three vertices.");
        }

        auto emitVertex = [&](const ObjIndex& index, const Vec3& fallbackNormal)
        {
            const int positionIndex =
                ResolveObjIndex(index.position, positions.size(), "Face at line " + std::to_string(lineNumber));
            const Vec3 position = positions[static_cast<size_t>(positionIndex)];

            Vec3 normal = fallbackNormal;
            if (index.normal != 0)
            {
                const int normalIndex =
                    ResolveObjIndex(index.normal, normals.size(), "Face at line " + std::to_string(lineNumber));
                normal = normals[static_cast<size_t>(normalIndex)];
            }

            const uint32_t vertexIndex = static_cast<uint32_t>(model.vertices.size());
            model.vertices.push_back(MakeModelVertex(position, normal));
            model.indices.push_back(vertexIndex);
        };

        // Fan-triangulate the face: anchor on vertex 0 and emit
        // (0, i, i+1) triangles. Works for convex polygons; concave
        // n-gons in malformed assets will produce visible artifacts but
        // are not a supported input.
        for (size_t i = 1; i + 1 < faceIndices.size(); ++i)
        {
            const int i0 = ResolveObjIndex(faceIndices[0].position,
                                           positions.size(),
                                           "Face at line " + std::to_string(lineNumber));
            const int i1 = ResolveObjIndex(faceIndices[i].position,
                                           positions.size(),
                                           "Face at line " + std::to_string(lineNumber));
            const int i2 = ResolveObjIndex(faceIndices[i + 1].position,
                                           positions.size(),
                                           "Face at line " + std::to_string(lineNumber));

            const Vec3 p0 = positions[static_cast<size_t>(i0)];
            const Vec3 p1 = positions[static_cast<size_t>(i1)];
            const Vec3 p2 = positions[static_cast<size_t>(i2)];
            const Vec3 faceNormal = NormalizeSafe(Cross(p1 - p0, p2 - p0));

            emitVertex(faceIndices[0], faceNormal);
            emitVertex(faceIndices[i], faceNormal);
            emitVertex(faceIndices[i + 1], faceNormal);
        }
    }

    if (model.vertices.empty() || model.indices.empty())
    {
        throw std::runtime_error("OBJ file does not contain any triangle geometry: " + filePath.string());
    }

    return model;
}
