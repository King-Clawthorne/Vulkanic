#pragma once

// Minimal OBJ mesh loader used by the path tracer. The loader is intentionally
// self-contained (no TinyObjLoader / Assimp dependency) and produces vertex
// buffers in the exact GPU-ready layout consumed by the ray-tracing pipeline.

#include "RuntimeConfig.h"

#include <cstdint>
#include <filesystem>
#include <vector>

// GPU-ready vertex layout. Position uses a vec4 with w=1 and normal pads to
// vec4 to satisfy std430 alignment on the device side without extra packing.
struct ModelVertex
{
    float position[4];
    float normal[4];
};

// Triangulated mesh: triangle list indices reference into vertices[].
struct ObjModel
{
    std::vector<ModelVertex> vertices;
    std::vector<uint32_t> indices;
};

// Search a small set of conventional locations (next to the executable, the
// working directory, etc.) and return the first matching path or {} if none
// is found. Accepts both absolute and relative file names.
std::filesystem::path ResolveModelFilePath(const std::string& fileName);

// Parse a Wavefront .obj file into an ObjModel. Supports v / vn / f records
// (with or without texture-coordinate slashes), triangulates n-gons via
// fan-triangulation, and synthesizes face normals when none are supplied.
// Throws std::runtime_error on malformed input or empty geometry.
ObjModel LoadObjModel(const std::filesystem::path& filePath);
