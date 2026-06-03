#pragma once

// RuntimeConfig — strongly-typed mirror of path_tracer_config.json.
//
// This header also defines the project's tiny self-contained math types
// (Vec3 and helpers). Keeping math here — rather than pulling in GLM —
// matches the project's "no third-party wrapper clutter" goal and keeps
// the same Vec3 layout shared between the OBJ loader, config parser, and
// the Vulkan front-end.

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Single source of truth for pi across the C++ side; shaders define their
// own copies in path_tracer_common.glsl.
inline constexpr float kPi = 3.14159265358979323846f;

// Plain 3-component float vector. Trivially copyable, matches std430 layout
// for a tightly-packed vec3 on the GPU once padded by the caller.
struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

inline bool operator==(const Vec3& left, const Vec3& right)
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

inline bool operator!=(const Vec3& left, const Vec3& right)
{
    return !(left == right);
}

inline Vec3 operator+(const Vec3& left, const Vec3& right)
{
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

inline Vec3 operator-(const Vec3& left, const Vec3& right)
{
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

inline Vec3 operator*(const Vec3& value, float scalar)
{
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

inline Vec3& operator+=(Vec3& left, const Vec3& right)
{
    left.x += right.x;
    left.y += right.y;
    left.z += right.z;
    return left;
}

inline float Length(const Vec3& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

inline Vec3 Normalize(const Vec3& value)
{
    const float length = Length(value);
    if (length <= 0.0f)
    {
        return {};
    }
    return value * (1.0f / length);
}

inline Vec3 Cross(const Vec3& left, const Vec3& right)
{
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

// Physical sky / atmosphere parameters fed to sky.comp. These follow the
// Bruneton/Nishita single-scattering parameterization: Rayleigh + Mie
// scattering against an Earth-sized sphere, plus a directional sun disk.
//
// Units are SI (metres) for the radii and scale heights; scattering
// coefficients are 1/m. secondarySamples / viewSteps / samples control
// numerical integration cost.
struct SkySpectralConfig
{
    std::array<float, 3> betaRayleigh{3.8e-6f, 13.5e-6f, 33.1e-6f};
    float betaMie = 21e-6f;
    float mieG = 0.76f;
    float earthRadius = 6360e3f;
    float atmosphereRadius = 6420e3f;
    float scaleHeightRayleigh = 7994.0f;
    float scaleHeightMie = 1200.0f;
    std::array<float, 3> sunRadiance{20.0f, 18.0f, 14.5f};
    std::array<float, 3> sunDirection{0.35f, 0.3f, 0.25f};
    float sunRadius = 0.1f;
    float sunAa = 0.01f;
    uint32_t secondarySamples = 1;
    uint32_t viewSteps = 5;
    uint32_t samples = 1;

    // ── Vector radiative transfer (polarized sky) ──
    // Rayleigh molecular depolarization factor (air ≈ 0.0279). Caps the
    // single-scatter degree of polarization below the ideal 1.0.
    float rayleighDepolarization = 0.0279f;
    // Aerosol model for the precomputed Lorenz–Mie scattering matrix.
    float aerosolRefractiveIndexReal = 1.33f;
    float aerosolRefractiveIndexImag = 0.0f;
    float aerosolMeanRadiusMicrometers = 0.2f; // log-normal geometric mean radius
    float aerosolSigma = 1.5f;                 // log-normal geometric std dev (> 1)
    std::array<float, 3> aerosolWavelengthsNmRgb{680.0f, 550.0f, 440.0f};
    uint32_t mieTableAngleBins = 181;          // scattering-angle samples in the Mie table
};

// True when any field that feeds the precomputed Lorenz–Mie scattering matrix
// changes — the table is sun-independent, so it only needs rebuilding here.
inline bool HasMieAerosolChanged(const SkySpectralConfig& left, const SkySpectralConfig& right)
{
    return left.aerosolRefractiveIndexReal != right.aerosolRefractiveIndexReal
           || left.aerosolRefractiveIndexImag != right.aerosolRefractiveIndexImag
           || left.aerosolMeanRadiusMicrometers != right.aerosolMeanRadiusMicrometers
           || left.aerosolSigma != right.aerosolSigma
           || left.aerosolWavelengthsNmRgb != right.aerosolWavelengthsNmRgb
           || left.mieTableAngleBins != right.mieTableAngleBins;
}

// Top-level configuration loaded from path_tracer_config.json. All fields
// have defaults so a missing config still produces a valid scene.
struct RuntimeConfig
{
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t frameCount = 2;
    uint32_t samplesPerPixel = 1;
    Vec3 initialPosition{0.0f, 0.35f, -6.5f};
    Vec3 initialLookAt{0.0f, -0.1f, 3.8f};
    float fovYDegrees = 40.0f;
    float moveSpeed = 4.0f;
    float fastMoveSpeed = 10.0f;
    float mouseSensitivity = 0.0035f;
    float keyLookSpeed = 1.8f;
    float polarizerRotateSpeed = 6.3f;
    float maxPitchDegrees = 89.0f;
    std::array<float, 3> skyBottomColor{0.03f, 0.04f, 0.06f};
    float skyExposure = 1.35f;
    std::array<float, 3> skyTopColor{0.55f, 0.72f, 0.95f};
    SkySpectralConfig skySpectral{};
};

// Resolve a runtime asset (config, SPIR-V blob, etc.) by checking the
// executable directory, its parent, and the current working directory.
// Returns {} when no candidate exists; the caller decides whether this is
// fatal.
std::filesystem::path ResolveRuntimeFilePath(const wchar_t* fileName);

// Load an entire text file into memory as a single std::string. Throws on
// open / read failure or empty file.
std::string LoadTextFile(const std::filesystem::path& filePath);

// Parse and fully validate a path_tracer_config.json document. Throws
// std::runtime_error with a descriptive context message on any structural
// or semantic error.
RuntimeConfig ParseRuntimeConfig(const std::string& jsonText);
