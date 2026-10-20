#pragma once

/// @file Settings.hpp
/// @brief Renderer settings and Rayleigh lookup table construction.
///
/// Defines the settings and CPU routines that prepare the
/// values consumed by the compute shader.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <execution>
#include <numbers>
#include <ranges>
#include <vector>
#include <glm/glm.hpp>

/// Single-precision pi used by the renderer and its camera calculations.
inline constexpr float kPi = std::numbers::pi_v<float>;

/// Parameters used to build atmospheric lookup tables and shader constants.
/// Sun direction is in world space. Radii and scale heights are in metres.
struct SkyConfig {
    float betaRayleighGreen = 13.5e-6f; ///< Green-channel Rayleigh scattering coefficient at 550 nm, in inverse metres.
    float earthRadius = 6360e3f; ///< Planet radius, in metres.
    float atmosphereRadius = 6420e3f; ///< Outer radius of the modeled atmosphere, in metres.
    float scaleHeightRayleigh = 7994.0f; ///< Rayleigh density scale height, in metres.
    float sunRadiance = 8317.742f; ///< Equal solar radiance per color channel, in renderer radiance units.
    glm::vec3 sunDirection{0.35f, 0.01f, 0.25f}; ///< Direction toward the Sun in world space.
    float sunRadius = 0.00465f; ///< Angular radius of the Sun, in radians.
    float rayleighDepolarization = 0.0279f; ///< Depolarization parameter used by the Rayleigh Mueller matrix.
    /// Compares all sky parameters for exact equality.
    [[nodiscard]] friend bool operator==(const SkyConfig&, const SkyConfig&) = default;
};

/// Output image width and height, in pixels, used for the window and swapchain.
struct RenderConfig {
    glm::uvec2 resolution = { 1280, 720 }; ///< Requested swapchain width and height, in pixels.
};

/// Initial camera pose and vertical field-of-view limits.
/// Position and look-at coordinates are in world-space metres. Angles are in degrees.
struct CameraConfig {
    glm::vec3 initialPosition{0.0f, 2.0f, -10.0f}; ///< Initial camera position in world-space metres.
    glm::vec3 initialLookAt{0.35f, 2.0f, -9.75f}; ///< Initial world-space point toward which the camera looks.
    float fovYDegrees = 40.0f; ///< Vertical camera field of view, in degrees.
    float maxPitchDegrees = 89.0f; ///< Maximum absolute camera pitch, in degrees.
    float exposure = 10.0f; ///< Display exposure multiplier applied to integrated radiance.
};

/// Mouse-look sensitivity in radians per cursor pixel and polarizer rotation rate in radians per second.
struct InputConfig {
    float mouseSensitivity = 0.0035f; ///< Camera rotation in radians per cursor pixel.
    float polarizerRotateSpeed = 1.5f; ///< Polarizer rotation in radians per second.
};

/// Complete set of renderer settings consumed by the application.
struct RuntimeConfig {
    RenderConfig render{}; ///< Window and swapchain dimensions.
    CameraConfig camera{}; ///< Initial camera pose, field of view, and exposure.
    InputConfig input{}; ///< Mouse and polarizer control rates.
    SkyConfig sky{}; ///< Atmosphere, Sun, and scattering parameters.
};

/// Approximate Rayleigh scattering strength for red, green, and blue relative
/// to green, using an inverse fourth-power fit at 650, 550, and 450 nm.
inline constexpr glm::vec3 kRayleighScale = { 0.512f, 1.0f, 2.231f };

/// Number of altitude samples in the atmospheric transmittance table.
inline constexpr int kTransmittanceAltitudeBins = 64;
/// Number of direction-cosine samples in each transmittance row.
inline constexpr int kTransmittanceMuBins = 256;

namespace {
    /// Abscissae for 8-point Gauss-Legendre integration.
    constexpr std::array<double, 8> kGaussNodes = { -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363 };
    /// Weights paired with kGaussNodes for 8-point Gauss-Legendre integration.
    constexpr std::array<double, 8> kGaussWeights = { 0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763 };

    /// Runs an integer-indexed loop with the standard parallel execution policy.
    /// @tparam Body Callable invoked once for each index.
    /// @param count Number of indices in the half-open range [0, count).
    /// @param body Work to perform for each index. Calls may run concurrently.
    template <class Body>
    void ParallelFor(int count, Body body) {
        const auto indices = std::views::iota(0, count);
        std::for_each(std::execution::par, indices.begin(), indices.end(), body);
    }
}

/// Integrates the Rayleigh density column for each altitude and direction.
/// @param sky Atmospheric radii and Rayleigh scale height.
/// @return Altitude-major table with kTransmittanceAltitudeBins rows and
/// kTransmittanceMuBins direction samples per row.
inline std::vector<float> ComputeTransmittanceTable(const SkyConfig& sky) {
    const double re = sky.earthRadius;
    const double ra = sky.atmosphereRadius;
    std::vector<float> table(static_cast<size_t>(kTransmittanceAltitudeBins) * kTransmittanceMuBins);

    ParallelFor(kTransmittanceAltitudeBins, [&](int a) {
        const double x = static_cast<double>(a) / (kTransmittanceAltitudeBins - 1);
        const double r = re + x * x * (ra - re);

        for (int m = 0; m < kTransmittanceMuBins; m++) {
            const double v = 2.0 * m / (kTransmittanceMuBins - 1) - 1.0;
            const double mu = v * std::abs(v);
            const double b = r * mu;
            const double end = std::sqrt(std::max(b * b + (ra - r) * (ra + r), 0.0)) - b;
            std::vector<double> cuts { 0.0, end };

            if (-b > 0.0 && -b < end) cuts.push_back(-b);
            for (double height : { 0.0 }) {
                const double shell = re + height;
                const double discriminant = b * b + (shell - r) * (shell + r);
                if (discriminant < 0.0) continue;
                for (double t : { -b - std::sqrt(discriminant), -b + std::sqrt(discriminant) })
                    if (t > 0.0 && t < end) cuts.push_back(t);
            }

            std::ranges::sort(cuts);
            double column = 0.0;
            const double panelLength = 8.0 * static_cast<double>(sky.scaleHeightRayleigh);

            for (size_t segment = 1; segment < cuts.size(); segment++) {
                const double start = cuts[segment - 1], length = cuts[segment] - cuts[segment - 1];
                const double middle = start + 0.5 * length;
                if (r * r + middle * (middle + 2.0 * b) < re * re) {
                    column += length;
                    continue;
                }

                const int panels = std::max(1, static_cast<int>(std::ceil(length / panelLength)));
                const double half = 0.5 * length / panels;
                for (int panel = 0; panel < panels; panel++) {
                    for (size_t node = 0; node < kGaussNodes.size(); node++) {
                        const double t = start + (2 * panel + 1 + kGaussNodes[node]) * half;
                        const double radius = std::sqrt(r * r + t * (t + 2.0 * b));
                        const double altitude = std::max((r * r - re * re + t * (t + 2.0 * b)) / (radius + re), 0.0);
                        column += half * kGaussWeights[node] * std::exp(-altitude / sky.scaleHeightRayleigh);
                    }
                }
            }

            table[static_cast<size_t>(a) * kTransmittanceMuBins + static_cast<size_t>(m)] = static_cast<float>(column);
        }
    });

    return table;
}
