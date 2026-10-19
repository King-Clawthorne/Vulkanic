#pragma once

/// @file Settings.hpp
/// @brief Renderer settings and Rayleigh lookup table construction.
///
/// Defines the settings and CPU routines that prepare atmospheric and spectral
/// values consumed by the compute shader.

#include "DataTables.hpp"
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
    float betaRayleigh550 = 13.5e-6f; ///< Rayleigh scattering coefficient at 550 nm, in inverse metres.
    float earthRadius = 6360e3f; ///< Planet radius, in metres.
    float atmosphereRadius = 6420e3f; ///< Outer radius of the modeled atmosphere, in metres.
    float scaleHeightRayleigh = 7994.0f; ///< Rayleigh density scale height, in metres.
    float sunRadiance550 = 8317.742f; ///< Solar radiance reference at 550 nm, in renderer radiance units.
    glm::vec3 sunDirection{0.35f, 0.01f, 0.25f}; ///< Direction toward the Sun in world space.
    float sunRadius = 0.00465f; ///< Angular radius of the Sun, in radians.
    uint32_t viewSteps = 2; ///< Number of integration samples per primary view-ray segment.
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
    glm::vec3 initialLookAt{0.35f, 2.01f, -9.75f}; ///< Initial world-space point toward which the camera looks.
    float fovYDegrees = 40.0f; ///< Vertical camera field of view, in degrees.
    float maxPitchDegrees = 89.0f; ///< Maximum absolute camera pitch, in degrees.
    float exposure = 10.0f; ///< Display exposure multiplier applied to accumulated radiance.
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

/// Number of wavelength bands in the renderer's spectral lookup tables.
inline constexpr int kSpectralBandCount = 17;
/// First reference wavelength in the spectral grid, in nanometres.
inline constexpr double kSpectralLambdaMinNm = 380.0;
/// Wavelength spacing between adjacent spectral bands, in nanometres.
inline constexpr double kSpectralLambdaStepNm = 25.0;

/// Number of altitude samples in the atmospheric transmittance table.
inline constexpr int kTransmittanceAltitudeBins = 64;
/// Number of direction-cosine samples in each transmittance row.
inline constexpr int kTransmittanceMuBins = 256;

/// Wavelength-dependent values used to populate one shader spectral band.
/// betaRayleighScale is relative to 550 nm. The remaining fields hold the
/// solar irradiance scale and CIE XYZ matching values.
struct SpectralBand {
    double betaRayleighScale; ///< Rayleigh coefficient relative to its value at 550 nm.
    double sunIrradianceScale; ///< Interpolated solar irradiance sample from the reference spectrum.
    glm::dvec3 cie; ///< CIE 1931 XYZ color-matching values at this wavelength.
};

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

        for (int m = 0; m < kTransmittanceMuBins; ++m) {
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

            for (size_t segment = 1; segment < cuts.size(); ++segment) {
                const double start = cuts[segment - 1], length = cuts[segment] - cuts[segment - 1];
                const double middle = start + 0.5 * length;
                if (r * r + middle * (middle + 2.0 * b) < re * re) {
                    column += length;
                    continue;
                }

                const int panels = std::max(1, static_cast<int>(std::ceil(length / panelLength)));
                const double half = 0.5 * length / panels;
                for (int panel = 0; panel < panels; ++panel) {
                    for (size_t node = 0; node < kGaussNodes.size(); ++node) {
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

namespace {
    /// Samples the CIE 1931 matching functions on their 1 nm grid.
    /// @param wavelengthNm Wavelength in nanometres.
    /// @return Interpolated XYZ matching values, or zero outside 360 to 830 nm.
    glm::dvec3 Cie1931Xyz(double wavelengthNm) {
        if (wavelengthNm < 360.0 || wavelengthNm > 830.0) return { };
        const double position = wavelengthNm - 360.0;
        const auto lower = static_cast<size_t>(position);
        const size_t upper = std::min(lower + 1, kCie1931Xyz.size() - 1);
        return glm::mix(kCie1931Xyz[lower], kCie1931Xyz[upper], position - static_cast<double>(lower));
    }

    /// Evaluates the wavelength-dependent Rayleigh scattering shape factor.
    /// @param wavelengthNm Wavelength in nanometres.
    /// @return Relative shape factor before normalization to the 550 nm reference.
    double RayleighShape(double wavelengthNm) {
        const double sigma2 = 1.0e6 / (wavelengthNm * wavelengthNm);
        const double n = 1.0 + 1.0e-8 * (8060.51 + 2480990.0 / (132.274 - sigma2) + 17455.7 / (39.32957 - sigma2));
        const double n2 = n * n - 1.0;
        return n2 * n2 / std::pow(wavelengthNm, 4.0);
    }
}

/// Samples wavelength-dependent source datasets at the requested wavelength.
/// @param wavelengthNm Wavelength to sample, in nanometres.
/// @return Rayleigh scale, solar irradiance scale, and CIE XYZ values.
/// @pre wavelengthNm is in [360, 830), the interval supported by solar-spectrum interpolation.
inline SpectralBand ComputeSpectralAtWavelength(double wavelengthNm) {
    SpectralBand result { .betaRayleighScale = 0.0, .sunIrradianceScale = 0.0, .cie = { } };
    result.betaRayleighScale = RayleighShape(wavelengthNm) / RayleighShape(550.0);
    const double solarPosition = wavelengthNm - 360.0;
    const auto solarLower = static_cast<size_t>(solarPosition);
    const double solarFraction = solarPosition - static_cast<double>(solarLower);
    result.sunIrradianceScale = std::lerp(kSolarIrradiance[solarLower], kSolarIrradiance[solarLower + 1], solarFraction);
    result.cie = Cie1931Xyz(wavelengthNm);
    return result;
}
