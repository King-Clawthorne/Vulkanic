#pragma once

/// @file Settings.hpp
/// @brief Renderer configuration types, spectral dimensions, and table interfaces.
///
/// Declares the settings shared by the application and the CPU routines that
/// build the lookup tables consumed by the compute shader.
/// @ingroup renderer_settings
/// @ingroup lookup_table_generation

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>
#include <glm/glm.hpp>

/// Single-precision pi used by the renderer and its camera calculations.
/// @ingroup renderer_settings
inline constexpr float kPi = std::numbers::pi_v<float>;

/// One aerosol population used by the atmospheric Mie-scattering model.
///
/// Aerosol density decays exponentially with altitude. The radius distribution
/// is log-normal and the complex refractive index is relative to air.
/// @ingroup renderer_settings
struct AerosolConfig {
    float beta; ///< Extinction scale in inverse metres, normalized at the reference Mie band.
    float scaleHeight; ///< Exponential density scale height, in metres.
    float refractiveIndexReal; ///< Real part of the relative particle refractive index.
    float refractiveIndexImag; ///< Imaginary part of the relative particle refractive index.
    float meanRadiusMicrometers; ///< Geometric-mean particle radius for the log-normal distribution, in micrometres.
    float sigma; ///< Geometric standard deviation of the log-normal radius distribution.

    /// Compares all aerosol parameters for exact equality.
    [[nodiscard]] friend bool operator==(const AerosolConfig&, const AerosolConfig&) = default;
};

/// Parameters used to build atmospheric lookup tables and shader constants.
/// Sun direction is in world space. Radii and scale heights are in metres.
/// @ingroup renderer_settings
struct SkyConfig {
    float betaRayleigh550 = 13.5e-6f; ///< Rayleigh scattering coefficient at 550 nm, in inverse metres.
    float earthRadius = 6360e3f; ///< Planet radius, in metres.
    float atmosphereRadius = 6420e3f; ///< Outer radius of the modeled atmosphere, in metres.
    float scaleHeightRayleigh = 7994.0f; ///< Rayleigh density scale height, in metres.
    float sunRadiance550 = 8317.742f; ///< Solar radiance reference at 550 nm, in renderer radiance units.
    glm::vec3 sunDirection{0.35f, 0.01f, 0.25f}; ///< Direction toward the Sun in world space.
    float sunRadius = 0.00465f; ///< Angular radius of the Sun, in radians.
    uint32_t secondarySamples = 2; ///< Number of path samples per secondary-ray segment.
    uint32_t viewSteps = 2; ///< Number of integration samples per primary view-ray segment.
    uint32_t scatteringOrders = 2; ///< Maximum scattering order evaluated by the path tracer.
    float rayleighDepolarization = 0.0279f; ///< Depolarization parameter used by the Rayleigh Mueller matrix.
    uint32_t mieTableAngleBins = 721; ///< Angular samples per Mie phase row over [0, pi].

    /// Fine and coarse aerosol populations, in the order expected by the shader.
    std::array<AerosolConfig, 2> aerosols{{{.beta = 21e-6f,
                                            .scaleHeight = 1200.0f,
                                            .refractiveIndexReal = 1.33f,
                                            .refractiveIndexImag = 0.0f,
                                            .meanRadiusMicrometers = 0.2f,
                                            .sigma = 1.5f},
                                           {.beta = 4.0e-6f,
                                            .scaleHeight = 1500.0f,
                                            .refractiveIndexReal = 1.53f,
                                            .refractiveIndexImag = 0.008f,
                                            .meanRadiusMicrometers = 1.0f,
                                            .sigma = 2.0f}}};

    /// Compares all sky parameters for exact equality.
    [[nodiscard]] friend bool operator==(const SkyConfig&, const SkyConfig&) = default;
};

/// Output image width and height, in pixels, used for the window and swapchain.
/// @ingroup renderer_settings
struct RenderConfig {
    glm::uvec2 resolution = { 1280, 720 }; ///< Requested swapchain width and height, in pixels.
};

/// Initial camera pose and vertical field-of-view limits.
/// Position and look-at coordinates are in world-space metres. Angles are in degrees.
/// @ingroup renderer_settings
struct CameraConfig {
    glm::vec3 initialPosition{0.0f, 2.0f, -10.0f}; ///< Initial camera position in world-space metres.
    glm::vec3 initialLookAt{0.35f, 2.01f, -9.75f}; ///< Initial world-space point toward which the camera looks.
    float fovYDegrees = 40.0f; ///< Vertical camera field of view, in degrees.
    float maxPitchDegrees = 89.0f; ///< Maximum absolute camera pitch, in degrees.
    float exposure = 10.0f; ///< Display exposure multiplier applied to accumulated radiance.
};

/// Mouse-look sensitivity in radians per cursor pixel and polarizer rotation rate in radians per second.
/// @ingroup renderer_settings
struct InputConfig {
    float mouseSensitivity = 0.0035f; ///< Camera rotation in radians per cursor pixel.
    float polarizerRotateSpeed = 1.5f; ///< Polarizer rotation in radians per second.
};

/// Complete set of renderer settings consumed by the application.
/// @ingroup renderer_settings
struct RuntimeConfig {
    RenderConfig render{}; ///< Window and swapchain dimensions.
    CameraConfig camera{}; ///< Initial camera pose, field of view, and exposure.
    InputConfig input{}; ///< Mouse and polarizer control rates.
    SkyConfig sky{}; ///< Atmosphere, Sun, and scattering parameters.
};

/// Number of wavelength bands in the renderer's spectral lookup tables.
/// @ingroup lookup_table_generation
inline constexpr int kSpectralBandCount = 17;
/// First reference wavelength in the spectral grid, in nanometres.
/// @ingroup lookup_table_generation
inline constexpr double kSpectralLambdaMinNm = 380.0;
/// Wavelength spacing between adjacent spectral bands, in nanometres.
/// @ingroup lookup_table_generation
inline constexpr double kSpectralLambdaStepNm = 25.0;

/// Builds normalized Stokes phase data for one aerosol at each spectral band.
/// Rows are band-major, with one vec4 per angular bin.
/// @param sky Atmospheric and spectral parameters used to build the table.
/// @param aerosol Index of the aerosol population in sky.aerosols (0 or 1).
/// @param crossSections Receives per-band extinction (x) and scattering (y) cross sections.
/// @return Band-major Stokes phase table.
/// @ingroup lookup_table_generation
std::vector<glm::vec4> ComputeMieScatteringTable(const SkyConfig& sky, int aerosol, std::array<glm::dvec2, kSpectralBandCount>& crossSections);

/// Appends a normalized angular sampling CDF for a band-major phase table.
///
/// Reads one row of `bins` intensity samples for each spectral band starting at
/// `firstEntry`, combines them using `bandWeights`, then appends `bins` vec4
/// entries. Each appended entry stores its CDF value in x and fitted within-bin
/// density shape in y. The CDF integrates over solid angle. Its cosine-linear
/// within-bin density fit reduces the scalar phase proposal's quadrature estimate
/// of the second moment.
/// @param[in,out] table Phase rows followed by the appended CDF and shape entries.
/// @param[in] bins Number of angular samples in each phase row.
/// @param[in] firstEntry Index of the first phase sample in table.
/// @param[in] bandWeights Relative displayed-radiance contribution of each spectral band.
/// @pre bins is at least 2 and table contains kSpectralBandCount * bins samples from firstEntry.
/// @post Exactly bins CDF and shape entries are appended to table.
/// @ingroup lookup_table_generation
void AppendSamplingCdf(std::vector<glm::vec4>& table, int bins, size_t firstEntry, const std::array<double, kSpectralBandCount>& bandWeights);

/// Number of altitude samples in the atmospheric transmittance table.
/// @ingroup lookup_table_generation
inline constexpr int kTransmittanceAltitudeBins = 64;
/// Number of direction-cosine samples in each transmittance row.
/// @ingroup lookup_table_generation
inline constexpr int kTransmittanceMuBins = 256;

/// Integrates atmospheric density columns for each altitude and direction.
/// @param sky Atmospheric radii, scale heights, and aerosol parameters.
/// @return Altitude-major table with kTransmittanceAltitudeBins rows and
/// kTransmittanceMuBins direction samples per row. Each vec4 stores Rayleigh,
/// fine-aerosol, reserved zero, and coarse-aerosol density columns, in that order.
/// @ingroup lookup_table_generation
std::vector<glm::vec4> ComputeTransmittanceTable(const SkyConfig& sky);

/// Wavelength-dependent values used to populate one shader spectral band.
/// betaRayleighScale is relative to 550 nm. The remaining fields hold the
/// solar irradiance scale and CIE XYZ matching values.
/// @ingroup lookup_table_generation
struct SpectralBand {
    double betaRayleighScale; ///< Rayleigh coefficient relative to its value at 550 nm.
    double sunIrradianceScale; ///< Interpolated solar irradiance sample from the reference spectrum.
    glm::dvec3 cie; ///< CIE 1931 XYZ color-matching values at this wavelength.
};

/// Samples wavelength-dependent source datasets at the requested wavelength.
/// @param wavelengthNm Wavelength to sample, in nanometres.
/// @return Rayleigh scale, solar irradiance scale, and CIE XYZ values.
/// @pre wavelengthNm is in [360, 830), the interval supported by solar-spectrum interpolation.
/// @ingroup lookup_table_generation
SpectralBand ComputeSpectralAtWavelength(double wavelengthNm);

/// Integrates the intensity component of a tabulated phase function for normalization.
/// @param phase Stokes phase samples on a uniform angular grid spanning [0, pi].
/// @return Solid-angle normalization under the renderer's phase convention.
/// @pre phase contains at least two angular samples.
/// @ingroup lookup_table_generation
inline double PhaseNormalization(const std::vector<glm::dvec4>& phase) {
    const int bins = static_cast<int>(phase.size());
    const double dTheta = std::numbers::pi / (bins - 1);

    double integral = 0.0;
    for (int i = 0; i + 1 < bins; ++i) { integral += phase[i].x * std::sin(i * dTheta) + phase[i + 1].x * std::sin((i + 1) * dTheta); }
    return 0.25 * integral * dTheta;
}
