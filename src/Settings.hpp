#pragma once

/// @file Settings.hpp
/// @brief Renderer configuration types, spectral dimensions, and table interfaces.
///
/// Declares the settings shared by the application and the CPU routines that
/// build the lookup tables consumed by the compute shader.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>
#include <glm/glm.hpp>

/// Single-precision pi used by the renderer and its camera calculations.
inline constexpr float kPi = std::numbers::pi_v<float>;

/// One aerosol population used by the atmospheric Mie-scattering model.
/// Scale heights are in metres. meanRadiusMicrometers is in micrometres.
struct AerosolConfig {
    float beta, scaleHeight, refractiveIndexReal, refractiveIndexImag, meanRadiusMicrometers, sigma;
    [[nodiscard]] friend bool operator==(const AerosolConfig&, const AerosolConfig&) = default;
};

/// Parameters used to build atmospheric lookup tables and shader constants.
/// Sun direction is in world space. Radii and scale heights are in metres.
struct SkySpectralConfig {
    float betaRayleigh550 = 13.5e-6f;
    float earthRadius = 6360e3f, atmosphereRadius = 6420e3f, scaleHeightRayleigh = 7994.0f, sunRadiance550 = 8317.742f;
    glm::vec3 sunDirection{0.35f, 0.01f, 0.25f};
    float sunRadius = 0.00465f, sunAa = 0.0005f;
    uint32_t secondarySamples = 1, viewSteps = 1, samples = 1, scatteringOrders = 1;
    float rayleighDepolarization = 0.0279f, ozoneDobsonUnits = 300.0f;
    uint32_t mieTableAngleBins = 181;

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

    [[nodiscard]] friend bool operator==(const SkySpectralConfig&, const SkySpectralConfig&) = default;
};

/// Geometry, optical coefficients, and integration settings for the rain volume.
/// Radii and distance are in metres. Droplet size is in micrometres.
struct RainbowConfig {
    uint32_t enabled = 0;
    float distance = 4000.0f, height = 0.0f;
    glm::vec3 radii{6000.0f, 6000.0f, 1500.0f};
    float edgeSoftness = 0.15f, scatteringCoefficient = 5.0e-4f, extinctionCoefficient = 5.0e-4f;
    float effectiveRadiusMicrometers = 500.0f, effectiveVariance = 0.08f;
    uint32_t angleBins = 4097, viewSteps = 24, includeSecondary = 1;
    uint32_t scatteringOrders = 2, multipleScatteringSamples = 1, multipleScatteringSteps = 8;
    [[nodiscard]] friend bool operator==(const RainbowConfig&, const RainbowConfig&) = default;
};

/// Output image width and height, in pixels, used for the window and swapchain.
struct RenderConfig {
    glm::uvec2 resolution = { 1280, 720 };
};

/// Initial camera pose and vertical field-of-view limits.
/// Position and look-at coordinates are in world-space metres. Angles are in degrees.
struct CameraConfig {
    glm::vec3 initialPosition{0.0f, 2.0f, -10.0f};
    glm::vec3 initialLookAt{0.0f, 0.5f, 0.0f};
    float fovYDegrees = 40.0f, maxPitchDegrees = 89.0f;
};

/// Mouse-look sensitivity in radians per cursor pixel and polarizer rotation rate in radians per second.
struct InputConfig {
    float mouseSensitivity = 0.0035f, polarizerRotateSpeed = 1.5f;
};

/// Display exposure and atmospheric parameters used by the application.
struct SkyConfig {
    float exposure = 10.0f;
    SkySpectralConfig spectral{};
};

/// Complete set of renderer settings consumed by the application.
struct RuntimeConfig {
    RenderConfig render{};
    CameraConfig camera{};
    InputConfig input{};
    RainbowConfig rainbow{};
    SkyConfig sky{};
};

/// Number of wavelength bands in the renderer's spectral lookup tables.
inline constexpr int kSpectralBandCount = 17;
/// First reference wavelength in the spectral grid, in nanometres.
inline constexpr double kSpectralLambdaMinNm = 380.0;
/// Wavelength spacing between adjacent spectral bands, in nanometres.
inline constexpr double kSpectralLambdaStepNm = 25.0;

/// Builds normalized Stokes phase data for one aerosol at each spectral band.
/// Rows are band-major, with one vec4 per angular bin.
/// @param sky Atmospheric and spectral parameters used to build the table.
/// @param aerosol Index of the aerosol population in sky.aerosols (0 or 1).
/// @param crossSections Receives per-band extinction (x) and scattering (y) cross sections.
/// @return Band-major Stokes phase table.
std::vector<glm::vec4> ComputeMieScatteringTable(const SkySpectralConfig& sky, int aerosol, std::array<glm::dvec2, kSpectralBandCount>& crossSections);

/// Builds band-major rainbow phase data over [0, pi], including finite-sun blur.
/// @param rainbow Droplet geometry, optical properties, and integration settings.
/// @param sunRadius Angular radius of the sun, in radians.
/// @return Stokes phase table with one angular row per spectral band.
std::vector<glm::vec4> ComputeRainbowScatteringTable(const RainbowConfig& rainbow, float sunRadius);

/// Appends a normalized angular sampling CDF for a band-major phase table.
/// @param table Phase data to sample from. The CDF data is appended to this vector.
/// @param bins Number of angular bins in each phase row.
/// @param firstEntry Offset of the first phase entry for the selected table.
/// @param bandWeights Relative displayed-radiance contribution of each spectral band.
///
/// The CDF integrates over solid angle. Its cosine-linear within-bin density is
/// fitted to reduce the scalar phase proposal's quadrature estimate of the
/// second moment.
void AppendSamplingCdf(std::vector<glm::vec4>& table, int bins, size_t firstEntry, const std::array<double, kSpectralBandCount>& bandWeights);

/// Number of altitude samples in the atmospheric transmittance table.
inline constexpr int kTransmittanceAltitudeBins = 64;
/// Number of direction-cosine samples in each transmittance row.
inline constexpr int kTransmittanceMuBins = 256;

/// Integrates atmospheric density columns for each altitude and direction.
/// @param sky Atmospheric radii, scale heights, and aerosol parameters.
/// @return Table whose vec4 entries contain Rayleigh, fine-aerosol, ozone, and
/// coarse-aerosol density columns, in that order.
std::vector<glm::vec4> ComputeTransmittanceTable(const SkySpectralConfig& sky);

/// Evaluates the piecewise-linear relative ozone density profile used by the LUT.
/// @param altitude Altitude above the ground, in metres.
/// @return Nonnegative profile value, peaking at 25 km and reaching zero outside
/// the 10 to 40 km interval.
inline double OzoneProfile(double altitude) { return std::max(0.0, std::min(altitude / 15000.0 - 2.0 / 3.0, 8.0 / 3.0 - altitude / 15000.0)); }

/// Wavelength-dependent values used to populate one shader spectral band.
/// betaRayleighScale is relative to 550 nm. The remaining fields hold the ozone
/// cross section, solar irradiance scale, and CIE XYZ matching values.
struct SpectralBand {
    double betaRayleighScale, ozoneCrossSection, sunIrradianceScale;
    glm::dvec3 cie;
};

/// Samples wavelength-dependent source datasets at the requested wavelength.
/// @param wavelengthNm Wavelength to sample, in nanometres.
/// @return Rayleigh scale, ozone cross section, solar irradiance scale, and CIE XYZ values.
SpectralBand ComputeSpectralAtWavelength(double wavelengthNm);

/// Integrates the intensity component of a tabulated phase function for normalization.
/// @param phase Stokes phase samples on a uniform angular grid spanning [0, pi].
/// @return Solid-angle normalization under the renderer's phase convention.
/// @pre phase contains at least two angular samples.
inline double PhaseNormalization(const std::vector<glm::dvec4>& phase) {
    const int bins = static_cast<int>(phase.size());
    const double dTheta = std::numbers::pi / (bins - 1);

    double integral = 0.0;
    for (int i = 0; i + 1 < bins; ++i) { integral += phase[i].x * std::sin(i * dTheta) + phase[i + 1].x * std::sin((i + 1) * dTheta); }
    return 0.25 * integral * dTheta;
}
