#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>
#include <glm/glm.hpp>

inline constexpr float kPi = std::numbers::pi_v<float>;

struct AerosolConfig {
    float beta, scaleHeight, refractiveIndexReal, refractiveIndexImag, meanRadiusMicrometers, sigma;
    [[nodiscard]] friend bool operator==(const AerosolConfig&, const AerosolConfig&) = default;
};

struct SkySpectralConfig {
    float betaRayleigh550 = 13.5e-6f;
    float earthRadius = 6360e3f, atmosphereRadius = 6420e3f, scaleHeightRayleigh = 7994.0f, sunRadiance550 = 8317.742f;
    glm::vec3 sunDirection{0.35f, 0.01f, 0.25f};
    float sunRadius = 0.00465f, sunAa = 0.0005f;
    uint32_t secondarySamples = 1, viewSteps = 1, samples = 1, scatteringOrders = 3;

    float rayleighDepolarization = 0.0279f, ozoneDobsonUnits = 300.0f;
    uint32_t mieTableAngleBins = 181;

    std::array<AerosolConfig, 2> aerosols{
        {{.beta = 21e-6f, .scaleHeight = 1200.0f, .refractiveIndexReal = 1.33f, .refractiveIndexImag = 0.0f, .meanRadiusMicrometers = 0.2f, .sigma = 1.5f},
         {.beta = 4.0e-6f, .scaleHeight = 1500.0f, .refractiveIndexReal = 1.53f, .refractiveIndexImag = 0.008f, .meanRadiusMicrometers = 1.0f, .sigma = 2.0f}}};

    [[nodiscard]] friend bool operator==(const SkySpectralConfig&, const SkySpectralConfig&) = default;
};

struct RainbowConfig {
    uint32_t enabled = 1;
    float distance = 4000.0f, height = 0.0f;
    glm::vec3 radii{6000.0f, 6000.0f, 1500.0f};
    float edgeSoftness = 0.15f, scatteringCoefficient = 5.0e-4f, extinctionCoefficient = 5.0e-4f;
    float effectiveRadiusMicrometers = 500.0f, effectiveVariance = 0.08f;
    uint32_t angleBins = 4097, viewSteps = 24, includeSecondary = 1;
    uint32_t scatteringOrders = 2, multipleScatteringSamples = 1, multipleScatteringSteps = 8;

    [[nodiscard]] friend bool operator==(const RainbowConfig&, const RainbowConfig&) = default;
};

struct RenderConfig {
    uint32_t width = 960, height = 540, samplesPerPixel = 1;
};

struct CameraConfig {
    glm::vec3 initialPosition{0.0f, 2.0f, -10.0f};
    glm::vec3 initialLookAt{0.0f, 0.5f, 0.0f};
    float fovYDegrees = 40.0f, maxPitchDegrees = 89.0f;
};

struct InputConfig {
    float mouseSensitivity = 0.0035f, polarizerRotateSpeed = 1.5f;
};

struct SkyConfig {
    float exposure = 10.0f;
    SkySpectralConfig spectral{};
};

struct RuntimeConfig {
    RenderConfig render{};
    CameraConfig camera{};
    InputConfig input{};
    RainbowConfig rainbow{};
    SkyConfig sky{};
};

inline constexpr int kSpectralBandCount = 17;
inline constexpr double kSpectralLambdaMinNm = 380.0;
inline constexpr double kSpectralLambdaStepNm = 25.0;

std::vector<glm::vec4> ComputeMieScatteringTable(const SkySpectralConfig& sky, int aerosol, std::array<glm::dvec2, kSpectralBandCount>& crossSections);

std::vector<glm::vec4> ComputeRainbowScatteringTable(const RainbowConfig& rainbow, float sunRadius);

void AppendSamplingCdf(std::vector<glm::vec4>& table, int bins, size_t firstEntry);

inline constexpr int kTransmittanceAltitudeBins = 64;
inline constexpr int kTransmittanceMuBins = 256;

std::vector<glm::vec4> ComputeTransmittanceTable(const SkySpectralConfig& sky);

inline double OzoneProfile(double altitude) { return std::max(0.0, std::min((altitude / 15000.0) - (2.0 / 3.0), (8.0 / 3.0) - (altitude / 15000.0))); }

struct SpectralBand {
    double betaRayleighScale, ozoneCrossSection, sunIrradianceScale;
    glm::dvec3 cie;
};

SpectralBand ComputeSpectralBand(int band);

inline double PhaseNormalization(const std::vector<glm::dvec4>& phase) {
    const int bins = static_cast<int>(phase.size());
    const double dTheta = std::numbers::pi / (bins - 1);
    double integral = 0.0;
    for (int i = 0; i + 1 < bins; ++i) integral += (phase[i].x * std::sin(i * dTheta)) + (phase[i + 1].x * std::sin((i + 1) * dTheta));
    return 0.25 * integral * dTheta;
}
