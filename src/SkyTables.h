#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

inline constexpr float kPi = std::numbers::pi_v<float>;

struct Vec3 {
    float x, y, z;
    friend bool operator==(const Vec3&, const Vec3&) = default;
};

struct SkySpectralConfig {
    float betaRayleigh550 = 13.5e-6f;
    float betaMie = 21e-6f;
    float earthRadius = 6360e3f;
    float atmosphereRadius = 6420e3f;
    float scaleHeightRayleigh = 7994.0f;
    float scaleHeightMie = 1200.0f;
    float sunRadiance550 = 8317.742f;
    std::array<float, 3> sunDirection{0.35f, 0.01f, 0.25f};
    float sunRadius = 0.00465f;
    float sunAa = 0.0005f;
    uint32_t secondarySamples = 1;
    uint32_t viewSteps = 1;
    uint32_t samples = 1;
    uint32_t scatteringOrders = 3;

    float rayleighDepolarization = 0.0279f;
    float aerosolRefractiveIndexReal = 1.33f;
    float aerosolRefractiveIndexImag = 0.0f;
    float aerosolMeanRadiusMicrometers = 0.2f;
    float aerosolSigma = 1.5f;
    uint32_t mieTableAngleBins = 181;
    float ozoneDobsonUnits = 300.0f;

    float betaMie2 = 4.0e-6f;
    float scaleHeightMie2 = 1500.0f;
    float aerosol2RefractiveIndexReal = 1.53f;
    float aerosol2RefractiveIndexImag = 0.008f;
    float aerosol2MeanRadiusMicrometers = 1.0f;
    float aerosol2Sigma = 2.0f;

    [[nodiscard]] friend bool operator==(const SkySpectralConfig&, const SkySpectralConfig&) = default;
};

struct RainbowConfig {
    uint32_t enabled = 1;
    float distance = 4000.0f;
    float height = 0.0f;
    Vec3 radii{.x = 6000.0f, .y = 6000.0f, .z = 1500.0f};
    float edgeSoftness = 0.15f;
    float scatteringCoefficient = 5.0e-4f;
    float extinctionCoefficient = 5.0e-4f;
    float effectiveRadiusMicrometers = 500.0f;
    float effectiveVariance = 0.08f;
    uint32_t angleBins = 4097;
    uint32_t viewSteps = 24;
    uint32_t includeSecondary = 1;
    uint32_t scatteringOrders = 2;
    uint32_t multipleScatteringSamples = 1;
    uint32_t multipleScatteringSteps = 8;

    [[nodiscard]] friend bool operator==(const RainbowConfig&, const RainbowConfig&) = default;
};

struct RenderConfig {
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t samplesPerPixel = 1;
};

struct CameraConfig {
    Vec3 initialPosition{.x = 0.0f, .y = 2.0f, .z = -10.0f};
    Vec3 initialLookAt{.x = 0.0f, .y = 0.5f, .z = 0.0f};
    float fovYDegrees = 40.0f;
    float maxPitchDegrees = 89.0f;
};

struct InputConfig {
    float mouseSensitivity = 0.0035f;
    float polarizerRotateSpeed = 1.5f;
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

inline constexpr int kSpectralBandCount = 13;
inline constexpr double kSpectralLambdaMinNm = 400.0;
inline constexpr double kSpectralLambdaStepNm = 25.0;

struct MieMatrixEntry {
    float f11;
    float f12;
    float f33;
    float f34;
};

struct MieCrossSection {
    double extinction;
    double scattering;
};

std::vector<MieMatrixEntry> ComputeMieScatteringTable(const SkySpectralConfig& sky, int aerosol, std::array<MieCrossSection, kSpectralBandCount>& crossSections);

std::vector<MieMatrixEntry> ComputeRainbowScatteringTable(const RainbowConfig& rainbow, float sunRadius);

void AppendRainbowSamplingCdf(std::vector<MieMatrixEntry>& table, int bins);

inline constexpr int kTransmittanceAltitudeBins = 64;
inline constexpr int kTransmittanceMuBins = 256;

std::vector<std::array<float, 4>> ComputeTransmittanceTable(const SkySpectralConfig& sky);

inline double OzoneProfile(double altitude) { return std::max(0.0, std::min((altitude / 15000.0) - (2.0 / 3.0), (8.0 / 3.0) - (altitude / 15000.0))); }

struct SpectralBand {
    double betaRayleighScale;
    double limbDarkening;
    double ozoneCrossSection;
    double sunIrradianceScale;
    std::array<double, 3> cie;
};

SpectralBand ComputeSpectralBand(int band);

inline double PhaseNormalization(const std::vector<double>& f11) {
    const int bins = static_cast<int>(f11.size());
    const double dTheta = std::numbers::pi / (bins - 1);
    double integral = 0.0;
    for (int i = 0; i + 1 < bins; ++i) integral += 0.5 * ((f11[i] * std::sin(i * dTheta)) + (f11[i + 1] * std::sin((i + 1) * dTheta))) * dTheta;
    return 0.5 * integral;
}
