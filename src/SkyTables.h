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
    float sunTemperatureKelvin = 5778.0f;
    float sunRadiance550 = 8317.742f;
    std::array<float, 3> sunDirection{0.35f, 0.01f, 0.25f};
    float sunRadius = 0.00465f;
    float sunAa = 0.0005f;
    uint32_t secondarySamples = 1;
    uint32_t viewSteps = 1;
    uint32_t samples = 1;
    uint32_t scatteringOrders = 1;

    float rayleighDepolarization = 0.0279f;
    float aerosolRefractiveIndexReal = 1.33f;
    float aerosolRefractiveIndexImag = 0.0f;
    float aerosolMeanRadiusMicrometers = 0.2f;
    float aerosolSigma = 1.5f;
    uint32_t mieTableAngleBins = 181;

    [[nodiscard]] friend bool operator==(const SkySpectralConfig&, const SkySpectralConfig&) = default;
};

struct RainbowConfig {
    uint32_t enabled = 0;
    Vec3 center{0.0f, 1200.0f, 0.0f};
    Vec3 radii{5000.0f, 1800.0f, 5000.0f};
    float edgeSoftness = 0.15f;
    float scatteringCoefficient = 1.2e-4f;
    float extinctionCoefficient = 2.0e-4f;
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
    Vec3 initialPosition{0.0f, 2.0f, -10.0f};
    Vec3 initialLookAt{0.0f, 0.5f, 0.0f};
    float fovYDegrees = 40.0f;
    float maxPitchDegrees = 89.0f;
};

struct InputConfig {
    float mouseSensitivity = 0.0035f;
    float polarizerRotateSpeed = 1.5f;
};

struct SkyConfig {
    float exposure = 1.0f;
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

std::vector<MieMatrixEntry> ComputeMieScatteringTable(const SkySpectralConfig& sky);

std::vector<MieMatrixEntry> ComputeRainbowScatteringTable(const RainbowConfig& rainbow, float sunRadius);

void AppendRainbowSamplingCdf(std::vector<MieMatrixEntry>& table, int bins);

inline double PhaseNormalization(const std::vector<double>& f11) {
    const int bins = int(f11.size());
    const double dTheta = std::numbers::pi / (bins - 1);
    double integral = 0.0;
    for (int i = 0; i + 1 < bins; ++i) integral += 0.5 * (f11[i] * std::sin(i * dTheta) + f11[i + 1] * std::sin((i + 1) * dTheta)) * dTheta;
    return 0.5 * integral;
}
