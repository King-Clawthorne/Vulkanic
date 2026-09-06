#include "RainbowScattering.h"
#include "DropOptics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace
{
constexpr double kPi = std::numbers::pi_v<double>;

// Visible liquid-water dispersion sampled at the renderer's thirteen bands.
// Values follow the smooth room-temperature Hale-Querry visible trend; linear
// interpolation keeps this routine usable if the spectral grid is refined.
double WaterIor(double wavelengthNm)
{
    constexpr std::array<double, kSpectralBandCount> values = {
        1.34350, 1.34055, 1.33795, 1.33570, 1.33370, 1.33225, 1.33110,
        1.33020, 1.32945, 1.32885, 1.32835, 1.32795, 1.32760,
    };
    const double position = std::clamp((wavelengthNm - kSpectralLambdaMinNm) / kSpectralLambdaStepNm, 0.0,
                                       static_cast<double>(kSpectralBandCount - 1));
    const int lower = static_cast<int>(std::floor(position));
    const int upper = std::min(lower + 1, kSpectralBandCount - 1);
    return std::lerp(values[static_cast<size_t>(lower)], values[static_cast<size_t>(upper)], position - lower);
}

using Histogram = std::vector<std::array<double, 6>>;

void Deposit(Histogram &histogram, const drop_optics::Beam &beam)
{
    const auto matrix = drop_optics::Mueller(beam);
    if (matrix[0] <= 1.0e-16)
        return;
    const double bin = std::acos(std::clamp(beam.direction.z, -1.0, 1.0)) * (histogram.size() - 1) / kPi;
    const size_t first = std::min(static_cast<size_t>(bin), histogram.size() - 1);
    const size_t second = std::min(first + 1, histogram.size() - 1);
    const double fraction = bin - first;
    for (int c = 0; c < 6; ++c)
    {
        histogram[first][c] += matrix[c] * (1.0 - fraction);
        histogram[second][c] += matrix[c] * fraction;
    }
}
} // namespace

std::vector<RainbowMatrixEntry> ComputeRainbowScatteringTable(const RainbowScatteringParams &params)
{
    if (params.angleBins < 16 || !std::isfinite(params.effectiveRadiusMicrometers) ||
        params.effectiveRadiusMicrometers <= 0 || !std::isfinite(params.effectiveVariance) ||
        params.effectiveVariance < 0 || params.raySamples < 256)
        throw std::runtime_error("Invalid rainbow scattering-table parameters.");
    const int bins = params.angleBins;
    std::vector<RainbowMatrixEntry> table(static_cast<size_t>(kSpectralBandCount) * bins);
    // Geometry depends on wavelength, not radius. Trace once per
    // band, then convolve the angular-energy histogram with the size mixture.
    // This avoids repeating every ray for nine radii and Gaussian bin taps.
    constexpr int radiusSamples = 9;
    std::array<double, radiusSamples> radii{}, sizeWeights{};
    const double sigmaLn = std::sqrt(std::log1p(params.effectiveVariance));
    const double geometricRadius = params.effectiveRadiusMicrometers / std::exp(2.5 * sigmaLn * sigmaLn);
    double sizeSum = 0;
    for (int r = 0; r < radiusSamples; ++r)
    {
        const double z = sigmaLn > 1e-8 ? -3.5 + 7.0 * (r + 0.5) / radiusSamples : 0.0;
        radii[r] = geometricRadius * std::exp(sigmaLn * z) * 1e-6;
        sizeWeights[r] = std::exp(-0.5 * z * z) * radii[r] * radii[r];
        sizeSum += sizeWeights[r];
    }
    for (double &w : sizeWeights)
        w /= sizeSum;
    using namespace drop_optics;
    const int rays = std::max(16384, params.raySamples / 4);
    for (int band = 0; band < kSpectralBandCount; ++band)
    {
        Histogram raw(bins), filtered(bins);
        const double wavelength = kSpectralLambdaMinNm + kSpectralLambdaStepNm * band;
        const double n = WaterIor(wavelength);
        for (int i = 0; i < rays; ++i)
        {
            const double impact = std::sqrt((i + 0.5) / rays);
            const Sphere drop;
            Vector position{impact, 0, -2};
            Beam incoming;
            const double entry = drop.EntryDistance(position, incoming.direction);
            if (entry < 0)
                continue;
            position = position + incoming.direction * entry;
            const Vector normal = Unit(drop.Gradient(position));
            Deposit(raw, Interface(incoming, normal, 1.0, n, true)); // external reflection
            Beam internal = Interface(incoming, normal, 1.0, n, false);
            const int lastOrder = params.includeSecondary ? 2 : 1;
            for (int order = 0; order <= lastOrder; ++order)
            {
                position = position + internal.direction * 1e-8;
                const double distance = drop.ExitDistance(position, internal.direction);
                if (distance <= 0)
                    break;
                position = position + internal.direction * distance;
                const Vector exitNormal = Unit(drop.Gradient(position));
                Deposit(raw, Interface(internal, exitNormal, n, 1.0, false));
                internal = Interface(internal, exitNormal, n, 1.0, true);
            }
        }
        // Positive, normalized convolution of angular ENERGY (not phase
        // density) preserves integral power, including endpoint bins.
        const double binScale = (bins - 1) / kPi;
        std::vector<double> kernel;
        for (int r = 0; r < radiusSamples; ++r)
        {
            const double airy = 0.55 * std::pow(wavelength * 1e-9 / radii[r], 2.0 / 3.0);
            const double sigma = std::max(airy * binScale, 0.65);
            const int radius = static_cast<int>(std::ceil(4 * sigma));
            if (kernel.size() < static_cast<size_t>(radius + 1))
                kernel.resize(radius + 1);
            double norm = 0;
            for (int k = -radius; k <= radius; ++k)
                norm += std::exp(-0.5 * k * k / (sigma * sigma));
            for (int k = 0; k <= radius; ++k)
                kernel[k] += sizeWeights[r] * std::exp(-0.5 * k * k / (sigma * sigma)) / norm;
        }
        for (int i = 0; i < bins; ++i)
        {
            const int first = std::max(0, i - static_cast<int>(kernel.size()) + 1);
            const int last = std::min(bins - 1, i + static_cast<int>(kernel.size()) - 1);
            double norm = 0;
            for (int j = first; j <= last; ++j)
                norm += kernel[std::abs(j - i)];
            for (int j = first; j <= last; ++j)
            {
                const double weight = kernel[std::abs(j - i)] / norm;
                for (int c = 0; c < 6; ++c)
                    filtered[j][c] += raw[i][c] * weight;
            }
        }
        // Convert bin energy to phase density, then normalize with the same
        // trapezoidal solid-angle convention used by the GPU interpolant.
        for (int i = 0; i < bins; ++i)
        {
            const double lo = kPi * std::max(0.0, i - 0.5) / (bins - 1);
            const double hi = kPi * std::min(double(bins - 1), i + 0.5) / (bins - 1);
            const double solidAngle = 2 * kPi * (std::cos(lo) - std::cos(hi));
            for (double &c : filtered[i])
                c /= solidAngle;
        }
        double integral = 0;
        for (int i = 0; i < bins - 1; ++i)
            integral += kPi * kPi / (bins - 1) *
                        (filtered[i][0] * std::sin(kPi * i / (bins - 1)) +
                         filtered[i + 1][0] * std::sin(kPi * (i + 1) / (bins - 1)));
        if (!std::isfinite(integral) || integral <= 0)
            throw std::runtime_error("Rainbow phase table has invalid energy normalization.");
        const double scale = 4 * kPi / integral;
        for (int i = 0; i < bins; ++i)
        {
            auto &m = filtered[i];
            for (double &c : m)
                c *= scale;
            for (double c : m)
                if (!std::isfinite(c))
                    throw std::runtime_error("Non-finite droplet Mueller entry.");
            if (m[0] < 0 || std::abs(m[1]) > m[0] * 1.000001)
                throw std::runtime_error("Non-physical droplet Mueller entry.");
            table[static_cast<size_t>(band) * bins + i] = {float(m[0]), float(m[1]), float(m[2]),
                                                           float(m[3]), float(m[4]), float(m[5])};
        }
        double cumulative = 0.0;
        for (int i = 1; i < bins; ++i)
        {
            cumulative += kPi / (4.0 * (bins - 1)) *
                (filtered[i - 1][0] * std::sin(kPi * (i - 1) / (bins - 1)) +
                 filtered[i][0] * std::sin(kPi * i / (bins - 1)));
            table[static_cast<size_t>(band) * bins + i].cdf = static_cast<float>(cumulative);
        }
        table[static_cast<size_t>(band + 1) * bins - 1].cdf = 1.0f;
    }
    return table;
}
