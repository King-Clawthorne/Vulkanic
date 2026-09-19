#include "SkyTables.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <execution>
#include <mdspan>
#include <numbers>
#include <numeric>
#include <ranges>
#include <array>
#include <stdexcept>

namespace {
    using Complex = std::complex<double>;

    int MieTermCount(double x) {
        const int n = static_cast<int>(x + (4.0 * std::cbrt(std::max(x, 1e-8))) + 2.0);
        return n + 2;
    }

    void MieCoefficients(double x, Complex m, std::vector<Complex>& a, std::vector<Complex>& b) {
        const int nmax = MieTermCount(x);
        const Complex mx = m * x;

        const int nstart = nmax + 15;
        std::vector<Complex> D(static_cast<size_t>(nstart) + 1u, Complex(0.0, 0.0));
        for (int n = nstart; n > 0; --n) {
            const Complex nOverMx = Complex(static_cast<double>(n), 0.0) / mx;
            D[static_cast<size_t>(n) - 1u] = nOverMx - Complex(1.0, 0.0) / (D[static_cast<size_t>(n)] + nOverMx);
        }

        a.assign(static_cast<size_t>(nmax) + 1u, Complex(0.0, 0.0));
        b.assign(static_cast<size_t>(nmax) + 1u, Complex(0.0, 0.0));

        double psiPrev = std::cos(x);
        double psi = std::sin(x);
        double chiPrev = -std::sin(x);
        double chi = std::cos(x);

        for (int n : std::views::iota(1, nmax + 1)) {
            const auto dn = static_cast<double>(n);
            const double psiN = (((2.0 * dn) - 1.0) / x * psi) - psiPrev;
            const double chiN = (((2.0 * dn) - 1.0) / x * chi) - chiPrev;
            const Complex ksiN(psiN, -chiN);
            const Complex ksiPrev(psiPrev, -chiPrev);

            const Complex Dn = D[static_cast<size_t>(n)];
            const Complex ta = Dn / m + Complex(dn / x, 0.0);
            a[static_cast<size_t>(n)] = (ta * psiN - psiPrev) / (ta * ksiN - ksiPrev);
            const Complex tb = Dn * m + Complex(dn / x, 0.0);
            b[static_cast<size_t>(n)] = (tb * psiN - psiPrev) / (tb * ksiN - ksiPrev);

            psiPrev = psi;
            psi = psiN;
            chiPrev = chi;
            chi = chiN;
        }
    }
}

namespace {
    template <class Body>
    void ParallelFor(int count, Body body) {
        std::vector<int> indices(static_cast<size_t>(count));
        std::ranges::iota(indices, 0);
        std::for_each(std::execution::par, indices.begin(), indices.end(), body);
    }
}

std::vector<MieMatrixEntry> ComputeMieScatteringTable(const SkySpectralConfig& sky, std::array<MieCrossSection, kSpectralBandCount>& crossSections) {
    const int bins = std::max(2, static_cast<int>(sky.mieTableAngleBins));
    const Complex m(sky.aerosolRefractiveIndexReal, std::max(0.0, static_cast<double>(sky.aerosolRefractiveIndexImag)));

    std::vector<double> mu(static_cast<size_t>(bins));
    for (int i : std::views::iota(0, bins)) mu[i] = std::cos(std::numbers::pi * i / (bins - 1));

    const int radiusSamples = 48;
    const double lnSigma = std::log(std::max(1.0001, static_cast<double>(sky.aerosolSigma)));
    const double lnRg = std::log(std::max(1e-4, static_cast<double>(sky.aerosolMeanRadiusMicrometers)));
    const double lnMin = lnRg - (4.0 * lnSigma);
    const double lnMax = lnRg + (4.0 * lnSigma);
    const double dLn = (lnMax - lnMin) / static_cast<double>(radiusSamples - 1);

    std::vector<MieMatrixEntry> table(static_cast<size_t>(bins) * kSpectralBandCount);
    auto tableView = std::mdspan(table.data(), static_cast<size_t>(kSpectralBandCount), static_cast<size_t>(bins));

    ParallelFor(kSpectralBandCount, [&](int band) {
        const double lambdaUm = (kSpectralLambdaMinNm + (kSpectralLambdaStepNm * band)) * 1e-3;
        const double k = 2.0 * std::numbers::pi / lambdaUm;

        std::vector<double> p11(static_cast<size_t>(bins), 0.0);
        std::vector<double> p12(static_cast<size_t>(bins), 0.0);
        std::vector<double> p33(static_cast<size_t>(bins), 0.0);
        std::vector<double> p34(static_cast<size_t>(bins), 0.0);
        MieCrossSection cross{.extinction = 0.0, .scattering = 0.0};

        for (int rs : std::views::iota(0, radiusSamples)) {
            const double lnR = lnMin + (dLn * static_cast<double>(rs));
            const double z = (lnR - lnRg) / lnSigma;
            const double weight = std::exp(-0.5 * z * z) * dLn;
            if (weight < 1e-12) {
                continue;
            }

            const double r = std::exp(lnR);
            const double x = k * r;
            std::vector<Complex> a;
            std::vector<Complex> b;
            MieCoefficients(x, m, a, b);
            const int nmax = static_cast<int>(a.size()) - 1;
            for (int n : std::views::iota(1, nmax + 1)) {
                const double order = (2.0 * n) + 1.0;
                const double area = weight * 2.0 * std::numbers::pi / (k * k);
                cross.extinction += area * order * (a[static_cast<size_t>(n)] + b[static_cast<size_t>(n)]).real();
                cross.scattering += area * order * (std::norm(a[static_cast<size_t>(n)]) + std::norm(b[static_cast<size_t>(n)]));
            }

            for (int i : std::views::iota(0, bins)) {
                const double u = mu[static_cast<size_t>(i)];

                double piPrev = 0.0;
                double piCur = 1.0;
                Complex s1(0.0, 0.0);
                Complex s2(0.0, 0.0);

                for (int n : std::views::iota(1, nmax + 1)) {
                    const auto dn = static_cast<double>(n);
                    const double tauCur = (dn * u * piCur) - ((dn + 1.0) * piPrev);
                    const double fn = ((2.0 * dn) + 1.0) / (dn * (dn + 1.0));
                    s1 += fn * (a[static_cast<size_t>(n)] * piCur + b[static_cast<size_t>(n)] * tauCur);
                    s2 += fn * (a[static_cast<size_t>(n)] * tauCur + b[static_cast<size_t>(n)] * piCur);

                    const double piNext = ((((2.0 * dn) + 1.0) * u * piCur) - ((dn + 1.0) * piPrev)) / dn;
                    piPrev = piCur;
                    piCur = piNext;
                }

                const double i1 = std::norm(s1);
                const double i2 = std::norm(s2);
                const Complex cross = s2 * std::conj(s1);
                p11[static_cast<size_t>(i)] += weight * 0.5 * (i2 + i1);
                p12[static_cast<size_t>(i)] += weight * 0.5 * (i2 - i1);
                p33[static_cast<size_t>(i)] += weight * cross.real();
                p34[static_cast<size_t>(i)] += weight * cross.imag();
            }
        }

        crossSections[static_cast<size_t>(band)] = cross;
        const double norm = std::max(PhaseNormalization(p11), 1e-20);

        for (int i : std::views::iota(0, bins)) {
            MieMatrixEntry& entry = tableView[band, static_cast<size_t>(i)];
            entry.f11 = static_cast<float>(p11[static_cast<size_t>(i)] / norm);
            entry.f12 = static_cast<float>(p12[static_cast<size_t>(i)] / norm);
            entry.f33 = static_cast<float>(p33[static_cast<size_t>(i)] / norm);
            entry.f34 = static_cast<float>(p34[static_cast<size_t>(i)] / norm);
        }
    });

    return table;
}

namespace {

    double WaterIor(double wavelengthNm) {
        constexpr std::array<double, kSpectralBandCount> values = {
            1.34350,
            1.34055,
            1.33795,
            1.33570,
            1.33370,
            1.33225,
            1.33110,
            1.33020,
            1.32945,
            1.32885,
            1.32835,
            1.32795,
            1.32760,
        };
        const double position = std::clamp((wavelengthNm - kSpectralLambdaMinNm) / kSpectralLambdaStepNm,
                                           0.0, static_cast<double>(kSpectralBandCount - 1));
        const int lower = static_cast<int>(std::floor(position));
        const int upper = std::min(lower + 1, kSpectralBandCount - 1);
        return std::lerp(values[static_cast<size_t>(lower)], values[static_cast<size_t>(upper)],
                         position - lower);
    }

    struct FresnelPower {
        double reflectS;
        double reflectP;
        double transmitS;
        double transmitP;
    };

    FresnelPower AirToWaterFresnel(double incidence, double refraction, double n) {
        const double ci = std::cos(incidence);
        const double cr = std::cos(refraction);
        const double rs = (ci - (n * cr)) / (ci + (n * cr));
        const double rp = ((n * ci) - cr) / ((n * ci) + cr);
        const double reflectS = rs * rs;
        const double reflectP = rp * rp;
        return {.reflectS = reflectS, .reflectP = reflectP, .transmitS = 1.0 - reflectS, .transmitP = 1.0 - reflectP};
    }

    void Deposit(std::vector<double>& f11, std::vector<double>& f12,
                 std::vector<double>& f33, double theta, double weightS,
                 double weightP, double sigma, int bins) {
        theta = std::acos(std::clamp(std::cos(theta), -1.0, 1.0));
        const double scale = static_cast<double>(bins - 1) / std::numbers::pi;
        const double centre = theta * scale;
        const double sigmaBins = std::max(sigma * scale, 0.65);
        const int radius = std::max(2, static_cast<int>(std::ceil(4.0 * sigmaBins)));
        const int first = std::max(0, static_cast<int>(std::floor(centre)) - radius);
        const int last = std::min(bins - 1, static_cast<int>(std::floor(centre)) + radius);
        const double weight11 = 0.5 * (weightS + weightP);
        const double weight12 = 0.5 * (weightP - weightS);
        const double weight33 = std::sqrt(std::max(weightS * weightP, 0.0));
        for (int bin = first; bin <= last; ++bin) {
            const double x = (static_cast<double>(bin) - centre) / sigmaBins;
            const double kernel = std::exp(-0.5 * x * x);
            f11[static_cast<size_t>(bin)] += weight11 * kernel;
            f12[static_cast<size_t>(bin)] += weight12 * kernel;
            f33[static_cast<size_t>(bin)] += weight33 * kernel;
        }
    }

    double Airy(double x) {
        constexpr double lo = -22.0;
        constexpr double hi = 8.0;
        constexpr double h = 1.0e-3;
        static const std::vector<double> values = [] {
            const auto count = static_cast<size_t>(std::lround((hi - lo) / h)) + 1;
            std::vector<double> table(count);
            const double zeta = 2.0 / 3.0 * std::pow(hi, 1.5);
            const double scale = std::exp(-zeta) / (2.0 * std::sqrt(std::numbers::pi));
            double y = scale / std::pow(hi, 0.25) * (1.0 - (5.0 / (72.0 * zeta)) + (385.0 / (10368.0 * zeta * zeta)));
            double v = -scale * std::pow(hi, 0.25) * (1.0 + (7.0 / (72.0 * zeta)) - (455.0 / (10368.0 * zeta * zeta)));
            for (size_t i = count; i-- > 0;) {
                table[i] = y;
                const double x0 = lo + (static_cast<double>(i) * h);
                const double s = -h;
                const double k1y = v;
                const double k1v = x0 * y;
                const double k2y = v + (0.5 * s * k1v);
                const double k2v = (x0 + (0.5 * s)) * (y + (0.5 * s * k1y));
                const double k3y = v + (0.5 * s * k2v);
                const double k3v = (x0 + (0.5 * s)) * (y + (0.5 * s * k2y));
                const double k4y = v + (s * k3v);
                const double k4v = (x0 + s) * (y + (s * k3y));
                y += s / 6.0 * (k1y + (2.0 * k2y) + (2.0 * k3y) + k4y);
                v += s / 6.0 * (k1v + (2.0 * k2v) + (2.0 * k3v) + k4v);
            }
            return table;
        }();
        if (x >= hi) return 0.0;
        const double f = (std::max(x, lo) - lo) / h;
        const auto i = std::min(static_cast<size_t>(f), values.size() - 2);
        return std::lerp(values[i], values[i + 1], f - static_cast<double>(i));
    }

    double AiryBlend(double z) {
        const double t = std::clamp((20.0 - z) / 8.0, 0.0, 1.0);
        return t * t * (3.0 - (2.0 * t));
    }

    double Deviation(int k, double b, double n) {
        return (k * std::numbers::pi) + (2.0 * std::asin(b)) - (2.0 * (k + 1) * std::asin(b / n));
    }

    struct Bow {
        double b0, dMin, curvature;
    };

    Bow RainbowRay(int k, double n) {
        const double b0 = std::sqrt(1.0 - (((n * n) - 1.0) / (k * (k + 2.0))));
        constexpr double h = 1.0e-4;
        const double dMin = Deviation(k, b0, n);
        return {.b0 = b0, .dMin = dMin, .curvature = (Deviation(k, b0 + h, n) - (2.0 * dMin) + Deviation(k, b0 - h, n)) / (h * h)};
    }
}

std::vector<MieMatrixEntry> ComputeRainbowScatteringTable(const RainbowConfig& rainbow, float sunRadius) {
    const int bins = static_cast<int>(rainbow.angleBins);
    std::vector<MieMatrixEntry> table(static_cast<size_t>(kSpectralBandCount * bins));
    constexpr int subWavelengths = 5;
    constexpr int raySamples = 16384 / subWavelengths;
    constexpr int radiusSamples = 9;

    ParallelFor(kSpectralBandCount, [&](int band) {
        std::vector<double> f11(static_cast<size_t>(bins), 0.0);
        std::vector<double> f12(static_cast<size_t>(bins), 0.0);
        std::vector<double> f33(static_cast<size_t>(bins), 0.0);

        const double sigmaLn = std::sqrt(std::log1p(static_cast<double>(rainbow.effectiveVariance)));
        const double geometricRadiusUm = static_cast<double>(rainbow.effectiveRadiusMicrometers) / std::exp(2.5 * sigmaLn * sigmaLn);
        double radiusWeightSum = 0.0;
        std::array<double, radiusSamples> radiiUm{};
        std::array<double, radiusSamples> radiusWeights{};
        for (int radiusIndex = 0; radiusIndex < radiusSamples; ++radiusIndex) {
            const double z = sigmaLn > 1.0e-8
                                 ? -3.5 + (7.0 * (static_cast<double>(radiusIndex) + 0.5) / radiusSamples)
                                 : 0.0;
            const double radiusUm = geometricRadiusUm * std::exp(sigmaLn * z);
            const double numberWeight = sigmaLn > 1.0e-8 ? std::exp(-0.5 * z * z) : 1.0;
            const double weight = numberWeight * radiusUm * radiusUm;
            radiiUm[static_cast<size_t>(radiusIndex)] = radiusUm;
            radiusWeights[static_cast<size_t>(radiusIndex)] = weight;
            radiusWeightSum += weight;
        }

        const double solarSigma = 0.5 * static_cast<double>(sunRadius);
        const double binWidth = std::numbers::pi / (bins - 1);
        const double kernelSum = std::max(solarSigma / binWidth, 0.65) * std::sqrt(2.0 * std::numbers::pi);
        const int bows = rainbow.includeSecondary != 0 ? 2 : 1;
        std::vector<double> a11(static_cast<size_t>(bins), 0.0);
        std::vector<double> a12(static_cast<size_t>(bins), 0.0);
        std::vector<double> a33(static_cast<size_t>(bins), 0.0);
        for (int sub = 0; sub < subWavelengths; ++sub) {
            const double wavelengthNm = kSpectralLambdaMinNm + (kSpectralLambdaStepNm * (band + ((sub + 0.5) / subWavelengths) - 0.5));
            const double n = WaterIor(wavelengthNm);
            const std::array<Bow, 2> bow = {RainbowRay(1, n), RainbowRay(2, n)};
            for (int radiusIndex = 0; radiusIndex < radiusSamples; ++radiusIndex) {
                const double radiusM = radiiUm[static_cast<size_t>(radiusIndex)] * 1.0e-6;
                const double radiusWeight = radiusWeights[static_cast<size_t>(radiusIndex)] / radiusWeightSum;
                const double ka = 2.0 * std::numbers::pi * radiusM / (wavelengthNm * 1.0e-9);
                std::array<double, 2> zScale{};
                for (int k = 0; k < bows; ++k)
                    zScale[k] = std::pow(ka, 2.0 / 3.0) * std::cbrt(2.0 / bow[k].curvature);

                for (int sample = 0; sample < raySamples; ++sample) {
                    const double b = std::sqrt((static_cast<double>(sample) + 0.5) / raySamples);
                    const double incidence = std::asin(std::min(b, 1.0));
                    const FresnelPower fr = AirToWaterFresnel(incidence, std::asin(b / n), n);
                    double weightS = radiusWeight * fr.transmitS * fr.transmitS;
                    double weightP = radiusWeight * fr.transmitP * fr.transmitP;
                    for (int k = 0; k < bows; ++k) {
                        weightS *= fr.reflectS;
                        weightP *= fr.reflectP;
                        const double deviation = Deviation(k + 1, b, n);
                        const double fade = 1.0 - AiryBlend(zScale[k] * (deviation - bow[k].dMin));
                        if (fade > 0.0) Deposit(f11, f12, f33, deviation, weightS * fade, weightP * fade, solarSigma, bins);
                    }
                }

                for (int k = 0; k < bows; ++k) {
                    const double b0 = bow[k].b0;
                    const double i0 = std::asin(b0);
                    const FresnelPower fr = AirToWaterFresnel(i0, std::asin(b0 / n), n);
                    const double weightS = radiusWeight * fr.transmitS * fr.transmitS * std::pow(fr.reflectS, k + 1);
                    const double weightP = radiusWeight * fr.transmitP * fr.transmitP * std::pow(fr.reflectP, k + 1);
                    const double amplitude = raySamples * 8.0 * std::numbers::pi * b0 * std::sqrt(zScale[k]) / std::sqrt(2.0 * bow[k].curvature) * binWidth * kernelSum;
                    for (int bin = 0; bin < bins; ++bin) {
                        const double theta = bin * binWidth;
                        const double z = zScale[k] * ((k == 0 ? theta : (2.0 * std::numbers::pi) - theta) - bow[k].dMin);
                        if (z < -6.0 || z > 20.0) continue;
                        const double ai = Airy(-z);
                        const double profile = amplitude * ai * ai * AiryBlend(z);
                        a11[static_cast<size_t>(bin)] += 0.5 * (weightS + weightP) * profile;
                        a12[static_cast<size_t>(bin)] += 0.5 * (weightP - weightS) * profile;
                        a33[static_cast<size_t>(bin)] += std::sqrt(weightS * weightP) * profile;
                    }
                }
            }
        }

        const double sigmaBins = std::max(solarSigma / binWidth, 0.65);
        const int reach = static_cast<int>(std::ceil(4.0 * sigmaBins));
        for (int bin = 0; bin < bins; ++bin) {
            if (a11[static_cast<size_t>(bin)] == 0.0 && a33[static_cast<size_t>(bin)] == 0.0) continue;
            for (int offset = -reach; offset <= reach; ++offset) {
                const int target = std::clamp(bin + offset, 0, bins - 1);
                const double x = offset / sigmaBins;
                const double kernel = std::exp(-0.5 * x * x) / kernelSum;
                f11[static_cast<size_t>(target)] += a11[static_cast<size_t>(bin)] * kernel;
                f12[static_cast<size_t>(target)] += a12[static_cast<size_t>(bin)] * kernel;
                f33[static_cast<size_t>(target)] += a33[static_cast<size_t>(bin)] * kernel;
            }
        }

        const double normalization = PhaseNormalization(f11);
        for (int i = 0; i < bins; ++i) {
            const double normalizedF11 = f11[static_cast<size_t>(i)] / normalization;
            const double normalizedF12 = f12[static_cast<size_t>(i)] / normalization;
            const double normalizedF33 = f33[static_cast<size_t>(i)] / normalization;
            table[(static_cast<size_t>(band) * static_cast<size_t>(bins)) + static_cast<size_t>(i)] = {
                .f11 = static_cast<float>(normalizedF11),
                .f12 = static_cast<float>(normalizedF12),
                .f33 = static_cast<float>(normalizedF33),
                .f34 = 0.0f,
            };
        }
    });

    return table;
}

void AppendRainbowSamplingCdf(std::vector<MieMatrixEntry>& table, int bins) {
    std::vector<double> cdf(bins, 0.0);
    for (int i = 1; i < bins; ++i) {
        const double a = table[(6 * bins) + i - 1].f11 * std::sin(std::numbers::pi * (i - 1) / (bins - 1));
        const double b = table[(6 * bins) + i].f11 * std::sin(std::numbers::pi * i / (bins - 1));
        cdf[i] = cdf[i - 1] + std::max(0.0, a + b);
    }
    for (double value : cdf) table.push_back({.f11 = static_cast<float>(value / cdf.back()), .f12 = 0, .f33 = 0, .f34 = 0});
}

std::vector<std::array<float, 2>> ComputeTransmittanceTable(const SkySpectralConfig& sky) {
    constexpr int steps = 256;
    const double re = sky.earthRadius;
    const double ra = sky.atmosphereRadius;
    std::vector<std::array<float, 2>> table(static_cast<size_t>(kTransmittanceAltitudeBins) * kTransmittanceMuBins);
    ParallelFor(kTransmittanceAltitudeBins, [&](int a) {
        const double x = static_cast<double>(a) / (kTransmittanceAltitudeBins - 1);
        const double r = re + (x * x * (ra - re));
        for (int m = 0; m < kTransmittanceMuBins; ++m) {
            const double v = (2.0 * m / (kTransmittanceMuBins - 1)) - 1.0;
            const double mu = v * std::abs(v);
            const double b = r * mu;
            const double ds = (-b + std::sqrt(std::max((b * b) - (r * r) + (ra * ra), 0.0))) / steps;
            double rayleigh = 0.0;
            double mie = 0.0;
            for (int i = 0; i < steps; ++i) {
                const double t = (i + 0.5) * ds;
                const double altitude = std::max(std::sqrt((r * r) + (t * t) + (2.0 * r * mu * t)) - re, 0.0);
                rayleigh += std::exp(-altitude / sky.scaleHeightRayleigh) * ds;
                mie += std::exp(-altitude / sky.scaleHeightMie) * ds;
            }
            table[(static_cast<size_t>(a) * kTransmittanceMuBins) + static_cast<size_t>(m)] = {static_cast<float>(rayleigh), static_cast<float>(mie)};
        }
    });
    return table;
}

namespace {

    double RayleighShape(double wavelengthNm) {
        const double sigma2 = 1.0e6 / (wavelengthNm * wavelengthNm);
        const double n = 1.0 + (1.0e-8 * (8060.51 + (2480990.0 / (132.274 - sigma2)) + (17455.7 / (39.32957 - sigma2))));
        const double n2 = (n * n) - 1.0;
        return n2 * n2 / std::pow(wavelengthNm, 4.0);
    }
}

SpectralBand ComputeSpectralBand(int band) {
    const double centre = kSpectralLambdaMinNm + (kSpectralLambdaStepNm * band);
    SpectralBand result{.betaRayleighScale = 0.0, .limbDarkening = -0.023 + (0.292e3 / centre)};
    for (int offset = -10; offset <= 10; offset += 5) {
        const double wavelength = centre + offset;
        result.betaRayleighScale += RayleighShape(wavelength) / RayleighShape(550.0) / 5.0;
    }
    return result;
}
