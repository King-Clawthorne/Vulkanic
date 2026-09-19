#include "SkyTables.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <mdspan>
#include <numbers>
#include <ranges>
#include <array>
#include <stdexcept>

namespace {
    using Complex = std::complex<double>;

    int MieTermCount(double x) {
        const int n = static_cast<int>(x + 4.0 * std::cbrt(std::max(x, 1e-8)) + 2.0);
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
            const double dn = static_cast<double>(n);
            const double psiN = (2.0 * dn - 1.0) / x * psi - psiPrev;
            const double chiN = (2.0 * dn - 1.0) / x * chi - chiPrev;
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

std::vector<MieMatrixEntry> ComputeMieScatteringTable(const SkySpectralConfig& sky) {
    const int bins = std::max(2, int(sky.mieTableAngleBins));
    const Complex m(sky.aerosolRefractiveIndexReal, std::max(0.0, double(sky.aerosolRefractiveIndexImag)));

    std::vector<double> mu(static_cast<size_t>(bins));
    for (int i : std::views::iota(0, bins)) mu[i] = std::cos(std::numbers::pi * i / (bins - 1));

    const int radiusSamples = 48;
    const double lnSigma = std::log(std::max(1.0001, double(sky.aerosolSigma)));
    const double lnRg = std::log(std::max(1e-4, double(sky.aerosolMeanRadiusMicrometers)));
    const double lnMin = lnRg - 4.0 * lnSigma;
    const double lnMax = lnRg + 4.0 * lnSigma;
    const double dLn = (lnMax - lnMin) / static_cast<double>(radiusSamples - 1);

    std::vector<MieMatrixEntry> table(static_cast<size_t>(bins) * kSpectralBandCount);
    auto tableView = std::mdspan(table.data(), static_cast<size_t>(kSpectralBandCount), static_cast<size_t>(bins));

    for (int band : std::views::iota(0, kSpectralBandCount)) {
        const double lambdaUm = (kSpectralLambdaMinNm + kSpectralLambdaStepNm * band) * 1e-3;
        const double k = 2.0 * std::numbers::pi / lambdaUm;

        std::vector<double> p11(static_cast<size_t>(bins), 0.0);
        std::vector<double> p12(static_cast<size_t>(bins), 0.0);
        std::vector<double> p33(static_cast<size_t>(bins), 0.0);
        std::vector<double> p34(static_cast<size_t>(bins), 0.0);

        for (int rs : std::views::iota(0, radiusSamples)) {
            const double lnR = lnMin + dLn * static_cast<double>(rs);
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

            for (int i : std::views::iota(0, bins)) {
                const double u = mu[static_cast<size_t>(i)];

                double piPrev = 0.0;
                double piCur = 1.0;
                Complex s1(0.0, 0.0);
                Complex s2(0.0, 0.0);

                for (int n : std::views::iota(1, nmax + 1)) {
                    const double dn = static_cast<double>(n);
                    const double tauCur = dn * u * piCur - (dn + 1.0) * piPrev;
                    const double fn = (2.0 * dn + 1.0) / (dn * (dn + 1.0));
                    s1 += fn * (a[static_cast<size_t>(n)] * piCur + b[static_cast<size_t>(n)] * tauCur);
                    s2 += fn * (a[static_cast<size_t>(n)] * tauCur + b[static_cast<size_t>(n)] * piCur);

                    const double piNext = ((2.0 * dn + 1.0) * u * piCur - (dn + 1.0) * piPrev) / dn;
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

        const double norm = std::max(PhaseNormalization(p11), 1e-20);

        for (int i : std::views::iota(0, bins)) {
            MieMatrixEntry& entry = tableView[band, static_cast<size_t>(i)];
            entry.f11 = static_cast<float>(p11[static_cast<size_t>(i)] / norm);
            entry.f12 = static_cast<float>(p12[static_cast<size_t>(i)] / norm);
            entry.f33 = static_cast<float>(p33[static_cast<size_t>(i)] / norm);
            entry.f34 = static_cast<float>(p34[static_cast<size_t>(i)] / norm);
        }
    }

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
        const double rs = (ci - n * cr) / (ci + n * cr);
        const double rp = (n * ci - cr) / (n * ci + cr);
        const double reflectS = rs * rs;
        const double reflectP = rp * rp;
        return {reflectS, reflectP, 1.0 - reflectS, 1.0 - reflectP};
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
}

std::vector<MieMatrixEntry> ComputeRainbowScatteringTable(const RainbowConfig& rainbow, float sunRadius) {
    const int bins = int(rainbow.angleBins);
    std::vector<MieMatrixEntry> table(static_cast<size_t>(kSpectralBandCount * bins));
    constexpr int raySamples = 16384;
    constexpr int radiusSamples = 9;

    for (int band = 0; band < kSpectralBandCount; ++band) {
        const double wavelengthNm = kSpectralLambdaMinNm + kSpectralLambdaStepNm * band;
        const double n = WaterIor(wavelengthNm);
        std::vector<double> f11(static_cast<size_t>(bins), 0.0);
        std::vector<double> f12(static_cast<size_t>(bins), 0.0);
        std::vector<double> f33(static_cast<size_t>(bins), 0.0);

        const double sigmaLn = std::sqrt(std::log1p(double(rainbow.effectiveVariance)));
        const double geometricRadiusUm = double(rainbow.effectiveRadiusMicrometers) / std::exp(2.5 * sigmaLn * sigmaLn);
        double radiusWeightSum = 0.0;
        std::array<double, radiusSamples> radiiUm{};
        std::array<double, radiusSamples> radiusWeights{};
        for (int radiusIndex = 0; radiusIndex < radiusSamples; ++radiusIndex) {
            const double z = sigmaLn > 1.0e-8
                                 ? -3.5 + 7.0 * (static_cast<double>(radiusIndex) + 0.5) / radiusSamples
                                 : 0.0;
            const double radiusUm = geometricRadiusUm * std::exp(sigmaLn * z);
            const double numberWeight = sigmaLn > 1.0e-8 ? std::exp(-0.5 * z * z) : 1.0;
            const double weight = numberWeight * radiusUm * radiusUm;
            radiiUm[static_cast<size_t>(radiusIndex)] = radiusUm;
            radiusWeights[static_cast<size_t>(radiusIndex)] = weight;
            radiusWeightSum += weight;
        }

        for (int radiusIndex = 0; radiusIndex < radiusSamples; ++radiusIndex) {
            const double radiusM = radiiUm[static_cast<size_t>(radiusIndex)] * 1.0e-6;
            const double radiusWeight = radiusWeights[static_cast<size_t>(radiusIndex)] / radiusWeightSum;
            const double airyWidth = 0.55 * std::pow(wavelengthNm * 1.0e-9 / radiusM, 2.0 / 3.0);
            const double solarSigma = 0.5 * double(sunRadius);
            const double blur = std::sqrt(solarSigma * solarSigma + airyWidth * airyWidth);

            for (int sample = 0; sample < raySamples; ++sample) {
                const double b = std::sqrt((static_cast<double>(sample) + 0.5) / raySamples);
                const double incidence = std::asin(std::min(b, 1.0));
                const double refraction = std::asin(b / n);
                const FresnelPower fr = AirToWaterFresnel(incidence, refraction, n);
                const double primary = std::numbers::pi + 2.0 * incidence - 4.0 * refraction;
                const double primaryS = radiusWeight * fr.transmitS * fr.transmitS * fr.reflectS;
                const double primaryP = radiusWeight * fr.transmitP * fr.transmitP * fr.reflectP;
                Deposit(f11, f12, f33, primary, primaryS, primaryP, blur, bins);

                if ((rainbow.includeSecondary != 0)) {
                    const double secondary = 2.0 * std::numbers::pi + 2.0 * incidence - 6.0 * refraction;
                    const double secondaryS = primaryS * fr.reflectS;
                    const double secondaryP = primaryP * fr.reflectP;
                    Deposit(f11, f12, f33, secondary, secondaryS, secondaryP, blur, bins);
                }
            }
        }

        const double normalization = PhaseNormalization(f11);
        for (int i = 0; i < bins; ++i) {
            const double normalizedF11 = f11[static_cast<size_t>(i)] / normalization;
            const double normalizedF12 = f12[static_cast<size_t>(i)] / normalization;
            const double normalizedF33 = f33[static_cast<size_t>(i)] / normalization;
            table[static_cast<size_t>(band * bins + i)] = {
                static_cast<float>(normalizedF11),
                static_cast<float>(normalizedF12),
                static_cast<float>(normalizedF33),
                0.0f,
            };
        }
    }

    return table;
}

void AppendRainbowSamplingCdf(std::vector<MieMatrixEntry>& table, int bins) {
    std::vector<double> cdf(bins, 0.0);
    for (int i = 1; i < bins; ++i) {
        const double a = table[6 * bins + i - 1].f11 * std::sin(std::numbers::pi * (i - 1) / (bins - 1));
        const double b = table[6 * bins + i].f11 * std::sin(std::numbers::pi * i / (bins - 1));
        cdf[i] = cdf[i - 1] + std::max(0.0, a + b);
    }
    for (double value : cdf) table.push_back({static_cast<float>(value / cdf.back()), 0, 0, 0});
}
