/// @file Baker.cpp
/// @brief Builds the spectral lookup tables consumed by the path tracer.
///
/// The implementation computes polarized Mie phase functions, angular
/// sampling distributions, atmospheric density columns, and spectral values
/// interpolated from the tables in DataTables.hpp.
#include "DataTables.hpp"
#include "Settings.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <execution>
#include <numbers>
#include <ranges>

namespace {
	/// Abscissae for 8-point Gauss-Legendre integration.
	constexpr std::array<double, 8> kGaussNodes = { -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363 };
	/// Weights paired with kGaussNodes for 8-point Gauss-Legendre integration.
	constexpr std::array<double, 8> kGaussWeights = { 0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620, 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763 };

	/// Computes the spherical-particle Mie coefficients for one size parameter.
	/// @param x Particle size parameter.
	/// @param m Complex relative refractive index.
	/// @param a Output electric-scattering coefficients, indexed by order.
	/// @param b Output magnetic-scattering coefficients, indexed by order.
	/// @param D Output logarithmic-derivative recurrence used to form the coefficients.
	void MieCoefficients(double x, std::complex<double> m, std::vector<std::complex<double>>& a, std::vector<std::complex<double>>& b, std::vector<std::complex<double>>& D) {
		const int nmax = static_cast<int>(x + 4.0 * std::cbrt(std::max(x, 1e-8)) + 4.0);
		const std::complex<double> mx = m * x;
		const int nstart = nmax + 15;
		D.resize(static_cast<size_t>(nstart) + 1u);
		D[static_cast<size_t>(nstart)] = { 0.0, 0.0 };

		for (int n = nstart; n > 0; --n) {
			const std::complex<double> nOverMx = std::complex<double>(static_cast<double>(n), 0.0) / mx;
			D[static_cast<size_t>(n) - 1u] = nOverMx - std::complex<double>(1.0, 0.0) / (D[static_cast<size_t>(n)] + nOverMx);
		}

		a.resize(static_cast<size_t>(nmax) + 1u);
		b.resize(static_cast<size_t>(nmax) + 1u);
		a[0] = b[0] = { 0.0, 0.0 };
		double psiPrev = std::cos(x), psi = std::sin(x);
		double chiPrev = -std::sin(x), chi = std::cos(x);

		for (int n : std::views::iota(1, nmax + 1)) {
			const auto dn = static_cast<double>(n);
			const double psiN = (2.0 * dn - 1.0) / x * psi - psiPrev;
			const double chiN = (2.0 * dn - 1.0) / x * chi - chiPrev;
			const std::complex<double> ksiN(psiN, -chiN);
			const std::complex<double> ksiPrev(psiPrev, -chiPrev);
			const std::complex<double> Dn = D[static_cast<size_t>(n)];
			const std::complex<double> ta = Dn / m + std::complex<double>(dn / x, 0.0);
			a[static_cast<size_t>(n)] = (ta * psiN - psiPrev) / (ta * ksiN - ksiPrev);
			const std::complex<double> tb = Dn * m + std::complex<double>(dn / x, 0.0);
			b[static_cast<size_t>(n)] = (tb * psiN - psiPrev) / (tb * ksiN - ksiPrev);

			psiPrev = psi;
			psi = psiN;
			chiPrev = chi;
			chi = chiN;
		}
	}
}

namespace {
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

std::vector<glm::vec4> ComputeMieScatteringTable(const SkyConfig& sky, int aerosol, std::array<glm::dvec2, kSpectralBandCount>& crossSections) {
	const int bins = std::max(2, static_cast<int>(sky.mieTableAngleBins));
	const AerosolConfig& species = sky.aerosols[static_cast<size_t>(aerosol)];
	const std::complex<double> m(species.refractiveIndexReal, std::max(0.0, static_cast<double>(species.refractiveIndexImag)));

	std::vector<double> mu(static_cast<size_t>(bins));
	for (int i : std::views::iota(0, bins)) mu[i] = std::cos(std::numbers::pi * i / (bins - 1));

	constexpr int radiusPanels = 32;
	constexpr int radiusSamples = radiusPanels * 8;
	const double lnSigma = std::log(std::max(1.0001, static_cast<double>(species.sigma)));
	const double lnRg = std::log(std::max(1e-4, static_cast<double>(species.meanRadiusMicrometers)));
	const double halfPanel = 4.0 * lnSigma / radiusPanels;
	std::vector<glm::vec4> table(static_cast<size_t>(bins) * kSpectralBandCount);

	ParallelFor(kSpectralBandCount, [&](int band) {
		const double lambdaUm = (kSpectralLambdaMinNm + kSpectralLambdaStepNm * band) * 1e-3;
		const double k = 2.0 * std::numbers::pi / lambdaUm;
		std::vector<glm::dvec4> phase(static_cast<size_t>(bins));
		glm::dvec2 cross { 0.0, 0.0 };
		std::vector<std::complex<double>> a, b, D;

		for (int rs : std::views::iota(0, radiusSamples)) {
			const int panel = rs / 8;
			const size_t node = static_cast<size_t>(rs % 8);
			const double lnR = lnRg - 4.0 * lnSigma + (2 * panel + 1 + kGaussNodes[node]) * halfPanel;
			const double z = (lnR - lnRg) / lnSigma;
			const double weight = std::exp(-0.5 * z * z) * halfPanel * kGaussWeights[node];
			const double r = std::exp(lnR);
			const double x = k * r;
			MieCoefficients(x, m, a, b, D);
			const int nmax = static_cast<int>(a.size()) - 1;

			for (int n : std::views::iota(1, nmax + 1)) {
				const double order = 2.0 * n + 1.0;
				const double area = weight * 2.0 * std::numbers::pi / (k * k);
				cross.x += area * order * (a[static_cast<size_t>(n)] + b[static_cast<size_t>(n)]).real();
				cross.y += area * order * (std::norm(a[static_cast<size_t>(n)]) + std::norm(b[static_cast<size_t>(n)]));
			}

			for (int i : std::views::iota(0, bins)) {
				const double u = mu[static_cast<size_t>(i)];
				double piPrev = 0.0, piCur = 1.0;
				std::complex<double> s1(0.0, 0.0), s2(0.0, 0.0);
				for (int n : std::views::iota(1, nmax + 1)) {
					const auto dn = static_cast<double>(n);
					const double tauCur = dn * u * piCur - (dn + 1.0) * piPrev;
					const double fn = (2.0 * dn + 1.0) / (dn * (dn + 1.0));
					s1 += fn * (a[static_cast<size_t>(n)] * piCur + b[static_cast<size_t>(n)] * tauCur);
					s2 += fn * (a[static_cast<size_t>(n)] * tauCur + b[static_cast<size_t>(n)] * piCur);
					const double piNext = ((2.0 * dn + 1.0) * u * piCur - (dn + 1.0) * piPrev) / dn;
					piPrev = piCur;
					piCur = piNext;
				}

				const double i1 = std::norm(s1), i2 = std::norm(s2);
				const std::complex<double> cross = s2 * std::conj(s1);
				phase[static_cast<size_t>(i)] += glm::dvec4 { weight * 0.5 * (i2 + i1), weight * 0.5 * (i2 - i1), weight * cross.real(), weight * cross.imag() };
			}
		}

		crossSections[static_cast<size_t>(band)] = cross;
		const double norm = std::max(PhaseNormalization(phase), 1e-20);
		for (int i : std::views::iota(0, bins)) {
			const glm::dvec4 q = phase[static_cast<size_t>(i)] / norm;
			table[static_cast<size_t>(band) * static_cast<size_t>(bins) + static_cast<size_t>(i)] = glm::vec4 { static_cast<float>(q.x), static_cast<float>(q.y), static_cast<float>(q.z), static_cast<float>(q.w) };
		}
	});

	return table;
}

/// Builds a scalar angular sampling proposal from the weighted phase intensities.
///
/// Reads band-major phase rows starting at firstEntry, with bins samples per row,
/// weights each band by bandWeights, and appends normalized CDF values and fitted
/// within-bin shape parameters to table. The shape fit reduces the proposal's
/// second-moment estimate during angular quadrature.
void AppendSamplingCdf(std::vector<glm::vec4>& table, int bins, size_t firstEntry, const std::array<double, kSpectralBandCount>& bandWeights) {
	const auto phase = [&](int i) {
		double sum = 0.0;
		for (int band = 0; band < kSpectralBandCount; ++band) sum += std::max(0.0, bandWeights[band]) * table[firstEntry + static_cast<size_t>((band * bins) + i)].x;
		return std::max(0.0, sum);
	};

	std::vector<double> cdf(bins, 0.0), phaseValues(bins), shapes(bins, 0.5), masses(bins - 1);
	for (int i = 0; i < bins; ++i) phaseValues[i] = phase(i);

	double a = phaseValues[0];
	const double halfWidth = std::numbers::pi / (2.0 * (bins - 1));
	const double h2 = halfWidth * halfWidth;
	const double slopeIntegral = h2 * (1.0 / 3.0 + h2 * (-1.0 / 30.0 + h2 * (1.0 / 840.0 - h2 / 45360.0)));

	for (int i = 1; i < bins; ++i) {
		const double b = phaseValues[i];
		const double midpoint = (2.0 * i - 1.0) * halfWidth;
		const double slopeTerm = halfWidth < 0.1 ? slopeIntegral : std::sin(halfWidth) / halfWidth - std::cos(halfWidth);
		const double mass = (a + b) * std::sin(midpoint) * std::sin(halfWidth) + (b - a) * std::cos(midpoint) * slopeTerm;
		cdf[i] = cdf[i - 1] + std::max(0.0, mass);
		masses[i - 1] = std::max(0.0, mass);
		a = b;
	}

	for (int i = 0; i < bins; ++i) {
		const double left = phaseValues[i], right = phaseValues[std::min(i + 1, bins - 1)];
		double shape = left + right > 0.0 ? left / (left + right) : 0.5;

		if (i + 1 < bins && left + right > 0.0) {
			constexpr int nodes = 64;
			std::array<double, nodes> fractions { }, contributions { };
			const double theta0 = 2.0 * i * halfWidth;
			const double width = 2.0 * std::sin((2.0 * i + 1.0) * halfWidth) * std::sin(halfWidth);

			for (int j = 0; j < nodes; ++j) {
				const double t = (j + 0.5) / nodes;
				const double theta = theta0 + 2.0 * halfWidth * t;
				fractions[j] = 2.0 * std::sin(theta0 + halfWidth * t) * std::sin(halfWidth * t) / width;
				const double p = std::lerp(left, right, t) / (left + right);
				contributions[j] = p * p * std::sin(theta);
			}

			const auto secondMoment = [&](double candidate) {
				double sum = 0.0;
				for (int j = 0; j < nodes; ++j) sum += contributions[j] / (2.0 * std::lerp(candidate, 1.0 - candidate, fractions[j]));
				return sum;
			};

			double lo = 0.0, hi = 1.0;
			for (int iteration = 0; iteration < 24; ++iteration) {
				const double a = std::lerp(lo, hi, 1.0 / 3.0);
				const double b = std::lerp(lo, hi, 2.0 / 3.0);
				if (secondMoment(a) < secondMoment(b)) hi = b;
				else lo = a;
			}

			const double fitted = 0.5 * (lo + hi);
			if (secondMoment(fitted) < secondMoment(shape)) shape = fitted;
			masses[i] = (left + right) * std::sqrt(width * 2.0 * halfWidth / nodes * secondMoment(shape));
		}

		shapes[i] = shape;
	}

	cdf[0] = 0.0;
	for (int i = 1; i < bins; ++i) cdf[i] = cdf[i - 1] + masses[i - 1];

	for (int i = 0; i < bins; ++i) {
		const double value = cdf.back() > 0.0 ? cdf[i] / cdf.back() : 0.5 * (1.0 - std::cos(std::numbers::pi * i / (bins - 1)));
		table.emplace_back(static_cast<float>(value), static_cast<float>(shapes[i]), 0.0f, 0.0f);
	}
}

/// Integrates atmospheric density columns for each altitude and direction sample.
///
/// sky supplies the spherical bounds and density scale heights. Each vec4 result
/// stores the Rayleigh, first-aerosol, reserved zero, and second-aerosol columns. Ray paths
/// are split at ground and density-shell intersections before Gaussian quadrature.
std::vector<glm::vec4> ComputeTransmittanceTable(const SkyConfig& sky) {
	const double re = sky.earthRadius;
	const double ra = sky.atmosphereRadius;
	std::vector<glm::vec4> table(static_cast<size_t>(kTransmittanceAltitudeBins) * kTransmittanceMuBins);

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
			glm::dvec4 column(0.0);
			const double panelLength = 8.0 * std::min({ static_cast<double>(sky.scaleHeightRayleigh), static_cast<double>(sky.aerosols[0].scaleHeight), static_cast<double>(sky.aerosols[1].scaleHeight) });

			for (size_t segment = 1; segment < cuts.size(); ++segment) {
				const double start = cuts[segment - 1], length = cuts[segment] - start;
				const double middle = start + 0.5 * length;
				if (r * r + middle * (middle + 2.0 * b) < re * re) {
					column += glm::dvec4(length, length, 0.0, length);
					continue;
				}

				const int panels = std::max(1, static_cast<int>(std::ceil(length / panelLength)));
				const double half = 0.5 * length / panels;
				for (int panel = 0; panel < panels; ++panel) {
					for (size_t node = 0; node < kGaussNodes.size(); ++node) {
						const double t = start + (2 * panel + 1 + kGaussNodes[node]) * half;
						const double radius = std::sqrt(r * r + t * (t + 2.0 * b));
						const double altitude = std::max((r * r - re * re + t * (t + 2.0 * b)) / (radius + re), 0.0);
						column += half * kGaussWeights[node] * glm::dvec4 { std::exp(-altitude / sky.scaleHeightRayleigh), std::exp(-altitude / sky.aerosols[0].scaleHeight), 0.0, std::exp(-altitude / sky.aerosols[1].scaleHeight) };
					}
				}
			}

			table[static_cast<size_t>(a) * kTransmittanceMuBins + static_cast<size_t>(m)] = glm::vec4(column);
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

/// Samples wavelength-dependent coefficients and matching values for one spectral band.
///
/// wavelengthNm is in nanometres. The result contains a Rayleigh factor relative to
/// 550 nm, interpolated solar data, and CIE 1931 XYZ matching values.
SpectralBand ComputeSpectralAtWavelength(double wavelengthNm) {
	SpectralBand result { .betaRayleighScale = 0.0, .sunIrradianceScale = 0.0, .cie = { } };
	result.betaRayleighScale = RayleighShape(wavelengthNm) / RayleighShape(550.0);
	const double solarPosition = wavelengthNm - 360.0;
	const auto solarLower = static_cast<size_t>(solarPosition);
	const double solarFraction = solarPosition - static_cast<double>(solarLower);
	result.sunIrradianceScale = std::lerp(kSolarIrradiance[solarLower], kSolarIrradiance[solarLower + 1], solarFraction);
	result.cie = Cie1931Xyz(wavelengthNm);
	return result;
}
