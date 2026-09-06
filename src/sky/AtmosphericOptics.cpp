#include "AtmosphericOptics.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace
{
struct Ray
{
    double x, y, dx, dy;
};
Ray Derivative(Ray r, const SkySpectralConfig &s)
{
    const double radius = std::hypot(r.x, r.y);
    const double refractivity =
        s.seaLevelRefractivity * std::exp(-std::max(radius - s.earthRadius, 0.0) / s.refractionScaleHeight);
    const double gradient = -refractivity / ((1.0 + refractivity) * s.refractionScaleHeight * radius);
    const double gx = gradient * r.x, gy = gradient * r.y;
    const double parallel = r.dx * gx + r.dy * gy;
    return {r.dx, r.dy, gx - r.dx * parallel, gy - r.dy * parallel};
}
Ray Add(Ray a, Ray b, double scale)
{
    return {a.x + b.x * scale, a.y + b.y * scale, a.dx + b.dx * scale, a.dy + b.dy * scale};
}
Ray Advance(Ray r, double ds, const SkySpectralConfig &s)
{
    if (s.seaLevelRefractivity == 0)
        return {r.x + r.dx * ds, r.y + r.dy * ds, r.dx, r.dy};
    const Ray a = Derivative(r, s), b = Derivative(Add(r, a, ds * 0.5), s);
    const Ray c = Derivative(Add(r, b, ds * 0.5), s), d = Derivative(Add(r, c, ds), s);
    r = Add(Add(Add(Add(r, a, ds / 6), b, ds / 3), c, ds / 3), d, ds / 6);
    const double norm = std::hypot(r.dx, r.dy);
    r.dx /= norm;
    r.dy /= norm;
    return r;
}
double Height(Ray r, const SkySpectralConfig &s)
{
    return std::hypot(r.x, r.y) - s.earthRadius;
}
struct Node
{
    double distance;
    Ray ray;
};
} // namespace

bool HasAtmosphereGeometryChanged(const SkySpectralConfig &a, const SkySpectralConfig &b)
{
    return a.earthRadius != b.earthRadius || a.atmosphereRadius != b.atmosphereRadius ||
           a.scaleHeightRayleigh != b.scaleHeightRayleigh || a.scaleHeightMie != b.scaleHeightMie ||
           a.seaLevelRefractivity != b.seaLevelRefractivity || a.refractionScaleHeight != b.refractionScaleHeight;
}

std::vector<AtmosphereRayEntry> ComputeAtmosphereRayTable(const SkySpectralConfig &sky)
{
    constexpr int angles = 2 * kAtmosphereAnglesPerBranch;
    constexpr int stride = 1 + kAtmosphereRaySamples;
    const double atmosphereHeight = sky.atmosphereRadius - sky.earthRadius;
    const double pi = std::numbers::pi_v<double>;
    std::vector<AtmosphereRayEntry> table(static_cast<size_t>(kAtmosphereHeights) * angles * stride);
    std::vector<Node> trajectory;
    trajectory.reserve(4096);
    for (int hIndex = 0; hIndex < kAtmosphereHeights; ++hIndex)
    {
        const double h = std::max(0.01, std::expm1(std::log1p(atmosphereHeight) * hIndex / (kAtmosphereHeights - 1)));
        const double radius = sky.earthRadius + h;
        const double n = 1 + sky.seaLevelRefractivity * std::exp(-h / sky.refractionScaleHeight);
        const double critical = (1 + sky.seaLevelRefractivity) * sky.earthRadius / (n * radius);
        const double horizon = -std::acos(std::clamp(critical, 0.0, 1.0));
        for (int angle = 0; angle < angles; ++angle)
        {
            const bool ground = angle < kAtmosphereAnglesPerBranch;
            const int angularIndex = angle % kAtmosphereAnglesPerBranch;
            const double u = double(angularIndex) / (kAtmosphereAnglesPerBranch - 1);
            // Concentrate both branches near the horizon. The tiny offset
            // chooses a side of the tangent ray without a discontinuous mix.
            const double elevation = ground ? horizon - (pi / 2 + horizon) * (1 - u) * (1 - u) - 1e-7
                                            : horizon + (pi / 2 - horizon) * u * u + 1e-7;
            Ray ray{0, radius, std::cos(elevation), std::sin(elevation)};
            trajectory.clear();
            trajectory.push_back({0, ray});
            double distance = 0, depthR = 0, depthM = 0;
            for (int iteration = 0; iteration < 32768; ++iteration)
            {
                const double altitude = Height(ray, sky);
                double ds = std::clamp(0.08 * std::max(altitude, 0.0) + 100.0, 100.0, 2000.0);
                Ray next = Advance(ray, ds, sky);
                const double nextAltitude = Height(next, sky);
                const bool hit = nextAltitude < 0 || nextAltitude >= atmosphereHeight;
                if (hit)
                {
                    double lo = 0, hi = ds;
                    for (int j = 0; j < 24; ++j)
                    {
                        const double mid = (lo + hi) * 0.5;
                        const double height = Height(Advance(ray, mid, sky), sky);
                        if (height < 0 || height >= atmosphereHeight)
                            hi = mid;
                        else
                            lo = mid;
                    }
                    ds = (lo + hi) * 0.5;
                    next = Advance(ray, ds, sky);
                }
                const double midAltitude = std::max(Height(Advance(ray, ds * 0.5, sky), sky), 0.0);
                depthR += std::exp(-midAltitude / sky.scaleHeightRayleigh) * ds;
                depthM += std::exp(-midAltitude / sky.scaleHeightMie) * ds;
                distance += ds;
                ray = next;
                trajectory.push_back({distance, ray});
                if (hit)
                    break;
            }
            const size_t base = (static_cast<size_t>(hIndex) * angles + angle) * stride;
            table[base] = {float(distance), float(depthR), float(depthM), ground ? 1.0f : 0.0f};
            size_t interval = 1;
            for (int sample = 0; sample < kAtmosphereRaySamples; ++sample)
            {
                const double uSample = double(sample) / (kAtmosphereRaySamples - 1);
                const double target = distance * uSample * uSample;
                while (interval + 1 < trajectory.size() && trajectory[interval].distance < target)
                    ++interval;
                const Node &start = trajectory[interval - 1];
                const Ray point = Advance(start.ray, std::max(0.0, target - start.distance), sky);
                table[base + 1 + sample] = {float(point.x), float(point.y - radius), float(point.dx), float(point.dy)};
            }
        }
    }
    return table;
}
