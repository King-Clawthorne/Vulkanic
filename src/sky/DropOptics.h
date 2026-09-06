#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

// Geometric ray optics in a unit sphere. All lengths are scaled by
// droplet radius, so geometry is reusable across the size distribution.
namespace drop_optics
{
using Complex = std::complex<double>;
struct Vector
{
    double x, y, z;
    Vector operator+(Vector b) const
    {
        return {x + b.x, y + b.y, z + b.z};
    }
    Vector operator-(Vector b) const
    {
        return {x - b.x, y - b.y, z - b.z};
    }
    Vector operator*(double s) const
    {
        return {x * s, y * s, z * s};
    }
};
inline double Dot(Vector a, Vector b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vector Cross(Vector a, Vector b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vector Unit(Vector a)
{
    return a * (1.0 / std::sqrt(Dot(a, a)));
}
using Jones = std::array<Complex, 4>; // rows: outgoing p/s, columns: initial x/y
struct Beam
{
    Vector direction{0, 0, 1}, a{1, 0, 0}, b{0, 1, 0};
    Jones j{1.0, 0.0, 0.0, 1.0};
};
struct Sphere
{
    Vector Gradient(Vector p) const
    {
        return p;
    }
    double ExitDistance(Vector p, Vector d) const
    {
        const double a = Dot(d, Gradient(d));
        const double b = Dot(p, Gradient(d));
        const double c = Dot(p, Gradient(p)) - 1.0;
        const double discriminant = b * b - a * c;
        return discriminant < 0.0 ? -1.0 : (-b + std::sqrt(discriminant)) / a;
    }
    double EntryDistance(Vector p, Vector d) const
    {
        const double a = Dot(d, Gradient(d));
        const double b = Dot(p, Gradient(d));
        const double c = Dot(p, Gradient(p)) - 1.0;
        const double discriminant = b * b - a * c;
        return discriminant < 0.0 ? -1.0 : (-b - std::sqrt(discriminant)) / a;
    }
};

// Signed Fresnel amplitudes, including complex TIR phase. Transmission uses
// flux-normalized amplitudes sqrt(1-|r|^2); a complete air/water/air path has
// no residual refractive-index throughput factor.
inline Beam Interface(const Beam &incoming, Vector normal, double n1, double n2, bool reflection)
{
    const Vector d = incoming.direction;
    if (Dot(d, normal) > 0.0)
        normal = normal * -1.0;
    const double ci = std::clamp(-Dot(d, normal), 0.0, 1.0);
    const double eta = n1 / n2;
    const Complex ct = std::sqrt(Complex(1.0 - eta * eta * (1.0 - ci * ci), 0.0));
    const Complex rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    const Complex rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct);
    Vector s = Cross(d, normal);
    s = Dot(s, s) > 1.0e-20 ? Unit(s) : incoming.b;
    const Vector p = Cross(s, d);
    Beam result;
    result.direction = reflection ? d + normal * (2.0 * ci) : d * eta + normal * (eta * ci - ct.real());
    result.direction = Unit(result.direction);
    result.a = Cross(s, result.direction);
    result.b = s;
    const Complex ap = reflection ? rp : Complex(std::sqrt(std::max(0.0, 1.0 - std::norm(rp))), 0);
    const Complex as = reflection ? rs : Complex(std::sqrt(std::max(0.0, 1.0 - std::norm(rs))), 0);
    for (int column = 0; column < 2; ++column)
    {
        result.j[column] = ap * (Dot(p, incoming.a) * incoming.j[column] + Dot(p, incoming.b) * incoming.j[2 + column]);
        result.j[2 + column] =
            as * (Dot(s, incoming.a) * incoming.j[column] + Dot(s, incoming.b) * incoming.j[2 + column]);
    }
    return result;
}

// Mueller components in the scattering-plane basis. Spherical trajectories
// satisfy F22=F11 and F44=F33.
inline std::array<double, 6> Mueller(const Beam &beam)
{
    const Vector din{0, 0, 1};
    Vector s = Cross(din, beam.direction);
    s = Dot(s, s) > 1.0e-20 ? Unit(s) : Vector{0, 1, 0};
    const Vector pi = Cross(s, din), po = Cross(s, beam.direction);
    Jones j;
    for (int col = 0; col < 2; ++col)
    {
        j[col] = Dot(po, beam.a) * beam.j[col] + Dot(po, beam.b) * beam.j[2 + col];
        j[2 + col] = Dot(s, beam.a) * beam.j[col] + Dot(s, beam.b) * beam.j[2 + col];
    }
    const Complex a = j[0] * pi.x + j[1] * pi.y;
    const Complex b = j[0] * s.x + j[1] * s.y;
    const Complex c = j[2] * pi.x + j[3] * pi.y;
    const Complex d = j[2] * s.x + j[3] * s.y;
    const double aa = std::norm(a), bb = std::norm(b), cc = std::norm(c), dd = std::norm(d);
    return {0.5 * (aa + bb + cc + dd),
            0.5 * (aa - dd),
            0.5 * (aa - bb - cc + dd),
            (a * std::conj(d) + b * std::conj(c)).real(),
            (a * std::conj(d) - b * std::conj(c)).imag(),
            (a * std::conj(d) - b * std::conj(c)).real()};
}
} // namespace drop_optics
