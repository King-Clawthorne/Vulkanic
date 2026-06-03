#pragma once

// Lorenz–Mie scattering-matrix precompute for the polarized (vector radiative
// transfer) sky. Implemented from scratch — no SciPy/miepython/BHMIE
// dependency — to match the project's "no third-party wrappers" ethos.
//
// The full Mie scattering matrix cannot be evaluated live in a shader (the
// series runs over tens of terms per particle and must be integrated over a
// particle size distribution, per wavelength). Instead we bake it once on the
// CPU into a table of the four independent scattering-matrix elements
// (F11, F12, F33, F34) versus scattering angle, for three wavelength bands
// (R, G, B), and sample that table in sky.comp.
//
// For spherical particles the 4x4 single-scattering (Mueller) matrix is block
// structured with F22 = F11 and F44 = F33:
//   [ F11  F12   0    0  ]
//   [ F12  F22   0    0  ]
//   [  0    0   F33  F34 ]
//   [  0    0  -F34  F44 ]
// We bake all six independent elements so non-spherical aerosols can break the
// F22=F11 / F44=F33 identities (the spherical case keeps them equal). F11 is
// normalized as a phase function in the (1/4π)∫P dΩ = 1 convention (i.e.
// ∫F11 dΩ = 4π), matching the Rayleigh phase matrix used in sky.comp.

#include <cstdint>
#include <vector>

// Aerosol model + sampling controls for the Mie table bake. Defaults model a
// mild continental haze (water-ish droplets, sub-micron, weakly absorbing).
struct MieAerosolParams
{
    double refractiveIndexReal = 1.33; // real part of the relative refractive index
    double refractiveIndexImag = 0.0;  // imaginary part (absorption; >= 0)
    double meanRadiusMicrometers = 0.2; // log-normal geometric mean radius r_g (µm)
    double sigma = 1.5;                 // log-normal geometric standard deviation (> 1)
    double wavelengthsNmRgb[3] = {680.0, 550.0, 440.0}; // representative R/G/B wavelengths (nm)
    double nonSphericity = 0.0;         // [0,1] aerosol non-sphericity (0 = perfect spheres)
    int angleBins = 181;                // scattering-angle samples over [0, π] inclusive
};

// One table entry: the six independent normalized scattering-matrix elements at
// a single scattering angle, padded to two GPU vec4s. The shader reads
// entries[2*idx] = (f11, f12, f33, f34) and entries[2*idx+1] = (f22, f44, 0, 0).
struct MieMatrixEntry
{
    float f11;
    float f12;
    float f33;
    float f34;
    float f22;
    float f44;
    float pad0;
    float pad1;
};

// Compute the ensemble-averaged, phase-normalized Mie scattering-matrix table.
// Returns params.angleBins entries per band, concatenated band-major:
//   index(band, bin) = band * angleBins + bin
// where bin i corresponds to scattering angle theta = i/(angleBins-1) * π.
// The three bands correspond to wavelengthsNmRgb[0..2].
std::vector<MieMatrixEntry> ComputeMieScatteringTable(const MieAerosolParams& params);
