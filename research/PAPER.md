---
title: GPU workload of finite-Sun integration in polarized atmospheric rendering
author: Cohen Thomas
---

## Abstract

This paper measures GPU workload for nested deterministic quadrature over a horizon-clipped solar disk in a single-scattering polarized sky model. A coarse-to-fine Stokes norm difference guides azimuthal refinement at each camera-path point. Across 20 Sun and view geometries, including 12 held-out cases, the strictest adaptive setting reduces median GPU workload duration from 440.35 ms for a fixed 256-azimuth rule to 143.37 ms for a 320-ray batch on an NVIDIA GeForce RTX 5060 Ti. The adaptive rule reaches its azimuth cap in 12 of 20 geometries. These measurements characterize computational cost and cap events. They do not establish accuracy or an end-to-end error bound.

**Keywords:** polarized rendering, atmospheric scattering, finite solar disk, adaptive quadrature, GPU integration, Stokes radiance

## 1. Introduction

Single-scattering sky models evaluate radiance by integrating contributions along a camera ray and over incident solar directions. A point-source Sun removes the second integral, but it cannot represent partial visibility of the solar disk at the geometric horizon. A finite source also changes the scattering plane across the disk, so the source integral affects the output Stokes vector rather than only its scalar magnitude.

Prior atmospheric-rendering studies have compared midpoint, trapezoid, and Simpson rules for single-scattering sky integrals [4], while adaptive step sizing has been applied to participating-media ray marching [1]. Graphics work on finite spherical lights includes stratified sampling of their visible projected solid angle [5]. MYSTIC studies have modeled finite solar disks with wavelength-dependent limb darkening in circumsolar radiance calculations [2]. Broader radiative-transfer packages such as libRadtran and MONKI compute polarized radiances [3,8]. OpenSky and PySkyLumos address simulation workflows for sky-polarization measurements, including polarimetric-camera outputs [6,7]. These efforts establish related methods and applications, while addressing different questions. Here we measure the GPU workload of nested deterministic source-direction quadrature in a small polarized sky renderer, including planet occlusion across the disk and a held-out geometry set.

The contribution is an empirical characterization of GPU workload and cap events for nested source-direction quadrature. We compare adaptive and fixed disk rules under the same physical and camera-path assumptions. The experiment does not establish their accuracy.

## 2. Model and integration method

### 2.1 Single-scattering Stokes integral

Let **x**(t) be a point on a camera ray with propagation direction **v**, and let **d** range over the apparent solar disk Ω_s. For color channel c, the integrated Stokes radiance is

$$
\mathbf{S}_c = \int_{t_0}^{t_1} \rho(\mathbf{x}(t))\,T_{v,c}(t)
\int_{\Omega_s} V(\mathbf{x}(t),\mathbf{d})\,T_{s,c}(\mathbf{x}(t),\mathbf{d})\,
\mathbf{M}(\mathbf{d},\mathbf{v})\,\mathbf{S}_{\odot,c}\,d\Omega\,dt,
$$

where density is normalized molecular density, T_v and T_s are camera and solar transmittance, V is planet visibility, M is the Rayleigh Mueller response including depolarization, and the incident solar Stokes vector is unpolarized. The atmosphere is spherical with exponential density. Only single Rayleigh scattering is included.

The apparent solar disk is uniform. Direct solar-disk raster coverage is computed by the renderer separately and is not part of this atmospheric in-scattering integral. Aerosols, clouds, multiple scattering, surface reflection, limb darkening, and wavelength sampling are outside the model.

### 2.2 Horizon-clipped disk rule

At a scattering point, the disk direction is parameterized as

$$
\mathbf{d}(\theta,\phi)=\cos\theta\,\mathbf{L}
+\sin\theta\,(\cos\phi\,\mathbf{a}+\sin\phi\,\mathbf{b}),
$$

where **L** is the solar-center direction, **a** is aligned with the local horizon tangent projected onto the disk plane, and **b** = **L** × **a**. For each azimuth, the planet horizon determines a visible polar interval from $\theta_{\min}$ to $\theta_{\max}$. A bisection locates a boundary when the disk is partly occluded. A four-point Gauss-Legendre rule integrates this interval with the solid-angle measure $\sin\theta\,d\theta$. A periodic trapezoid rule integrates azimuth.

The fixed rule evaluates $N$ azimuths. The adaptive rule starts at $N=8$ and repeatedly doubles a nested grid. If $\mathbf{S}_N$ and $\mathbf{S}_{2N}$ are the local RGB-Stokes integrals at a path point, the reported local indicator is

$$
E_N = \frac{\lVert \mathbf{S}_{2N}-\mathbf{S}_{N}\rVert_2}
{\max(\lVert\mathbf{S}_{2N}\rVert_2,10^{-30})}.
$$

Refinement stops when $E_N$ is at most the requested value or the maximum azimuth count is reached. The shader reports the maximum local indicator across camera-path nodes, the maximum selected azimuth count, and accumulated source-node candidates. For each azimuth candidate, the counter adds four radial-node candidates before rejecting a fully occluded interval. It excludes azimuth visibility tests, horizon bisection, and camera-path work, so it is a source-rule work proxy rather than a count of actual quadrature evaluations. GPU workload duration measures the resulting batch cost. Reaching the cap is recorded separately. $E_N$ is an empirical indicator, not a formal error bound for the whole camera ray.

### 2.3 Baseline camera and optical-depth rules

All GPU source-rule comparisons use 48 composite four-point Gauss-Legendre panels along the camera ray, for 192 path evaluations. Optical depth uses composite four-point panels with panel width no greater than four density scale heights. These rules are fixed across point, fixed-disk, and adaptive-disk methods, so the source-rule comparison does not mix changes to the camera-path sampling policy.

Optical depth uses the same fixed composite four-point rule in every GPU configuration. The main experiment compares source integration at a fixed path rule. The retained workflow does not test camera-path convergence, so the comparison does not establish a 10^-4 end-to-end camera-ray error.

## 3. Evaluation protocol

### 3.1 Geometry and recorded outputs

The experiment contains 20 fixed geometries. Eight are labeled pilot and 12 are held out. Sun elevation spans −9° to +20°, view elevation spans 0° to 90°, and view azimuth is 90° for every case. The GPU harness records raw Stokes values before display clamping and tone mapping, the local refinement indicator, and source-node candidate counts.

### 3.2 GPU timing

The measured device is an NVIDIA GeForce RTX 5060 Ti under Windows and Vulkan. Nsight Systems 2026.3.2 records Vulkan workload events. Each of ten configurations executes three warmup submissions and 21 measured submissions. Each submission processes a 320-ray batch formed by 16 replicas of the 20 geometries. Reported times are GPU workload duration for this batch, not frame time. A second independent profile repeats all configurations. The raw Stokes outputs are bit-identical across runs, and the per-configuration median times differ by at most 1.40 percent. We report both run medians and retain means and 95th percentiles in the result tables.

\newpage

## 4. Results

Table 1 reports the GPU timings and source-node candidate counts for two fixed rules and three adaptive settings. The strictest adaptive setting reduces median GPU workload duration from 440.35 ms for fixed 256 azimuths to 143.37 ms per 320-ray batch in the first profile, a 3.07× speedup. In the repeat profile the corresponding times are 443.92 ms and 144.89 ms, a 3.06× speedup. Its mean source-node candidate count is 17,859 per ray, compared with 196,608 for fixed 256.

| Method | Median GPU time per batch, run 1 / run 2 | Mean source-node candidates per ray | Cap reached |
| --- | ---: | ---: | ---: |
| Fixed, 128 azimuths | 220.69 / 221.80 ms | 98,304 | Not applicable |
| Fixed, 256 azimuths | 440.35 / 443.92 ms | 196,608 | Not applicable |
| Adaptive, requested 10^-2 | 47.51 / 48.18 ms | 13,533 | 4 of 20 |
| Adaptive, requested 10^-3 | 96.01 / 97.06 ms | 16,067 | 12 of 20 |
| Adaptive, requested 10^-4 | 143.37 / 144.89 ms | 17,859 | 12 of 20 |

**Table 1.** GPU workload for a 320-ray batch and source-node candidates per ray, the work proxy defined in Section 2.2. A cap event means at least one path node reached 256 azimuths. The local indicator may or may not meet the requested tolerance at that count. The 95th-percentile duration for adaptive 10^-4 is 264.54 ms in run 1 and 145.17 ms in run 2.

The local indicator guides refinement but is not a proven bound on whole-ray error. At the strictest setting, 12 geometries reach the azimuth cap, so those cases do not establish that the requested stopping condition was met at every path node. The measurements alone do not establish equivalent accuracy between methods.

## 5. Discussion

The tested adaptive settings use fewer source-node candidates and less GPU time than the fixed 256-azimuth rule. These measurements show a cost difference for the recorded configurations. They do not establish a work-accuracy tradeoff without an independent accuracy evaluation.

The method is complementary to adaptive camera-path integration [1] and prior comparisons of camera-path rules for atmospheric rendering [4]. Finite-light sampling work targets variance reduction for surface illumination [5], while this experiment evaluates deterministic angular quadrature inside a polarized volume-scattering integral. OpenSky and PySkyLumos model sky-polarization measurements [6,7], and MYSTIC, libRadtran, and MONKI address polarized atmospheric radiative transfer at broader physical scopes [2,3,8].

## 6. Limitations

The model excludes multiple scattering and non-Rayleigh components, assumes a uniform solar disk, and uses one fixed camera-path quadrature. Camera-path convergence and end-to-end accuracy are not established. There is no comparison with an external solver or measurement. The geometry set has 20 rays, timing was measured on one GPU, and the 320-ray batch is a microbenchmark rather than a full-frame benchmark. The local indicator has no proof of conservativeness. Reaching the cap alone does not establish whether the requested tolerance was met at every path node.

## 7. Conclusion

Nested adaptive integration over a finite solar disk reduced GPU workload duration compared with fixed 256-azimuth integration in this experiment. The strict adaptive setting reached its cap in 12 of 20 geometries. Further accuracy and convergence evaluation is needed before drawing conclusions about the quality of its output.

## Code and data availability

The shader variant, standalone build, Vulkan harness, measured GPU CSV data, and timing tables are included in `research/`. `REPRODUCE.md` gives the build, profiling, and analysis commands. Nsight traces can be regenerated under `research/build/`.

## References

\footnotesize

1. A. Muñoz, “Higher Order Ray Marching,” *Computer Graphics Forum*, vol. 33, no. 8, pp. 167-176, 2014. [doi:10.1111/cgf.12424](https://doi.org/10.1111/cgf.12424).
2. B. Reinhardt, R. Buras, L. Bugliaro, S. Wilbert, and B. Mayer, “Determination of circumsolar radiation from Meteosat Second Generation,” *Atmospheric Measurement Techniques*, vol. 7, pp. 823-838, 2014. [doi:10.5194/amt-7-823-2014](https://doi.org/10.5194/amt-7-823-2014).
3. C. Emde et al., “The libRadtran software package for radiative transfer calculations (version 2.0.1),” *Geoscientific Model Development*, vol. 9, pp. 1647-1672, 2016. [doi:10.5194/gmd-9-1647-2016](https://doi.org/10.5194/gmd-9-1647-2016).
4. T. Gałaj and A. Wojciechowski, “A study on numerical integration methods for rendering atmospheric scattering phenomenon,” *Open Physics*, vol. 17, no. 1, pp. 241-249, 2019. [doi:10.1515/phys-2019-0025](https://doi.org/10.1515/phys-2019-0025).
5. C. Ureña and I. Georgiev, “Stratified Sampling of Projected Spherical Caps,” *Computer Graphics Forum*, vol. 37, no. 4, pp. 13-20, 2018. [doi:10.1111/cgf.13471](https://doi.org/10.1111/cgf.13471).
6. A. Moutenet, L. Poughon, B. Toulon, J. R. Serres, and S. Viollet, “OpenSky: A Modular and Open-Source Simulator of Sky Polarization Measurements,” *IEEE Transactions on Instrumentation and Measurement*, vol. 73, pp. 1-16, 2024. [doi:10.1109/TIM.2024.3374965](https://doi.org/10.1109/TIM.2024.3374965).
7. T.-A. Ciochirca, D. J. Chadwick, I. Sandall, and J. F. Ralph, “PySkyLumos: A Python-Based Polarized Skylight Simulator,” *Modelling*, vol. 7, no. 5, article 222, 2026. [doi:10.3390/modelling7050222](https://doi.org/10.3390/modelling7050222).
8. V. J. H. Trees et al., “MONKI: a three-dimensional Monte Carlo simulator for transfer of polarised light in planetary atmospheres,” *Geoscientific Model Development*, vol. 19, pp. 9677-9728, 2026. [doi:10.5194/gmd-19-9677-2026](https://doi.org/10.5194/gmd-19-9677-2026).
