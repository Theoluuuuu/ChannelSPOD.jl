# ChannelSPOD.jl

[![CI](https://github.com/Davide-Lasagna-s-Lab/ChannelSPOD.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Davide-Lasagna-s-Lab/ChannelSPOD.jl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small channel-flow spectral proper orthogonal decomposition package. It uses
Welch blocks, a temporal Fourier transform and a weighted thin SVD at each
frequency, following the SPOD framework of
[Towne, Schmidt & Colonius (2018)](https://arxiv.org/abs/1708.04393).

## Contents

- [Installation](#installation)
- [Interface](#interface)
- [Norm and time convention](#norm-and-time-convention)
- [Welch estimate and normalisation](#welch-estimate-and-normalisation)
- [Frequency-by-frequency projections](#frequency-by-frequency-projections)
- [Pipe flow](#pipe-flow)
- [Verification](#verification)

## Installation

Requires Julia 1.12 or newer. The packages are installed directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChebUtils.jl")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ChannelSPOD.jl")
```

Dependencies are [ChebUtils.jl](https://github.com/Davide-Lasagna-s-Lab/ChebUtils.jl)
for wall-normal quadrature, [FDGrids.jl](https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl)
for radial quadrature in the pipe, [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) for
temporal transforms, and Julia's `LinearAlgebra` standard library for the SVD.
DNS profiles may come from [ChannelFlow.jl](https://github.com/Davide-Lasagna-s-Lab/ChannelFlow.jl),
but ChannelFlow is not a dependency.

## Interface

```julia
import ChannelSPOD

# One synthetic spatial Fourier coefficient, sampled at Lobatto nodes.
Ny, Nt, dt = 33, 1024, 0.1
y = cospi.((0:Ny-1) ./ (Ny-1))
velocity_history = zeros(ComplexF64, Ny, 3, Nt)
for n in 1:Nt
    velocity_history[:, 1, n] .= (1 .- y.^2) .* exp(-im * 0.5 * (n-1) * dt)
end

problem = ChannelSPOD.SPODProblem(Ny)
result = ChannelSPOD.analyse(problem, velocity_history, dt;
                            nperseg=256, nmodes=3)
result.omega       # sorted, signed angular frequencies
result.energies    # min(3Ny, nblocks) × nperseg: all spectral energy densities
result.modes       # Ny × 3 × nmodes × nperseg
result.nblocks
```

`velocity_history` must contain the **complex spatial Fourier coefficients** of
one `(kx,kz)` pair, evaluated at the wall-normal Lobatto points, with shape
`(Ny,3,Nt)`. Components are `(u,v,w)` and nodes run from +1 to -1.

Samples must be equally spaced in time by `dt`. Run an analysis separately for each spatial pair. The zero spatial pair needs no special algebra here; the data, rather than a differential operator, determine its modes. Use full velocity fluctuations, including any plane-mean fluctuations you want to study. The default temporal-mean subtraction removes a stationary
base profile as well as any stationary component of the input.

Both this package and ChannelResolvent export `analyse`. When using both in one
session, prefer `import ChannelSPOD, ChannelResolvent` and qualify the calls.

## Norm and time convention

The norm matches ChannelResolvent:

$$
\|u\|_W^2=\frac12\int_{-1}^{1}u^*u\,dy
\approx u^*Wu,
\qquad W=\mathrm{diag}(w,w,w),
$$

where `w=chebws(Ny)/2`. This is physical Clenshaw–Curtis quadrature, not a
Chebyshev-weighted polynomial norm. The modes have unit weighted norm.

The convention is `exp(im*(α*x + β*z - ω*t))`. The analysis transform therefore
uses the **positive** temporal exponent (`FFTW.bfft`, unnormalised): a signal
`exp(-im*ω*t)` peaks at positive returned `omega`. Results are two-sided even
for real input. No positive-frequency doubling or spatial conjugate-pair factor
is applied. For even segment length the unique Nyquist bin is labelled negative.

## Welch estimate and normalisation

`nperseg` is required. By default, `noverlap=nperseg÷2`, the window is a periodic
Hann window, and `demean=true` removes the full-record temporal mean separately
at every node and component. A real window vector may be supplied explicitly.
Incomplete final blocks are discarded. Only supply uniformly sampled records;
interpolation, missing-data handling and restart-gap repair are outside this API.

For block b, let $\hat q_b(\omega)$ be its unnormalised windowed transform and
let $s_w=\sum_j w_j^2$ denote the window's squared sum (distinct from spatial
quadrature weights). The matrix decomposed at a frequency is

$$
X_\omega=\sqrt{\frac{\Delta t}{2\pi s_w N_b}}
W^{1/2}[\hat q_1,\ldots,\hat q_{N_b}].
$$

If $X_\omega=U\Sigma V^*$, return energies $\lambda_j=\Sigma_{jj}^2$ and
modes $\psi_j=W^{-1/2}U_j$. Energies are densities **per unit angular frequency**,
not resolvent amplitude gains. All eigenvalues are always retained, independently
of `nmodes`, so
$\sum_{\omega,j}\lambda_j(\omega)\Delta\omega$ equals the average window-weighted
block norm after mean subtraction. It is not necessarily the unweighted
full-record sample variance, because windowing, overlap and discarded tails
change sample weights. There is no extra kinetic-energy factor 1/2.

At most `min(3Ny,nblocks)` modes can be estimated per frequency. `nmodes=nothing`
retains up to three spatial modes; a request above the available rank raises an error. Returned modes
at zero spectral energy have no unique interpretation. Mode phases are arbitrary;
repeated eigenvalues require subspace comparisons. More modes cannot compensate
for too few statistically independent blocks.

For spectra and energy fractions without mode shapes, use
`analyse(problem, velocity_history, dt; nperseg=256, compute_modes=false)`.
This returns all energies and `modes=nothing`, using singular values only.
The denominator of a captured-energy fraction must sum **all** eigenvalues,
not only the modes displayed in a plot.

This first implementation stores the block transforms in memory and computes a
thin dense SVD per frequency (singular values only when `compute_modes=false`). It is serial, allocating and deliberately small;
it does not implement streaming SPOD, confidence intervals or spatial symmetry
reductions. Results can be compared with resolvent response modes using the
same weighted norm, but agreement depends on the forcing statistics.

## Frequency-by-frequency projections

Pass `frequency_callback=(omega, X) -> ...` to `analyse` to process each weighted,
normalised Fourier matrix before its factorisation. Treat `X` as **read-only**
and do not retain it: the storage is reused. For a basis `Psi` satisfying
`Psi' * W * Psi = I`, the rows of `A = (sqrt.(diag(W)) .* Psi)' * X` contain
its projected realisations; `sum(abs2, A; dims=2)` gives their spectral energies.
This avoids constructing SPOD modes when comparing another basis against the
full DNS covariance. Use `compute_modes=false` in that case.

The implementation views the input and reuses FFT and weighted-matrix buffers.
It still stores all Welch-block transforms, requiring approximately
`16 * 3Ny * nperseg * nblocks` bytes in complex double precision, plus work and
outputs. It is not a streaming estimator. Process spatial pairs sequentially
when analysing large archives.

## Pipe flow

`PipeSPODProblem(Nr)` and its `analyse` method (`src/pipe_spod.jl`) write out
the same Welch estimator for the radial grid and the pipe inner product:

```julia
import ChannelSPOD, FDGrids

problem = ChannelSPOD.PipeSPODProblem(64)
r = FDGrids.grid(64, 1.0, FDGrids.HalfChebyshevGrid()).xs   # radii of the input profiles
result = ChannelSPOD.analyse(problem, velocity_history, dt; nperseg=256)
```

`velocity_history` has shape `(Nr,3,Nt)`: the complex Fourier coefficients of
one `(k_θ,k_z)` pair with components `(u_r,u_θ,u_z)` on the radial grid `r`,
wall at `r[end]=1` and no point on the axis; `PipeSPODProblem` takes the
weights `ws` of `∫ f r dr` on the same grid, so

$$
\|u\|_W^2=\int_0^1\left(|u_r|^2+|u_\theta|^2+|u_z|^2\right)r\,dr\approx u^*Wu,
\qquad W=\mathrm{diag}(w_s,w_s,w_s),
$$

without the factor 2 of an area average. This is the norm of the pipe
resolvent response modes, so a basis `Psi` from them can be projected through
`frequency_callback` as above. The convention is `exp(im*(k_θ*θ + α*k_z*z - ω*t))`,
so `omega` is the report's `ω_k`; Resolvent.jl's `PipeResolvent` takes
`exp(+im*Ω*t)` with `Ω = -ω`. The HalfChebyshev grid is that of
PipeResolventModes' `resolvent_grid(Nr)`, so the same `Nr` gives the same
radii as the resolvent modes. Its weights integrate the kink of `|r|` at the axis
and are only second-order accurate for `∫ r dr`: the norm is consistent, so
orthonormality and the energy identities hold exactly in it, but absolute
energies carry that quadrature error.

Data on another radial grid, such as openpipeflow's, need not be interpolated:
`PipeSPODProblem(ws)` takes the positive weights `ws` of `∫ f r dr` on that grid,
in the order of the input profiles, and the norm, orthonormality and energy
identities then hold in that quadrature. Interpolating onto the HalfChebyshev
grid, with the axis parity of each component, is outside this package.

## Verification

The tests check a bin-centred travelling harmonic's frequency sign, weighted
mode norm, rectangular/Hann normalisation, input preservation, zero demeaned
data, odd segment lengths and invalid dimensions/settings.

The pipe tests, on a HalfChebyshev grid in the `∫ r dr` norm, check that

- a `k_θ = 1` mode `exp(i(k_θθ + αk_z z - ωt))` peaks at `+ω` with its
  `∫ |u|² r dr` as energy, under rectangular and Hann windows, and on a
  midpoint grid given its `∫ f r dr` weights;
- two structures orthonormal in `∫ r dr` but not in the Euclidean product,
  uncorrelated across blocks, come back as the first two modes with their
  energies, which the plain or `dr`-weighted norm cannot do;
- the response `H f` of an operator to forcing white in the `∫ r dr` norm has
  SPOD energies `σ²/B` and modes equal to the response modes of
  `W^{1/2} H W^{-1/2}`, the Towne, Schmidt & Colonius (2018) correspondence the
  pipe resolvent comparison rests on, and that projecting onto those response
  modes through `frequency_callback` returns exactly the SPOD energies;
- at `(k_θ, k_z) = (0, 0)` demeaning removes a Hagen–Poiseuille profile and
  nothing else, and the spectrum integrates to the window-weighted
  `∫ |u|² r dr` with and without the mean.

Replacing the weights by ones or by the `dr` weights, flipping the transform
sign, returning modes without `W^{-1/2}` or dropping the window energy each
makes several of these tests fail. These checks verify
the estimator conventions; they do not establish convergence for a turbulent
record of a particular duration.

Run the suite from a checkout with:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

Distributed under the [MIT licence](LICENSE).
