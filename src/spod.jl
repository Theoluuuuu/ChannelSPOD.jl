"""
    SPODProblem(Ny)

Define the velocity inner product on `Ny` Chebyshev–Lobatto points ordered
from +1 to -1. Clenshaw–Curtis weights approximate the wall-normal average
`⟨u,v⟩ = (1/2)∫₋₁¹ u(y)ᴴv(y) dy`, summed over three velocity components.
The input profiles are nodal values, not Chebyshev coefficients.
"""
struct SPODProblem
    # Scalar-profile quadrature weights; repeated for u, v and w during analysis.
    weights::Vector{Float64}

    function SPODProblem(Ny::Integer)
        Ny >= 2 || throw(ArgumentError("at least two wall-normal points are required"))
        return new(chebws(Int(Ny)) ./ 2)
    end
end

raw"""
    analyse(problem, velocity, dt; nperseg, noverlap=nperseg÷2,
            nmodes=nothing, compute_modes=true, demean=true, window=nothing,
            frequency_callback=nothing)

Compute Welch SPOD of one spatial Fourier pair from uniformly sampled nodal
velocities of shape `(Ny, 3, Nt)`. Real or complex input is accepted and preserved.

# Mathematical definition

Let ``q_n ∈ ℂ^{3N_y}`` contain the three velocity profiles at time ``nΔt``.
For `demean=true`, subtract the full-record mean ``μ = N_t^{-1}∑_n q_n``;
otherwise use ``μ=0``. This is not separate mean subtraction within each block.

Split the record into ``B`` overlapping blocks of length ``L = nperseg`` and
stride ``L-noverlap``, discarding an incomplete final block. For block start
``s_b`` and real window ``w_n``, define the unnormalised Fourier realisation

```math
\hat q_b(ω_k) = \sum_{n=0}^{L-1} w_n (q_{s_b+n}-μ) e^{+2πikn/L},
\qquad ω_k = \frac{2πk}{LΔt}.
```

Thus a signal ``e^{-iωt}`` appears at positive ``ω``. Frequencies are returned
in increasing signed order, with the even-length Nyquist bin on the negative
side. The default window is periodic Hann, ``w_n=(1-\cos(2πn/L))/2``.

Let ``W`` be the diagonal wall-normal quadrature matrix, repeated for the three
components, and ``Q_k`` collect the ``B`` Fourier realisations as columns.
The cross-spectral density estimate and its weighted factor are

```math
c = \frac{Δt}{2πB\sum_n w_n^2},\qquad
S_k = c Q_k Q_k^H,\qquad X_k=\sqrt{c}\,W^{1/2}Q_k.
```

The thin SVD ``X_k=U_kΣ_kV_k^H`` gives energies ``λ_j=Σ_{jj}^2`` and velocity
modes ``ψ_j=W^{-1/2}U_{k,j}``. Hence ``S_kWψ_j=λ_jψ_j`` and
``ψ_i^HWψ_j=δ_{ij}``. Energies are two-sided densities **per unit angular
frequency**, without an additional kinetic-energy factor ``1/2`` or Fourier
conjugacy multiplicity. With ``Δω=2π/(LΔt)``, Parseval's identity gives

```math
\sum_{k,j}λ_j(ω_k)Δω =
\frac{\sum_{b,n} w_n^2\lVert q_{s_b+n}-μ\rVert_W^2}
     {B\sum_n w_n^2}.
```

This is a window-weighted block average, not necessarily the unweighted
full-record variance. At each frequency the fraction captured by ``r`` modes
is ``\sum_{j=1}^r λ_j / \sum_{j=1}^{R} λ_j``, where ``R=min(3N_y,B)``.
It is undefined at zero total energy. Implicit eigenvalues beyond ``R`` are zero.

An optional `frequency_callback(omega, X)` is called for each frequency before
factorisation, with the weighted matrix ``X_k`` defined above. Treat this matrix
as read-only and do not retain it: its storage is reused. This permits direct
projection onto another weighted-orthonormal basis without forming SPOD modes.

# Return values

Return `(omega, energies, modes, nblocks)`:
- `omega`: `nperseg` signed angular frequencies.
- `energies`: `(min(3Ny, nblocks), nperseg)`, always including every eigenvalue
  supported by the sample rank, independently of `nmodes`.
- `modes`: `(Ny, 3, nr, nperseg)`, where `nr=nmodes` or at most three by default.
  `compute_modes=false` returns `nothing` and ignores `nmodes`.
- `nblocks`: number of complete Welch blocks (overlap does not make them independent).

An explicit mode count must lie in `1:min(3Ny, nblocks)`. Modes at zero energy
are arbitrary, and repeated eigenvalues determine a subspace, not unique modes.
"""
function analyse(problem::SPODProblem, velocity::AbstractArray{<:Number,3}, dt::Real;
                 nperseg::Integer, noverlap::Integer=nperseg÷2,
                 nmodes::Union{Nothing,Integer}=nothing, compute_modes::Bool=true,
                 demean::Bool=true, window=nothing, frequency_callback=nothing)

    # Validate shape, sample spacing and Welch settings before allocating work.
    Ny, nc, Nt = size(velocity)
    Ny == length(problem.weights) && nc == 3 ||
        throw(DimensionMismatch("velocity must have shape (Ny, 3, Nt)"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    2 <= nperseg <= Nt || throw(ArgumentError("require 2 ≤ nperseg ≤ Nt"))
    0 <= noverlap < nperseg || throw(ArgumentError("invalid overlap"))
    all(isfinite, velocity) || throw(ArgumentError("velocity must be finite"))

    win = isnothing(window) ? 0.5 .- 0.5cos.(2π .* (0:nperseg-1) ./ nperseg) : Float64.(window)
    length(win) == nperseg && all(isfinite, win) && sum(abs2, win) > 0 ||
        throw(ArgumentError("window must be finite, nonzero and have length nperseg"))
    starts = 1:(nperseg-noverlap):(Nt-nperseg+1)
    nblocks = length(starts)
    rankmax = min(3Ny, nblocks)
    nr = !compute_modes || isnothing(nmodes) ? min(3, rankmax) : nmodes
    1 <= nr <= rankmax || throw(ArgumentError("nmodes must lie in 1:min(3Ny, nblocks)"))

    # View the input as spatial rows and temporal columns, without copying it.
    # Demeaning is fused with windowing below; only the mean profile is stored.
    data = reshape(velocity, 3Ny, Nt)
    meanprofile = zeros(ComplexF64, 3Ny)
    if demean
        for t in axes(data, 2)
            @views meanprofile .+= data[:, t]
        end
        meanprofile ./= Nt
    end

    # Reuse one contiguous block and one in-place FFT plan for all realisations.
    # The full transform ensemble is retained so each frequency can later access
    # every block. This is the main remaining memory cost, O(3Ny * L * B).
    block = Matrix{ComplexF64}(undef, 3Ny, nperseg)
    plan = plan_bfft!(block, 2; flags=FFTW.ESTIMATE)
    spectra = Array{ComplexF64}(undef, 3Ny, nperseg, nblocks)

    for (b, start) in enumerate(starts)
        @views block .= (data[:, start:start+nperseg-1] .- meanprofile) .* transpose(win)
        plan * block
        @views spectra[:, :, b] .= block
    end

    # Convert FFT bin order to signed angular frequencies. The normalisation
    # includes the window energy, number of blocks and conversion from Hz to rad/s.
    bins = [j <= (nperseg-1)÷2 ? j : j-nperseg for j in 0:nperseg-1]
    order = sortperm(bins)
    omega = (2π/(nperseg*dt)) .* bins[order]
    rootweights = sqrt.(repeat(problem.weights, 3))
    scale = sqrt(dt/(2π*sum(abs2, win)*nblocks))

    energies = Matrix{Float64}(undef, rankmax, nperseg)
    modes = compute_modes ? Array{ComplexF64}(undef, Ny, 3, nr, nperseg) : nothing
    weighted = Matrix{ComplexF64}(undef, 3Ny, nblocks)
    modecolumns = compute_modes ? reshape(modes, 3Ny, nr, nperseg) : nothing

    # Refill one contiguous matrix before each destructive LAPACK factorisation.
    # Views avoid copying slices of spectra or the returned singular vectors.
    # LAPACK's Julia wrappers still allocate their outputs and internal workspace.
    for (j, k) in enumerate(order)
        @views @. weighted = rootweights * spectra[:, k, :] * scale

        # Consumers may project these realisations before LAPACK overwrites them.
        isnothing(frequency_callback) || frequency_callback(omega[j], weighted)

        if compute_modes
            F = svd!(weighted)
            @views energies[:, j] .= abs2.(F.S)
            @views modecolumns[:, :, j] .= F.U[:, 1:nr] ./ rootweights
        else
            # Energy-only analysis never constructs singular vectors or modes.
            @views energies[:, j] .= abs2.(svdvals!(weighted))
        end
    end

    return (; omega, energies, modes, nblocks)
end
