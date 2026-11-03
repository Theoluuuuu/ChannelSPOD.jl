# Definition of the pipe-flow SPOD, in the structure of the channel `SPODProblem`
# and `analyse`: the same Welch estimator, written out for the radial grid and
# the inner product of the pipe.
#
# Velocities stack (u_r, u_θ, u_z) on the Nr HalfChebyshev points of
# FDGrids.grid(Nr, 1.0, HalfChebyshevGrid()) (wall at r[end] = 1, no point on the
# axis), the grid of PipeResolventModes' resolvent_grid(Nr). The inner product is
#
#     ⟨u, v⟩ = ∫₀¹ (u_r* v_r + u_θ* v_θ + u_z* v_z) r dr ≈ uᴴ W v,
#
# without the factor 2 of an area average, so that SPOD modes are orthonormal
# in the norm of the pipe resolvent response modes (9-month report, Section
# 3.3.3) and can be projected onto them through `frequency_callback`.
#
# Sign convention: that of the report, a mode exp(i(k_θθ + αk_z z - ω_k t)), so
# a signal exp(-iω_k t) peaks at the positive returned `omega` = ω_k. Resolvent.jl's
# `PipeResolvent` takes exp(+iΩt), so its Ω is -omega.

"""
    PipeSPODProblem(Nr)

Define the velocity inner product `⟨u,v⟩ = ∫₀¹ uᴴv r dr` on the `Nr` points of
`FDGrids.grid(Nr, 1.0, HalfChebyshevGrid())`, ordered from the axis to the wall
at `r = 1`, summed over the three velocity components `(u_r, u_θ, u_z)`. The
input profiles are nodal values.
"""
struct PipeSPODProblem
    # Scalar-profile weights of ∫ f r dr; repeated for u_r, u_θ and u_z during analysis.
    weights::Vector{Float64}

    function PipeSPODProblem(Nr::Integer)
        Nr >= 2 || throw(ArgumentError("at least two radial points are required"))
        return new(FDGrids.grid(Int(Nr), 1.0, FDGrids.HalfChebyshevGrid()).ws)
    end
end

raw"""
    analyse(problem::PipeSPODProblem, velocity, dt; nperseg, noverlap=nperseg÷2,
            nmodes=nothing, compute_modes=true, demean=true, window=nothing,
            frequency_callback=nothing)

Compute Welch SPOD of one spatial Fourier pair ``(k_θ, k_z)`` of pipe flow from
uniformly sampled nodal velocities of shape `(Nr, 3, Nt)`, components
``(u_r, u_θ, u_z)``. Real or complex input is accepted and preserved.

# Mathematical definition

Let ``q_n ∈ ℂ^{3N_r}`` contain the three velocity profiles at time ``nΔt``.
For `demean=true`, subtract the full-record mean ``μ = N_t^{-1}∑_n q_n``, which
at ``(k_θ, k_z) = (0, 0)`` removes the mean profile; otherwise use ``μ=0``.

Split the record into ``B`` overlapping blocks of length ``L = nperseg`` and
stride ``L-noverlap``, discarding an incomplete final block. For block start
``s_b`` and real window ``w_n``, define the unnormalised Fourier realisation

```math
\hat q_b(ω_k) = \sum_{n=0}^{L-1} w_n (q_{s_b+n}-μ) e^{+2πikn/L},
\qquad ω_k = \frac{2πk}{LΔt},
```

so a mode ``e^{i(k_θθ + αk_z z - ω t)}`` appears at positive ``ω``. Frequencies
are returned in increasing signed order, with the even-length Nyquist bin on
the negative side. The default window is periodic Hann.

Let ``W`` be the diagonal matrix of the weights of ``∫ f r dr``, repeated for
the three components, and ``Q_k`` collect the ``B`` Fourier realisations as
columns. The cross-spectral density estimate and its weighted factor are

```math
c = \frac{Δt}{2πB\sum_n w_n^2},\qquad
S_k = c Q_k Q_k^H,\qquad X_k=\sqrt{c}\,W^{1/2}Q_k.
```

The thin SVD ``X_k=U_kΣ_kV_k^H`` gives energies ``λ_j=Σ_{jj}^2`` and velocity
modes ``ψ_j=W^{-1/2}U_{k,j}``, so ``S_kWψ_j=λ_jψ_j`` and
``\int_0^1 ψ_i^Hψ_j\,r\,dr ≈ ψ_i^HWψ_j=δ_{ij}``, the norm of the pipe resolvent
response modes. Energies are two-sided densities per unit angular frequency,
and with ``Δω=2π/(LΔt)``

```math
\sum_{k,j}λ_j(ω_k)Δω =
\frac{\sum_{b,n} w_n^2 (q_{s_b+n}-μ)^H W (q_{s_b+n}-μ)}
     {B\sum_n w_n^2}.
```

An optional `frequency_callback(omega, X)` is called for each frequency before
factorisation, with the weighted matrix ``X_k``. Treat it as read-only and do
not retain it. For response modes `Psi` with `Psi' * W * Psi = I`, the rows of
`(sqrt.(diag(W)) .* Psi)' * X` are their projected realisations.

# Return values

Return `(omega, energies, modes, nblocks)`:
- `omega`: `nperseg` signed angular frequencies ``ω_k``.
- `energies`: `(min(3Nr, nblocks), nperseg)`, every eigenvalue supported by
  the sample rank, independently of `nmodes`.
- `modes`: `(Nr, 3, nr, nperseg)`, where `nr=nmodes` or at most three by default.
  `compute_modes=false` returns `nothing` and ignores `nmodes`.
- `nblocks`: number of complete Welch blocks.
"""
function analyse(problem::PipeSPODProblem, velocity::AbstractArray{<:Number,3}, dt::Real;
                 nperseg::Integer, noverlap::Integer=nperseg÷2,
                 nmodes::Union{Nothing,Integer}=nothing, compute_modes::Bool=true,
                 demean::Bool=true, window=nothing, frequency_callback=nothing)

    # Validate shape, sample spacing and Welch settings before allocating work.
    Nr, nc, Nt = size(velocity)
    Nr == length(problem.weights) && nc == 3 ||
        throw(DimensionMismatch("velocity must have shape (Nr, 3, Nt)"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    2 <= nperseg <= Nt || throw(ArgumentError("require 2 ≤ nperseg ≤ Nt"))
    0 <= noverlap < nperseg || throw(ArgumentError("invalid overlap"))
    all(isfinite, velocity) || throw(ArgumentError("velocity must be finite"))

    win = isnothing(window) ? 0.5 .- 0.5cos.(2π .* (0:nperseg-1) ./ nperseg) : Float64.(window)
    length(win) == nperseg && all(isfinite, win) && sum(abs2, win) > 0 ||
        throw(ArgumentError("window must be finite, nonzero and have length nperseg"))
    starts = 1:(nperseg-noverlap):(Nt-nperseg+1)
    nblocks = length(starts)
    rankmax = min(3Nr, nblocks)
    nr = !compute_modes || isnothing(nmodes) ? min(3, rankmax) : nmodes
    1 <= nr <= rankmax || throw(ArgumentError("nmodes must lie in 1:min(3Nr, nblocks)"))

    # View the input as (u_r; u_θ; u_z) rows and temporal columns, without copying it.
    # Demeaning is fused with windowing below; only the mean profile is stored.
    data = reshape(velocity, 3Nr, Nt)
    meanprofile = zeros(ComplexF64, 3Nr)
    if demean
        for t in axes(data, 2)
            @views meanprofile .+= data[:, t]
        end
        meanprofile ./= Nt
    end

    # Reuse one contiguous block and one in-place FFT plan for all realisations.
    # The full transform ensemble is retained, O(3Nr * L * B).
    block = Matrix{ComplexF64}(undef, 3Nr, nperseg)
    plan = plan_bfft!(block, 2; flags=FFTW.ESTIMATE)
    spectra = Array{ComplexF64}(undef, 3Nr, nperseg, nblocks)

    for (b, start) in enumerate(starts)
        @views block .= (data[:, start:start+nperseg-1] .- meanprofile) .* transpose(win)
        plan * block
        @views spectra[:, :, b] .= block
    end

    # Convert FFT bin order to signed angular frequencies ω_k. The normalisation
    # includes the window energy, number of blocks and conversion from Hz to rad/s.
    bins = [j <= (nperseg-1)÷2 ? j : j-nperseg for j in 0:nperseg-1]
    order = sortperm(bins)
    omega = (2π/(nperseg*dt)) .* bins[order]
    # Square roots of the weights of ∫ f r dr, repeated for u_r, u_θ and u_z.
    rootweights = sqrt.(repeat(problem.weights, 3))
    scale = sqrt(dt/(2π*sum(abs2, win)*nblocks))

    energies = Matrix{Float64}(undef, rankmax, nperseg)
    modes = compute_modes ? Array{ComplexF64}(undef, Nr, 3, nr, nperseg) : nothing
    weighted = Matrix{ComplexF64}(undef, 3Nr, nblocks)
    modecolumns = compute_modes ? reshape(modes, 3Nr, nr, nperseg) : nothing

    # Refill one contiguous matrix before each destructive LAPACK factorisation.
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
