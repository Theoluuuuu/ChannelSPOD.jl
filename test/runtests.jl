using Test, LinearAlgebra, Random, ChannelSPOD

@testset "Channel SPOD" begin
    Ny=9; L=32; Nt=4L; dt=0.1
    p=SPODProblem(Ny); y=cospi.((0:Ny-1)/(Ny-1))
    q=1 .- y.^2; q ./= sqrt(sum(p.weights.*abs2.(q)))
    ω=2π*3/(L*dt)
    data=zeros(ComplexF64,Ny,3,Nt)
    for t in 1:Nt
        data[:,1,t]=q.*exp(-im*ω*(t-1)*dt)
    end
    saved=copy(data)
    # A rectangular window and bin-centred travelling signal isolate its signed frequency.
    r=analyse(p,data,dt;nperseg=L,noverlap=0,nmodes=2,window=ones(L),demean=false)
    @test data==saved
    @test size(r.modes)==(Ny,3,2,L)
    peak=argmax(r.energies[1,:]); @test r.omega[peak] ≈ ω
    weights=repeat(p.weights,3)
    mode=vec(r.modes[:,:,1,peak])
    @test real(dot(mode,weights.*mode)) ≈ 1
    @test abs(dot(q,p.weights.*r.modes[:,1,1,peak])) ≈ 1
    @test sum(r.energies)*(2π/(L*dt)) ≈ 1
    @test r.energies[2,peak] < 1e-20
    # Hann-window normalisation preserves the energy of a constant-amplitude harmonic.
    hann=analyse(p,data,dt;nperseg=L,demean=false)
    @test sum(hann.energies)*(2π/(L*dt)) ≈ 1
    centered=analyse(p,ones(Ny,3,Nt),dt;nperseg=L)
    @test iszero(sum(centered.energies))
    negative=analyse(p,conj.(data),dt;nperseg=L,window=ones(L),demean=false)
    @test negative.omega[argmax(negative.energies[1,:])] ≈ -ω
    # Odd block lengths retain both signed end bins.
    odd=analyse(p,data,dt;nperseg=31)
    @test length(odd.omega)==31 && issorted(odd.omega)
    @test_throws ArgumentError analyse(p,data,0;nperseg=L)
    @test_throws ArgumentError analyse(p,data,dt;nperseg=L,noverlap=L)
    @test_throws DimensionMismatch analyse(p,data[:,1:2,:],dt;nperseg=L)
end

@testset "Full energy spectrum without mode storage" begin
    p = SPODProblem(5)
    # Multiple independent temporal signals ensure discarded eigenvalues carry energy.
    data = reshape(sin.(1:5*3*128) .+ cos.(sqrt(2) .* (1:5*3*128).^2), 5, 3, 128)
    full = analyse(p, data, 0.1; nperseg=32, nmodes=7)
    truncated = analyse(p, data, 0.1; nperseg=32, nmodes=1)
    energyonly = analyse(p, data, 0.1; nperseg=32, compute_modes=false)
    @test size(truncated.modes, 3) == 1
    @test size(truncated.energies) == (7, 32)
    @test truncated.energies ≈ full.energies
    @test energyonly.modes === nothing
    @test energyonly.omega == full.omega
    @test energyonly.energies ≈ full.energies
    @test sum(full.energies[2:end, :]) > 0
end

@testset "Frequency callback and input views" begin
    p = SPODProblem(5)
    backing = reshape(complex.(sin.(1:5*3*2*128)), 5, 3, 2, 128)
    data = view(backing, :, :, 1, :)
    saved = copy(backing)
    frequencies, traces = Float64[], Float64[]
    callback = (omega, X) -> begin
        push!(frequencies, omega)
        push!(traces, sum(abs2, X))
    end
    result = analyse(p, data, 0.1; nperseg=32, compute_modes=false,
                     frequency_callback=callback)
    # The weighted CSD trace equals the sum of every eigenvalue, independently
    # of whether modes are constructed. A view must leave its parent unchanged.
    @test frequencies == result.omega
    @test traces ≈ vec(sum(result.energies; dims=1))
    @test backing == saved
end

# Pipe tests: velocities (u_r, u_θ, u_z) on the midpoint radial grid, standing for
# a DNS grid and its weights of ∫ f r dr, in the norm ∫ uᴴu r dr of the pipe resolvent.
midpoint(Nr) = (r = ((1:Nr) .- 0.5) ./ Nr; (r, r ./ Nr))

@testset "Pipe: travelling mode convention in the r dr norm" begin
    Nr=16; L=32; Nt=4L; dt=0.1
    r, ws = midpoint(Nr)
    p=PipeSPODProblem(ws); W=Diagonal(repeat(ws,3))
    @test p.weights == ws
    # A k_θ = 1 profile, regular at the axis and no-slip at the wall, of unit ∫ |u|² r dr.
    phi=[1 .- r.^2; im .* (1 .- r.^2); r .* (1 .- r.^2)]
    phi ./= sqrt(real(dot(phi, W*phi)))
    ω=2π*3/(L*dt)
    data=zeros(ComplexF64,Nr,3,Nt)
    for t in 1:Nt
        data[:,:,t]=reshape(phi.*exp(-im*ω*(t-1)*dt), Nr, 3)
    end
    saved=copy(data)
    # exp(i(k_θθ + αk_z z - ωt)) must appear at +ω, the report's ω_k.
    res=analyse(p,data,dt;nperseg=L,noverlap=0,window=ones(L),demean=false)
    @test data==saved
    peak=argmax(res.energies[1,:]); @test res.omega[peak] ≈ ω
    psi=vec(res.modes[:,:,1,peak])
    @test real(dot(psi, W*psi)) ≈ 1
    @test abs(dot(phi, W*psi)) ≈ 1
    # A unit-norm harmonic carries unit energy, all of it in one mode and one bin.
    @test res.energies[1,peak]*(2π/(L*dt)) ≈ 1
    @test sum(res.energies)*(2π/(L*dt)) ≈ 1
    @test res.energies[2,peak] < 1e-20
    hann=analyse(p,data,dt;nperseg=L,demean=false)
    @test sum(hann.energies)*(2π/(L*dt)) ≈ 1
    backward=analyse(p,conj.(data),dt;nperseg=L,noverlap=0,window=ones(L),demean=false)
    @test backward.omega[argmax(backward.energies[1,:])] ≈ -ω
end

@testset "Pipe: uncorrelated structures are separated in the r dr norm" begin
    Nr=16; L=32; B=4; Nt=B*L; dt=0.1
    r, ws = midpoint(Nr)
    p=PipeSPODProblem(ws); W=Diagonal(repeat(ws,3))
    # Two k_θ = 1 structures, orthonormal in ∫ uᴴv r dr but not in the plain
    # Euclidean product, so only the r dr weights can return both as SPOD modes.
    phi1=[1 .- r.^2; im .* (1 .- r.^2); r .* (1 .- r.^2)]
    phi1 ./= sqrt(real(dot(phi1, W*phi1)))
    phi2=ComplexF64[(1 .- r.^2).^2; (1 .- r.^2) .* (1 .- 3r.^2); r.^3 .* (1 .- r.^2)]
    phi2 .-= phi1 .* dot(phi1, W*phi2)
    phi2 ./= sqrt(real(dot(phi2, W*phi2)))
    @test abs(dot(phi1, W*phi2)) < 1e-12
    @test abs(dot(phi1, phi2)) > 0.1
    # Same frequency, amplitudes 2 and 1, phases uncorrelated across the blocks:
    # in block b, phi2 is shifted by 2πb/B and Σ_b exp(2πib/B) = 0.
    A1, A2 = 2.0, 1.0
    ω=2π*3/(L*dt)
    data=zeros(ComplexF64,Nr,3,Nt)
    for b in 0:B-1, n in 1:L
        t=b*L+n
        data[:,:,t]=reshape((A1 .* phi1 .+ A2*exp(2π*im*b/B) .* phi2) .* exp(-im*ω*(t-1)*dt), Nr, 3)
    end
    res=analyse(p,data,dt;nperseg=L,noverlap=0,nmodes=2,window=ones(L),demean=false)
    Δω=2π/(L*dt)
    peak=argmax(res.energies[1,:]); @test res.omega[peak] ≈ ω
    # Energies are the structures' ∫ |u|² r dr and the modes are the structures.
    @test res.energies[1,peak]*Δω ≈ A1^2
    @test res.energies[2,peak]*Δω ≈ A2^2
    @test res.energies[3,peak]*Δω < 1e-20
    psi1=vec(res.modes[:,:,1,peak]); psi2=vec(res.modes[:,:,2,peak])
    @test abs(dot(phi1, W*psi1)) ≈ 1
    @test abs(dot(phi2, W*psi2)) ≈ 1
    @test abs(dot(psi1, W*psi2)) < 1e-12
end

@testset "Pipe: response to white forcing gives resolvent gains and response modes" begin
    Nr=8; L=8; dt=0.1
    r, ws = midpoint(Nr)
    p=PipeSPODProblem(ws); W=Diagonal(repeat(ws,3))
    # H stands for the pipe resolvent at one (k_θ, k_z, ω): a non-normal operator
    # whose gains and response modes are those of its SVD in the r dr norm, as in
    # Resolvent.jl's svd(H, ws, M): W^{1/2} H W^{-1/2} = U Σ V^H, ψ = W^{-1/2} U.
    rng=Xoshiro(1)
    H=randn(rng, ComplexF64, 3Nr, 3Nr)
    F=svd(sqrt(W)*H/sqrt(W))
    sigma=F.S; Psi=sqrt(W)\F.U
    # B = 3Nr forcing realisations orthonormal in the r dr norm make the forcing
    # cross-spectral density white in that norm, Σ_b f_b f_bᴴ = W^{-1}.
    B=3Nr; Nt=B*L
    Z=Matrix(qr(randn(rng, ComplexF64, 3Nr, 3Nr)).Q)
    forcing=sqrt(W)\Z
    @test forcing'*W*forcing ≈ I
    ω=2π*2/(L*dt)
    data=zeros(ComplexF64,Nr,3,Nt)
    for b in 1:B, n in 1:L
        t=(b-1)*L+n
        data[:,:,t]=reshape(H*forcing[:,b] .* exp(-im*ω*(t-1)*dt), Nr, 3)
    end
    # Project every frequency onto the resolvent response modes before the SVD.
    frequencies=Float64[]; projected=Vector{Float64}[]
    callback=(omega, X) -> begin
        push!(frequencies, omega)
        push!(projected, vec(sum(abs2, (sqrt.(diag(W)) .* Psi)' * X; dims=2)))
    end
    res=analyse(p,data,dt;nperseg=L,noverlap=0,nmodes=3,window=ones(L),demean=false,
                frequency_callback=callback)
    Δω=2π/(L*dt)
    peak=argmax(res.energies[1,:]); @test res.omega[peak] ≈ ω
    # SPOD energies are the squared gains, per unit forcing energy per realisation.
    @test res.energies[:,peak]*Δω*B ≈ sigma.^2
    for j in 1:3
        @test abs(dot(Psi[:,j], W*vec(res.modes[:,:,j,peak]))) ≈ 1
    end
    # Each response mode carries exactly its SPOD energy, and no other bin has any.
    @test frequencies == res.omega
    @test projected[peak] ≈ res.energies[:,peak]
    @test all(maximum(projected[j]) < 1e-20*maximum(projected[peak]) for j in eachindex(projected) if j != peak)
end

@testset "Pipe: mean profile removal and energy budget at (k_θ, k_z) = (0, 0)" begin
    Nr=16; L=32; Nt=8L; dt=0.1
    r, ws = midpoint(Nr)
    p=PipeSPODProblem(ws); W=Diagonal(repeat(ws,3))
    # Hagen–Poiseuille profile U = 2(1 - r²) e_z in bulk units, plus a real axial
    # and swirl fluctuation with an integer number of periods over the record.
    U=[zeros(Nr); zeros(Nr); 2 .* (1 .- r.^2)]
    ω1=2π*5/(Nt*dt); ω2=2π*13/(Nt*dt)
    fluctuation=zeros(Nr,3,Nt); total=zeros(Nr,3,Nt)
    for t in 1:Nt
        fluctuation[:,2,t]=r .* (1 .- r.^2) .* sin(ω2*(t-1)*dt)
        fluctuation[:,3,t]=(1 .- r.^2) .* (1 .- 2r.^2) .* cos(ω1*(t-1)*dt)
        total[:,:,t]=reshape(U, Nr, 3) .+ fluctuation[:,:,t]
    end
    # Demeaning removes the mean profile and nothing else.
    centred=analyse(p,total,dt;nperseg=L,compute_modes=false)
    alone=analyse(p,fluctuation,dt;nperseg=L,compute_modes=false,demean=false)
    @test centred.energies ≈ alone.energies
    # The spectrum integrates to the window-weighted block average of ∫ |u|² r dr,
    # with and without the mean profile.
    raw=analyse(p,total,dt;nperseg=L,compute_modes=false,demean=false)
    win=0.5 .- 0.5cos.(2π .* (0:L-1) ./ L)
    starts=1:(L-L÷2):(Nt-L+1)
    budget_fluctuation=0.0; budget_total=0.0
    for s in starts, n in 1:L
        q=vec(fluctuation[:,:,s+n-1]); qt=vec(total[:,:,s+n-1])
        budget_fluctuation += win[n]^2*real(dot(q, W*q))
        budget_total += win[n]^2*real(dot(qt, W*qt))
    end
    budget_fluctuation /= length(starts)*sum(abs2, win)
    budget_total /= length(starts)*sum(abs2, win)
    Δω=2π/(L*dt)
    @test sum(centred.energies)*Δω ≈ budget_fluctuation
    @test sum(raw.energies)*Δω ≈ budget_total
    @test budget_total > 10*budget_fluctuation
end

@testset "Pipe: invalid inputs" begin
    @test_throws ArgumentError PipeSPODProblem([0.5])
    @test_throws ArgumentError PipeSPODProblem([0.5, -0.1])
    @test_throws ArgumentError PipeSPODProblem([0.5, NaN])
    p=PipeSPODProblem(midpoint(8)[2]); data=zeros(8,3,64)
    @test_throws DimensionMismatch analyse(p,data[1:end-1,:,:],0.1;nperseg=16)
    @test_throws DimensionMismatch analyse(p,data[:,1:2,:],0.1;nperseg=16)
    @test_throws ArgumentError analyse(p,data,0.1;nperseg=16,noverlap=0,nmodes=5)
    @test_throws ArgumentError analyse(p,data,0;nperseg=16)
end
