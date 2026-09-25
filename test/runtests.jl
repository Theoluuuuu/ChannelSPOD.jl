using Test, LinearAlgebra, ChannelSPOD

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
