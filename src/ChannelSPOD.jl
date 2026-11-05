module ChannelSPOD

using LinearAlgebra, FFTW, ChebUtils

export SPODProblem, PipeSPODProblem, analyse

include("spod.jl")
include("pipe_spod.jl")

end
