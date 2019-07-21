module ScaLAPACK

export svdvals!, A_mul_B!

using Compat
using LinearAlgebra: BlasFloat, BlasReal
using MPI, Distributed, DistributedArrays
using MPIArrays

import DistributedArrays: DArray, defaultdist
# this should only be a temporary solution until procs returns a type that encodes more information about the processes
DArray(init, dims, manager::MPIManager, args...) = DArray(init, dims, collect(values(manager.mpi2j))[sortperm(collect(keys(manager.mpi2j)))], args...)
function defaultdist(sz::Int, nc::Int)
    if sz >= nc
        d, r = divrem(sz, nc)
        if r == 0
            return vcat(1:d:sz+1)
        end
        return vcat(vcat(1:d+1:sz+1), [sz+1])
    else
        return vcat(vcat(1:(sz+1)), zeros(Int, nc-sz))
    end
end

struct ScaLAPACKException <: Exception
    info::Int32
end

moduledir = pathof(ScaLAPACK) |> dirname |> dirname
const libscalapack = joinpath(moduledir, "deps", "libscalapack.so")

include("blacs.jl")
include("scalapackWrappers.jl")
include("convenience.jl")
include("convinience_mpi.jl")

end # module
