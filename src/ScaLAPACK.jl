module ScaLAPACK


using Compat
using LinearAlgebra: BlasFloat, BlasReal
using MPI
using MPIArrays

struct ScaLAPACKException <: Exception
    info::Int32
end

moduledir = pathof(ScaLAPACK) |> dirname |> dirname
const libscalapack = joinpath(moduledir, "deps", "libscalapack.so")

include("blacs.jl")
include("scalapackWrappers.jl")
include("convenience_mpi.jl")

end # module
