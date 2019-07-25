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

include("slarrays.jl")
include("blacs.jl")
include("pblas.jl")
include("scalapackWrappers.jl")
include("convenience_mpi.jl")
include("linalg.jl")
include("hessenberg.jl")
include("schur.jl")

end # module
