debug = true
N = 9

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK
using LinearAlgebra

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

for elty in (Float32, Float64, ComplexF32, ComplexF64)
    A = CyclicMPIArray(elty, N, N)
    forlocalpart!(x->rand!(x), A)
    sync(A)

    A_test = convert(Array, A)

    if rank == 0 && debug
        show(stdout, "text/plain", A)
        println()
    end

    hessenberg!(A)
    hessenberg!(A_test)
    B = convert(Array, A)

    if rank == 0 && debug
        show(stdout, "text/plain", A)
        println()
        show(stdout, "text/plain", A_test)
        println()
        show(stdout, "text/plain", B - A_test)
        println()
    end

    @test all(abs.(B - A_test) .< 1e-5)

    free(A)
end

MPI.Finalize()