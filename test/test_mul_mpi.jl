debug = true
p = 4
n, m = 8, 8

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)


for eltype in [Float32, Float64, ComplexF32, ComplexF64]
    if rank == 0
        println("A_mul_B! tests")
        println("eltype: $eltype")
    end

    procs_grid = (2, 2)
    blocksize = (2, 2)
    alpha, beta = 1.0, 0.0

    A = CyclicMPIArray(eltype, proc_grids=procs_grid, blocksizes=blocksize, m, n)
    B = CyclicMPIArray(eltype, proc_grids=procs_grid, blocksizes=blocksize, m, n)
    C = CyclicMPIArray(eltype, proc_grids=procs_grid, blocksizes=blocksize, m, n)

    forlocalpart!(x->fill!(x, rank), A)
    forlocalpart!(x->fill!(x, rank), B)
    sync(A, B)

    A_test = convert(Array, A)
    B_test = convert(Array, B)

    A_mul_B!(eltype(alpha), A, B, eltype(beta), C)

    if rank == 0
        println("array A")
        show(stdout, "text/plain", A)
        println()

        println("array B")
        show(stdout, "text/plain", B)
        println()

        println("A_mul_B! C = A * B")
        show(stdout, "text/plain", C)
        println()

        println("LinearAlgebra C = A * B")
        show(stdout, "text/plain", A_test*B_test)
        println()
    end

    @test convert(Array, C) == alpha * A_test * B_test
    free(A); free(B); free(C)
end




MPI.Finalize()
