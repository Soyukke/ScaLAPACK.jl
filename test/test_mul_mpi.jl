debug = true
p = 4

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)


for eltype in [Float32, Float64, ComplexF32, ComplexF64]
    n, m = 9, 9
    if rank == 0
        println("A_mul_B! tests")
        println("eltype: $eltype")
    end

    procs_grid = (2, 2)
    alpha, beta = 1.0, 0.0

    A = SLArray(eltype, m, n, proc_grids=procs_grid)
    B = SLArray(eltype, n, m, proc_grids=procs_grid)
    C = SLArray(eltype, n, n, proc_grids=procs_grid)

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


for eltype in [Float32, Float64, ComplexF32, ComplexF64]
    n, m = 9, 9
    if rank == 0
        println("A_mul_B! tests")
        println("eltype: $eltype")
    end

    procs_grid = (2, 2)
    blocksizes = (2, 2)
    alpha, beta = 1.0, 0.0

    A = SLArray{eltype}(m, n, proc_grids=procs_grid, blocksizes=blocksizes)
    B = SLArray{eltype}(n, m, proc_grids=procs_grid, blocksizes=blocksizes)
    C = SLArray{eltype}(n, n, proc_grids=procs_grid, blocksizes=blocksizes)

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

for eltype in [Float32, Float64, ComplexF32, ComplexF64]
    n, m = 9, 9
    if rank == 0
        println("A_mul_B! tests")
        println("eltype: $eltype")
    end

    procs_grid = (2, 2)
    blocksizes = (1, 1)
    alpha, beta = 1.0, 0.0

    A = SLArray{eltype}(m, n, proc_grids=procs_grid, blocksizes=blocksizes)
    B = SLArray{eltype}(n, m, proc_grids=procs_grid, blocksizes=blocksizes)
    C = SLArray{eltype}(n, n, proc_grids=procs_grid, blocksizes=blocksizes)

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
