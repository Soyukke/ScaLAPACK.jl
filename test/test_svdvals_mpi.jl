debug = false
# matrix size (m, n)
m, n = 7, 9

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK
using LinearAlgebra

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

for eltype in [Float32, Float64, ComplexF32, ComplexF64]
    if rank == 0
        println("SVD tests")
        println("eltype: $eltype")
    end
    procs_grid = (2, 2)
    blocksizes = (2, 2)
    A = SLMatrix{eltype}(m, n, proc_grids=procs_grid, blocksizes=blocksizes)
    # A = MPIArray{eltype}(comm, (2, 2), m, n)
    forlocalpart!(A) do lA
        m_local, n_local = size(lA)
        for i in 1:m_local, j in 1:n_local
            if eltype <: Complex
                lA[i, j] = eltype(rand() + im*rand())
            else
                lA[i, j] = eltype(rand())
            end
        end
    end
    sync(A)
    B = rma!(A) do
        convert(Array, A)
    end

    rma!(A) do
        if rank == 0 && debug
            show(stdout, "text/plain", A)
            println()
            show(stdout, "text/plain", B)
            println()
        end
    end

    diff_svdvals = norm(svdvals!(A) - svdvals!(B))
    if rank == 0
        println(diff_svdvals)
        @test diff_svdvals < 1e-5
    end
    free(A)
end

MPI.Finalize()
