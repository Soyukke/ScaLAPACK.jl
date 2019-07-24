debug = false
# matrix size (m, n)
N = 93

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
        println("eigen_hermitian tests")
        println("eltype: $eltype")
    end

    procs_grid = (2, 2)
    blocksize = (2, 2)
    A = CyclicMPIArray(eltype, proc_grids=procs_grid, blocksizes=blocksize, N, N)
    forlocalpart!(A) do localarray
        range_i, range_j = localindices(A, rank)
        for (i, gi) in enumerate(range_i)
            for (j, gj) in enumerate(range_j)
                if gi==gj
                    localarray[i, j] = gi
                elseif eltype <: Complex
                    if gi < gj
                        localarray[i, j] = gi + im*gj
                    else
                        localarray[i, j] = gj - im*gi
                    end
                elseif eltype <: AbstractFloat
                    localarray[i, j] = gi + gj
                end
            end
        end
    end
    sync(A)
    B = convert(Array, A)

    if rank == 0 && debug
        show(stdout, "text/plain", A)
        println()
        show(stdout, "text/plain", B)
        println()
    end

    eigenvalues_A, eigenvectors = eigen_hermitian(A)
    eigenvalues_B, _ = eigen(B)
    eigenvalues_B = sort(real(eigenvalues_B))

    if rank == 0 && debug
        show(stdout, "text/plain", eigenvalues_A)
        println()
        show(stdout, "text/plain", eigenvalues_B)
        println()
    end

    diff_error = norm(eigenvalues_A - eigenvalues_B) / N
    if rank == 0
        println(diff_error)
        @test diff_error < 1e-4
        # @test all(eigenvalues_A .== typeof(real(eltype(0))).(eigenvalues_B))
    end
    free(A)
    free(eigenvectors)
end

MPI.Finalize()

