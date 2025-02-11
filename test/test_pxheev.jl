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

    procs_grid = Int.((sqrt(n_proc), sqrt(n_proc)))
    A = SLArray(eltype, proc_grids=procs_grid, N, N)
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

    eigenvalues_A, eigenvectors = eigen_hermitian(A)

    if rank == 0
        eigenvalues_B, _ = eigen(B)
        eigenvalues_B = sort(real(eigenvalues_B))
        diff_error = norm(eigenvalues_A - eigenvalues_B) / N

        show(stdout, "text/plain", eigenvalues_A)
        println()
        show(stdout, "text/plain", eigenvalues_B)
        println()

        println(diff_error)
        @test diff_error < 1e-4
        # @test all(eigenvalues_A .== typeof(real(eltype(0))).(eigenvalues_B))
    end
end

MPI.Finalize()

