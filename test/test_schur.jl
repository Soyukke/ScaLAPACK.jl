debug = true
N = 64

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
# for elty in [ComplexF64]
    A = CyclicMPIArray(elty, N, N, proc_grids=(2, 2), blocksizes=(8, 8))
    forlocalpart!(x->rand!(x), A)
    sync(A)

    # if rank == 0 && debug
    #     show(stdout, "text/plain", A)
    #     println()
    # end

    A_test = convert(Array, A)
    eigenvalues, _ = eigen(A_test)

    hessenberg!(A)
    hessenberg!(A_test)

    W, Z = eigen_schur!(A)
    
    B = convert(Array, A)

    # if rank == 0 && debug
    #     show(stdout, "text/plain", A)
    #     println()
    #     show(stdout, "text/plain", A_test)
    #     println()
    #     show(stdout, "text/plain", B - A_test)
    #     println()
    # end

    if rank == 0 && debug
        # println("eigenvalues")
        # show(stdout, "text/plain", W)
        # println()

        # println("true eigenvalues")
        # show(stdout, "text/plain", eigenvalues)
        # println()
        for (e1, e2) in zip(W[sortperm(real(W))], eigenvalues[sortperm(real(eigenvalues))])
            println("(hess->schur_eig, LinearAlgebra.eigen) = $e1, $e2, diff=$(e1 - e2)")
        end

    end

    # @test all(abs.(B - A_test) .< 1e-5)

    free(A)
    free(Z)
end

MPI.Finalize()