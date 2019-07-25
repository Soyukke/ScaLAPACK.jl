debug = true
N = 73

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
    A = SLArray(elty, N, N, proc_grids=(2, 2))
    forlocalpart!(x->rand!(x), A)
    sync(A)

    A_test = convert(Array, A)
    eigenvalues, _ = eigen(A_test)

    hessenberg!(A)
    hessenberg!(A_test)

    W, Z = eigen_schur!(A)
    
    B = convert(Array, A)

    if rank == 0 && debug
        for (e1, e2) in zip(W[sortperm(real(W))], eigenvalues[sortperm(real(eigenvalues))])
            println("(hess->schur_eig, LinearAlgebra.eigen) = $e1, $e2, diff=$(e1 - e2)")
        end
    end
    # @test all(abs.(B - A_test) .< 1e-5)
    free(A, Z)
end

MPI.Finalize()