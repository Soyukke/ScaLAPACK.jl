using MPI, MPIArrays, Random, Printf, ScaLAPACK
using LinearAlgebra
comm = MPI.COMM_WORLD
MPI.Init()
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

N = 12
A = SLMatrix{ComplexF64}(N, N, proc_grids=(2, 2), blocksizes=(6, 6))
forlocalpart!(x->rand!(x), A)
sync(A)
A_test = convert(Array, A)

# hessenberg -> schur
eigenvalues = eigen(A)
eigenvalues_test, _ = LinearAlgebra.eigen(A_test)

if rank == 0
    eigenvalues = eigenvalues[sortperm(real(eigenvalues))]
    eigenvalues_test = eigenvalues_test[sortperm(real(eigenvalues_test))]


    for (e1, e2) in zip(eigenvalues, eigenvalues_test)
        @show e1 e2
        println()
    end
end

MPI.Finalize()