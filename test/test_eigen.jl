using MPI, MPIArrays, Random, Printf, ScaLAPACK
using LinearAlgebra
comm = MPI.COMM_WORLD
MPI.Init()
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

N = 50
A = SLMatrix{ComplexF64}(N, N, proc_grids=(2, 2), blocksizes=(6, 6))
forlocalpart!(x->rand!(x), A)
sync(A)
A_test = convert(Array, A)

# hessenberg -> schur
eigenvalues, eigenvectors = eigen(A)
eigenvectors_ = convert(Array, eigenvectors)

eigenvalues_test, eigenvectors_test = eigen(A_test)

indices_test = sortperm(real(eigenvalues_test))
indices = sortperm(real(eigenvalues))
eigenvalues_test = eigenvalues_test[indices_test]
eigenvalues= eigenvalues[indices]

eigenvectors_ = eigenvectors_[:, indices]
eigenvectors_test = eigenvectors_test[:, indices_test]
eigenvalues_matrix = eigenvectors_test' * A_test * eigenvectors_test

diff_eq = eigenvectors_*Diagonal(eigenvalues) - A_test*eigenvectors_
diff_eq_test = eigenvectors_test*Diagonal(eigenvalues_test) - A_test*eigenvectors_test

# show(stdout, "text/plain", eigenvalues-eigenvalues_test)
if rank == 0 
    show(stdout, "text/plain", norm(diff_eq))
    println()
    show(stdout, "text/plain", norm(diff_eq_test))
    println()
    # show(stdout, "text/plain", eigenvectors)
    println()
end

# if rank == 0
#     eigenvalues_test, _ = LinearAlgebra.eigen(A_test)
#     eigenvalues = eigenvalues[sortperm(real(eigenvalues))]
#     eigenvalues_test = eigenvalues_test[sortperm(real(eigenvalues_test))]


#     for (e1, e2) in zip(eigenvalues, eigenvalues_test)
#         @show e1 e2
#         println()
#     end
# end

MPI.Finalize()