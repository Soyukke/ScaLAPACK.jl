N = 1024
using Test
using Random
using MPI, MPIArrays
using ScaLAPACK
using LinearAlgebra

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

rank == 0 && print("test pxheevd\n")

# make Hermitian
H = MPIArray{ComplexF64}(N, N)
for i in 1:N
    for j in 1:i
        if i == j
            H[i, i] = rand()
        else
            H[i, j] = rand() + im*rand()
            H[j, i] = H[i, j]'
        end
    end
end
sync(H)
H_test = convert(Array, H)

eigenvalues, eigenvectors = eigen_hermitian(H)
if rank == 0
    # println("array H")
    # show(stdout, "text/plain", H)
    # println()

    # println("eigenvalues")
    # show(stdout, "text/plain", eigenvalues)
    # println()

    # println("eigenvectors")
    # show(stdout, "text/plain", eigenvectors)
    # println() 


    # println("LinearAlgebra.eigen")
    # show(stdout, "text/plain", eigen(H_test))
    # println() 
    # @test convert(Array, C) == alpha * A_test * B_test

    # LinearAlgebra.eigen
    eigenvalues_test, _ = eigen(H_test)
    eigenvalues_test = sort(real(eigenvalues_test))
    eigen_diff = eigenvalues - eigenvalues_test
    show(stdout, "text/plain", eigen_diff)
    println() 
    @test norm(eigen_diff) < 1e-8
end


MPI.Finalize()
