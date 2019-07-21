debug = true
p = 4
n_grid, m_grid = 2, 2
n, m = 8, 8

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)


alpha, beta = 1.0, 1.0

A = MPIArray{Float64}(comm, (n_grid, m_grid), n, m)
B = MPIArray{Float64}(comm, (n_grid, m_grid), n, m)
C = MPIArray{Float64}(comm, (n_grid, m_grid), n, m)

forlocalpart!(x->fill!(x, rank), A)
forlocalpart!(x->fill!(x, rank), B)
sync(A, B)

A_mul_B!(alpha, A, B, beta, C)

if rank == 0
    println("A_mul_B! is worked")
    show(stdout, "text/plain", C)
end

MPI.Finalize()