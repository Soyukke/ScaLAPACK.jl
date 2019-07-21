debug = true
p = 4
n_grid, m_grid = 2, 2
n, m = 80, 80

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

debug && println("SVD tests")
debug && println("eltype: Float64")
A = MPIArray{Float64}(comm, (n_grid, m_grid), n, m)
forlocalpart!(x->rand!(x), A)
B = convert(Array, A)
@test svdvals!(A) == svdvals!(B)

# debug && println("eltype: Float32")
# A = DArray(I -> float32(randn(map(length,I))), (m, n), manager)
# B = convert(Array, A)
# @test svdvals!(A) == svdvals!(B)

# debug && println("eltype: Complex64")
# A = DArray(I -> complex64(complex(randn(map(length,I)), randn(map(length,I)))), (m, n), manager)
# B = convert(Array, A)
# @test svdvals!(A) == svdvals!(B)

# debug && println("eltype: Complex128")
# A = DArray(I -> complex(randn(map(length,I)), randn(map(length,I))), (m, n), manager)
# B = convert(Array, A)
# @test svdvals!(A) == svdvals!(B)


MPI.Finalize()