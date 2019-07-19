debug = true

p = 4
m, n = 80, 80

using Test
using MPI
using DistributedArrays
using Distributed

MPI.Init()
manager = MPI.start_main_loop(MPI_TRANSPORT_ALL)
println("MPI.workers: $(MPI.workers())")

@everywhere using ScaLAPACK

debug && println("SVD tests")
debug && println("eltype: Float64")
A = DArray(I -> randn(map(length,I)), (m, n), manager)
B = convert(Array, A)
@test svdvals!(A) == svdvals!(B)

debug && println("eltype: Float32")
A = DArray(I -> float32(randn(map(length,I))), (m, n), manager)
B = convert(Array, A)
@test svdvals!(A) == svdvals!(B)

debug && println("eltype: Complex64")
A = DArray(I -> complex64(complex(randn(map(length,I)), randn(map(length,I)))), (m, n), manager)
B = convert(Array, A)
@test svdvals!(A) == svdvals!(B)

debug && println("eltype: Complex128")
A = DArray(I -> complex(randn(map(length,I)), randn(map(length,I))), (m, n), manager)
B = convert(Array, A)
@test svdvals!(A) == svdvals!(B)

MPI.stop_main_loop(manager)