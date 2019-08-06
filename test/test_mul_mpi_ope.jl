debug = true
p = 4

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)
nprow = Int(sqrt(nproc))


for eltype in [Float32, Float64, ComplexF32, ComplexF64]
    n, m = 9, 9
    # if rank == 0
    #     println("A_mul_B! tests")
    #     println("eltype: $eltype")
    # end
    
    procs_grid = (nprow, nprow)

    A = SLMatrix{eltype}(m, n, proc_grids=procs_grid, blocksizes=(2, 2))
    B = SLMatrix{eltype}(n, m, proc_grids=procs_grid, blocksizes=(2, 2))

    forlocalpart!(x->fill!(x, rank), A)
    forlocalpart!(x->fill!(x, rank), B)
    sync(A, B)

    # if comment out under 4 line , this script is not work
    A2 = convert(Array, A)
    MPI.Barrier(comm)
    B2 = convert(Array, B)
    MPI.Barrier(comm)

    # A2 = convert(Array, A)
    # MPI.Barrier(comm)
    # B2 = convert(Array, B)
    # MPI.Barrier(comm)
    # print(A2)

    C = A * B * B * A * B'
    print("ok\n")

    # if comment out under 4 line, this script is work

    C2 = convert(Array, C)
    MPI.Barrier(comm)
    print("ok2\n")

    # if rank == 0
        println("array A")
        show(stdout, "text/plain", A2)
        println()

        println("array B")
        show(stdout, "text/plain", B2)
        println()

        println("A_mul_B! C = A * B")
        show(stdout, "text/plain", C2)
        println()

        println("C - C2")
        show(stdout, "text/plain", C2 - A2*B2*B2*A2)
        println()
    # end
    # free(A, B, C)
end




MPI.Finalize()
