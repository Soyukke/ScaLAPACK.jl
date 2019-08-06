using Test
using Random
using MPI, MPIArrays
using BenchmarkTools
using ScaLAPACK
using Dates
using LinearAlgebra
using Printf

using Distributed
LinearAlgebra.BLAS.set_num_threads(1)

macro root_time(ex)
    quote
        local n_loop = 1
        local time_start = Dates.now()
        for i in 1:n_loop
            $(esc(ex))
        end
        local time_stop = Dates.now()
        teval = (time_stop - time_start).value  / n_loop
        if MPI.Comm_rank(comm) == 0
            println(teval, " msec")
        end
        MPI.Barrier(comm)
        teval
    end
end


comm = MPI.COMM_WORLD
MPI.Init()
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)
np_row = Int(sqrt(nproc))
np_col = Int(sqrt(nproc))

print("host: $(gethostname())\n")
MPI.Barrier(comm)
# rank == 0 && (f = open("bench.dat", "w"))

for N in 2 .^ (5:6)
for elty in (Float32, Float64, ComplexF32, ComplexF64)
    proc_grid = (np_row, np_col)
    blocksize = floor.(Int, N ./ proc_grid)
    rank == 0 && print("TYPE = $elty, N = $N\n")
    MPI.Barrier(comm)
    A = SLMatrix{elty}(N, N, proc_grids=proc_grid, blocksizes=blocksize)
    forlocalpart!(x->rand!(x), A)
    sync(A)
    time_ = @root_time eigen(A)
    print("ok\n")
    if rank == 0
        # println("ok in rank 0")
        show(stdout, "text/plain", A)
        println()
    end
    # if rank == 0
    #     @printf(f, "N:%d, type:%20s, time: %.4f msec\n", N, eltype, time_)
    # end
    # MPI.Barrier(comm)
    free(A)
end
end

MPI.Finalize()
