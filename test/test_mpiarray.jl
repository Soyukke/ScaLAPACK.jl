using MPI, MPIArrays

MPI.Init()
N = 8
comm = MPI.COMM_WORLD
num_process = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

n_grid = sqrt(num_process)
@assert n_grid == round(n_grid)
n_grid = round(Int64, n_grid)

A = MPIArray{Float64}(comm, (n_grid, n_grid), N, N)
forlocalpart!(x->fill!(x, rank), A)
sync(A)
if rank == 0
    show(stdout, "text/plain", A)
    println()
    println("size A: $(size(A))")
end

# clean up
free(A)
MPI.Finalize()