using Random
using MPI, MPIArrays
using ScaLAPACK
using LinearAlgebra
using Dates
using Printf
using Distributed

print("hostname: $(gethostname())\n")

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

function bench_pxheevd(N::Integer)
    rank == 0 && print("bench pxheevd ComplexF64 in procs: $n_proc \n")
    # make Hermitian
    H = MPIArray{ComplexF64}(N, N)
    if rank == 0
        for i in 1:N, j in 1:i
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
    time_start = Dates.now()
    eigenvalues, eigenvectors = eigen_hermitian(H)
    time_stop = Dates.now()
    n_elem = N^2
    time_msec = time_stop - time_start
    if rank == 0
        println("Hermitian Matrix ($N, $N), n_elem: $n_elem , time $time_msec", )
    end
    free(H)
    free(eigenvectors)
    return n_proc, n_elem, time_msec
end

function main()
    n_loop = 10
    n_elem = 0
    if rank == 0
        f = open("bench_pxheevd.dat", "w")
        @printf(f, "number of process, number of matrix element, time [msec]\n")
    end
    for N in  2 .^ (4:20)
        time_msec_sum = 0
        for i in 1:n_loop
            MPI.Barrier(comm)
            n_proc, n_elem, time_msec = bench_pxheevd(N)
            MPI.Barrier(comm)
            time_msec_sum += time_msec.value
        end
        if rank == 0
            @printf(f, "%d %d %d\n", n_proc, n_elem, time_msec_sum/n_loop)
        end
    end
    if rank == 0
        close(f)
    end
end

main()
MPI.Finalize()