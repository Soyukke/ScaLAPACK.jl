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
    forlocalpart!(H) do localarray
        range_i, range_j = localindices(H, rank)
        for (i, gi) in enumerate(range_i)
            for (j, gj) in enumerate(range_j)
                if i==j
                    localarray[i, j] = gi
                elseif i < j
                    localarray[i, j] = gi + im*gj
                else
                    localarray[i, j] = gj - im*gi
                end
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
    n_loop = 2
    n_elem = 0
    for N in  2 .^ (8:20)
        time_msec_sum = 0
        for i in 1:n_loop
            MPI.Barrier(comm)
            n_proc, n_elem, time_msec = bench_pxheevd(N)
            MPI.Barrier(comm)
            time_msec_sum += time_msec.value
        end
    end
end

main()
MPI.Finalize()