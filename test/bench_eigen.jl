using Random
using LinearAlgebra
using Dates
using Printf


function bench_pxheevd(N::Integer)
    # make Hermitian
    H = Matrix{ComplexF64}(undef, N, N)
    for i in 1:N, j in 1:i
        if i == j
            H[i, i] = rand()
        else
            H[i, j] = rand() + im*rand()
            H[j, i] = H[i, j]'
        end
    end
    time_start = Dates.now()
    eigenvalues, eigenvectors = eigen(H)
    time_stop = Dates.now()
    n_elem = N^2
    time_msec = time_stop - time_start
    println("Hermitian Matrix ($N, $N), n_elem: $n_elem , time $time_msec", )
    return n_elem, time_msec
end

function main()
    n_loop = 10
    n_elem = 0
    f = open("bench_lineralgebra_eigen.dat", "w")
    @printf(f, "number of matrix element, time [msec]\n")
    for N in  2 .^ (2:16)
        time_msec_sum = 0
        for i in 1:n_loop
            n_elem, time_msec = bench_pxheevd(N)
            time_msec_sum += time_msec.value
        end
        @printf(f, "%d %d\n", n_elem, time_msec_sum/n_loop)
    end
    if rank == 0
        close(f)
    end
end

main()