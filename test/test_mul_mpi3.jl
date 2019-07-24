debug = true
m, n = 11, 9 

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)


alpha, beta = 1.0, 0.0

A = MPIArray{Float64}(m, n)
B = MPIArray{Float64}(n, m)
C = MPIArray{Float64}(m, m)


print("rank $rank $(localindices(B, rank))\n")

fill_array! = function(x, indices)
    nb, mb = size(x)
    cnt = 0
    for (i, gi) in zip(1:nb, indices[1])
        for (j, gj) in zip(1:mb, indices[2])
            x[i, j] = (gi-1)*m + gj
            cnt += 1
        end
    end
end

#= 単位行列を作る =#
I! = function(x, indices)
    n, m = size(x)
    for (i, gi) in zip(1:n, indices[1])
        for (j, gj) in zip(1:m, indices[2])
            if gi == gj
                x[i, j] = 1
            else
                x[i, j] = 0
            end
        end
    end
end



forlocalpart!(x->fill_array!(x, localindices(A, rank)), A)
# forlocalpart!(x->I!(x, localindices(B, rank)), B)
forlocalpart!(x->fill!(x, 1), B)
forlocalpart!(x->fill!(x, 0), C)
sync(A, B, C)

A_test = convert(Array, A)
B_test = convert(Array, B)

A_mul_B!(alpha, A, B, beta, C)

if rank == 0
    println("array A")
    show(stdout, "text/plain", A)
    println()

    println("array B")
    show(stdout, "text/plain", B)
    println()

    println("A_mul_B! is worked")
    show(stdout, "text/plain", C)
    println() 

    println("A_test * B_test")
    show(stdout, "text/plain", alpha * A_test * B_test)
    println() 

    # @test convert(Array, C) == alpha * A_test * B_test
end


MPI.Finalize()
