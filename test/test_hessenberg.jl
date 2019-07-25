debug = true
N = 5

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK
using LinearAlgebra

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

function test_hessenberg()
    for elty in (Float32, Float64, ComplexF32, ComplexF64)
        A = SLArray(elty, N, N, proc_grids=(2, 2))
        forlocalpart!(x->rand!(x), A)
        sync(A)

        A_test = convert(Array, A)

        if rank == 0 && debug
            show(stdout, "text/plain", A)
            println()
        end

        hessenberg!(A)
        hessenberg!(A_test)
        B = convert(Array, A)

        if rank == 0 && debug
            show(stdout, "text/plain", A)
            println()
            show(stdout, "text/plain", A_test)
            println()
            show(stdout, "text/plain", B - A_test)
            println()
        end

        @test all(abs.(B - A_test) .< 1e-5)

        free(A)
    end
end

function test_hessenberg_Q()
    for elty in (Float32, Float64, ComplexF32, ComplexF64)
        A = SLArray(elty, N, N, proc_grids=(2, 2))
        forlocalpart!(x->rand!(x), A)
        sync(A)

        A_test = convert(Array, A)

        if rank == 0 && debug
            show(stdout, "text/plain", A)
            println()
        end

        HA = hessenberg!(A)
        Q = HA.Q
        H = HA.H

        if rank == 0 && debug
            println("hessenberg!(A)")
            show(stdout, "text/plain", A)
            println()
            println("Q")
            show(stdout, "text/plain", Q)
            println()
            println("H")
            show(stdout, "text/plain", H)
            println()
        end

        free(A, Q, H)
    end
end

function test_hessenberg_remake()
    for elty in (Float32, Float64, ComplexF32, ComplexF64)
        A = SLArray(elty, N, N, proc_grids=(2, 2))
        forlocalpart!(x->rand!(x), A)
        sync(A)

        A_test = convert(Array, A)

        HA = hessenberg!(A)
        Q = HA.Q
        H = HA.H
        B = Q * H * Q'
        # get full array
        Q_ = convert(Array, Q)
        H_ = convert(Array, H)
        QHQT = convert(Array, B)
        @show norm(A_test - QHQT)
        # MPI.Barrier(comm)
        # if rank == 0
            # println("A")
            # show(stdout, "text/plain", A_test)
            # println()
            # println("Q")
            # show(stdout, "text/plain", Q_)
            # println()
            # println("H")
            # show(stdout, "text/plain", H)
            # println()
            # println("QHQ'")
            # show(stdout, "text/plain", QHQT)
            # println()
        # end
        free(A, Q, H, B)
    end
end



# test_hessenberg()
# test_hessenberg_Q()
test_hessenberg_remake()


MPI.Finalize()