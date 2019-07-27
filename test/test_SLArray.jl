using Test
using MPI
using MPIArrays
using ScaLAPACK
using Random
using LinearAlgebra


MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)


function show_slarray()
    A = SLArray(Float64, 7, 7)
    forlocalpart!(x->fill!(x, rank), A) 
    sync(A)
    if rank == 0
        show(stdout, "text/plain", A)
        println()
    end
    free(A)
end

function test_multiply()
    m, n = 7, 5
    proc_grid = (2, 2)
    A = SLArray(Float64, m, n, proc_grids=proc_grid)
    B = SLArray(Float64, n, m, proc_grids=proc_grid)
    forlocalpart!(x->fill!(x, rank), A) 
    forlocalpart!(x->fill!(x, rank), B) 
    sync(A, B)
    C = A * B
    if rank == 0
        println("A")
        show(stdout, "text/plain", A)
        println()
        println("B")
        show(stdout, "text/plain", B)
        println()
        println("C")
        show(stdout, "text/plain", C)
        println()

        C_true = convert(Array, A)*convert(Array, B)
        println("true C")
        show(stdout, "text/plain", C_true)
        println()

        @test C == C_true
    end
    free(A); free(B); free(C)
end


function test_multiply2()
    m, n = 7, 5
    A = SLArray(Float64, m, n)
    B = SLArray(Float64, n, m)
    forlocalpart!(x->fill!(x, rank), A) 
    forlocalpart!(x->fill!(x, rank), B) 
    sync(A, B)
    C = A * B
    if rank == 0
        println("A")
        show(stdout, "text/plain", A)
        println()
        println("B")
        show(stdout, "text/plain", B)
        println()
        println("C")
        show(stdout, "text/plain", C)
        println()

        C_true = convert(Array, A)*convert(Array, B)
        println("true C")
        show(stdout, "text/plain", C_true)
        println()

        @test C == C_true
    end
    free(A); free(B); free(C)
end

function test_multiply3()
    m, n = 5, 5
    proc_grid = (2, 2)
    A = SLArray(Float64, m, n, proc_grids=proc_grid)
    forlocalpart!(x->fill!(x, rank), A) 
    sync(A)
    C = A * A
    if rank == 0
        println("A")
        show(stdout, "text/plain", A)
        println("C")
        show(stdout, "text/plain", C)
        println()

        C_true = convert(Array, A)*convert(Array, A)
        println("true C")
        show(stdout, "text/plain", C_true)
        println()

        @test C == C_true
    end
    free(A, C)
end

function test_multiply4()
    m, n = 5, 5
    proc_grid = (2, 2)
    A = SLArray(Float64, m, n, proc_grids=proc_grid)
    forlocalpart!(x->fill!(x, rank), A) 
    B = SLArray(Float64, m, n, proc_grids=proc_grid)
    forlocalpart!(x->fill!(x, rank), B) 
    sync(A, B)
    A_test = convert(Array, A)
    B_test = convert(Array, B)
    C = B * A * A * B
    C_true = B_test * A_test * A_test * B_test
    C_test = convert(Array, C)

    print("ok1\n")
    MPI.Barrier(comm)
    print("ok2\n")
    MPI.Barrier(comm)


    if rank == 0
        # print("ok")
        # print(C_test)
        # show(stdout, "text/plain", A)
        show(stdout, "text/plain", C_test)
        # f = open("slshow.dat", "w")
        # show(f, "text/plain", C)
        # close(f)
    end

    print("ok3\n")
    MPI.Barrier(comm)

    # @show norm(C_test - C_true)
    free(A, B, C)
end



function test_adjoint()
    for elty in (Float32, Float64, ComplexF32, ComplexF64)
        m, n = 4, 9
        A = SLMatrix{elty}(m, n, proc_grids=(2, 2))
        forlocalpart!(x->rand!(x), A) 
        sync(A)
        C = A'
        if rank == 0
            println("A")
            show(stdout, "text/plain", A)
            println()
            println("C")
            show(stdout, "text/plain", C)
            println()

            C_true = convert(Array, A)'
            println("true C")
            show(stdout, "text/plain", C_true)
            println()
            @test C == C_true
        end
        free(A, C)
    end
end


# show_slarray()
# test_multiply()
# test_multiply2()
# test_multiply3()
test_multiply4()
# test_adjoint()

MPI.Finalize()

