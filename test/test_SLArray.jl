using Test
using MPI
using MPIArrays
using ScaLAPACK


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




# show_slarray()
# test_multiply()
test_multiply2()

MPI.Finalize()

