debug = true

using Test
using Random
using MPI, MPIArrays
using ScaLAPACK
using LinearAlgebra

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n_proc = MPI.Comm_size(comm)

function only_schur()
    for elty in (Float32, Float64, ComplexF32, ComplexF64)
        N = 73
        A = SLMatrix{elty}(N, N, proc_grids=(2, 2))
        forlocalpart!(x->rand!(x), A)
        sync(A)

        A_test = convert(Array, A)
        eigenvalues, _ = eigen(A_test)

        hessenberg!(A)
        hessenberg!(A_test)

        W, Z = eigen_schur!(A)
        
        B = convert(Array, A)

        if rank == 0 && debug
            for (e1, e2) in zip(W[sortperm(real(W))], eigenvalues[sortperm(real(eigenvalues))])
                println("(hess->schur_eig, LinearAlgebra.eigen) = $e1, $e2, diff=$(e1 - e2)")
            end
        end
        # @test all(abs.(B - A_test) .< 1e-5)
        free(A, Z)
    end
end

function only_schur2()
    for elty in (ComplexF32, ComplexF64)
        N = 12
        A = SLMatrix{elty}(N, N, proc_grids=(2, 2), blocksizes=(6, 6))
        forlocalpart!(x->rand!(x), A)
        sync(A)

        A_test = convert(Array, A)
        eigenvalues, _ = eigen(A_test)

        HA = hessenberg!(A)
        hessenberg!(A_test)

        H = HA.H
        SCH = schur!(H)
        
        B = convert(Array, A)

        if rank == 0
            println("Z")
            show(stdout, "text/plain", SCH.Z)
            println()
            println("T")
            show(stdout, "text/plain", SCH.T)
            println()
        end

        # for (e1, e2) in zip(SCH.values[sortperm(real(SCH.values))], eigenvalues[sortperm(real(eigenvalues))])
            # println("(hess->schur_eig, LinearAlgebra.eigen) = $e1, $e2, diff=$(e1 - e2)")
        # end
        # @test all(abs.(B - A_test) .< 1e-5)
        free(A, SCH.Z, H)
    end
end

only_schur2()

MPI.Finalize()