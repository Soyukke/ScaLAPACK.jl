using MPI, MPIArrays, Random, Printf, ScaLAPACK
using LinearAlgebra
comm = MPI.COMM_WORLD
MPI.Init()
rank = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

N = 12

function sortperm_complex(x::Vector)
    indices = sortperm(real(x))
    indices2 = copy(indices)
    for index in 1:length(x)-1
        iindex1 = indices[index]
        iindex2 = indices[index+1]
        if (real(x[iindex1]) == real(x[iindex2])) && (imag(x[iindex1]) > imag(x[iindex2]))
            print("P $(x[iindex1]) : $(x[iindex2])\n")
            # if 1 + 3im, 1 - 3im -> 1 - 3im, 1 + 3im
            indices[index] = iindex2
            indices[index+1] = iindex1
        end
    end
    return indices
end

for elty in (Float32, Float64, ComplexF32, ComplexF64)
    A = SLMatrix{elty}(N, N, proc_grids=(2, 2), blocksizes=(6, 6))
    forlocalpart!(x->rand!(x), A)
    sync(A)
    A_test = convert(Array, A)

    # hessenberg -> schur
    eigenvalues, eigenvectors = eigen(A)
    eigenvectors_ = convert(Array, eigenvectors)

    eigenvalues_test, eigenvectors_test = eigen(A_test)

    # indices_test = sortperm_complex(eigenvalues_test)
    # indices = sortperm_complex(eigenvalues)

    indices_test = sortperm(eigenvalues_test, lt = (x,y) -> real(x)==real(y) ? imag(x)<imag(y) : real(x)<real(y))
    indices = sortperm(eigenvalues, lt = (x,y) -> real(x)==real(y) ? imag(x)<imag(y) : real(x)<real(y))

    eigenvalues_test = eigenvalues_test[indices_test]
    eigenvalues = eigenvalues[indices]

    eigenvectors_ = eigenvectors_[:, indices]
    eigenvectors_test = eigenvectors_test[:, indices_test]
    eigenvalues_matrix = eigenvectors_test' * A_test * eigenvectors_test

    diff_eq = eigenvectors_*Diagonal(eigenvalues) - A_test*eigenvectors_
    diff_eq_test = eigenvectors_test*Diagonal(eigenvalues_test) - A_test*eigenvectors_test
    for (e1, e2) in zip(eigenvalues, eigenvalues_test)
        if rank == 0
            @printf("% .4f + % .4fim  :  % .4f + % .4fim\n", real(e1), imag(e1), real(e2), imag(e2))
        end
        MPI.Barrier(comm)
    end

    # show(stdout, "text/plain", eigenvalues-eigenvalues_test)
    if rank == 0 
        println("TEST:$elty")
        println("eigenvalues_ScaLAPACK - eigenvalues_LinearAlgebra = $(norm(eigenvalues_test - eigenvalues))")
        println("AX - XΛ = ScaLAPACK:$(norm(diff_eq)), LinearAlgebra:$(norm(diff_eq_test))")
    end
    MPI.Barrier(comm)

    # if rank == 0
    #     eigenvalues_test, _ = LinearAlgebra.eigen(A_test)
    #     eigenvalues = eigenvalues[sortperm(real(eigenvalues))]
    #     eigenvalues_test = eigenvalues_test[sortperm(real(eigenvalues_test))]


    #     for (e1, e2) in zip(eigenvalues, eigenvalues_test)
    #         @show e1 e2
    #         println()
    #     end
    # end
    free(A, eigenvectors)
end

MPI.Finalize()