export schur!
import LinearAlgebra: schur, schur!


struct SLSchur{T} <: Factorization{T}
    T::SLArray{T, 2} # schur
    Z::SLArray{T, 2} # vectors
    values::Vector{T} # eigenvalues
end

"""
after hessenberg, call this function Z is Q
"""
function schur!(A::SLMatrix{T}, Z::SLMatrix{T}) where T<:Complex
    @assert size(A) == size(Z)
    @assert pids(A) == pids(Z)
    @assert blocksizes(A) == blocksizes(Z)

    n, N = Cint.(size(A))
    NP, NQ = Cint.(size(pids(A)))
    NB, NB2 = Cint.(blocksizes(A))

    @assert n == N "m != n of matrix"
    @assert NB == NB2 && NB >= 6  "NB1 != NB2, NB1 >= 6"

    W = Vector{T}(undef, N)

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', NP, NQ) # process grid column major

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    LOCr_A = numroc(N, NB, myrow, 0, nprow)
    LOCr_Z = numroc(N, NB, myrow, 0, nprow)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(N, N, NB, NB, 0, 0, ic, LOCr_A)
        dZ = descinit(N, N, NB, NB, 0, 0, ic, LOCr_Z)
        pXlahqr!(N, A.localarray, dA, W, Z.localarray, dZ)
        # clean up
        BLACS.gridexit(ic)
    end
    return SLSchur(A, Z, W)
end

function schur!(A::SLMatrix{T}) where T<:Complex
    n, N = Cint.(size(A))
    NP, NQ = Cint.(size(pids(A)))
    NB, NB2 = Cint.(blocksizes(A))
    Z = SLMatrix{T}(N, N, proc_grids=(NP, NQ), blocksizes=(NB, NB2))
    # Identity matrix
    forlocalpart!(Z) do lZ
        one = T(1)
        zero = T(0)
        gi, gj = localindices(Z)
        for (li, gi) in enumerate(gi)
            for (lj, gj) in enumerate(gj)
                if gi == gj
                    lZ[li, lj] = one
                else
                    lZ[li, lj] = zero
                end
            end
        end
    end
    sync(Z)

    return schur!(A, Z)
end