export hessenberg!

import LinearAlgebra: Factorization, AbstractQ, hessenberg!

struct SLHessenberg{T} <: Factorization{T}
    factors::SLArray{T, 2}
    τ::Vector{T}
end

struct SLHessenbergQ{T} <: AbstractQ{T}
    Q::SLArray{T, 2}
    τ::Vector{T}

    function SLHessenbergQ{T}(Q, τ) where T
        new(Q, τ)
    end
end

function Base.getproperty(F::SLHessenberg, d::Symbol)
    d == :Q && return HessenbergQ(F)
    if d == :H
        H = triu(getfield(F, :factors), -1)
        MPI.Barrier(MPI.COMM_WORLD)
        return H
    end
    return getfield(F, d)
end

Base.propertynames(F::SLHessenberg, private::Bool=false) = (:Q, :H, (private ? fieldnames(typeof(F)) : ())...)


function hessenberg!(A::SLArray{T, 2}) where T<:BlasFloat
    n, N = Cint.(size(A))
    NP, NQ = Cint.(size(pids(A)))
    NB, _ = Cint.(blocksizes(A))

    @assert n == N "m != n of matrix"
    @assert NP == NQ "proccess grid NP != NQ"

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', NP, NQ) # process grid column major

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    LOCr_A = numroc(N, NB, myrow, 0, nprow)
    IA, JA = 1, 1
    LOCc_TAU = numroc(IA+N-2, NB, mycol, 0, npcol)
    TAU = Vector{T}(undef, LOCc_TAU)

    # Get Array infjjjo
    dA = descinit(N, N, NB, NB, 0, 0, ic, LOCr_A)
    pXgehrd!(N, A.localarray, dA, TAU)
    # clean up
    BLACS.gridexit(ic)
    return SLHessenberg(A, TAU)
end

# pXunmhr!(M::Cint, N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, TAU::Vector{$elty}, C::Matrix{$elty}, DESCC::Vector{Cint})
function HessenbergQ(A::SLHessenberg{T}) where T
    M, N = Cint.(size(A.factors))
    NP, NQ = Cint.(size(pids(A.factors)))
    NB, NB2 = Cint.(blocksizes(A.factors))

    @assert NP == NQ "proccess grid NP != NQ"

    # Identity matrix
    C = SLMatrix{T}(M, N, proc_grids=(NP, NQ), blocksizes=(NB, NB2))
    forlocalpart!(C) do lC
        gi, gj = localindices(C)
        for (li, gi) in enumerate(gi)
            for (lj, gj) in enumerate(gj)
                if gi == gj
                    lC[li, lj] = 1
                else
                    lC[li, lj] = 0
                end
            end
        end
    end
    sync(C)
    IC, JC = 1, 1

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', NP, NQ) # process grid column major

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    LOCr_A = numroc(N, NB, myrow, 0, nprow)
    LOCr_C = numroc(N, NB, myrow, 0, nprow)
    IA, JA = 1, 1
    # Get Array infjjjo
    dA = descinit(N, N, NB, NB, 0, 0, ic, LOCr_A)
    dC = descinit(N, N, NB, NB, 0, 0, ic, LOCr_C)
    # Q
    pXunmhr!(M, N, A.factors.localarray, dA, A.τ, C.localarray, dC)
    # clean up
    BLACS.gridexit(ic)

    return C
end