import LinearAlgebra.svdvals!
import LinearAlgebra.hessenberg!
export svdvals!, A_mul_B!, eigen_hermitian
export eigen_schur!

# BlasFloat is Union{Complex{Float32}, Complex{Float64}, Float32, Float64}
function A_mul_B!(α::T, A::SLArray{T}, B::SLArray{T}, β::T, C::SLArray{T}) where T <: BlasFloat
    # global matrix size
    m_A, n_A = size(A)
    m_B, n_B = size(B)
    m_C, n_C = size(C)
    k = n_A
    # process grid
    m_proc_grid, n_proc_grid = size(pids(A))

    # check
    if m_A != m_C || n_A != m_B || n_B != n_C
        throw(DimensionMismatch("shapes don't fit"))
    end

    id, nprocs = BLACS.pinfo()

    # blocksize of mpiarray
    m_blocksize_A, n_blocksize_A = blocksizes(A)
    m_blocksize_B, n_blocksize_B = blocksizes(B)
    m_blocksize_C, n_blocksize_C = blocksizes(C)
    # Assign MPI process with column major
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', m_proc_grid, n_proc_grid)

    # who am I?
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(m_A, m_blocksize_A, myrow, 0, nprow)
    npB = numroc(m_B, m_blocksize_B, myrow, 0, nprow)
    npC = numroc(m_C, m_blocksize_C, myrow, 0, nprow)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA    = descinit(m_A, n_A, m_blocksize_A, n_blocksize_A, 0, 0, ic, npA)
        dB    = descinit(m_B, n_B, m_blocksize_B, n_blocksize_B, 0, 0, ic, npB)
        dC    = descinit(m_C, n_C, m_blocksize_C, n_blocksize_C, 0, 0, ic, npC)
        # calculate
        pdgemm!('N', 'N', m_C, n_C, k, α, A.localarray, 1, 1, dA, B.localarray, 1, 1, dB, β, C.localarray, 1, 1, dC)
        # cleanup
        BLACS.gridexit(ic)
    end
    return nothing
end

function svdvals!(A::SLArray{T}) where T<:BlasFloat
    m, n = size(A)
    m_blocksize, n_blocksize = blocksizes(A)
    m_proc_grid, n_proc_grid = size(pids(A))

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', m_proc_grid, n_proc_grid)

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(m, m_blocksize, myrow, 0, nprow)
    nqA = numroc(n, n_blocksize, mycol, 0, npcol)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(m, n, m_blocksize, n_blocksize, 0, 0, ic, npA)
        # calculate DSVD
        U, s, Vt = pxgesvd!('N', 'N', m, n, A.localarray, 1, 1, dA, zeros(typeof(real(one(T))), min(m, n)), zeros(T, 0, 0), 0, 0, dA, zeros(T, 0, 0), 0, 0, dA)
        # clean up
        BLACS.gridexit(ic)
        return s
    end
    return nothing
end

function eigen_hermitian(A::SLArray{T}) where T<:BlasFloat
    n, N = Cint.(size(A))
    @assert n == N "m != n of matrix"
    NP, NQ = Cint.(size(pids(A)))
    NB, _ = Cint.(blocksizes(A))

    eigenvalues = Vector{typeof(real(T(0)))}(undef, N)
    eigenvectors = SLArray(T, proc_grids=(NP, NQ), N, N)

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', NP, NQ) # process grid column major

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    # hermitian
    npA = numroc(N, NB, myrow, 0, nprow)
    # eigenvectors
    npZ = numroc(N, NB, myrow, 0, nprow)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(N, N, NB, NB, 0, 0, ic, npA)
        dZ = descinit(N, N, NB, NB, 0, 0, ic, npZ)

        pXheev!(N, A.localarray, dA, eigenvalues, eigenvectors.localarray, dZ)
        # clean up
        BLACS.gridexit(ic)
        return eigenvalues, eigenvectors
    end
    return nothing
end


function eigen_schur!(A::SLArray{T, 2}) where T<:AbstractFloat
    n, N = Cint.(size(A))
    NP, NQ = Cint.(size(pids(A)))
    NB, NB2 = Cint.(blocksizes(A))

    @assert n == N "m != n of matrix"
    @assert NP == NQ "proccess grid NP != NQ"
    @assert NB == NB2 && NB >= 6  "bocksizes NB1 != NB2, NB1 >= 6"

    WR = Vector{T}(undef, N)
    WI = Vector{T}(undef, N)
    Z = SLArray(T, proc_grids=(NP, NQ), N, N)

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', NP, NQ) # process grid column major

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    LOCr_A = numroc(N, NB, myrow, 0, nprow)
    LOCr_Z = numroc(N, NB, myrow, 0, nprow)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(N, N, NB, NB, 0, 0, ic, LOCr_A)
        dZ = descinit(N, N, NB, NB, 0, 0, ic, LOCr_Z)
        pXlahqr!(N, A.localarray, dA, WR, WI, Z.localarray, dZ)
        # clean up
        BLACS.gridexit(ic)
    end
    return WR + im*WI, Z
end

function schur!(A::SLArray{T, 2}) where T<:Complex
    n, N = Cint.(size(A))
    NP, NQ = Cint.(size(pids(A)))
    NB, NB2 = Cint.(blocksizes(A))

    @assert n == N "m != n of matrix"
    @assert NB == NB2 && NB >= 6  "NB1 != NB2, NB1 >= 6"

    W = Vector{T}(undef, N)
    Z = SLArray(T, proc_grids=(NP, NQ), N, N)

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
    return W, Z
end