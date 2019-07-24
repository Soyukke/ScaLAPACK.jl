import LinearAlgebra.svdvals!

function A_mul_B!(α::T, A::MPIArray{T}, B::MPIArray{T}, β::T, C::MPIArray{T}) where T <: Union{BlasFloat, Complex}
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
    return C
end

function svdvals!(A::MPIArray{T}) where T<:BlasFloat
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


function eigen_symmetric(A::MPIArray{T}) where T<:BlasFloat
    n, N = Cint.(size(A))
    @assert n == N "m != n of matrix"
    NP, NQ = Cint.(size(pids(A)))
    NB, _ = Cint.(blocksizes(A))

    eigenvalues = Vector{T}(undef, N)
    eigenvectors = CyclicMPIArray(T, proc_grids=(NP, NQ), blocksizes=(NB, NB), N, N)

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', NP, NQ) # process grid column major

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)

    npA = numroc(N, NB, myrow, 0, nprow)
    nqA = numroc(N, NB, mycol, 0, npcol)
    # eigenvectors
    npZ = numroc(N, NB, myrow, 0, nprow)
    nqZ = numroc(N, NB, mycol, 0, npcol)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(N, N, NB, NB, 0, 0, ic, npA)
        dZ = descinit(N, N, NB, NB, 0, 0, ic, npZ)

        WORK = Vector{Complex{T}}(undef, 1)
        LWORK::Cint = -1
        RWORK = Vector{Complex{T}}(undef, 1)
        LRWORK::Cint = -1
        INFO::Cint = 0
        # GET LWORK, LRWORK
        pxheev!('V', 'U', N, A.localarray, Cint(1), Cint(1), dA, eigenvalues, eigenvectors.localarray, Cint(1), Cint(1), dZ, WORK, LWORK, RWORK, LRWORK, INFO)
        # allocate work space memory
        LWORK = real(WORK[1])
        LRWORK = real(RWORK[1])
        WORK = Vector{Complex{T}}(undef, LWORK)
        RWORK = Vector{Complex{T}}(undef, LRWORK)
        # calculate eigenvalues / eigenvectors of hermitian matrix A
        pxheev!('V', 'U', N, A.localarray, Cint(1), Cint(1), dA, eigenvalues, eigenvectors.localarray, Cint(1), Cint(1), dZ, WORK, LWORK, RWORK, LRWORK, INFO)
        # clean up
        BLACS.gridexit(ic)
        return eigenvalues, eigenvectors
    end
    return nothing
end

function eigen_hermitian(A::MPIArray{Complex{T}}) where T<:AbstractFloat
    n, N = Cint.(size(A))
    @assert n == N "m != n of matrix"
    NP, NQ = Cint.(size(pids(A)))
    NB, _ = Cint.(blocksizes(A))

    eigenvalues = Vector{T}(undef, N)
    eigenvectors = CyclicMPIArray(Complex{T}, proc_grids=(NP, NQ), blocksizes=(NB, NB), N, N)

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


