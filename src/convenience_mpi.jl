import LinearAlgebra.svdvals!

function block_size_mpiarray(A::MPIArray, n_proc::Integer)
    m, n = size(A)
    m_blocksize_A =  div(m, size(A.partitioning, 1))
    n_blocksize_A = div(n, size(A.partitioning, 2))
    return m_blocksize_A, n_blocksize_A
end

function A_mul_B!(α::T, A::MPIArray{T}, B::MPIArray{T}, β::T, C::MPIArray{T}; m_blocksize=1, n_blocksize=1) where T <: BlasFloat
    # global matrix size
    m_A, n_A = size(A)
    m_B, n_B = size(B)
    m_C, n_C = size(C)
    k = n_A
    # process grid
    m_proc_grid, n_proc_grid = size(A.partitioning)

    # check
    if m_A != m_C || n_A != m_B || n_B != n_C
        throw(DimensionMismatch("shapes don't fit"))
    end

    id, nprocs = BLACS.pinfo()

    # blocksize of mpiarray
    m_blocksize_A, n_blocksize_A = block_size_mpiarray(A, nprocs)
    m_blocksize_B, n_blocksize_B = block_size_mpiarray(B, nprocs)
    m_blocksize_C, n_blocksize_C = block_size_mpiarray(C, nprocs)
    if id == 0
        println("$(size(A))")
        println("$(size(B))")

        println("$(localindices(B, 0))")

        println("$(size(C))")

        println("$m_blocksize_A, $n_blocksize_A")
        println("$m_blocksize_B, $n_blocksize_B")
        println("$m_blocksize_C, $n_blocksize_C")
    end
    # Assign MPI process with column major
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', m_proc_grid, n_proc_grid)

    # who am I?
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(m_A, m_blocksize_A, myrow, 0, nprow)
    nqA = numroc(n_A, n_blocksize_A, mycol, 0, npcol)
    npB = numroc(m_B, m_blocksize_B, myrow, 0, nprow)
    nqB = numroc(n_B, n_blocksize_B, mycol, 0, npcol)
    npC = numroc(m_C, m_blocksize_C, myrow, 0, nprow)
    nqC = numroc(n_C, n_blocksize_C, mycol, 0, npcol)

    npAnew = numroc(m_A, m_blocksize, myrow, 0, nprow)
    nqAnew = numroc(n_A, n_blocksize, mycol, 0, npcol)
    npBnew = numroc(m_B, m_blocksize, myrow, 0, nprow)
    nqBnew = numroc(n_B, n_blocksize, mycol, 0, npcol)
    npCnew = numroc(m_C, m_blocksize, myrow, 0, nprow)
    nqCnew = numroc(n_C, n_blocksize, mycol, 0, npcol)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA    = descinit(m_A, n_A, m_blocksize_A, n_blocksize_A, 0, 0, ic, npA)
        dB    = descinit(m_B, n_B, m_blocksize_B, n_blocksize_B, 0, 0, ic, npB)
        dC    = descinit(m_C, n_C, m_blocksize_C, n_blocksize_C, 0, 0, ic, npC)
        # for redistribute
        dAnew = descinit(m_A, n_A, m_blocksize, n_blocksize, 0, 0, ic, npAnew)
        dBnew = descinit(m_B, n_B, m_blocksize, n_blocksize, 0, 0, ic, npBnew)
        dCnew = descinit(m_C, n_C, m_blocksize, n_blocksize, 0, 0, ic, npCnew)

        # print("myrank: $id, myrow: $myrow, mycol: $mycol, npAnew: $npAnew, npCnew: $npCnew, npCnew: $npCnew\n")

        # redistribute cyclic
        Anew = Matrix{T}(undef, npAnew, nqAnew)
        pxgemr2d!(m_A, n_A, A.localarray, 1, 1, dA, Anew, 1, 1, dAnew, ic)
        # print("myrow: $myrow, mycol: $mycol, npAnew: $npAnew, npCnew: $npCnew, npCnew: $npCnew\n")
        # if id == 0
        #     println("rank $id array Anew")
        #     show(stdout, "text/plain", Anew)
        #     println()
        # end
        Bnew = Matrix{T}(undef, npBnew, nqBnew)
        pxgemr2d!(m_B, n_B, B.localarray, 1, 1, dB, Bnew, 1, 1, dBnew, ic)
        Cnew = Matrix{T}(undef, npCnew, nqCnew)
        pxgemr2d!(m_C, n_C, C.localarray, 1, 1, dC, Cnew, 1, 1, dCnew, ic)

        # calculate
        pdgemm!('N', 'N', m_C, n_C, k, α, Anew, 1, 1, dAnew, Bnew, 1, 1, dBnew, β, Cnew, 1, 1, dCnew)

        # move result back to C
        pxgemr2d!(m_C, n_C, Cnew, 1, 1, dCnew, C.localarray, 1, 1, dC, ic)
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
        pxheevd!('V', 'U', N, A.localarray, Cint(1), Cint(1), dA, eigenvalues, eigenvectors.localarray, Cint(1), Cint(1), dZ, WORK, LWORK, RWORK, LRWORK, INFO)
        # allocate work space memory
        LWORK = real(WORK[1])
        LRWORK = real(RWORK[1])
        WORK = Vector{Complex{T}}(undef, LWORK)
        RWORK = Vector{Complex{T}}(undef, LRWORK)
        # calculate eigenvalues / eigenvectors of hermitian matrix A
        pxheevd!('V', 'U', N, A.localarray, Cint(1), Cint(1), dA, eigenvalues, eigenvectors.localarray, Cint(1), Cint(1), dZ, WORK, LWORK, RWORK, LRWORK, INFO)
        # clean up
        BLACS.gridexit(ic)
        return eigenvalues, eigenvectors
    end
    return nothing
end

