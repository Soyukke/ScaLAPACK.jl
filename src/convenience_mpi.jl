function A_mul_B!(α::T, A::MPIArray{T}, B::MPIArray{T}, β::T, C::MPIArray{T}) where T <: BlasFloat

    # extract
    mA, nA = size(A)
    mB, nB = size(B)
    mC, nC = size(C)
    k = nA
    mGrid, nGrid = size(A.partitioning)
    mbA, nbA = size(A.localarray)
    mbB, nbB = size(B.localarray)
    mbC, nbC = size(C.localarray)

    blocksize1, blocksize2 = 1, 1

    # check
    if mA != mC || nA != mB || nB != nC
        throw(DimensionMismatch("shapes don't fit"))
    end


    id, nprocs = BLACS.pinfo()
    # Assign MPI process with row major
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', mGrid, nGrid)

    # who am I?
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(mA, mbA, myrow, 0, nprow)
    nqA = numroc(nA, nbA, mycol, 0, npcol)
    npB = numroc(mB, mbB, myrow, 0, nprow)
    nqB = numroc(nB, nbB, mycol, 0, npcol)
    npC = numroc(mC, mbC, myrow, 0, nprow)
    nqC = numroc(nC, nbC, mycol, 0, npcol)

    npAnew = numroc(mA, blocksize1, myrow, 0, nprow)
    nqAnew = numroc(nA, blocksize2, mycol, 0, npcol)
    npBnew = numroc(mB, blocksize1, myrow, 0, nprow)
    nqBnew = numroc(nB, blocksize2, mycol, 0, npcol)
    npCnew = numroc(mC, blocksize1, myrow, 0, nprow)
    nqCnew = numroc(nC, blocksize2, mycol, 0, npcol)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("numroc")
        println("npA: $npA, nqA: $nqA")
        println("npB: $npB, nqB: $nqB")
        println("npC: $npC, nqC: $nqC")
    end

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA    = descinit(mA, nA, mbA, nbA, 0, 0, ic, npA)
        dB    = descinit(mB, nB, mbB, nbB, 0, 0, ic, npB)
        dC    = descinit(mC, nC, mbC, nbC, 0, 0, ic, npC)
        # for redistribute
        dAnew = descinit(mA, nA, blocksize1, blocksize2, 0, 0, ic, npAnew)
        dBnew = descinit(mB, nB, blocksize1, blocksize2, 0, 0, ic, npBnew)
        dCnew = descinit(mC, nC, blocksize1, blocksize2, 0, 0, ic, npCnew)

        # print("myrank: $id, myrow: $myrow, mycol: $mycol, npAnew: $npAnew, npCnew: $npCnew, npCnew: $npCnew\n")

        # redistribute cyclic
        Anew = Matrix{T}(undef, npAnew, nqAnew)
        pxgemr2d!(mA, nA, A.localarray, 1, 1, dA, Anew, 1, 1, dAnew, ic)
        # print("myrow: $myrow, mycol: $mycol, npAnew: $npAnew, npCnew: $npCnew, npCnew: $npCnew\n")
        # if id == 0
        #     println("rank $id array Anew")
        #     show(stdout, "text/plain", Anew)
        #     println()
        # end
        Bnew = Matrix{T}(undef, npBnew, nqBnew)
        pxgemr2d!(mB, nB, B.localarray, 1, 1, dB, Bnew, 1, 1, dBnew, ic)
        Cnew = Matrix{T}(undef, npCnew, nqCnew)
        pxgemr2d!(mC, nC, C.localarray, 1, 1, dC, Cnew, 1, 1, dCnew, ic)

        # calculate
        pdgemm!('N', 'N', mC, nC, k, α, Anew, 1, 1, dAnew, Bnew, 1, 1, dBnew, β, Cnew, 1, 1, dCnew)

        # move result back to C
        pxgemr2d!(mC, nC, Cnew, 1, 1, dCnew, C.localarray, 1, 1, dC, ic)
        # cleanup
        BLACS.gridexit(ic)
    end
    return C
end


function svdvals!(A::MPIArray{T}) where T<:BlasFloat

    # m, n = (行数，列数)
    m, n = size(A)
    # mGrid, nGrid =  行の分割数, 列の分割数
    mGrid, nGrid = size(A.partitioning)
    # 部分行列のサイズ (mbA, nbA)
    mbA, nbA = size(A.localarray)
    mbB = 1

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', mGrid, nGrid)

    # who am I?
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(m, mbA, myrow, 0, nprow)
    nqA = numroc(n, nbA, mycol, 0, npcol)

    npB = numroc(m, mbB, myrow, 0, nprow)
    nqB = numroc(n, mbB, mycol, 0, npcol)


    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(m, n, mbA, nbA, 0, 0, ic, npA)
        dB = descinit(m, n, mbB, mbB, 0, 0, ic, npB)
        # redistribute
        B = Matrix{T}(undef, npB, nqB)
        pxgemr2d!(m, n, A.localarray, 1, 1, dA, B, 1, 1, dB, ic)


        # calculate DSVD
        U, s, Vt = pxgesvd!('N', 'N', m, n, B, 1, 1, dB, zeros(typeof(real(one(T))), min(m, n)), zeros(T, 0, 0), 0, 0, dB, zeros(T, 0, 0), 0, 0, dB)
        # clean up
        BLACS.gridexit(ic)
        return s
    end
    return nothing
end


function eigen_hermitian(A::MPIArray{Complex{T}}) where T<:AbstractFloat
    n, N::Cint = size(A)
    @assert n == N "nrow != ncol of matrix"
    NP, NQ = size(A.partitioning)
    mbA, nbA = size(A.localarray) # submatrix size
    mbB = 1 # blocksize

    eigenvalues = Vector{T}(undef, n)
    eigenvectors = MPIArray{Complex{T}}(MPI.COMM_WORLD, (NP, NQ), N, N)

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', NP, NQ) # process grid column major

    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    # hermitian
    npA = numroc(N, mbA, myrow, 0, nprow)
    nqA = numroc(N, nbA, mycol, 0, npcol)
    npB = numroc(N, mbB, myrow, 0, nprow)
    nqB = numroc(N, mbB, mycol, 0, npcol)
    # eigenvectors
    npZ = numroc(N, mbA, myrow, 0, nprow)
    nqZ = numroc(N, mbA, mycol, 0, npcol)
    npZ_cyclic = numroc(N, mbB, myrow, 0, nprow)
    nqZ_cyclic = numroc(N, mbB, mycol, 0, npcol)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(N, N, mbA, nbA, 0, 0, ic, npA)
        dB = descinit(N, N, mbB, mbB, 0, 0, ic, npB)
        dZ = descinit(N, N, mbA, mbA, 0, 0, ic, npZ)
        dZ_cyclic = descinit(N, N, mbB, mbB, 0, 0, ic, npZ_cyclic)
        # redistribute cyclic matrix
        B = Matrix{Complex{T}}(undef, npB, nqB)
        Z_cyclic = Matrix{Complex{T}}(undef, npZ_cyclic, nqZ_cyclic)
        pxgemr2d!(N, N, A.localarray, 1, 1, dA, B, 1, 1, dB, ic)
        pxgemr2d!(N, N, eigenvectors.localarray, 1, 1, dZ, Z_cyclic, 1, 1, dZ_cyclic, ic)

        WORK = Vector{Complex{T}}(undef, 1)
        LWORK::Cint = -1
        RWORK = Vector{Complex{T}}(undef, 1)
        LRWORK::Cint = -1
        INFO::Cint = 0
        # GET LWORK, LRWORK
        pxheevd!('V', 'U', N, B, Cint(1), Cint(1), dB, eigenvalues, Z_cyclic, Cint(1), Cint(1), dZ_cyclic, WORK, LWORK, RWORK, LRWORK, INFO)
        # if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        #     println("WORK SIZE: $(real(WORK[1])), RWORK SIZE: $(real(RWORK[1]))")
        # end
        # allocate work space memory
        LWORK = real(WORK[1])
        LRWORK = real(RWORK[1])
        WORK = Vector{Complex{T}}(undef, LWORK)
        RWORK = Vector{Complex{T}}(undef, LRWORK)
        # calculate eigenvalues / eigenvectors of hermitian matrix A
        pxheevd!('V', 'U', N, B, Cint(1), Cint(1), dB, eigenvalues, Z_cyclic, Cint(1), Cint(1), dZ_cyclic, WORK, LWORK, RWORK, LRWORK, INFO)

        # move result back to C
        pxgemr2d!(N, N, Z_cyclic, 1, 1, dZ_cyclic, eigenvectors.localarray, 1, 1, dZ, ic)

        # clean up
        BLACS.gridexit(ic)
        return eigenvalues, eigenvectors
    end
    return nothing
end

