function svdvals!(A::MPIArray{T}, blocksize::Integer = max(10, round(Integer, minimum(size(A))/10))) where T<:BlasFloat

    # m, n = (行数，列数)
    m, n = size(A)
    # mGrid, nGrid =  行の分割数, 列の分割数
    mGrid, nGrid = size(A.partitioning)
    # 部分行列のサイズ (mbA, nbA)
    mbA, nbA = size(A.localarray)
    mbB = blocksize

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'c', mGrid, nGrid)

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

        B = zeros(T, npB, nqB)
        pxgemr2d!(m, n, A.localarray, 1, 1, dA, B, 1, 1, dB, ic)

        # calculate DSVD
        U, s, Vt = pxgesvd!('N', 'N', m, n, B, 1, 1, dB, zeros(typeof(real(one(T))), min(m, n)), zeros(T, 0, 0), 0, 0, dB, zeros(T, 0, 0), 0, 0, dB)

        # clean up
        BLACS.gridexit(ic)
        return s
    end
    return nothing
end


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

    # check
    if mA != mC || nA != mB || nB != nC
        throw(DimensionMismatch("shapes don't fit"))
    end


    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'c', mGrid, nGrid)

    # who am I?
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(mA, mbA, myrow, 0, nprow)
    nqA = numroc(nA, nbA, mycol, 0, npcol)
    npB = numroc(mB, mbB, myrow, 0, nprow)
    nqB = numroc(nB, nbB, mycol, 0, npcol)
    npC = numroc(mC, mbC, myrow, 0, nprow)
    nqC = numroc(nC, nbC, mycol, 0, npcol)

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
        # calculate
        pdgemm!('N', 'N', mC, nC, k, α, A.localarray, 1, 1, dA, B.localarray, 1, 1, dB, β, C.localarray, 1, 1, dC)
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

    id, nprocs = BLACS.pinfo()
    ic = BLACS.gridinit(BLACS.get(0, 0), 'c', mGrid, nGrid)

    # who am I?
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(m, mbA, myrow, 0, nprow)
    nqA = numroc(n, nbA, mycol, 0, npcol)

    if nprow >= 0 && npcol >= 0
        # Get Array info
        dA = descinit(m, n, mbA, nbA, 0, 0, ic, npA)
        # calculate DSVD
        U, s, Vt = pxgesvd!('N', 'N', m, n, A.localarray, 1, 1, dA, zeros(typeof(real(one(T))), min(m, n)), zeros(T, 0, 0), 0, 0, dA, zeros(T, 0, 0), 0, 0, dA)
        # clean up
        BLACS.gridexit(ic)
        return s
    end
    return nothing
end
