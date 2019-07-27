import LinearAlgebra.adjoint!
import LinearAlgebra.adjoint

Base.:*(A::SLArray{T, 2}, B::SLArray{T, 2}) where T = begin
    alpha::T = 1.0
    beta::T = 0.0
    m_A, n_A = size(A)
    m_B, n_B = size(B)
    proc_grid_A = size(pids(A))
    proc_grid_B = size(pids(B))
    blocksizes_A = blocksizes(A)
    blocksizes_B = blocksizes(B)
    @assert proc_grid_A == proc_grid_B

    C = SLMatrix{T}(m_A, n_B, proc_grids=proc_grid_A, blocksizes=blocksizes_A)
    A_mul_B!(alpha, A, B, beta, C)
    return C
end

function adjoint(A::SLArray{T}) where T <: BlasFloat
    m_A, n_A = Cint.(size(A))
    m_proc_grid, n_proc_grid = size(pids(A))
    m_blocksize_A, n_blocksize_A = blocksizes(A)

    C = SLMatrix{T}(n_A, m_A, proc_grids=(m_proc_grid, n_proc_grid), blocksizes=(n_blocksize_A, m_blocksize_A))
    m_C, n_C = Cint.(size(C))
    m_blocksize_C, n_blocksize_C = blocksizes(C)

    id, nprocs = BLACS.pinfo()
    # Assign MPI process with column major
    ic = BLACS.gridinit(BLACS.get(0, 0), 'C', m_proc_grid, n_proc_grid)
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
    npA = numroc(m_A, m_blocksize_A, myrow, 0, nprow)
    npC = numroc(m_C, m_blocksize_C, myrow, 0, nprow)
    # Get Array info
    dA    = descinit(m_A, n_A, m_blocksize_A, n_blocksize_A, 0, 0, ic, npA)
    dC    = descinit(m_C, n_C, m_blocksize_C, n_blocksize_C, 0, 0, ic, npC)
    # calculate
    α::T = 1.0
    β::T = 0.0
    pXtranc!(m_C, n_C, α, A.localarray, dA, β, C.localarray, dC)
    # cleanup
    BLACS.gridexit(ic)
    return C
end

function adjoint!(A::SLArray{T}) where T <: BlasFloat
    C = adjoint(A)
    forlocalpart!(A) do lA
        lA .= C.localarray
    end
    sync(A)
    free(C)
    return A
end