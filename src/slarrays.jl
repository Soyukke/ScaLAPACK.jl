import MPIArrays.AbstractMPIArray
import MPIArrays.Partitioning

export SLArray
# for implements operators
struct SLArray{T, N} <: AbstractMPIArray{T, N}
    sizes::NTuple{N,Int} # # global matrix size
    localarray::Array{T,N}
    partitioning::Partitioning{N}
    comm::MPI.Comm
    win::MPI.Win
    myrank::Int

    function SLArray(T::Type, m::Integer, n::Integer; comm=MPI.COMM_WORLD, proc_grids=(1, MPI.Comm_size(MPI.COMM_WORLD)))
        m_proc, n_proc = proc_grids
        m_blocksize = m ÷ m_proc
        n_blocksize = n ÷ n_proc
        blocksizes = (m_blocksize, n_blocksize)
        A = CyclicMPIArray(T, m, n, comm=comm, proc_grids=proc_grids, blocksizes=blocksizes)
        return new{T, 2}(A.sizes, A.localarray, A.partitioning, A.comm, A.win, A.myrank)
    end

    # function SLArray(T::Type, m::Integer, n::Integer; comm=MPI.COMM_WORLD, proc_grids=(MPI.Comm_size(MPI.COMM_WORLD), ones(Int, N-1)...), blocksizes)
    #     A = CyclicMPIArray(T, sizes..., comm=comm, proc_grids=proc_grids, blocksizes=blocksizes)
    #     return new{T, 2}(A.sizes, A.localarray, A.partitioning, A.comm, A.win, A.myrank)
    # end
end

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

    C = SLArray(T, m_A, n_B, proc_grids=proc_grid_A)
    A_mul_B!(alpha, A, B, beta, C)
    return C
end