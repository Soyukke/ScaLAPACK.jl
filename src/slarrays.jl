export SLArray, SLMatrix, SLVector

import MPIArrays.AbstractMPIArray
import MPIArrays.Partitioning

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

    function SLArray{T, 2}(m::Integer, n::Integer; comm=MPI.COMM_WORLD, proc_grids=(1, MPI.Comm_size(MPI.COMM_WORLD)), blocksizes=(1, 1)) where T
        A = CyclicMPIArray(T, Int(m), Int(n), comm=comm, proc_grids=Int.(proc_grids), blocksizes=Int.(blocksizes))
        return new{T, 2}(A.sizes, A.localarray, A.partitioning, A.comm, A.win, A.myrank)
    end
end

SLMatrix{T} = SLArray{T, 2} where T
SLVector{T} = SLArray{T, 1} where T


