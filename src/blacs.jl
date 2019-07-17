module BLACS

using ..ScaLAPACK: libscalapack

function get(icontxt::Integer, what::Integer)
    val = Int32[0]
    ccall((:blacs_get_, libscalapack), Cvoid, (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}), Ref{icontxt}, Ref{what}, val)
    return val[1]
end

function gridinit(icontxt::Integer, order::Char, nprow::Integer, npcol::Integer)
    icontxta = Int32[icontxt]
    ccall((:blacs_gridinit_, libscalapack), Cvoid,
        (Ptr{Int32}, Ptr{UInt8}, Ptr{Int32}, Ptr{Int32}),
        icontxta, Ref{order}, Ref{nprow}, Ref{npcol})
    icontxta[1]
end

function pinfo()
    mypnum, nprocs = Int32[0], Int32[0]
    ccall((:blacs_pinfo_, libscalapack), Cvoid,
        (Ptr{Int32}, Ptr{Int32}),
        mypnum, nprocs)
    return mypnum[1], nprocs[1]
end

function gridinfo(ictxt::Integer)
    nprow = Int32[0]
    npcol = Int32[0]
    myprow = Int32[0]
    mypcol = Int32[0]
    ccall((:blacs_gridinfo_, libscalapack), Cvoid,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
        Ref{ictxt}, nprow, npcol, myprow, mypcol)
    return nprow[1], npcol[1], myprow[1], mypcol[1]
end

gridexit(ictxt::Integer) = ccall((:blacs_gridexit_, libscalapack), Cvoid, (Ptr{Int32},), Ref{ictxt})

exit(cont = 0) = ccall((:blacs_exit_, libscalapack), Cvoid, (Ptr{Int},), Ref{cont})

end
