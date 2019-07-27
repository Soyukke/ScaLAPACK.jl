
# Initialize
function sl_init(nprow::Integer, npcol::Integer)
    ictxt = zeros(Cint, 1)
    ccall((:sl_init_, libscalapack), Cvoid,
        (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        ictxt, [nprow], [npcol])
    return ictxt[1]
end

# Calculate size of local array
function numroc(n::Integer, nb::Integer, iproc::Integer, isrcproc::Integer, nprocs::Integer)
    ccall((:numroc_, libscalapack), Cint,
        (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        Ref(Cint(n)), Ref(Cint(nb)), Ref(Cint(iproc)), Ref(Cint(isrcproc)), Ref(Cint(nprocs)))
end

# Array descriptor
function descinit(m::Integer, n::Integer, mb::Integer, nb::Integer, irsrc::Integer, icsrc::Integer, ictxt::Integer, lld::Integer)

    # extract values
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ictxt)
    locrm = numroc(m, mb, myrow, irsrc, nprow)

    # checks
    m >= 0 || throw(ArgumentError("first dimension must be non-negative"))
    n >= 0 || throw(ArgumentError("second dimension must be non-negative"))
    mb > 0 || throw(ArgumentError("first dimension blocking factor must be positive"))
    nb > 0 || throw(ArgumentError("second dimension blocking factor must be positive"))
    0 <= irsrc < nprow || throw(ArgumentError("process row must be positive and less that grid size"))
    0 <= irsrc < nprow || throw(ArgumentError("process column must be positive and less that grid size"))
    # lld >= locrm || throw(ArgumentError("leading dimension of local array is too small"))

    # allocation
    desc = zeros(Cint, 9)
    info = Cint[1]

    # ccall
    ccall((:descinit_, libscalapack), Cvoid,
        (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
         Ptr{Cint}, Ptr{Cint}),
        desc, Ref(Cint(m)), Ref(Cint(n)), Ref(Cint(mb)),
        Ref(Cint(nb)), Ref(Cint(irsrc)), Ref(Cint(icsrc)), Ref(Cint(ictxt)),
        Ref(Cint(lld)), info)

    info[1] == 0 || error("input argument $(info[1]) has illegal value")

    return desc
end

# Redistribute arrays
for (fname, elty) in ((:psgemr2d_, :Float32),
                      (:pdgemr2d_, :Float64),
                      (:pcgemr2d_, :ComplexF32),
                      (:pzgemr2d_, :ComplexF64))
    @eval begin
        function pxgemr2d!(m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{Cint}, B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{Cint}, ictxt::Integer)

            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                 Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                 Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                Ref(Cint(m)), Ref(Cint(n)), A, Ref(Cint(ia)),
                Ref(Cint(ja)), desca, B, Ref(Cint(ib)),
                Ref(Cint(jb)), descb, Ref(Cint(ictxt)))
        end
    end
end

##################
# Linear Algebra #
##################

# Matmul
for (fname, elty) in ((:psgemm_, :Float32),
                      (:pdgemm_, :Float64),
                      (:pcgemm_, :ComplexF32),
                      (:pzgemm_, :ComplexF64))
    @eval begin
        function pdgemm!(transa::Char, transb::Char, m::Integer, n::Integer, k::Integer, α::$elty, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{Cint}, B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{Cint}, β::$elty, C::Matrix{$elty}, ic::Integer, jc::Integer, descc::Vector{Cint})

            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{Cint}, Ptr{Cint},
                 Ptr{Cint}, Ptr{$elty}, Ptr{$elty}, Ptr{Cint},
                 Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                 Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{$elty},
                 Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                Ref(Cuchar(transa)), Ref(Cuchar(transb)), Ref(Cint(m)), Ref(Cint(n)),
                Ref(Cint(k)), Ref(α), A, Ref(Cint(ia)),
                Ref(Cint(ja)), desca, B, Ref(Cint(ib)),
                Ref(Cint(jb)), descb, Ref(β), C,
                Ref(Cint(ic)), Ref(Cint(jc)), descc)
        end
    end
end

# Eigensolves
for (fname, elty) in ((:psstedc_, :Float32),
                      (:pdstedc_, :Float64))
    @eval begin
        function pxstedc!(compz::Char, n::Integer, d::Vector{$elty}, e::Vector{$elty}, Q::Matrix{$elty}, iq::Integer, jq::Integer, descq::Vector{Cint})


            work    = $elty[0]
            lwork   = convert(Cint, -1)
            iwork   = Cint[0]
            liwork  = convert(Cint, -1)
            info    = Cint[0]

            for i = 1:2
                ccall(($(string(fname)), libscalapack), Cvoid,
                    (Ptr{UInt8}, Ptr{UInt8}, Ptr{$elty}, Ptr{$elty},
                     Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                     Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                     Ptr{$Cint}),
                    [Cuchar(compz)], [n], d, e,
                    Q, [iq], [jq], descq,
                    work, [lwork], iwork, [liwork],
                    info)

                if i == 1
                    lwork = convert(Cint, work[1])
                    work = zeros($elty, lwork)
                    liwork = convert(Cint, iwork[1])
                    iwork = zeros(Cint, liwork)
                end
            end

            return d, Q
        end
    end
end

# symmetric matrix eigenvalues
for (fname, elty) in ((:pssyev_, :Float32), (:pdsyev_, :Float64))
    @eval begin
        # only ccall
        function $fname(JOBZ::Cuchar, UPLO::Cuchar, N::Cint, 
            A::Matrix{$elty}, IA::Cint, JA::Cint, DESCA::Vector{Cint}, 
            W::Vector{$elty}, Z::Matrix{$elty}, IZ::Cint, JZ::Cint, DESCZ::Vector{Cint}, 
            WORK::Vector{$elty}, LWORK::Cint, INFO::Vector{Cint})
            """
            JOBZ    : 'N': eigenvalues only, 'V': eigenvalues and eigenvectors
            UPLO    : 'U': upper triangular, 'L': lower triangular
            N       : matrix row/col size
            A       : cyclic symmetric matrix
            IA      : index i
            JA      : index j
            DESCA   : 
            W       : out Float32/Float64, eigenvalues
            Z       : out Float32/Float64, eigenvectors
            IZ      : in
            JZ      : in
            DESCZ   : 
            WORK    : out elty array
            LWORK   : in Integer
            INFO    : out Integer = 0 success
            """
            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cuchar}, Ptr{Cuchar}, Ptr{Cint}, Ptr{$elty},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                Ref(JOBZ), Ref(UPLO), Ref(N), A,
                Ref(IA), Ref(JA), DESCA, W,
                Z, Ref(IZ), Ref(JZ), DESCZ,
                WORK, Ref(LWORK), INFO)
        end # function

        # wrap
        function pXsyev!(N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, W::Vector{$elty}, Z::Matrix{$elty}, DESCZ::Vector{Cint})
            LWORK::Cint = -1
            WORK = Vector{$elty}(undef, 1)
            INFO = Cint[0]
    
            $fname(Cuchar('V'), Cuchar('U'), N, A, Cint(1), Cint(1), DESCA, W, Z, Cint(1), Cint(1), DESCZ, WORK, LWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")

            # allocate work space memory
            LWORK = real(WORK[1])
            WORK = Vector{$elty}(undef, LWORK)

            $fname(Cuchar('V'), Cuchar('U'), N, A, Cint(1), Cint(1), DESCA, W, Z, Cint(1), Cint(1), DESCZ, WORK, LWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")
        end # wrap function

        function pXheev!(N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, W::Vector{$elty}, Z::Matrix{$elty}, DESCZ::Vector{Cint})
            pXsyev!(N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, W::Vector{$elty}, Z::Matrix{$elty}, DESCZ::Vector{Cint})
        end
    end # eval begin
end

# Hermitian Eigensolves
for (fname, elty) in ((:pcheev_, :ComplexF32),
                      (:pzheev_, :ComplexF64))
    @eval begin
        function $fname(JOBZ::Cuchar, UPLO::Cuchar, N::Cint, 
            A::Matrix{$elty}, IA::Cint, JA::Cint, DESCA::Vector{Cint}, 
            W::Vector{typeof(real($elty(0)))}, Z::Matrix{$elty}, IZ::Cint, JZ::Cint, DESCZ::Vector{Cint}, 
            WORK::Vector{$elty}, LWORK::Cint, RWORK::Vector{$elty}, LRWORK::Cint, INFO::Vector{Cint})
            """
            JOBZ    : 'N': eigenvalues only, 'V': eigenvalues and eigenvectors
            UPLO    : 'U': upper triangular, 'L': lower triangular
            N       : matrix row/col size
            A       : cyclic matrix Complex
            IA      : index i
            JA      : index j
            DESCA   : 
            W       : out Float32/Float64, eigenvalues
            Z       : out eigenvectors
            IZ      : in
            JZ      : in
            DESCZ   : 
            WORK    : out Complex array
            LWORK   : in Integer
            RWORK   : out Complex
            LRWORK  : in Integer
            INFO    : out Integer = 0 success
            """
            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cuchar}, Ptr{Cuchar}, Ptr{Cint}, Ptr{$elty},
                    Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{typeof(real($elty(0)))},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                Ref(JOBZ), Ref(UPLO), Ref(N), A,
                Ref(IA), Ref(JA), DESCA, W,
                Z, Ref(IZ), Ref(JZ), DESCZ,
                WORK, Ref(LWORK), RWORK, Ref(LRWORK), INFO)

        end # function

        # wrap
        function pXheev!(N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, W::Vector{typeof(real($elty(0)))}, Z::Matrix{$elty}, DESCZ::Vector{Cint})
            WORK = Vector{$elty}(undef, 1)
            LWORK::Cint = -1
            RWORK = Vector{$elty}(undef, 1)
            LRWORK::Cint = -1
            INFO = Cint[0]
    
            $fname(Cuchar('V'), Cuchar('U'), N, A, Cint(1), Cint(1), DESCA, W, Z, Cint(1), Cint(1), DESCZ, WORK, LWORK, RWORK, LRWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")

            # allocate work space memory
            LWORK = real(WORK[1])
            LRWORK = real(RWORK[1])
            WORK = Vector{$elty}(undef, LWORK)
            RWORK = Vector{$elty}(undef, LRWORK)

            $fname(Cuchar('V'), Cuchar('U'), N, A, Cint(1), Cint(1), DESCA, W, Z, Cint(1), Cint(1), DESCZ, WORK, LWORK, RWORK, LRWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")
        end # wrap function

    end # eval begin
end


# SVD solver
for (fname, elty) in ((:psgesvd_, :Float32),
                      (:pdgesvd_, :Float64))
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::StridedMatrix{$elty}, ia::Integer, ja::Integer, desca::Vector{Cint}, s::StridedVector{$elty}, U::StridedMatrix{$elty}, iu::Integer, ju::Integer, descu::Vector{Cint}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{Cint})
            # allocate
            info = zeros(Cint, 1)
            work = zeros($elty, 1)
            lwork::Cint = -1

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Cvoid,
                    (Ptr{UInt8}, Ptr{UInt8}, Ptr{Cint}, Ptr{Cint},
                     Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                     Ptr{$elty}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                     Ptr{Cint}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                     Ptr{Cint}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                    Ref(Cuchar(jobu)), Ref(Cuchar(jobvt)), Ref(Cint(m)), Ref(Cint(n)),
                    A, Ref(Cint(ia)), Ref(Cint(ja)), desca,
                    s, U, Ref(Cint(iu)), Ref(Cint(ju)),
                    descu, Vt, Ref(Cint(ivt)), Ref(Cint(jvt)),
                    descvt, work, Ref(lwork), info)
                if i == 1
                    lwork = convert(Cint, work[1])
                    work = zeros($elty, lwork)
                end
            end

            if 0 < info[1] <= min(m,n)
                throw(ScaLAPACKException(info[1]))
            end

            return U, s, Vt
        end
    end
end
for (fname, elty, relty) in ((:pcgesvd_, :ComplexF32, :Float32),
                             (:pzgesvd_, :ComplexF64, :Float64))
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{Cint}, s::Vector{$relty}, U::Matrix{$elty}, iu::Integer, ju::Integer, descu::Vector{Cint}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{Cint})
            # extract values

            # check

            # allocate
            info = zeros(Cint, 1)
            work = zeros($elty, 1)
            rwork = zeros($relty, 1 + 4*max(m, n))
            lwork = -1

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Cvoid,
                    (Ptr{UInt8}, Ptr{UInt8}, Ptr{Cint}, Ptr{Cint},
                     Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                     Ptr{$relty}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                     Ptr{Cint}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                     Ptr{Cint}, Ptr{$elty}, Ptr{Cint}, Ptr{$relty},
                     Ptr{Cint}),
                    [Cuchar(jobu)], [Cuchar(jobvt)], [m], [n],
                    A, [ia], [ja], desca,
                    s, U, [iu], [ju],
                    descu, Vt, [ivt], [jvt],
                    descvt, work, [lwork], rwork,
                    info)
                if i == 1
                    lwork = convert(Cint, work[1])
                    work = zeros($elty, lwork)
                end
            end

            info[1] > 0 && throw(ScaLAPACKException(info[1]))

            return U, s, Vt
        end
    end
end


# general matrix -> upper Hessenberg
for (fname, elty) in ((:psgehrd_, :Float32), (:pdgehrd_, :Float64), (:pcgehrd_, :ComplexF32), (:pzgehrd_, :ComplexF64))
    @eval begin
        function $fname(N::Cint, ILO::Cint, IHI::Cint, 
            A::Matrix{$elty}, IA::Cint, JA::Cint, DESCA::Vector{Cint}, TAU::Vector{$elty},
            WORK::Vector{$elty}, LWORK::Cint, INFO::Vector{Cint})
            """
            N       : matrix row/col size
            ILO     : upper triangular
            IHI     : upper triangular
            A       : cyclic matrix local matrix
            IA      : local index i
            JA      : local index j
            DESCA   : descriptor of A
            TAU     : (local output) size NUMROC(JA+N-2, NB_A, MYCOL, CSRC_A, NPCOL), 
            WORK    : out Complex array
            LWORK   : in Integer
            INFO    : out Integer = 0 success
            """
            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                Ref(N), Ref(ILO), Ref(IHI),
                A, Ref(IA), Ref(JA), DESCA, TAU,
                WORK, Ref(LWORK), INFO)
        end # function

        # wrap
        function pXgehrd!(N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, TAU::Vector{$elty})
            WORK = Vector{$elty}(undef, 1)
            LWORK::Cint = -1
            INFO = Cint[0]
            ILO::Cint = 1
            IHI::Cint = N
    
            $fname(N, ILO, IHI, A, Cint(1), Cint(1), DESCA, TAU, WORK, LWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")

            # allocate work space memory
            LWORK = real(WORK[1])
            WORK = Vector{$elty}(undef, LWORK)

            $fname(N, ILO, IHI, A, Cint(1), Cint(1), DESCA, TAU, WORK, LWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")
        end # wrap function

    end # eval begin
end

# pXgehrd -> get hessenberg Q
for (fname, elty) in ((:psormhr_, :Float32), (:pdormhr_, :Float64), (:pcunmhr_, :ComplexF32), (:pzunmhr_, :ComplexF64))
    @eval begin
        function $fname(SIDE::Cuchar, TRANS::Cuchar, M::Cint, N::Cint, ILO::Cint, IHI::Cint, 
            A::Matrix{$elty}, IA::Cint, JA::Cint, DESCA::Vector{Cint}, TAU::Vector{$elty},
            C::Matrix{$elty}, IC::Cint, JC::Cint, DESCC::Vector{Cint},
            WORK::Vector{$elty}, LWORK::Cint, INFO::Vector{Cint})
            """
            SIDE    : 'L', 'R'
            TRANS   : 'N', 'T' transpose Q
            M       : matrix row size C
            N       : matrix col size C
            ILO     : upper triangular
            IHI     : upper triangular
            A       : cyclic matrix local matrix
            IA      : local index i
            JA      : local index j
            DESCA   : descriptor of A
            TAU     : (local output) size NUMROC(JA+M-2, NB_A, MYCOL, CSRC_A, NPCOL) if SIDE='L', NUMROC(JA+M-2, NB_A, MYCOL, CSRC_A, NPCOL) if SIDE='R'
            C       : cyclic matrix local matrix
            IC      : local index i
            JC      : local index j
            DESCC   : descriptor of C
            WORK    : out Complex array
            LWORK   : in Integer
            INFO    : out Integer = 0 success
            """
            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cuchar}, Ptr{Cuchar}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}),
                Ref(SIDE), Ref(TRANS), Ref(M), Ref(N), Ref(ILO), Ref(IHI),
                A, Ref(IA), Ref(JA), DESCA, TAU,
                C, Ref(IC), Ref(JC), DESCC,
                WORK, Ref(LWORK), INFO)
        end # function

        # wrap
        function pXunmhr!(M::Cint, N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, TAU::Vector{$elty}, C::Matrix{$elty}, DESCC::Vector{Cint})
            WORK = Vector{$elty}(undef, 1)
            LWORK::Cint = -1
            INFO = Cint[0]
            ILO::Cint = 1
            IHI::Cint = M
    
            $fname(Cuchar('L'), Cuchar('N'), M, N, ILO, IHI, A, Cint(1), Cint(1), DESCA, TAU, C, Cint(1), Cint(1), DESCC, WORK, LWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")

            # allocate work space memory
            LWORK = real(WORK[1])
            WORK = Vector{$elty}(undef, LWORK)

            $fname(Cuchar('L'), Cuchar('N'), M, N, ILO, IHI, A, Cint(1), Cint(1), DESCA, TAU, C, Cint(1), Cint(1), DESCC, WORK, LWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")
        end # wrap function

    end # eval begin
end

for (fname, elty) in ((:pslahqr_, :Float32), (:pdlahqr_, :Float64))
    @eval begin
        function $fname(WANTT::Bool , WANTZ::Bool, N::Cint,
            ILO::Cint, IHI::Cint, A::Matrix{$elty}, DESCA::Vector{Cint},
            WR::Vector{$elty}, WI::Vector{$elty}, ILOZ::Cint, IHIZ::Cint, Z::Matrix{$elty}, DESCZ::Vector{Cint},
            WORK::Vector{$elty}, LWORK::Cint, IWORK::Vector{Cint}, ILWORK::Cint, INFO::Vector{Cint})
            """
            WANTT   : .TRUE. full Schur form T is required, .FALSE. only eigenvalues are required
            WANTZ   : .TRUE. required Schur vectors Z, .FALSE. not required
            N       : matrix row/col size
            ILO     : upper triangular
            IHI     : upper triangular
            A       : (global) upper hessenberg matrix
            DESCA   : descriptor of A
            WR       : eigenvalues real part
            WI       : eigenvalues imag part
            ILOZ    :
            IHIZ    :
            Z       : if WANTZ true
            DESCZ
            WORK    : out Complex array
            LWORK   : in Integer
            IWORK   : not use
            ILWORK  : not use
            INFO    : out Integer = 0 success
            """
            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                    Ptr{$elty}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                Ref(Cint(WANTT)), Ref(Cint(WANTZ)), Ref(N),
                Ref(ILO), Ref(IHI), A, DESCA,
                WR, WI, Ref(ILOZ), Ref(IHIZ), Z, DESCZ,
                WORK, Ref(LWORK), IWORK, Ref(ILWORK), INFO)
        end # function

        # wrap
        function pXlahqr!(N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, WR::Vector{$elty}, WI::Vector{$elty}, Z::Matrix{$elty}, DESCZ::Vector{Cint}, LWORK::Cint)
            WORK = Vector{$elty}(undef, LWORK)
            IWORK = Vector{Cint}(undef, 1)
            ILWORK::Cint = -1

            INFO = Cint[0]
            ILO::Cint = 1
            IHI::Cint = N
            ILOZ::Cint = 1
            IHIZ::Cint = N
    
            $fname(true, true, N, ILO, IHI, A, DESCA, WR, WI, ILOZ, IHIZ, Z, DESCZ, WORK, LWORK, IWORK, ILWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")
        end # wrap function

    end # eval begin
end

for (fname, elty) in ((:pclahqr_, :ComplexF32), (:pzlahqr_, :ComplexF64))
    @eval begin
        function $fname(WANTT::Bool , WANTZ::Bool, N::Cint,
            ILO::Cint, IHI::Cint, A::Matrix{$elty}, DESCA::Vector{Cint},
            W::Vector{$elty}, ILOZ::Cint, IHIZ::Cint, Z::Matrix{$elty}, DESCZ::Vector{Cint},
            WORK::Vector{$elty}, LWORK::Cint, IWORK::Vector{Cint}, ILWORK::Cint, INFO::Vector{Cint})
            """
            WANTT   : .TRUE. full Schur form T is required, .FALSE. only eigenvalues are required
            WANTZ   : .TRUE. required Schur vectors Z, .FALSE. not required
            N       : matrix row/col size
            ILO     : upper triangular
            IHI     : upper triangular
            A       : (global) upper hessenberg matrix
            DESCA   : descriptor of A
            W       : eigenvalues
            ILOZ    :
            IHIZ    :
            Z       : if WANTZ true
            DESCZ
            WORK    : out Complex array
            LWORK   : in Integer
            IWORK   : not use
            ILWORK  : not use
            INFO    : out Integer = 0 success
            """
            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                Ref(Cint(WANTT)), Ref(Cint(WANTZ)), Ref(N),
                Ref(ILO), Ref(IHI), A, DESCA,
                W, Ref(ILOZ), Ref(IHIZ), Z, DESCZ,
                WORK, Ref(LWORK), IWORK, Ref(ILWORK), INFO)
        end # function

        # wrap
        function pXlahqr!(N::Cint, A::Matrix{$elty}, DESCA::Vector{Cint}, W::Vector{$elty}, Z::Matrix{$elty}, DESCZ::Vector{Cint})
            WORK = Vector{$elty}(undef, 1)
            LWORK::Cint = -1
            IWORK = Vector{Cint}(undef, 1)
            ILWORK::Cint = -1

            INFO = Cint[0]
            ILO::Cint = 1
            IHI::Cint = N
            ILOZ::Cint = 1
            IHIZ::Cint = N
    
            $fname(true, true, N, ILO, IHI, A, DESCA, W, ILOZ, IHIZ, Z, DESCZ, WORK, LWORK, IWORK, ILWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")

            # allocate work space memory
            LWORK = real(WORK[1])
            WORK = Vector{$elty}(undef, LWORK)
            $fname(true, true, N, ILO, IHI, A, DESCA, W, ILOZ, IHIZ, Z, DESCZ, WORK, LWORK, IWORK, ILWORK, INFO)

            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")
        end # wrap function

    end # eval begin
end

for (fname, elty) in ((:pctrevc_, :ComplexF32), (:pztrevc_, :ComplexF64))
    @eval begin
        ```
            compute eigenvectors of a upper triangular matrix T
        ```
        function $fname(SIDE::Cuchar, HOWMNY::Cuchar, SELECT::Vector{Cint}, N::Cint,
            T::Matrix{$elty}, DESCT::Vector{Cint}, 
            VL::Matrix{$elty}, DESCVL::Vector{Cint},
            VR::Matrix{$elty}, DESCVR::Vector{Cint},
            MM::Cint, M::Vector{Cint},
            WORK::Vector{$elty}, RWORK::Vector{typeof(real($elty(0)))}, INFO::Vector{Cint})
            """
            SIDE    : 'R' compute right eigenvectors only, 'L' , 'B' compute both
            HOWMNY  : 'A': compute all eigenvectors, 'B':, 'S': compute selected eigenvectors
            SELECT  : logical array if HOWMNY = 'A' or 'B', SELECT is not refferenced.
            N       : T matrix row/col size
            T       : upper triangular matrix
            DESCT   : descriptor of T
            VL      : 
            DESCVL  :
            VR      : 
            DESCVR  :
            MM      : (global input) number of col VL and/or VR MM >= M
            M       : (global output)
            WORK    : out Complex array
            RWORK
            INFO    : out Integer = 0 success
            """
            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cuchar}, Ptr{Cuchar}, Ptr{Cint}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint},
                    Ptr{$elty}, Ptr{Cint},
                    Ptr{Cint}, Ptr{Cint},
                    Ptr{$elty}, Ptr{typeof(real($elty(0)))}, Ptr{Cint}),
                Ref(SIDE), Ref(HOWMNY), SELECT, Ref(N),
                T, DESCT,
                VL, DESCVL,
                VR, DESCVR,
                Ref(MM), M,
                WORK, RWORK, INFO)
        end # function

        # wrap
        function pXtrevc!(SIDE::Char, HOWMNY::Char, SELECT::Vector{Bool}, N::Integer, T::Matrix{$elty}, DESCT::Vector{Cint}, 
            VL::Matrix{$elty}, DESCVL::Vector{Cint}, VR::Matrix{$elty}, DESCVR::Vector{Cint}, LLD_T::Integer)
            MM = N
            M = Cint[-1]
            WORK = Vector{$elty}(undef, 2*LLD_T)
            RWORK = Vector{typeof(real($elty(0)))}(undef, LLD_T)
            INFO = Cint[-1]
    
            $fname(Cuchar(SIDE), Cuchar(HOWMNY), Cint.(SELECT), Cint(N),
            T, DESCT, VL, DESCVL, VR, DESCVR, 
            MM, M, WORK, RWORK, INFO)
            func_name = $(string(fname))
            INFO[1] != 0 && print("error in $(func_name) INFO=$(INFO[1])\n")
        end # wrap function

    end # eval begin
end

