# adjoint
for (fname, elty) in ((:pstran_, :Float32), (:pdtran_, :Float64), (:pctranc_, :ComplexF32), (:pztranc_, :ComplexF64))
    @eval begin
        function $fname(M::Cint, N::Cint, 
            ALPHA::$elty, A::Matrix{$elty}, IA::Cint, JA::Cint, DESCA::Vector{Cint}, 
            BETA::$elty, C::Matrix{$elty}, IC::Cint, JC::Cint, DESCC::Vector{Cint})

            ccall(($(string(fname)), libscalapack), Cvoid,
                (Ptr{Cint}, Ptr{Cint}, Ptr{$elty}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint},
                Ptr{$elty}, Ptr{$elty}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
                Ref(M), Ref(N), Ref(ALPHA), A, Ref(IA), Ref(JA), DESCA,
                Ref(BETA), C, Ref(IC), Ref(JC), DESCC)
        end

        function pXtranc!(M::Integer, N::Integer, α::$elty, A::Matrix{$elty}, DESCA::Vector{Cint}, β::$elty, C::Matrix{$elty}, DESCC::Vector{Cint})
            # C = β*C + α*A'
            IA, JA, IC, JC = 1, 1, 1, 1
            $fname(Cint(M), Cint(N), α, A, Cint(IA), Cint(JA), DESCA, β, C, Cint(IC), Cint(JC), DESCC)
        end

    end

end
