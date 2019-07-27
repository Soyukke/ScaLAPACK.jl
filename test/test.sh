#!/bin/bash
mpirun -np 4 julia --project=. ./test/test_mul_mpi.jl
mpirun -np 4 julia --project=. ./test/test_pxheev.jl
mpirun -np 4 julia --project=. ./test/test_svdvals_mpi.jl
mpirun -np 4 julia --project=. ./test/test_eigen.jl