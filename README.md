# ScaLAPACK.jl

ScaLAPACK wrapper for Julia 1.1.1

## Description

This module is forked from <https://github.com/JuliaParallel/ScaLAPACK.jl>
Not dependent on DistributedArrays.jl to use ScaLAPACK under HPC.

## Set up

put libscalapack.so in `deps` directory

Library creation example. It uses Intel mkl.
```
mpiifort -shared -o libscalapack.so -mkl -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_avx512
```

## Usage

run
```julia
$ mpiexec.hydra -np 4 julia sample.jl
```

### 1. Make MPIArray

```julia
```


## Example

Matrix product
./test/test_mul_mpi.jl

Eigen hermitian
./test/test_pxheevd.jl

SVD
./test/test_svdvals_mpi.jl
