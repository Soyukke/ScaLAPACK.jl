# ScaLAPACK.jl

ScaLAPACK wrapper for Julia 1.1.1

## Description

This module is forked from <https://github.com/JuliaParallel/ScaLAPACK.jl>
Not dependent on DistributedArrays.jl to use ScaLAPACK under HPC.



## Set up

### Add packages

```
pkg> add https://github.com/Soyukke/MPIArrays.jl
pkg> add https://github.com/Soyukke/ScaLAPACK.jl
```

### Make libscalapack.so

put libscalapack.so in `deps` directory

Library creation example. It uses Intel mkl.
```
mpiifort -shared -o libscalapack.so -mkl -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_avx512
```



## Usage

### 1. Make Block-Cyclic Distributed MPIArray

```julia
using ScaLAPACK, MPI, MPIArrays

MPI.Init()
A = CyclicMPIArray(Float64, 8, 8)

forlocalpart!(A) do localA
end
```

### 2. Use function implemented in convenience_mpi.jl

```
eigen_hermitian(A)
```

### 3. run

run
```julia
$ mpiexec -np 4 julia sample.jl
```




## Example

Matrix product
./test/test_mul_mpi.jl

Eigen hermitian
./test/test_pxheev.jl

SVD
./test/test_svdvals_mpi.jl
