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
# 12 x 12 matrix, process grid is 2 x 2, blocksize is 2 x 2
A = SLMatrix{Float64}(12, 12, proc_grids=(2, 2), blocksizes=(2, 2))

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




## code examples

[Matrix product](./test/test_mul_mpi.jl)

[Eigen hermitian](./test/test_pxheev.jl)

[Eigen general complex](./test/test_eigen.jl)

[SVD](./test/test_svdvals_mpi.jl)
