# deal.II CEED BP1/BP3 FEM Throughput benchmark

This is a simplified version of the [CEED bake-off problems](https://ceed.exascaleproject.org/bps/) using [deal.II](https://dealii.org) running on GPUs using the [Portable::MatrixFree class](https://dealii.org/developer/doxygen/deal.II/classPortable_1_1MatrixFree.html).

## Setup

1. Compile Kokkos with device support
2. Compile deal.II 9.8 or newer with p4est and Kokkos and device support
3. Configure this project and pointing it to the deal.II installation using DEAL_II_DIR.

Also see [GPU Instructions in the deal.II Wiki](https://github.com/dealii/dealii/wiki/GPU-support-in-deal.II).

## Usage

```
$ ./example -h
Usage: ./example --degree <p> --bp {0|1|3} --ts <teamsize>
$ ./example --degree 4 --bp 1 --ts 32
```

The Kokkos team size is an important tuning parameter that can lead to a 2-3x performance improvement over the default depending on the polynomial degree and the device used.
