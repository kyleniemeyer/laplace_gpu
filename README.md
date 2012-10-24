laplace_gpu
===========

Laplace solver running on GPU using [CUDA], with CPU version for comparison. In addition, the CPU version contains support for [OpenMP] and [OpenACC].

[CUDA]: http://www.nvidia.com/object/cuda_home_new.html
[OpenMP]: http://openmp.org/wp/
[OpenACC]: http://www.openacc-standard.org/

Usage
-----

Change the problem size by modifying `NUM` in `laplace_cpu.c` and `laplace_gpu.cu`. To make all the executables (CPU, OpenMP, OpenACC, and GPU) type `make`. 

To run each version, simply type

	$ ./laplace_cpu
	$ ./laplace_gpu
	$ ./laplace_omp
	$ ./laplace_acc

Requirements
------------

The makefile is set up to use `gcc` to compile the CPU version, `nvcc` (part of [CUDA]) to make the GPU version, and 'pgcc' (part of the [Portland Group]'s Accelerator compilers) to make the OpenMP and OpenACC versions. `gcc` may also be used to make the OpenMP version by changing the appropriate line in the makefile.

[CUDA]: http://www.nvidia.com/object/cuda_home_new.html
[Portland Group]: http://www.pgroup.com

Misc
-------

The most up-to-date version of `laplace_gpu` can be found at the [GitHub repository](https://github.com/kyleniemeyer/laplace_gpu) on GitHub.

License
-------

`laplace_gpu` is released under the modified BSD license, see LICENSE for details.

Author
------

Created by [Kyle Niemeyer](http://kyleniemeyer.com). Email address: [kyle.niemeyer@gmail.com](mailto:kyle.niemeyer@gmail.com)
