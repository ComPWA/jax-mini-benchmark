# Mini benchmark for JAX

This repository provides as set of benchmarks that can be used to profile JAX performance on a varying number of CPU cores. JAX does not provide control over the number of cores it uses, so a common trick is to work do this with [`taskset`](https://man7.org/linux/man-pages/man1/taskset.1.html).

To download an install this repository, [install Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) and run:

```shell
git clone https://github.com/ComPWA/jax-mini-benchmark
cd jax-mini-benchmark
conda env create
conda activate jax-mini-benchmark
bash benchmark-jax.sh
```

The resulting benchmark is rendered as an HTML page (`visualize-benchmark.html`).
