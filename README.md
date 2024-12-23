# Mini benchmark for JAX

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)

This package provides a set of benchmark scripts that can be used to profile JAX performance on a varying number of CPU cores. JAX does not provide control over the number of cores it uses, so a common trick is to work do this with [`taskset`](https://man7.org/linux/man-pages/man1/taskset.1.html).

The benchmarks can be run by installing the package with [`pip`](https://pypi.org/project/pip) and running it as follows:

```shell
python3 -m pip install git+https://github.com/ComPWA/jax-mini-benchmark@main
benchmark-jax
```

The resulting benchmark can be viewed in `jax-benchmark-$HOSTNAME.svg`. If you do not want to view the resulting plot directly, like when you run this command in a script, add the `--no-show` flag:

```shell
benchmark-jax --no-show
```

## Help developing

We recommend working with a virtual environment (more info here). If you have installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers), the project can easily be set up as follows:

```shell
git clone https://github.com/ComPWA/jax-mini-benchmark
cd jax-mini-benchmark
conda env create
conda activate jax-mini-benchmark
pre-commit install  # optional, but recommended
```

See ComPWA's [Help developing](https://compwa.github.io/develop) for more info.
