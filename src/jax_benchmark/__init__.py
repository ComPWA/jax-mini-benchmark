from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from jax.config import config
from tqdm.auto import tqdm

config.update("jax_enable_x64", True)
logging.getLogger("jax").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat",
        "-r",
        nargs=1,
        default=10,
        help="Number of repeats",
        type=int,
    )
    parser.add_argument(
        "--number",
        "-n",
        nargs=1,
        default=5,
        help="Number of times that the job should be run on a single repeat",
        type=int,
    )
    args = parser.parse_args()
    benchmarks = benchmark_cpu_range(
        repeat=args.repeat,
        number=args.number,
    )
    visualize(benchmarks)
    return 0


def benchmark_cpu_range(number: int, repeat: int) -> dict[int, dict]:
    n_available_cpus = os.cpu_count()
    print(
        f"Machine has {n_available_cpus} CPUs."
        " JAX will be benchmarked over a range of them."
    )
    benchmarks: dict[int, dict] = {}
    combinations = list(range(1, n_available_cpus + 1))
    for n_cpus in tqdm(combinations):
        benchmarks[n_cpus] = run_single_benchmark(
            n_cpus, number, repeat, shape=(5000, 100)
        )
    return benchmarks


def visualize(benchmarks: dict[int, dict]) -> None:
    fig, ax = plt.subplots()
    x = np.array(sorted(benchmarks))
    data = np.array([benchmarks[n_cpus]["results_in_seconds"] for n_cpus in x])
    ax.errorbar(
        x=x,
        y=data.mean(axis=1),
        yerr=data.std(axis=1),
    )
    filename = get_figure_filename()
    _, ymax = ax.get_ylim()
    hostname = socket.gethostname()
    fig.suptitle(f"JAX dot product on {hostname}")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Number of CPUs")
    ax.set_ylabel("Average time (s)")
    fig.savefig(filename)


def run_single_benchmark(
    n_cpus: int, number: int, repeat: int, shape: tuple[int, int]
) -> dict:
    filename = get_benchmark_filename(n_cpus)
    if not os.path.exists(filename):
        shape_str = "x".join(map(str, shape))
        filename.parent.mkdir(exist_ok=True)
        subprocess.call(
            f"taskset -c 0-{n_cpus-1}"
            " benchmark-jax-dot-product"
            f" --output={filename}"
            f" --number={number}"
            f" --repeat={repeat}"
            f" --shape={shape_str}",
            shell=True,
        )
    with open(filename) as f:
        return yaml.safe_load(f)


def get_benchmark_filename(n_cpus: int) -> Path:
    host_name = socket.gethostname()
    return Path(f"results/dot-product-{host_name}-{n_cpus}-cpus.yaml")


def get_figure_filename() -> str:
    host_name = socket.gethostname()
    return f"jax-benchmark-{host_name}.svg"


if __name__ == "__main__":
    raise SystemExit(main())
