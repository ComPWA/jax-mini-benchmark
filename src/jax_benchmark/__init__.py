from __future__ import annotations

import argparse
import os
import socket
import subprocess
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from jax_benchmark.io import mute_warnings


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
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show benchmark plot after benchmark run",
    )
    args = parser.parse_args()
    benchmarks = benchmark_cpu_range(
        repeat=args.repeat,
        number=args.number,
    )
    visualize(benchmarks, show=not args.no_show)
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


def visualize(benchmarks: dict[int, dict], show: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

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
    first_benchmark, *_ = benchmarks.values()
    jax_version = first_benchmark["machine_info"]["jax_version"]
    fig.suptitle(f"JAX v{jax_version} dot product on {hostname}")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Number of CPUs")
    ax.set_ylabel("Average time (s)")
    fig.savefig(filename)
    if show:
        plt.show()


def run_single_benchmark(
    n_cpus: int, number: int, repeat: int, shape: tuple[int, int]
) -> dict:
    filename = get_benchmark_filename(n_cpus)
    if not os.path.exists(filename):
        shape_str = "x".join(map(str, shape))
        filename.parent.mkdir(exist_ok=True, parents=True)
        subprocess.call(
            (
                f"taskset -c 0-{n_cpus-1}"
                " benchmark-jax-dot-product"
                f" --output={filename}"
                f" --number={number}"
                f" --repeat={repeat}"
                f" --shape={shape_str}"
            ),
            shell=True,
        )
    with open(filename) as f:
        return yaml.safe_load(f)


def get_benchmark_filename(n_cpus: int) -> Path:
    host_name = socket.gethostname()
    return Path(f"benchmark-results/{host_name}/dot-product-{n_cpus}-cpus.yaml")


def get_figure_filename() -> str:
    host_name = socket.gethostname()
    return f"jax-benchmark-{host_name}.svg"


if __name__ == "__main__":
    raise SystemExit(main())
